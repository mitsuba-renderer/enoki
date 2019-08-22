/*
    src/cuda/common.cpp -- CUDA backend (horizontal operations)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyrighe (c) 2019 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <vector>
#include "common.cuh"

NAMESPACE_BEGIN(enoki)

extern uint32_t cuda_log_level();

__global__ void arange(uint32_t n, uint32_t *out) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        out[i] = i;
}


ENOKI_EXPORT
size_t cuda_partition(size_t size, const void **ptrs_, void ***ptrs_unique_out,
                      uint32_t **counts_out, uint32_t ***perm_out) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_partition(size=" << size << ")" << std::endl;
#endif

    size_t     temp_size   = 0;
    void      *temp        = nullptr;

    uint32_t *perm         = (uint32_t *) cuda_malloc(size * sizeof(uint32_t)),
             *perm_sorted  = (uint32_t *) cuda_malloc(size * sizeof(uint32_t));

    uintptr_t *ptrs        = (uintptr_t *) ptrs_,
              *ptrs_sorted = (uintptr_t *) cuda_malloc(size * sizeof(uintptr_t));

    arange<<<256, 256>>>((uint32_t) size, perm);

    // Sort the key array
    cuda_check(cub::DeviceRadixSort::SortPairs(
        temp, temp_size, ptrs, ptrs_sorted, perm, perm_sorted, size));
    temp = cuda_malloc(temp_size);
    cuda_check_maybe_redo(cub::DeviceRadixSort::SortPairs(
        temp, temp_size, ptrs, ptrs_sorted, perm, perm_sorted, size));

    // Release memory that is no longer needed
    cuda_free(temp);
    cuda_free(perm);
    temp_size = 0; temp = nullptr;

    uintptr_t *ptrs_unique = (uintptr_t *) cuda_malloc(size * sizeof(uintptr_t));
    uint32_t *counts = (uint32_t *) cuda_malloc(size * sizeof(uint32_t));
    size_t *num_runs = (size_t *) cuda_malloc(sizeof(size_t));

    // RLE-encode the sorted pointer list
    cuda_check(cub::DeviceRunLengthEncode::Encode(
        temp, temp_size, ptrs_sorted, ptrs_unique, counts, num_runs, size));
    temp = cuda_malloc(temp_size);

    cuda_check_maybe_redo(cub::DeviceRunLengthEncode::Encode(
        temp, temp_size, ptrs_sorted, ptrs_unique, counts, num_runs, size));

    // Release memory that is no longer needed
    cuda_free(temp);
    cuda_free(ptrs_sorted);

    size_t num_runs_out = 0;
    cuda_check(cudaMemcpy(&num_runs_out, num_runs, sizeof(size_t), cudaMemcpyDeviceToHost));

    *ptrs_unique_out = (void **) malloc(sizeof(void *) * num_runs_out);
    *counts_out = (uint32_t *) malloc(sizeof(uint32_t) * num_runs_out);
    *perm_out = (uint32_t **) malloc(sizeof(uint32_t *) * num_runs_out);

    cuda_check(cudaMemcpy(*ptrs_unique_out, ptrs_unique, num_runs_out * sizeof(void *), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(*counts_out, counts, num_runs_out * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t *ptr = perm_sorted;
    for (size_t i = 0; i < num_runs_out; ++i) {
        size_t size = (*counts_out)[i];
        (*perm_out)[i] = (uint32_t *) cuda_malloc(size * sizeof(uint32_t));
        cuda_check(cudaMemcpyAsync((*perm_out)[i], ptr,
                                   size * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice));
        ptr += size;
    }

    cuda_free(num_runs);
    cuda_free(ptrs_unique);
    cuda_free(perm_sorted);
    cuda_free(counts);

    return num_runs_out;
}

template <typename T, std::enable_if_t<std::is_unsigned<T>::value, int> = 0>
void cuda_compress_impl(size_t size, const T *data, const bool *mask, T **out_data, size_t *out_size) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_compress(size=" << size << ")" << std::endl;
#endif

    size_t temp_size    = 0,
           *out_size_p  = nullptr;
    void *temp          = nullptr;

    T *result_p = nullptr;

    *out_data = (T *) cuda_malloc(size * sizeof(T));
    out_size_p = (size_t *) cuda_malloc(sizeof(size_t));

    cuda_check(cub::DeviceSelect::Flagged(temp, temp_size, data, mask, result_p, out_size_p, size));
    temp = cuda_malloc(temp_size);
    cuda_check_maybe_redo(cub::DeviceSelect::Flagged(temp, temp_size, data, mask, *out_data, out_size_p, size));
    cuda_check(cudaMemcpy(out_size, out_size_p, sizeof(size_t), cudaMemcpyDeviceToHost));
    cuda_free(temp);
    cuda_free(out_size_p);
}

template <typename T, std::enable_if_t<!std::is_unsigned<T>::value && sizeof(T) == 4, int> = 0>
void cuda_compress_impl(size_t size, const T *data, const bool *mask, T **out_data, size_t *out_size) {
    cuda_compress_impl(size, (const uint32_t *) data, mask, (uint32_t **) out_data, out_size);
}

template <typename T, std::enable_if_t<!std::is_unsigned<T>::value && sizeof(T) == 8, int> = 0>
void cuda_compress_impl(size_t size, const T *data, const bool *mask, T **out_data, size_t *out_size) {
    cuda_compress_impl(size, (const uint64_t *) data, mask, (uint64_t **) out_data, out_size);
}

template <typename T> void cuda_compress(size_t size, const T *data, const bool *mask, T **out_data, size_t *out_size) {
    cuda_compress_impl(size, data, mask, out_data, out_size);
}

template <typename T> T* cuda_hsum(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hsum(size=" << size << ")" << std::endl;
#endif

    size_t temp_size  = 0;
    void *temp        = nullptr;

    T *result_p = nullptr;

    cuda_check(cub::DeviceReduce::Sum(temp, temp_size, data, result_p, size));
    temp = cuda_malloc(temp_size);
    result_p = (T *) cuda_malloc(sizeof(T));
    cuda_check_maybe_redo(cub::DeviceReduce::Sum(temp, temp_size, data, result_p, size));
    cuda_free(temp);

    return result_p;
}

struct ReductionOpMul {
    template <typename T>
    __device__ __forceinline__
    T operator()(T a, T b) const {
        return a * b;
    }
};

template <typename T> T* cuda_hprod(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hprod(size=" << size << ")" << std::endl;
#endif

    size_t temp_size = 0;
    void *temp       = nullptr;
    T *result_p      = nullptr;

    ReductionOpMul mul_op;
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         mul_op, T(1)));
    temp = cuda_malloc(temp_size);
    result_p = (T *) cuda_malloc(sizeof(T));
    cuda_check_maybe_redo(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                                    mul_op, T(1)));
    cuda_free(temp);

    return result_p;
}

template <typename T> T* cuda_hmax(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hmax(size=" << size << ")" << std::endl;
#endif

    size_t temp_size = 0;
    void *temp       = nullptr;
    T *result_p      = nullptr;

    cuda_check(cub::DeviceReduce::Max(temp, temp_size, data, result_p, size));
    temp = cuda_malloc(temp_size);
    result_p = (T *) cuda_malloc(sizeof(T));
    cuda_check_maybe_redo(cub::DeviceReduce::Max(temp, temp_size, data, result_p, size));
    cuda_free(temp);

    return result_p;
}

template <typename T> T* cuda_hmin(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hmin(size=" << size << ")" << std::endl;
#endif

    size_t temp_size = 0;
    void *temp       = nullptr;
    T *result_p      = nullptr;

    cuda_check(cub::DeviceReduce::Min(temp, temp_size, data, result_p, size));
    temp = cuda_malloc(temp_size);
    result_p = (T *) cuda_malloc(sizeof(T));
    cuda_check_maybe_redo(cub::DeviceReduce::Min(temp, temp_size, data, result_p, size));
    cuda_free(temp);

    return result_p;
}

struct ReductionOpAll {
    __device__ __forceinline__
    bool operator()(bool a, bool b) const {
        return a && b;
    }
};

struct ReductionOpAny {
    __device__ __forceinline__
    bool operator()(bool a, bool b) const {
        return a || b;
    }
};

ENOKI_EXPORT bool cuda_all(size_t size, const bool *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_all(size=" << size << ")" << std::endl;
#endif

    size_t temp_size  = 0;
    void *temp        = nullptr;

    bool result = false, *result_p = nullptr;

    ReductionOpAll all_op;
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         all_op, true));
    temp = cuda_malloc(temp_size);
    result_p = (bool *) cuda_malloc(sizeof(bool));
    cuda_check_maybe_redo(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                                    all_op, true));
    cuda_free(temp);
    cuda_check(cudaMemcpy(&result, result_p, sizeof(bool), cudaMemcpyDeviceToHost));
    cuda_free(result_p);

    return result;
}

ENOKI_EXPORT bool cuda_any(size_t size, const bool *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_any(size=" << size << ")" << std::endl;
#endif

    size_t temp_size   = 0;
    void *temp        = nullptr;

    bool result = false, *result_p = nullptr;

    ReductionOpAny any_op;
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         any_op, false));
    temp = cuda_malloc(temp_size);
    result_p = (bool *) cuda_malloc(sizeof(bool));
    cuda_check_maybe_redo(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                                    any_op, false));
    cuda_free(temp);
    cuda_check(cudaMemcpy(&result, result_p, sizeof(bool), cudaMemcpyDeviceToHost));
    cuda_free(result_p);

    return result;
}

ENOKI_EXPORT size_t cuda_count(size_t size, const bool *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_count(size=" << size << ")" << std::endl;
#endif

    size_t temp_size  = 0;
    void *temp        = nullptr;

    size_t result = 0, *result_p = nullptr;

    cuda_check(cub::DeviceReduce::Sum(temp, temp_size, data, result_p, size));
    temp = cuda_malloc(temp_size);
    result_p = (size_t *) cuda_malloc(sizeof(size_t));
    cuda_check_maybe_redo(cub::DeviceReduce::Sum(temp, temp_size, data, result_p, size));
    cuda_free(temp);
    cuda_check(cudaMemcpy(&result, result_p, sizeof(size_t), cudaMemcpyDeviceToHost));
    cuda_free(result_p);

    return result;
}

template ENOKI_EXPORT int32_t*  cuda_hsum(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t* cuda_hsum(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t*  cuda_hsum(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t* cuda_hsum(size_t, const uint64_t *);
template ENOKI_EXPORT float*    cuda_hsum(size_t, const float *);
template ENOKI_EXPORT double*   cuda_hsum(size_t, const double *);

template ENOKI_EXPORT int32_t*  cuda_hprod(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t* cuda_hprod(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t*  cuda_hprod(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t* cuda_hprod(size_t, const uint64_t *);
template ENOKI_EXPORT float*    cuda_hprod(size_t, const float *);
template ENOKI_EXPORT double*   cuda_hprod(size_t, const double *);

template ENOKI_EXPORT int32_t*  cuda_hmax(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t* cuda_hmax(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t*  cuda_hmax(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t* cuda_hmax(size_t, const uint64_t *);
template ENOKI_EXPORT float*    cuda_hmax(size_t, const float *);
template ENOKI_EXPORT double*   cuda_hmax(size_t, const double *);

template ENOKI_EXPORT int32_t*  cuda_hmin(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t* cuda_hmin(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t*  cuda_hmin(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t* cuda_hmin(size_t, const uint64_t *);
template ENOKI_EXPORT float*    cuda_hmin(size_t, const float *);
template ENOKI_EXPORT double*   cuda_hmin(size_t, const double *);

template ENOKI_EXPORT void cuda_compress(size_t, const bool *,     const bool *mask, bool     **out_ptr, size_t *out_size);
template ENOKI_EXPORT void cuda_compress(size_t, const int32_t *,  const bool *mask, int32_t  **out_ptr, size_t *out_size);
template ENOKI_EXPORT void cuda_compress(size_t, const uint32_t *, const bool *mask, uint32_t **out_ptr, size_t *out_size);
template ENOKI_EXPORT void cuda_compress(size_t, const int64_t *,  const bool *mask, int64_t  **out_ptr, size_t *out_size);
template ENOKI_EXPORT void cuda_compress(size_t, const uint64_t *, const bool *mask, uint64_t **out_ptr, size_t *out_size);
template ENOKI_EXPORT void cuda_compress(size_t, const float *,    const bool *mask, float    **out_ptr, size_t *out_size);
template ENOKI_EXPORT void cuda_compress(size_t, const double *,   const bool *mask, double   **out_ptr, size_t *out_size);

NAMESPACE_END(enoki)
