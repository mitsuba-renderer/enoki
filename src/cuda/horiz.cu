/*
    src/cuda/common.cpp -- CUDA backend (horizontal operations)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyrighe (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

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

__global__ void arange(size_t n, size_t *out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        out[i] = i;
}

ENOKI_EXPORT
std::pair<std::vector<std::pair<void *, size_t>>, size_t *>
cuda_partition(size_t size, const void **ptrs_) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_partition(size=" << size << ")" << std::endl;
#endif

    size_t     temp_size   = 0,
              *perm        = nullptr,
              *perm_sorted = nullptr;
    uintptr_t *ptrs        = (uintptr_t *) ptrs_,
              *ptrs_sorted = nullptr;
    void      *temp        = nullptr;


    cuda_check(cudaMalloc(&perm, size * sizeof(size_t)));
    arange<<<256, 256>>>(size, perm);

    cuda_check(cudaMalloc(&perm_sorted, size * sizeof(size_t)));
    cuda_check(cudaMalloc(&ptrs_sorted, size * sizeof(uintptr_t)));

    // Sort the key array
    cuda_check(cub::DeviceRadixSort::SortPairs(temp, temp_size, ptrs, ptrs_sorted,
                                               perm, perm_sorted, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cub::DeviceRadixSort::SortPairs(temp, temp_size, ptrs, ptrs_sorted,
                                               perm, perm_sorted, size));

    // Release memory that is no longer needed
    cuda_check(cudaFree(temp));
    cuda_check(cudaFree(perm));
    temp_size = 0; temp = nullptr;

    uintptr_t *unique_p = nullptr;
    size_t *counts_p = nullptr,
           *num_runs_p = nullptr;

    cuda_check(cudaMalloc(&num_runs_p, sizeof(size_t)));
    cuda_check(cudaMalloc(&unique_p, size*sizeof(uintptr_t)));
    cuda_check(cudaMalloc(&counts_p, size*sizeof(size_t)));

    // RLE-encode the sorted pointer list
    cuda_check(cub::DeviceRunLengthEncode::Encode(
        temp, temp_size, ptrs_sorted, unique_p, counts_p, num_runs_p, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cub::DeviceRunLengthEncode::Encode(
        temp, temp_size, ptrs_sorted, unique_p, counts_p, num_runs_p, size));

    // Release memory that is no longer needed
    cuda_check(cudaFree(temp));
    cuda_check(cudaFree(ptrs_sorted));

    size_t num_runs = 0;
    cuda_check(cudaMemcpy(&num_runs, num_runs_p, sizeof(size_t), cudaMemcpyDeviceToHost));

    uintptr_t *unique = new uintptr_t[num_runs];
    size_t *counts = new size_t[num_runs];
    cuda_check(cudaMemcpy(&unique, unique_p, sizeof(size_t), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&counts, counts_p, sizeof(uintptr_t), cudaMemcpyDeviceToHost));

    std::vector<std::pair<void *, size_t>> result(num_runs);
    for (size_t i = 0; i < num_runs; ++i)
        result[i] = std::make_pair((void *) unique[i], counts[i]);

    cuda_check(cudaFree(num_runs_p));
    cuda_check(cudaFree(unique_p));
    cuda_check(cudaFree(counts_p));
    cuda_sync();
    delete[] unique;
    delete[] counts;

    return std::pair<std::vector<std::pair<void *, size_t>>, size_t *>(
        std::move(result),
        perm_sorted
    );
}

template <typename T, std::enable_if_t<std::is_unsigned<T>::value, int> = 0>
std::pair<T *, size_t> cuda_compress_impl(size_t size, const T *data, const bool *mask) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_compress(size=" << size << ")" << std::endl;
#endif

    size_t temp_size    = 0,
           out_size     = 0,
           *out_size_p  = nullptr;
    void *temp          = nullptr;

    T *result_p = nullptr;

    cuda_check(cub::DeviceSelect::Flagged(temp, temp_size, data, mask, result_p, out_size_p, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, size * sizeof(T)));
    cuda_check(cudaMalloc(&out_size_p, sizeof(size_t)));
    cuda_check(cub::DeviceSelect::Flagged(temp, temp_size, data, mask, result_p, out_size_p, size));
    cuda_check(cudaMemcpy(&out_size, out_size_p, sizeof(size_t), cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(temp));
    cuda_check(cudaFree(out_size_p));
    cuda_sync();

    return std::make_pair(result_p, out_size);
}

template <typename T, std::enable_if_t<!std::is_unsigned<T>::value && sizeof(T) == 4, int> = 0>
std::pair<T *, size_t> cuda_compress_impl(size_t size, const T *data, const bool *mask) {
    auto result = cuda_compress_impl(size, (const uint32_t *) data, mask);
    return std::make_pair((T *) result.first, result.second);
}

template <typename T, std::enable_if_t<!std::is_unsigned<T>::value && sizeof(T) == 8, int> = 0>
std::pair<T *, size_t> cuda_compress_impl(size_t size, const T *data, const bool *mask) {
    auto result = cuda_compress_impl(size, (const uint32_t *) data, mask);
    return std::make_pair((T *) result.first, result.second);
}

template <typename T> std::pair<T *, size_t> cuda_compress(size_t size, const T *data, const bool *mask) {
    return cuda_compress_impl(size, data, mask);
}


template <typename T> T cuda_hsum(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hsum(size=" << size << ")" << std::endl;
#endif

    size_t temp_size  = 0;
    void *temp        = nullptr;

    T result = 0, *result_p = nullptr;

    cuda_check(cub::DeviceReduce::Sum(temp, temp_size, data, result_p, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(T)));
    cuda_check(cub::DeviceReduce::Sum(temp, temp_size, data, result_p, size));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(T),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));
    cuda_sync();

    return result;
}

struct ReductionOpMul {
    template <typename T>
    __device__ __forceinline__
    T operator()(T a, T b) const {
        return a * b;
    }
};

template <typename T> T cuda_hprod(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hprod(size=" << size << ")" << std::endl;
#endif

    size_t temp_size  = 0;
    void *temp        = nullptr;

    T result = T(0), *result_p = nullptr;

    ReductionOpMul mul_op;
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         mul_op, T(1)));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(T)));
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         mul_op, T(1)));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(T),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));
    cuda_sync();

    return result;
}

template <typename T> T cuda_hmax(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hmax(size=" << size << ")" << std::endl;
#endif

    size_t temp_size   = 0;
    void *temp        = nullptr;

    T result = 0, *result_p = nullptr;

    cuda_check(cub::DeviceReduce::Max(temp, temp_size, data, result_p, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(T)));
    cuda_check(cub::DeviceReduce::Max(temp, temp_size, data, result_p, size));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(T),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));

    return result;
}

template <typename T> T cuda_hmin(size_t size, const T *data) {
#if !defined(NDEBUG)
    if (cuda_log_level() >= 4)
        std::cerr << "cuda_hmin(size=" << size << ")" << std::endl;
#endif

    size_t temp_size   = 0;
    void *temp        = nullptr;

    T result = 0, *result_p = nullptr;

    cuda_check(cub::DeviceReduce::Min(temp, temp_size, data, result_p, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(T)));
    cuda_check(cub::DeviceReduce::Min(temp, temp_size, data, result_p, size));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(T),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));
    cuda_sync();

    return result;
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

bool cuda_all(size_t size, const bool *data) {
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
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(bool)));
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         all_op, true));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(bool),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));
    cuda_sync();

    return result;
}

bool cuda_any(size_t size, const bool *data) {
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
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(bool)));
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         any_op, false));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(bool),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));
    cuda_sync();

    return result;
}

template ENOKI_EXPORT int32_t  cuda_hsum(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t cuda_hsum(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t  cuda_hsum(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t cuda_hsum(size_t, const uint64_t *);
template ENOKI_EXPORT float    cuda_hsum(size_t, const float *);
template ENOKI_EXPORT double   cuda_hsum(size_t, const double *);

template ENOKI_EXPORT int32_t  cuda_hprod(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t cuda_hprod(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t  cuda_hprod(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t cuda_hprod(size_t, const uint64_t *);
template ENOKI_EXPORT float    cuda_hprod(size_t, const float *);
template ENOKI_EXPORT double   cuda_hprod(size_t, const double *);

template ENOKI_EXPORT int32_t  cuda_hmax(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t cuda_hmax(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t  cuda_hmax(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t cuda_hmax(size_t, const uint64_t *);
template ENOKI_EXPORT float    cuda_hmax(size_t, const float *);
template ENOKI_EXPORT double   cuda_hmax(size_t, const double *);

template ENOKI_EXPORT int32_t  cuda_hmin(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t cuda_hmin(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t  cuda_hmin(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t cuda_hmin(size_t, const uint64_t *);
template ENOKI_EXPORT float    cuda_hmin(size_t, const float *);
template ENOKI_EXPORT double   cuda_hmin(size_t, const double *);

template ENOKI_EXPORT std::pair<bool *,     size_t> cuda_compress(size_t, const bool *,     const bool *mask);
template ENOKI_EXPORT std::pair<int32_t *,  size_t> cuda_compress(size_t, const int32_t *,  const bool *mask);
template ENOKI_EXPORT std::pair<uint32_t *, size_t> cuda_compress(size_t, const uint32_t *, const bool *mask);
template ENOKI_EXPORT std::pair<int64_t *,  size_t> cuda_compress(size_t, const int64_t *,  const bool *mask);
template ENOKI_EXPORT std::pair<uint64_t *, size_t> cuda_compress(size_t, const uint64_t *, const bool *mask);
template ENOKI_EXPORT std::pair<float *,    size_t> cuda_compress(size_t, const float *,    const bool *mask);
template ENOKI_EXPORT std::pair<double *,   size_t> cuda_compress(size_t, const double *,   const bool *mask);

NAMESPACE_END(enoki)
