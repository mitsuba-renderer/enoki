#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_reduce.cuh>
#include <vector>
#include "common.cuh"

NAMESPACE_BEGIN(enoki)

__global__ void arange(size_t n, size_t *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        out[i] = i;
}

ENOKI_EXPORT
std::pair<std::vector<std::pair<void *, size_t>>, size_t *>
cuda_partition(size_t size, const void **ptrs_) {
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

    uintptr_t *unique = nullptr;
    size_t *counts = nullptr,
           *num_runs = nullptr;

    cuda_check(cudaMallocManaged(&num_runs, sizeof(size_t)));
    cuda_check(cudaMallocManaged(&unique, size*sizeof(uintptr_t)));
    cuda_check(cudaMallocManaged(&counts, size*sizeof(size_t)));

    // RLE-encode the sorted pointer list
    cuda_check(cub::DeviceRunLengthEncode::Encode(
        temp, temp_size, ptrs_sorted, unique, counts, num_runs, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cub::DeviceRunLengthEncode::Encode(
        temp, temp_size, ptrs_sorted, unique, counts, num_runs, size));

    // Release memory that is no longer needed
    cuda_check(cudaFree(temp));
    cuda_check(cudaFree(ptrs_sorted));

    std::vector<std::pair<void *, size_t>> result(*num_runs);
    for (size_t i = 0; i < *num_runs; ++i)
        result[i] = std::make_pair((void *) unique[i], counts[i]);

    cuda_check(cudaFree(num_runs));
    cuda_check(cudaFree(unique));
    cuda_check(cudaFree(counts));

    return std::pair<std::vector<std::pair<void *, size_t>>, size_t *>(
        std::move(result),
        perm_sorted
    );
}

template <typename T> T cuda_hsum(size_t size, const T *data) {
    size_t temp_size   = 0;
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

    return result;
}

template <typename T> T cuda_hmax(size_t size, const T *data) {
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

template <typename T> T cuda_all(size_t size, const bool *data) {
    size_t temp_size   = 0;
    void *temp        = nullptr;

    bool result = false, *result_p = nullptr;

    ReductionOpAll all_op;
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         all_op, true));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(T)));
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         all_op, true));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(T),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));

    return result;
}

template <typename T> T cuda_any(size_t size, const bool *data) {
    size_t temp_size   = 0;
    void *temp        = nullptr;

    bool result = false, *result_p = nullptr;

    ReductionOpAny any_op;
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         any_op, false));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cudaMalloc(&result_p, sizeof(T)));
    cuda_check(cub::DeviceReduce::Reduce(temp, temp_size, data, result_p, size,
                                         any_op, false));
    cuda_check(cudaFree(temp));
    cuda_check(cudaMemcpy(&result, result_p, sizeof(T),
               cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(result_p));

    return result;
}

template ENOKI_EXPORT int32_t  cuda_hsum(size_t, const int32_t *);
template ENOKI_EXPORT uint32_t cuda_hsum(size_t, const uint32_t *);
template ENOKI_EXPORT int64_t  cuda_hsum(size_t, const int64_t *);
template ENOKI_EXPORT uint64_t cuda_hsum(size_t, const uint64_t *);
template ENOKI_EXPORT float    cuda_hsum(size_t, const float *);
template ENOKI_EXPORT double   cuda_hsum(size_t, const double *);

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

template ENOKI_EXPORT bool     cuda_all(size_t, const bool *);
template ENOKI_EXPORT bool     cuda_any(size_t, const bool *);

NAMESPACE_END(enoki)
