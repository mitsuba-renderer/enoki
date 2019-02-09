#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif
#
#if defined(_MSC_VER)
#  define ENOKI_EXPORT                 __declspec(dllexport)
#  define ENOKI_LIKELY(x)              x
#  define ENOKI_UNLIKELY(x)            x
#else
#  define ENOKI_EXPORT                 __attribute__ ((visibility("default")))
#  define ENOKI_LIKELY(x)              __builtin_expect(!!(x), 1)
#  define ENOKI_UNLIKELY(x)            __builtin_expect(!!(x), 0)
#endif

NAMESPACE_BEGIN(enoki)

enum EnokiType { Invalid = 0, Int8, UInt8, Int16, UInt16,
                 Int32, UInt32, Int64, UInt64, Float16,
                 Float32, Float64, Bool, Pointer };


struct CUDAErrorList {
    CUresult id;
    const char *value;
};

static CUDAErrorList __cuda_error_list[] = {
    { CUDA_SUCCESS,
     "CUDA_SUCCESS"},
    { CUDA_ERROR_INVALID_VALUE,
     "CUDA_ERROR_INVALID_VALUE"},
    { CUDA_ERROR_OUT_OF_MEMORY,
     "CUDA_ERROR_OUT_OF_MEMORY"},
    { CUDA_ERROR_NOT_INITIALIZED,
     "CUDA_ERROR_NOT_INITIALIZED"},
    { CUDA_ERROR_DEINITIALIZED,
     "CUDA_ERROR_DEINITIALIZED"},
    { CUDA_ERROR_PROFILER_DISABLED,
     "CUDA_ERROR_PROFILER_DISABLED"},
    { CUDA_ERROR_PROFILER_NOT_INITIALIZED,
     "CUDA_ERROR_PROFILER_NOT_INITIALIZED"},
    { CUDA_ERROR_PROFILER_ALREADY_STARTED,
     "CUDA_ERROR_PROFILER_ALREADY_STARTED"},
    { CUDA_ERROR_PROFILER_ALREADY_STOPPED,
     "CUDA_ERROR_PROFILER_ALREADY_STOPPED"},
    { CUDA_ERROR_NO_DEVICE,
     "CUDA_ERROR_NO_DEVICE"},
    { CUDA_ERROR_INVALID_DEVICE,
     "CUDA_ERROR_INVALID_DEVICE"},
    { CUDA_ERROR_INVALID_IMAGE,
     "CUDA_ERROR_INVALID_IMAGE"},
    { CUDA_ERROR_INVALID_CONTEXT,
     "CUDA_ERROR_INVALID_CONTEXT"},
    { CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
     "CUDA_ERROR_CONTEXT_ALREADY_CURRENT"},
    { CUDA_ERROR_MAP_FAILED,
     "CUDA_ERROR_MAP_FAILED"},
    { CUDA_ERROR_UNMAP_FAILED,
     "CUDA_ERROR_UNMAP_FAILED"},
    { CUDA_ERROR_ARRAY_IS_MAPPED,
     "CUDA_ERROR_ARRAY_IS_MAPPED"},
    { CUDA_ERROR_ALREADY_MAPPED,
     "CUDA_ERROR_ALREADY_MAPPED"},
    { CUDA_ERROR_NO_BINARY_FOR_GPU,
     "CUDA_ERROR_NO_BINARY_FOR_GPU"},
    { CUDA_ERROR_ALREADY_ACQUIRED,
     "CUDA_ERROR_ALREADY_ACQUIRED"},
    { CUDA_ERROR_NOT_MAPPED,
     "CUDA_ERROR_NOT_MAPPED"},
    { CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
     "CUDA_ERROR_NOT_MAPPED_AS_ARRAY"},
    { CUDA_ERROR_NOT_MAPPED_AS_POINTER,
     "CUDA_ERROR_NOT_MAPPED_AS_POINTER"},
    { CUDA_ERROR_ECC_UNCORRECTABLE,
     "CUDA_ERROR_ECC_UNCORRECTABLE"},
    { CUDA_ERROR_UNSUPPORTED_LIMIT,
     "CUDA_ERROR_UNSUPPORTED_LIMIT"},
    { CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
     "CUDA_ERROR_CONTEXT_ALREADY_IN_USE"},
    { CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
     "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"},
    { CUDA_ERROR_INVALID_PTX,
     "CUDA_ERROR_INVALID_PTX"},
    { CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
     "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT"},
    { CUDA_ERROR_NVLINK_UNCORRECTABLE,
     "CUDA_ERROR_NVLINK_UNCORRECTABLE"},
    { CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
     "CUDA_ERROR_JIT_COMPILER_NOT_FOUND"},
    { CUDA_ERROR_INVALID_SOURCE,
     "CUDA_ERROR_INVALID_SOURCE"},
    { CUDA_ERROR_FILE_NOT_FOUND,
     "CUDA_ERROR_FILE_NOT_FOUND"},
    { CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
     "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"},
    { CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
     "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"},
    { CUDA_ERROR_OPERATING_SYSTEM,
     "CUDA_ERROR_OPERATING_SYSTEM"},
    { CUDA_ERROR_INVALID_HANDLE,
     "CUDA_ERROR_INVALID_HANDLE"},
    { CUDA_ERROR_NOT_FOUND,
     "CUDA_ERROR_NOT_FOUND"},
    { CUDA_ERROR_NOT_READY,
     "CUDA_ERROR_NOT_READY"},
    { CUDA_ERROR_ILLEGAL_ADDRESS,
     "CUDA_ERROR_ILLEGAL_ADDRESS"},
    { CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
     "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"},
    { CUDA_ERROR_LAUNCH_TIMEOUT,
     "CUDA_ERROR_LAUNCH_TIMEOUT"},
    { CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"},
    { CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
     "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"},
    { CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
     "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"},
    { CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
     "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"},
    { CUDA_ERROR_CONTEXT_IS_DESTROYED,
     "CUDA_ERROR_CONTEXT_IS_DESTROYED"},
    { CUDA_ERROR_ASSERT,
     "CUDA_ERROR_ASSERT"},
    { CUDA_ERROR_TOO_MANY_PEERS,
     "CUDA_ERROR_TOO_MANY_PEERS"},
    { CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
     "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"},
    { CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
     "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"},
    { CUDA_ERROR_HARDWARE_STACK_ERROR,
     "CUDA_ERROR_HARDWARE_STACK_ERROR"},
    { CUDA_ERROR_ILLEGAL_INSTRUCTION,
     "CUDA_ERROR_ILLEGAL_INSTRUCTION"},
    { CUDA_ERROR_MISALIGNED_ADDRESS,
     "CUDA_ERROR_MISALIGNED_ADDRESS"},
    { CUDA_ERROR_INVALID_ADDRESS_SPACE,
     "CUDA_ERROR_INVALID_ADDRESS_SPACE"},
    { CUDA_ERROR_INVALID_PC,
     "CUDA_ERROR_INVALID_PC"},
    { CUDA_ERROR_LAUNCH_FAILED,
     "CUDA_ERROR_LAUNCH_FAILED"},
    { CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
     "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE"},
    { CUDA_ERROR_NOT_PERMITTED,
     "CUDA_ERROR_NOT_PERMITTED"},
    { CUDA_ERROR_NOT_SUPPORTED,
     "CUDA_ERROR_NOT_SUPPORTED"},
    { CUDA_ERROR_UNKNOWN,
     "CUDA_ERROR_UNKNOWN"},
    { (CUresult) -1, nullptr }
};

inline const char *getCudaDrvErrorString(CUresult id) {
    int index = 0;

    while (__cuda_error_list[index].id != id &&
           __cuda_error_list[index].id != (CUresult) -1)
        index++;

    if (__cuda_error_list[index].id == id)
        return __cuda_error_list[index].value;
    else
        return "Invalid CUDA error status!";
}

#define cuda_check(err)  __cuda_check (err, __FILE__, __LINE__)

inline void __cuda_check(CUresult errval, const char *file, const int line) {
    if (errval != CUDA_SUCCESS) {
        const char *err_msg = getCudaDrvErrorString(errval);
        fprintf(stderr,
                "cuda_check(): driver API error = %04d \"%s\" in "
                "%s:%i.\n", (int) errval, err_msg, file, line);
        exit(EXIT_FAILURE);
    }
}

inline void __cuda_check(cudaError_t errval, const char *file, const int line) {
    if (errval != cudaSuccess) {
        const char *err_msg = cudaGetErrorName(errval);
        fprintf(stderr,
                "cuda_check(): runtime API error = %04d \"%s\" in "
                "%s:%i.\n", (int) errval, err_msg, file, line);
        exit(EXIT_FAILURE);
    }
}

inline uint32_t next_power_of_two(uint32_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

extern void* cuda_malloc(size_t size);
extern void cuda_free(void *ptr);

NAMESPACE_END(enoki)
