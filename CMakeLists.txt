cmake_minimum_required (VERSION 2.8.12)

if (POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # MACOSX_RPATH is enabled by default
endif()

set(CMAKE_MACOSX_RPATH ON)

option(ENOKI_CUDA     "Build Enoki CUDA library?" OFF)
option(ENOKI_AUTODIFF "Build Enoki automatic differentation library?" OFF)
option(ENOKI_PYTHON   "Build pybind11 interface to CUDA & automatic differntiation libraries?" OFF)

if (ENOKI_CUDA)
  project(enoki CXX CUDA)
  add_definitions(-DENOKI_BUILD_CUDA=1)
else()
  project(enoki CXX)
endif()

set(ENOKI_MASTER_PROJECT OFF)
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  set(ENOKI_MASTER_PROJECT ON)
endif()

option(ENOKI_TEST "Build Enoki test suite?" ${ENOKI_MASTER_PROJECT})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" ENOKI_U_CMAKE_BUILD_TYPE)

set(ENOKI_CXXFLAGS_BACKUP ${CMAKE_CXX_FLAGS})
set(ENOKI_NATIVE_FLAGS "")

if (MSVC)
  set(ENOKI_ARCHFLAGS_FILE archflags_win32.cpp)
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(WARNING "Enoki does not support vectorization on 32-bit Windows due to various")
    message(WARNING "platform limitations (unaligned stack, calling conventions don't allow")
    message(WARNING "passing vector registers, etc.). Switching to scalar mode.")
  endif()
else()
  set(ENOKI_ARCHFLAGS_FILE archflags_unix.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Clang refuses -march=native on ARM machines
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
  set(ENOKI_ARCHFLAGS -march=armv7-a -mtune=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee)
  set(ENOKI_NATIVE_FLAGS ${ENOKI_ARCHFLAGS})
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(ENOKI_ARCHFLAGS -march=armv8-a+simd -mtune=cortex-a53)
  set(ENOKI_NATIVE_FLAGS ${ENOKI_ARCHFLAGS})
elseif (CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$")
  # Emscripten
else()
  try_run(
    ENOKI_ARCHFLAGS_RETVAL ENOKI_ARCHFLAGS_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/${ENOKI_ARCHFLAGS_FILE}
    COMPILE_OUTPUT_VARIABLE ENOKI_ARCHFLAGS_MSG
    RUN_OUTPUT_VARIABLE ENOKI_ARCHFLAGS
  )

  if (NOT ${ENOKI_ARCHFLAGS_COMPILE_RESULT})
    message(FATAL_ERROR "Failed to compile 'archflags' binary: ${ENOKI_ARCHFLAGS_MSG}")
  endif()
endif()

set(CMAKE_CXX_FLAGS ${ENOKI_CXXFLAGS_BACKUP})

# Optimize for current architecture
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if (NOT ENOKI_NATIVE_FLAGS)
    set(ENOKI_NATIVE_FLAGS -march=native)
  endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(ENOKI_NATIVE_FLAGS -xHost)
elseif (MSVC)
  # MSVC doesn't have a flag equivalent to -march=native
  # The 'archflags' executable provides this functionality
  set(ENOKI_NATIVE_FLAGS ${ENOKI_ARCHFLAGS})
endif()

if (NOT ENOKI_MASTER_PROJECT)
  set(ENOKI_NATIVE_FLAGS ${ENOKI_NATIVE_FLAGS} PARENT_SCOPE)
endif()

set(ENOKI_HOST "INTEL")
set(ENOKI_TEST_SSE42 OFF)
set(ENOKI_TEST_AVX OFF)
set(ENOKI_TEST_AVX2 OFF)
set(ENOKI_TEST_KNL OFF)
set(ENOKI_TEST_SKX OFF)
set(ENOKI_TEST_ARM OFF)

string(TOUPPER "${ENOKI_ARCHFLAGS}" ENOKI_ARCHFLAGS)

if (MSVC OR ENOKI_ARCHFLAGS MATCHES "SSE")
  set(ENOKI_TEST_SSE42 ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "AVX")
  set(ENOKI_TEST_SSE42 ON)
  set(ENOKI_TEST_AVX ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "AVX2")
  set(ENOKI_TEST_SSE42 ON)
  set(ENOKI_TEST_AVX ON)
  set(ENOKI_TEST_AVX2 ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "KNL")
  set(ENOKI_TEST_SSE42 ON)
  set(ENOKI_TEST_AVX ON)
  set(ENOKI_TEST_AVX2 ON)
  set(ENOKI_TEST_KNL ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "SKX")
  set(ENOKI_TEST_SSE42 ON)
  set(ENOKI_TEST_AVX ON)
  set(ENOKI_TEST_AVX2 ON)
  set(ENOKI_TEST_SKX ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "ARMV")
  set(ENOKI_HOST "ARM")
  set(ENOKI_TEST_ARM ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/resources")

macro(enoki_set_native_flags)
  add_compile_options(${ENOKI_NATIVE_FLAGS})
  string(REPLACE ";" " " ENOKI_NATIVE_FLAGS_STR "${ENOKI_NATIVE_FLAGS}")
  # Some linkers want to know the architecture flags (for LTO)
  if (NOT MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ENOKI_NATIVE_FLAGS_STR}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ENOKI_NATIVE_FLAGS_STR}")
  endif()
endmacro()

macro(enoki_set_compile_flags)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
      CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
      CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" ENOKI_U_CMAKE_BUILD_TYPE)
    if (NOT (${ENOKI_U_CMAKE_BUILD_TYPE} MATCHES "DEB"))
      if (NOT (${CMAKE_CXX_FLAGS} MATCHES "fsanitize"))
        # Don't use stack security features in release mode
        add_compile_options(-fno-stack-protector)

        # In release mode, don't keep the frame pointer in a dedicated register unless needed
        add_compile_options(-fomit-frame-pointer)
      endif()
    endif()

    # Never update the 'errno' variable due to arithmetic exceptions
    add_compile_options(-fno-math-errno)

    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      # Look for opportunities to fuse additions & multiplications into FMA operations
      add_compile_options(-ffp-contract=fast)
    endif()
  endif()

  # Disable overly aggressive FP optimization in the Intel compiler
  if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(-fp-model precise)
  endif()

  if (MSVC)
    # Disable buffer security check cookie
    set(Configurations RELEASE RELWITHDEBINFO MINSIZEREL)
    foreach(Configuration ${Configurations})
      string(REPLACE "/GS"  "" CMAKE_CXX_FLAGS_${Configuration} "${CMAKE_CXX_FLAGS_${Configuration}}")
    endforeach()
    string(REPLACE "/GS"  "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_compile_options("$<$<NOT:$<CONFIG:Debug>>:/GS->")

    # Enable intrinsic functions
    add_compile_options("$<$<CONFIG:Release>:/Oi>")

    # Honor __forceinline statements even in debug mode, needed to avoid internal compiler errors on MSVC
    string(REPLACE "/Ob0" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    add_compile_options("$<$<CONFIG:Debug>:/Ob1>")
  endif()
endmacro()

include_directories(include)

set(ENOKI_HEADERS
    ${PROJECT_SOURCE_DIR}/include/enoki/array.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_avx.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_avx2.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_avx512.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_base.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_call.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_fallbacks.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_generic.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_idiv.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_intrin.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_kmask.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_masked.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_macro.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_math.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_recursive.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_round.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_router.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_sse42.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_static.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_struct.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_traits.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_utils.h
    ${PROJECT_SOURCE_DIR}/include/enoki/autodiff.h
    ${PROJECT_SOURCE_DIR}/include/enoki/color.h
    ${PROJECT_SOURCE_DIR}/include/enoki/complex.h
    ${PROJECT_SOURCE_DIR}/include/enoki/dynamic.h
    ${PROJECT_SOURCE_DIR}/include/enoki/fwd.h
    ${PROJECT_SOURCE_DIR}/include/enoki/half.h
    ${PROJECT_SOURCE_DIR}/include/enoki/matrix.h
    ${PROJECT_SOURCE_DIR}/include/enoki/morton.h
    ${PROJECT_SOURCE_DIR}/include/enoki/python.h
    ${PROJECT_SOURCE_DIR}/include/enoki/quaternion.h
    ${PROJECT_SOURCE_DIR}/include/enoki/random.h
    ${PROJECT_SOURCE_DIR}/include/enoki/sh.h
    ${PROJECT_SOURCE_DIR}/include/enoki/special.h
    ${PROJECT_SOURCE_DIR}/include/enoki/stl.h
    ${PROJECT_SOURCE_DIR}/include/enoki/transform.h
)

if (ENOKI_TEST)
  enable_testing()
  add_subdirectory(tests)
endif()

add_definitions(-DENOKI_BUILD=1)

if (ENOKI_CUDA)
  include_directories(ext/cub)
  string(APPEND CMAKE_CUDA_FLAGS "-gencode arch=compute_75,code=sm_75")
  set(CMAKE_SHARED_LINKER_FLAGS_BACKUP "${CMAKE_SHARED_LINKER_FLAGS}")
  string(REPLACE "-stdlib=libc++" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  add_library(enoki-cuda SHARED
      src/cuda/common.cuh
      src/cuda/common.cu
      src/cuda/horiz.cu
      src/cuda/jit.cu
  )
endif()

enoki_set_native_flags()
enoki_set_compile_flags()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
elseif (MSVC)
  add_compile_options("/std:c++17")
endif()

if (ENOKI_AUTODIFF)
  add_library(enoki-autodiff SHARED
      ${PROJECT_SOURCE_DIR}/include/enoki/autodiff.h
      ${PROJECT_SOURCE_DIR}/src/autodiff/autodiff.cpp
  )
endif()

if (ENOKI_PYTHON)
  add_subdirectory(ext/pybind11)
  set(CMAKE_CXX_STANDARD 17)
  pybind11_add_module(
      enoki
      ${PROJECT_SOURCE_DIR}/src/python/common.h
      ${PROJECT_SOURCE_DIR}/src/python/main.cpp
      ${PROJECT_SOURCE_DIR}/src/python/cuda_1d.cpp
      ${PROJECT_SOURCE_DIR}/src/python/cuda_2d.cpp
      ${PROJECT_SOURCE_DIR}/src/python/cuda_3d.cpp
      ${PROJECT_SOURCE_DIR}/src/python/autodiff_1d.cpp
      ${PROJECT_SOURCE_DIR}/src/python/autodiff_2d.cpp
      ${PROJECT_SOURCE_DIR}/src/python/autodiff_3d.cpp
  )
  target_link_libraries(enoki PRIVATE enoki-cuda enoki-autodiff cuda)
endif()

# Build the documentation
if (ENOKI_MASTER_PROJECT)
  find_package(Sphinx)

  if (Sphinx_FOUND)
    set(SPHINX_INPUT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

    add_custom_target(mkdoc
        ${SPHINX_EXECUTABLE} -b html "${SPHINX_INPUT_DIR}" "${SPHINX_OUTPUT_DIR}"
        COMMENT "Building HTML documentation with Sphinx"
        USES_TERMINAL)
  endif()
endif()
