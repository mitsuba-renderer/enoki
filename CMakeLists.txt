cmake_minimum_required (VERSION 2.8.12)
project(enoki CXX)

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

# Use memkind library (high-bandwidth memory on Xeon Phi) if available
find_package(Memkind QUIET)

if (MEMKIND_FOUND)
  if (NOT ENOKI_MASTER_PROJECT)
    set(MEMKIND_INCLUDE_DIRS ${MEMKIND_INCLUDE_DIRS} PARENT_SCOPE)
    set(MEMKIND_LIBRARIES ${MEMKIND_LIBRARIES} PARENT_SCOPE)
  endif()
  message(STATUS "Enoki: Using memkind library for heap memory allocation.")
else()
  message(STATUS "Enoki: Using C library for heap memory allocation.")
endif()

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
    if (NOT (${ENOKI_U_CMAKE_BUILD_TYPE} MATCHES "DEBUG"))
      if (NOT (${CMAKE_CXX_FLAGS} MATCHES "fsanitize"))
        # Don't use stack security features in release mode
        add_compile_options(-fno-stack-protector)
        # In release mode, don't keep the frame pointer in a dedicated register unless needed
        add_compile_options(-fomit-frame-pointer)
      endif()

      # Never update the 'errno' variable due to arithmetic exceptions
      add_compile_options(-fno-math-errno)

      if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        # Look for opportunities to fuse additions & multiplications into FMA operations
        add_compile_options(-ffp-contract=fast)
      endif()
    endif()
  endif()

  # Disable overly aggressive FP optimization in the Intel compiler
  if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(-fp-model precise)
  endif()

  if (MEMKIND_INCLUDE_DIRS)
    include_directories(${MEMKIND_INCLUDE_DIRS})
    add_definitions(-DENOKI_USE_MEMKIND)
    set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} ${MEMKIND_LIBRARIES}")
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
  ${PROJECT_SOURCE_DIR}/include/enoki/alloc.h
  ${PROJECT_SOURCE_DIR}/include/enoki/fwd.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_avx.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_avx2.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_avx512.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_base.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_dynamic.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_generic.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_idiv.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_misc.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_macro.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_neon.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_recursive.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_round.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_router.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_sse42.h
  ${PROJECT_SOURCE_DIR}/include/enoki/quaternion.h
  ${PROJECT_SOURCE_DIR}/include/enoki/stl.h
  ${PROJECT_SOURCE_DIR}/include/enoki/special.h
  ${PROJECT_SOURCE_DIR}/include/enoki/transform.h
  ${PROJECT_SOURCE_DIR}/include/enoki/matrix.h
  ${PROJECT_SOURCE_DIR}/include/enoki/complex.h
  ${PROJECT_SOURCE_DIR}/include/enoki/common.h
  ${PROJECT_SOURCE_DIR}/include/enoki/half.h
  ${PROJECT_SOURCE_DIR}/include/enoki/python.h
  ${PROJECT_SOURCE_DIR}/tests/test.h
)


if (MSVC)
  set(ENOKI_NONE_FLAGS "/DENOKI_DISABLE_VECTORIZATION")
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ENOKI_SSE42_FLAGS "/D__SSE4_2__")
  else()
    set(ENOKI_SSE42_FLAGS "/arch:SSE2 /D__SSE4_2__")
  endif()
  set(ENOKI_AVX_FLAGS "/arch:AVX")
  set(ENOKI_AVX2_FLAGS "/arch:AVX2")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  add_compile_options(-wd11074 -wd11076)
  set(ENOKI_NONE_FLAGS "-DENOKI_DISABLE_VECTORIZATION")
  set(ENOKI_SSE42_FLAGS "-xSSE4.2")
  set(ENOKI_AVX_FLAGS "-xCORE-AVX-I")
  set(ENOKI_AVX2_FLAGS "-xCORE-AVX2")
  set(ENOKI_AVX512_KNL_FLAGS "-xMIC-AVX512")
  set(ENOKI_AVX512_SKX_FLAGS "-xCORE-AVX512")
else()
  set(ENOKI_NONE_FLAGS "-DENOKI_DISABLE_VECTORIZATION")
  set(ENOKI_SSE42_FLAGS "-msse4.2")
  set(ENOKI_AVX_FLAGS "-mavx")
  set(ENOKI_AVX2_FLAGS "-mavx2 -mfma -mf16c -mbmi2 -mlzcnt")
  if (APPLE AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(ENOKI_AVX512_KNL_FLAGS "-march=knl -Wa,-march=knl")
    set(ENOKI_AVX512_SKX_FLAGS "-march=skylake-avx512 -Wa,-march=skx")
  else()
    set(ENOKI_AVX512_KNL_FLAGS "-march=knl")
    set(ENOKI_AVX512_SKX_FLAGS "-march=skylake-avx512")
  endif()
  set(ENOKI_ARM_FLAGS "")
  if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
    set(ENOKI_ARM_FLAGS "-march=armv7-a -mtune=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee")
  elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(ENOKI_ARM_FLAGS "-march=armv8-a+simd -mtune=cortex-a53 ")
  endif()
endif()

function(enoki_test NAME)
  add_executable(${NAME}_none ${ARGN} ${ENOKI_HEADERS})
  add_test(${NAME}_none_test ${NAME}_none)
  set_tests_properties(${NAME}_none_test PROPERTIES LABELS "none")
  set_target_properties(${NAME}_none PROPERTIES FOLDER ${NAME})
  set_target_properties(${NAME}_none PROPERTIES COMPILE_FLAGS ${ENOKI_NONE_FLAGS})

  if (ENOKI_HOST MATCHES "INTEL")
    add_executable(${NAME}_sse42 ${ARGN} ${ENOKI_HEADERS})
    set_target_properties(${NAME}_sse42 PROPERTIES COMPILE_FLAGS ${ENOKI_SSE42_FLAGS})
    set_target_properties(${NAME}_sse42 PROPERTIES FOLDER ${NAME})
    if (ENOKI_TEST_SSE42)
      add_test(${NAME}_sse42_test ${NAME}_sse42)
      set_tests_properties(${NAME}_sse42_test PROPERTIES LABELS "sse42")
    endif()

    add_executable(${NAME}_avx ${ARGN} ${ENOKI_HEADERS})
    set_target_properties(${NAME}_avx PROPERTIES COMPILE_FLAGS ${ENOKI_AVX_FLAGS})
    set_target_properties(${NAME}_avx PROPERTIES FOLDER ${NAME})
    if (ENOKI_TEST_AVX)
      add_test(${NAME}_avx_test ${NAME}_avx)
      set_tests_properties(${NAME}_avx_test PROPERTIES LABELS "avx")
    endif()

    add_executable(${NAME}_avx2 ${ARGN} ${ENOKI_HEADERS})
    set_target_properties(${NAME}_avx2 PROPERTIES COMPILE_FLAGS ${ENOKI_AVX2_FLAGS})
    set_target_properties(${NAME}_avx2 PROPERTIES FOLDER ${NAME})
    if (ENOKI_TEST_AVX2)
      add_test(${NAME}_avx2_test ${NAME}_avx2)
      set_tests_properties(${NAME}_avx2_test PROPERTIES LABELS "avx2")
    endif()

    if (NOT MSVC)
      add_executable(${NAME}_avx512_knl ${ARGN} ${ENOKI_HEADERS})
      set_target_properties(${NAME}_avx512_knl PROPERTIES COMPILE_FLAGS ${ENOKI_AVX512_KNL_FLAGS})
      set_target_properties(${NAME}_avx512_knl PROPERTIES FOLDER ${NAME})
      if (ENOKI_TEST_KNL)
        add_test(${NAME}_avx512_knl_test ${NAME}_avx512_knl)
        set_tests_properties(${NAME}_avx512_knl_test PROPERTIES LABELS "avx512_knl")
      endif()

      add_executable(${NAME}_avx512_skx ${ARGN} ${ENOKI_HEADERS})
      set_target_properties(${NAME}_avx512_skx PROPERTIES COMPILE_FLAGS ${ENOKI_AVX512_SKX_FLAGS})
      set_target_properties(${NAME}_avx512_skx PROPERTIES FOLDER ${NAME})

      if (ENOKI_TEST_SKX)
        add_test(${NAME}_avx512_skx_test ${NAME}_avx512_skx)
        set_tests_properties(${NAME}_avx512_skx_test PROPERTIES LABELS "avx512_skx")
      endif()
    endif()
  endif()

  if (ENOKI_HOST MATCHES "ARM")
    add_executable(${NAME}_arm ${ARGN} ${ENOKI_HEADERS})
    set_target_properties(${NAME}_arm PROPERTIES COMPILE_FLAGS ${ENOKI_ARM_FLAGS})
    set_target_properties(${NAME}_arm PROPERTIES FOLDER ${NAME})
    if (ENOKI_TEST_ARM)
      add_test(${NAME}_arm_test ${NAME}_arm)
      set_tests_properties(${NAME}_arm_test PROPERTIES LABELS "arm")
    endif()
  endif()
endfunction()

if (ENOKI_TEST)
  enable_testing()
  enoki_set_compile_flags()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
      CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
      CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    if (NOT ${ENOKI_U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_compile_options( -fvisibility=hidden)
    endif()
    add_compile_options(-Wall -Wextra -Wconversion)
  elseif(WIN32)
    add_compile_options("/std:c++14")
    add_compile_options("/bigobj")
  endif()

  include(CheckCXXSourceRuns)
  include(CheckCXXCompilerFlag)

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    CHECK_CXX_COMPILER_FLAG(-faligned-new ALIGNED_NEW)
    if (ALIGNED_NEW)
      add_compile_options(-faligned-new)
    endif()
  endif()

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  macro(CHECK_CXX_COMPILER_AND_LINKER_FLAGS _RESULT _CXX_FLAGS _LINKER_FLAGS)
    set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
    set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_source_runs("#include <iostream>\nint main(int argc, char **argv) { std::cout << \"test\"; return 0; }" ${_RESULT})
    set(CMAKE_REQUIRED_FLAGS "")
    set(CMAKE_REQUIRED_LIBRARIES "")
  endmacro()

  # Prefer libc++ in conjunction with Clang
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
      message(STATUS "NanoGUI: using libc++.")
    else()
      CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
      if (HAS_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
        message(STATUS "NanoGUI: using libc++.")
      else()
        message(STATUS "NanoGUI: NOT using libc++.")
      endif()
    endif()
  endif()

  add_subdirectory(tests)

  add_custom_target(check
          ${CMAKE_COMMAND} -E echo CWD=${CMAKE_BINARY_DIR}
          COMMAND ${CMAKE_COMMAND} -E echo CMD=${CMAKE_CTEST_COMMAND} -C $<CONFIG>
          COMMAND ${CMAKE_COMMAND} -E echo ----------------------------------
          COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
              ${CMAKE_CTEST_COMMAND} -C $<CONFIG>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPENDS ALL_BUILD USES_TERMINAL
  )
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
