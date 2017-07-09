cmake_minimum_required (VERSION 2.8.12)
project(enoki CXX)

set(ENOKI_MASTER_PROJECT OFF)
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  set(ENOKI_MASTER_PROJECT ON)
endif()

option(ENOKI_TEST "Build Enoki test suite?" ${ENOKI_MASTER_PROJECT})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" ENOKI_U_CMAKE_BUILD_TYPE)

set(ENOKI_CXXFLAGS_BACKUP ${CMAKE_CXX_FLAGS})

if (MSVC)
  set(ENOKI_ARCHFLAGS_FILE archflags_win32.cpp)
else()
  set(ENOKI_ARCHFLAGS_FILE archflags_unix.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

try_run(
  ENOKI_ARCHFLAGS_RETVAL ENOKI_ARCHFLAGS_COMPILE_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/resources/${ENOKI_ARCHFLAGS_FILE}
  RUN_OUTPUT_VARIABLE ENOKI_ARCHFLAGS
)
if(NOT ${ENOKI_ARCHFLAGS_COMPILE_RESULT})
   message(FATAL_ERROR "Failed to compile 'archflags' binary")
endif()

set(CMAKE_CXX_FLAGS ${ENOKI_CXXFLAGS_BACKUP})

# Optimize for current architecture
set(ENOKI_NATIVE_FLAGS "")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(ENOKI_NATIVE_FLAGS "-march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(ENOKI_NATIVE_FLAGS "-xHost")
elseif(MSVC)
  # MSVC doesn't have a flag equivalent to -march=native
  # The 'archflags' executable provides this functionality
  set(ENOKI_NATIVE_FLAGS ${ENOKI_ARCHFLAGS})
endif()

if (NOT ENOKI_MASTER_PROJECT)
  set(ENOKI_NATIVE_FLAGS ${ENOKI_NATIVE_FLAGS} PARENT_SCOPE)
endif()

set(TEST_SSE42 OFF)
set(TEST_AVX OFF)
set(TEST_AVX2 OFF)
set(TEST_KNL OFF)
set(TEST_SKX OFF)

string(TOUPPER "${ENOKI_ARCHFLAGS}" ENOKI_ARCHFLAGS)

if (MSVC OR ENOKI_ARCHFLAGS MATCHES "SSE")
  set(TEST_SSE42 ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "AVX")
  set(TEST_SSE42 ON)
  set(TEST_AVX ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "AVX2")
  set(TEST_SSE42 ON)
  set(TEST_AVX ON)
  set(TEST_AVX2 ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "KNL")
  set(TEST_SSE42 ON)
  set(TEST_AVX ON)
  set(TEST_AVX2 ON)
  set(TEST_KNL ON)
endif()

if (ENOKI_ARCHFLAGS MATCHES "SKX")
  set(TEST_SSE42 ON)
  set(TEST_AVX ON)
  set(TEST_AVX2 ON)
  set(TEST_SKX ON)
endif()

macro(enoki_set_native_flags)
  add_compile_options(${ENOKI_NATIVE_FLAGS})
endmacro()

macro(enoki_set_compile_flags)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
      CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
      CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" ENOKI_U_CMAKE_BUILD_TYPE)
    if (NOT ${ENOKI_U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_compile_options(-fno-stack-protector -fomit-frame-pointer -fno-math-errno)
      if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        add_compile_options(-ffp-contract=fast)
      endif()
    endif()
  endif()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(-fp-model precise)
  endif()

  if (MSVC)
    # Disable buffer security check cookie
    string(REPLACE "/GS" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/Ob0" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    add_compile_options("$<$<NOT:$<CONFIG:Debug>>:/GS->")
    # Enable intrinsic functions
    add_compile_options("$<$<CONFIG:Release>:/Oi>")
    # Honor __forceinline statements even in debug mode, needed to avoid internal compiler errors on MSVC
    add_compile_options("$<$<CONFIG:Debug>:/Ob1>")
  endif()
endmacro()

include_directories(include)

set(ENOKI_HEADERS
  ${PROJECT_SOURCE_DIR}/include/enoki/alloc.h
  ${PROJECT_SOURCE_DIR}/include/enoki/definitions.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_avx.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_avx2.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_avx512.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_base.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_dynamic.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_generic.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_idiv.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_misc.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_recursive.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_round.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_router.h
  ${PROJECT_SOURCE_DIR}/include/enoki/array_sse42.h
  ${PROJECT_SOURCE_DIR}/include/enoki/quaternion.h
  ${PROJECT_SOURCE_DIR}/include/enoki/stl.h
  ${PROJECT_SOURCE_DIR}/include/enoki/special.h
  ${PROJECT_SOURCE_DIR}/include/enoki/matrix.h
  ${PROJECT_SOURCE_DIR}/include/enoki/complex.h
  ${PROJECT_SOURCE_DIR}/include/enoki/common.h
  ${PROJECT_SOURCE_DIR}/include/enoki/half.h
  ${PROJECT_SOURCE_DIR}/include/enoki/python.h
  ${PROJECT_SOURCE_DIR}/tests/test.h
)

if (MSVC)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(SSE42_FLAGS "/D__SSE4_2__")
  else()
    set(SSE42_FLAGS "/arch:SSE2 /D__SSE4_2__")
  endif()
  set(AVX_FLAGS "/arch:AVX")
  set(AVX2_FLAGS "/arch:AVX2 /D__F16C__")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  add_compile_options(-wd11074 -wd11076)
  set(SSE42_FLAGS "-xSSE4.2")
  set(AVX_FLAGS "-xCORE-AVX-I")
  set(AVX2_FLAGS "-xCORE-AVX2")
  set(AVX512_KNL_FLAGS "-xMIC-AVX512")
  set(AVX512_SKX_FLAGS "-xCORE-AVX512")
else()
  set(SSE42_FLAGS "-msse4.2")
  set(AVX_FLAGS "-mavx")
  set(AVX2_FLAGS "-mavx2 -mfma -mf16c -mbmi2")
  if (APPLE AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(AVX512_KNL_FLAGS "-march=knl -Wa,-march=knl")
    set(AVX512_SKX_FLAGS "-march=skylake-avx512 -Wa,-march=skx")
  else()
    set(AVX512_KNL_FLAGS "-march=knl")
    set(AVX512_SKX_FLAGS "-march=skylake-avx512")
  endif()
endif()

function(enoki_test NAME)
  add_executable(${NAME}_none ${ARGN} ${ENOKI_HEADERS})
  add_test(${NAME}_none_test ${NAME}_none)
  set_tests_properties(${NAME}_none_test PROPERTIES LABELS "none")
  set_target_properties(${NAME}_none PROPERTIES FOLDER ${NAME})

  add_executable(${NAME}_sse42 ${ARGN} ${ENOKI_HEADERS})
  set_target_properties(${NAME}_sse42 PROPERTIES COMPILE_FLAGS ${SSE42_FLAGS})
  set_target_properties(${NAME}_sse42 PROPERTIES FOLDER ${NAME})
  if (TEST_SSE42)
    add_test(${NAME}_sse42_test ${NAME}_sse42)
    set_tests_properties(${NAME}_sse42_test PROPERTIES LABELS "sse42")
  endif()

  add_executable(${NAME}_avx ${ARGN} ${ENOKI_HEADERS})
  set_target_properties(${NAME}_avx PROPERTIES COMPILE_FLAGS ${AVX_FLAGS})
  set_target_properties(${NAME}_avx PROPERTIES FOLDER ${NAME})
  if (TEST_AVX)
    add_test(${NAME}_avx_test ${NAME}_avx)
    set_tests_properties(${NAME}_avx_test PROPERTIES LABELS "avx")
  endif()

  add_executable(${NAME}_avx2 ${ARGN} ${ENOKI_HEADERS})
  set_target_properties(${NAME}_avx2 PROPERTIES COMPILE_FLAGS ${AVX2_FLAGS})
  set_target_properties(${NAME}_avx2 PROPERTIES FOLDER ${NAME})
  if (TEST_AVX2)
    add_test(${NAME}_avx2_test ${NAME}_avx2)
    set_tests_properties(${NAME}_avx2_test PROPERTIES LABELS "avx2")
  endif()

  if (NOT MSVC)
    add_executable(${NAME}_avx512_knl ${ARGN} ${ENOKI_HEADERS})
    set_target_properties(${NAME}_avx512_knl PROPERTIES COMPILE_FLAGS ${AVX512_KNL_FLAGS})
    set_target_properties(${NAME}_avx512_knl PROPERTIES FOLDER ${NAME})
    if (TEST_KNL)
      add_test(${NAME}_avx512_knl_test ${NAME}_avx512_knl)
      set_tests_properties(${NAME}_avx512_knl_test PROPERTIES LABELS "avx512_knl")
    endif()

    add_executable(${NAME}_avx512_skx ${ARGN} ${ENOKI_HEADERS})
    set_target_properties(${NAME}_avx512_skx PROPERTIES COMPILE_FLAGS ${AVX512_SKX_FLAGS})
    set_target_properties(${NAME}_avx512_skx PROPERTIES FOLDER ${NAME})

    if (TEST_SKX)
      add_test(${NAME}_avx512_skx_test ${NAME}_avx512_skx)
      set_tests_properties(${NAME}_avx512_skx_test PROPERTIES LABELS "avx512_skx")
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
    check_cxx_source_runs("int main(int argc, char **argv) { return 0; }" ${_RESULT})
    set(CMAKE_REQUIRED_FLAGS "")
    set(CMAKE_REQUIRED_LIBRARIES "")
  endmacro()

  # Always use libc++ on Clang
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
    if (HAS_LIBCPP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
      CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPPABI "-stdlib=libc++" "-stdlib=libc++ -lc++abi")
      if(HAS_LIBCPPABI)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++abi")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lc++abi")
        message(STATUS "Enoki: using libc++ and libc++abi.")
      else()
        message(STATUS "Enoki: using libc++.")
      endif()
    else()
      message(STATUS "Enoki: NOT using libc++.")
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
      DEPENDS ALL_BUILD
  )
endif()

# Build the documentation
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources")
find_package(Sphinx)

if (Sphinx_FOUND)
  set(SPHINX_INPUT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

  add_custom_target(mkdoc
      ${SPHINX_EXECUTABLE} -b html "${SPHINX_INPUT_DIR}" "${SPHINX_OUTPUT_DIR}"
      COMMENT "Building HTML documentation with Sphinx"
      USES_TERMINAL)
endif()
