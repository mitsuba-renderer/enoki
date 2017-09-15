# - Try to find memkind
# Once done this will define
#  MEMKIND_FOUND - System has memkind
#  MEMKIND_INCLUDE_DIRS - The memkind include directories
#  MEMKIND_LIBRARIES - The libraries needed to use memkind

find_path(MEMKIND_INCLUDE_DIR memkind.h)
find_library(MEMKIND_LIBRARY NAMES memkind)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MEMKIND_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Memkind DEFAULT_MSG
                                  MEMKIND_LIBRARY MEMKIND_INCLUDE_DIR)

mark_as_advanced(MEMKIND_INCLUDE_DIR MEMKIND_LIBRARY)

set(MEMKIND_LIBRARIES ${MEMKIND_LIBRARY})
set(MEMKIND_INCLUDE_DIRS ${MEMKIND_INCLUDE_DIR})
