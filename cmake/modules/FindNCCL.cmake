#  Copyright (c) 2019 ETH Zurich, Simon Frasch
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


#.rst:
# FindNCCL
# -----------
#
# This module looks for the nccl library.
#
# The following variables are set
#
# ::
#
#   NCCL_FOUND           - True if double precision nccl library is found
#   NCCL_LIBRARIES       - The required libraries
#   NCCL_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   NCCL::NCCL



# set paths to look for library
set(_NCCL_PATHS ${NCCL_ROOT} $ENV{NCCL_ROOT})
set(_NCCL_INCLUDE_PATHS)

set(_NCCL_DEFAULT_PATH_SWITCH)

if(_NCCL_PATHS)
    # disable default paths if ROOT is set
    set(_NCCL_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    # try to detect location with pkgconfig
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(PKG_NCCL QUIET "nccl")
    endif()
    set(_NCCL_PATHS ${PKG_NCCL_LIBRARY_DIRS})
    set(_NCCL_INCLUDE_PATHS ${PKG_NCCL_INCLUDE_DIRS})
endif()


find_library(
    NCCL_LIBRARIES
    NAMES "nccl"
    HINTS ${_NCCL_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_NCCL_DEFAULT_PATH_SWITCH}
)
find_path(NCCL_INCLUDE_DIRS
    NAMES "nccl.h"
    HINTS ${_NCCL_PATHS} ${_NCCL_INCLUDE_PATHS}
    PATH_SUFFIXES "include" "include/nccl"
    ${_NCCL_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL REQUIRED_VARS NCCL_INCLUDE_DIRS NCCL_LIBRARIES )

# add target to link against
if(NCCL_FOUND)
    if(NOT TARGET NCCL::NCCL)
        add_library(NCCL::NCCL INTERFACE IMPORTED)
    endif()
    set_property(TARGET NCCL::NCCL PROPERTY INTERFACE_LINK_LIBRARIES ${NCCL_LIBRARIES})
    set_property(TARGET NCCL::NCCL PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(NCCL_FOUND NCCL_LIBRARIES NCCL_INCLUDE_DIRS pkgcfg_lib_PKG_NCCL_nccl3)
