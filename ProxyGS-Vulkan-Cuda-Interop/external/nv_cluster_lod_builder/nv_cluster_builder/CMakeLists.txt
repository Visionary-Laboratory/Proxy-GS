# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(nv_cluster_builder VERSION 2.0)

set(SOURCES
  src/clusterizer.cpp
  src/connections.cpp
  src/nvcluster.cpp
)
file(GLOB HEADERS_INTERNAL
  src/*.hpp
)
file(GLOB HEADERS_PUBLIC
  include/nvcluster/*.h
  include/nvcluster/*.hpp
)

source_group("public_include" FILES ${HEADERS_PUBLIC})
source_group("source" FILES ${SOURCES} ${HEADERS_INTERNAL})

# Optionally build as a shared library
include(CMakeDependentOption)
cmake_dependent_option(
  NVCLUSTER_BUILDER_SHARED  # option variable
  "Build shared library"    # description
  OFF                       # default value if exposed; user can override
  "NOT BUILD_SHARED_LIBS"   # condition to expose option
  ON                        # value if not exposed; user can't override
)

if (NVCLUSTER_BUILDER_SHARED)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
  add_library(nv_cluster_builder SHARED ${SOURCES} ${HEADERS_INTERNAL} ${HEADERS_PUBLIC})
  target_compile_definitions(nv_cluster_builder PUBLIC NVCLUSTER_BUILDER_SHARED)
else()
  add_library(nv_cluster_builder STATIC ${SOURCES} ${HEADERS_INTERNAL} ${HEADERS_PUBLIC})
endif ()
target_compile_features(nv_cluster_builder PUBLIC cxx_std_20)
target_include_directories(nv_cluster_builder PUBLIC include)
target_include_directories(nv_cluster_builder PRIVATE src)
target_compile_definitions(nv_cluster_builder PRIVATE NVCLUSTER_BUILDER_COMPILING)

# All the warnings. Branch on COMPILE_LANGUAGE to avoid passing unknowns to nvcc
if(MSVC)
    target_compile_options(nv_cluster_builder PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/WX>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4127> # 'conditional expression is constant' unhelpful when mixing c.t. and dynamic
    )
    target_compile_definitions(nv_cluster_builder PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
else()
    target_compile_options(nv_cluster_builder PRIVATE
        -fno-math-errno
        -fno-trapping-math
        #-funsafe-math-optimizations
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(nv_cluster_builder PRIVATE
            $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
            # Do not use ABI breaking _GLIBCXX_DEBUG or _GLIBCXX_DEBUG_BACKTRACE
        )
    endif()
endif()

option(NVCLUSTER_MULTITHREADED "Build with multithreaded cluster generation support" ON)
if(NVCLUSTER_MULTITHREADED)
  target_compile_definitions(nv_cluster_builder PRIVATE NVCLUSTER_MULTITHREADED=1)

  # Optional TBB for std::execution on linux
  if(NOT MSVC)
    find_library(TBB_LIBRARIES NAMES tbb HINTS ${TBB_DIR})
    if(TBB_LIBRARIES)
        message(STATUS "TBB: ${TBB_LIBRARIES}")
        target_link_libraries(nv_cluster_builder PRIVATE ${TBB_LIBRARIES})
    else()
        message(STATUS "TBB not found for std::execution")
    endif()
  endif()
else()
  target_compile_definitions(nv_cluster_builder PRIVATE NVCLUSTER_MULTITHREADED=0)
endif()

if(BUILD_TESTING)
  option(BUILD_NV_CLUSTER_BUILDER_TESTING "Build nv_cluster_builder tests" ON)
  if(BUILD_NV_CLUSTER_BUILDER_TESTING)
    enable_testing()
    add_subdirectory(test)
  endif()
endif()

install(TARGETS nv_cluster_builder)
