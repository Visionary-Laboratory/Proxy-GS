# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

find_package(nanobench QUIET)
if(NOT nanobench_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      nanobench
      GIT_REPOSITORY https://github.com/martinus/nanobench.git
      GIT_TAG v4.3.11
      GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(nanobench)
endif()

add_executable(nv_cluster_builder_tests
  src/test_clusterizer.c
  src/test_clusterizer.cpp
  src/test_meshes.cpp
  src/test_perf.cpp
)

option(NVCLUSTER_TEST_MESHES "FetchContent cgltf to test meshes in current directory" OFF)
if(NVCLUSTER_TEST_MESHES)
  if(NOT TARGET cgltf_static)
    set(CGLTF_INCLUDE "${CMAKE_BINARY_DIR}/cgltf")
    file(MAKE_DIRECTORY "${CGLTF_INCLUDE}")
    file(DOWNLOAD https://raw.githubusercontent.com/jkuhlmann/cgltf/refs/tags/v1.15/cgltf.h "${CGLTF_INCLUDE}/cgltf.h")
    file(WRITE "${CMAKE_BINARY_DIR}/cgltf.cpp" "#define CGLTF_IMPLEMENTATION\n#include <cgltf.h>\n")
    add_library(cgltf_static "${CMAKE_BINARY_DIR}/cgltf.cpp")
    target_include_directories(cgltf_static PUBLIC "${CGLTF_INCLUDE}")
  endif()
  target_compile_definitions(nv_cluster_builder_tests PRIVATE TEST_MESHES)
  target_link_libraries(nv_cluster_builder_tests PRIVATE cgltf_static)
endif()

target_include_directories(nv_cluster_builder_tests PRIVATE src ../src) # adds internal src directory to allow unit testing
target_link_libraries(nv_cluster_builder_tests PRIVATE nv_cluster_builder gtest_main gmock_main nanobench)

if(MSVC)
    target_compile_options(nv_cluster_builder_tests PRIVATE
      /W4
      /WX
    )
    target_compile_definitions(nv_cluster_builder_tests PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
else()
    target_compile_options(nv_cluster_builder_tests PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow
      -Wconversion
      -Werror
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(nv_cluster_builder_tests PRIVATE
            $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
        )
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(nv_cluster_builder_tests)
