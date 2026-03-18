/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

/*
  
  Shader Description
  ==================
  
    Note: The sample showcases two ways to manage CLAS memory on the device.
    One using a persistent allocator system (`stream_allocator...` files),
    and one using a simple compaction scheme (`stream_compaction...` files).
    This file is part of the allocator system.
  
  This compute shader prepares the ranges of free gaps
  based on their size. It is required to handle the size-based
  binding within `stream_allocator_freegaps_insert.comp.glsl`.
  
  One thread represent one free gap size
  
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_subgroup_extended_types_int64 : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

////////////////////////////////////////////

layout(local_size_x=STREAM_ALLOCATOR_SETUP_INSERTION_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint threadID = gl_GlobalInvocationID.x;
  bool valid    = threadID < streaming.clasAllocator.maxAllocationSize;
  
  if (valid)
  {
    // from the previous kernel `stream_allocator_build_freegaps.comp.glsl` we know how
    // many slots the size-binned array will need
    uint rangeCount  = uint(streaming.clasAllocator.freeSizeRanges.d[threadID].count);
    // get an offset into `streaming.clasAllocator.freeGapsPosBinned` for the list of
    uint rangeOffset = atomicAdd(streamingRW.clasAllocator.freeGapsCounter, rangeCount);
    // setup range offset
    streaming.clasAllocator.freeSizeRanges.d[threadID].offset = rangeOffset;
    // reset to zero for insertion done in `stream_allocator_freegaps_insert.comp.glsl`
    streaming.clasAllocator.freeSizeRanges.d[threadID].count  = 0;
  }
}

