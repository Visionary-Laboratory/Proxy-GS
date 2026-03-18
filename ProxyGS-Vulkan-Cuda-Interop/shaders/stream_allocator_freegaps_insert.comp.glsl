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
  
  This compute shader bins the free gaps based on their size.
  It enables the allocator to provide empty gaps of certain sizes during
  the allocation process within `stream_allocator_load_groups.comp.glsl`.
  
  We read `streaming.clasAllocator.freeGapsPos` and `streaming.clasAllocator.freeGapsSize`
  and bin into `streaming.clasAllocator.freeGapsPosBinned` using the appropriate
  `streaming.clasAllocator.freeSizeRanges.d[freeGapSize-1].offset`
  
  One thread operates on one free gap
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

layout(local_size_x=STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint threadID = gl_GlobalInvocationID.x;
  bool valid    = threadID < streaming.clasAllocator.freeGapsCounter;

  if (valid)
  {
    // get the details of the free gap, it was computed in
    // `stream_allocator_build_freegaps.comp.glsl`.

    uint freeGapPos  = streaming.clasAllocator.freeGapsPos.d[threadID];
    uint freeGapSize = streaming.clasAllocator.freeGapsSize.d[threadID];
  
    // bin the gap into `streaming.clasAllocator.freeGapsPosBinned` based on size
    int32_t rangeIndex = atomicAdd(streaming.clasAllocator.freeSizeRanges.d[freeGapSize-1].count, 1);
    uint rangeOffset   = streaming.clasAllocator.freeSizeRanges.d[freeGapSize-1].offset;
    
    uint storeOffset   = rangeIndex + uint(rangeOffset);
    streaming.clasAllocator.freeGapsPosBinned.d[storeOffset] = freeGapPos;
  }
}