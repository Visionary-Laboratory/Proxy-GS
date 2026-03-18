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
  
  This compute shader handles de-allocation of clas memory space
  of unloaded groups.
  
  It marks the appropriate bits of the memory regions as empty again.
  `streaming.clasAllocator.usedBits` is modified accordingly.
  
  One thread represents an unloaded group
  
  TODO might want to improve divergence in the loops
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

layout(local_size_x=STREAM_ALLOCATOR_UNLOAD_GROUPS_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint threadID             = gl_GlobalInvocationID.x;
  bool valid                = threadID < streaming.update.patchUnloadGroupsCount;
  
  // unloads come first in patches
  StreamingPatch spatch = streaming.update.patches.d[threadID];
  
  if (valid)
  {  
    Group group = Group_in(geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupIndex]).d;
    
    // get the first clas address of the group, as all clas of a 
    // group are allocated together
    uint64_t firstClasAddress = streaming.resident.clasAddresses.d[group.clusterResidentID];
    // then convert this into a relative address compared to the clas base address
    uint64_t firstClasOffset  = firstClasAddress - streaming.resident.clasBaseAddress;
    
    // recreate the allocation properties of the group
    // get allocation position in units
    uint allocPos   = uint(firstClasOffset >> streaming.clasAllocator.granularityByteShift);
    // retrieve the size of allocation as well as the associated memory waste
    uvec2 groupSize = streaming.resident.groupClasSizes.d[group.residentID];
    // allocation size was stored in units, which is what we need here, but wasted size in bytes
    uint allocSize  = groupSize.x;
    uint wastedByteSize = groupSize.y;
    
  #if USE_MEMORY_STATS
    atomicAdd(streamingRW.clasAllocator.stats.d.allocatedSize, -int64_t(allocSize << streaming.clasAllocator.granularityByteShift));
    atomicAdd(streamingRW.clasAllocator.stats.d.wastedSize, -int64_t(wastedByteSize));
  #endif
    
    // for allocation management, tag bits as unusued
    //
    // allocPos and allocSize are in minimum granularity,
    // which is what we use to tag the appropriate bits.
    
    uint startPos = allocPos;
    uint endPos   = allocPos + allocSize - 1;
    
    uint startBit = (startPos) & 31;
    uint endBit   = (endPos) & 31;
    
    uint start32 = startPos / 32;
    uint end32   = endPos / 32;
    
    uint startMask = ~0;
    uint endMask   = ~0;
    
    if (startBit != 0)
    {
      startMask = ~((1u << (startBit))-1);
    }
    if (endBit != 31)
    {
      endMask =  (1u << (endBit + 1))-1;
    }
    
    bool single32 = start32 == end32;      
    if (single32)
    {
      startMask = endMask | startMask;
    }
    
    // start and end of an allocated region may end up in the same u32,
    // hence we need atomics for start and end
    
    uint oldMask = atomicAnd(streaming.clasAllocator.usedBits.d[start32], ~startMask);
  #if STREAMING_DEBUG_FREEGAPS_OVERLAP
    // for debugging we test if the region was indeed fully used
    bool hadError = false;
    if ((oldMask & startMask) != startMask){
      hadError = true;
    }
  #endif
    
    if (!single32) 
    {
      // process the region that is exclusively covered by this allocation
      for (uint32_t i = start32 + 1; i < end32; i++)
      {
      #if STREAMING_DEBUG_FREEGAPS_OVERLAP
        if(streaming.clasAllocator.usedBits.d[i] == 0){
          hadError = true;
        }
      #endif
        streaming.clasAllocator.usedBits.d[i] = 0;
      }
      
      oldMask = atomicAnd(streaming.clasAllocator.usedBits.d[end32], ~endMask);
    #if STREAMING_DEBUG_FREEGAPS_OVERLAP
      if ((oldMask & endMask) != endMask){
        hadError = true;
      }
    #endif
    }
  #if STREAMING_DEBUG_FREEGAPS_OVERLAP
    if (hadError){
      streamingRW.request.errorClasDealloc = 1 + threadID;
    }
  #endif
  }
}

