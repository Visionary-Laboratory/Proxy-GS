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
    This file is part of the compaction scheme.
  
  This compute shader compacts / defrags cluster CLAS storage
  of all previously active resident groups.
  
  A thread represents one resident group.
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

layout(local_size_x=STREAM_COMPACTION_OLD_CLAS_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_MOVE_OLD_CLAS_WORKGROUP

  uint threadID        = gl_GlobalInvocationID.x;
  uint groupResidentID = streaming.resident.activeGroups.d[threadID];
  
  // old resident groups come first, then after this offset are the newly loaded,
  // which we can ignore here.
  bool valid           = threadID < streaming.update.loadActiveGroupsOffset;
  
  if (valid)
  {
    // Walk over all old resident groups' clusters and compact their clas
    // objects storage so that the newly built clas can be appended to the
    // end.
    
    // This will result in a lot of movement of clas and is not recommended,
    // but avoids a more sophisticated clas allocation scheme.
    
    Group group = streaming.resident.groups.d[groupResidentID].group.d;
    
    // TODO improve divergence
    for (uint c = 0; c < group.clusterCount; c++)
    {
      uint clusterResidentID = group.clusterResidentID + c;
      
      uint clasSize        = streaming.resident.clasSizes.d[clusterResidentID];
      uint64_t clasAddress = streaming.resident.clasAddresses.d[clusterResidentID];
      
      uint64_t clasNewAddress = atomicAdd(streamingRW.update.moveClasSize, uint64_t(clasSize)) +
                                streaming.resident.clasBaseAddress;
      
      // don't move identical addresses (in reality this will hardly happen due to
      // non-deterministic nature of atomicAdd)
      bool move       = clasNewAddress != clasAddress;      
      uint moveOffset = atomicAdd(streamingRW.update.moveClasCounter, move ? 1 : 0);
      
      if (move) {
        // set up move to new destination
        streaming.update.moveClasSrcAddresses.d[moveOffset] = clasAddress;
        streaming.update.moveClasDstAddresses.d[moveOffset] = clasNewAddress;
        // update internal state of destination
        streaming.resident.clasAddresses.d[clusterResidentID] = clasNewAddress;
        streaming.resident.clasSizes.d[clusterResidentID]     = clasSize;
      }
    }
  }  
}

