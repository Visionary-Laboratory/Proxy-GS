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
  
  This compute shader writes the streaming request for
  groups to be unloaded. We determine this based on an
  age since the group has been used last.
  
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

layout(local_size_x=STREAM_AGEFILTER_GROUPS_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_AGEFILTER_CLUSTERS_WORKGROUP
  uint residentID = streaming.resident.activeGroups.d[gl_GlobalInvocationID.x];
  if (gl_GlobalInvocationID.x < streaming.resident.activeGroupsCount)
  {
  #if STREAMING_DEBUG_ADDRESSES
    if (uint64_t(streaming.resident.groups.d[residentID].group) >= STREAMING_INVALID_ADDRESS_START)
    {
      streamingRW.request.errorAgeFilter = residentID;
      return;
    }
  #endif
  
    // increase the age of a resident group  
    int age = ++streaming.resident.groups.d[residentID].age;      
    
    // detect if we are over the age limit and request the group to be unloaded
    if (age > streaming.ageThreshold)
    {    
      uint unloadOffset = atomicAdd(streamingRW.request.unloadCounter, 1);
      if (unloadOffset <= streaming.request.maxUnloads) {
        Group_in groupRef = streaming.resident.groups.d[residentID].group;
        streaming.request.unloadGeometryGroups.d[unloadOffset] = uvec2(groupRef.d.geometryID, groupRef.d.groupID);
      }
    }
  }
}

