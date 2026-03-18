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
  
  This compute shader handles updating the scene.
  Previous requests to load/unload have been completed and
  are provided for patching the scene.
  
  Effectively we are manipulating the geometries' 
  `streamingGroupAddresses` array that points to the resident
  memory location of a group (or tags it invalid).
  
  Furthermore when ray tracing is required we prepare building
  new CLAS for the loaded groups' clusters.
  
  After building is completed we run the `stream_move_new_clas.comp.glsl`
  to move them from temporary to final location.

  A thread represents a single patch operation, which takes care of
  one group.
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

layout(local_size_x=STREAM_UPDATE_SCENE_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_UPDATE_SCENE_WORKGROUP
  
  uint threadID = gl_GlobalInvocationID.x;  

  // works for both load and unload
  StreamingPatch spatch = streaming.update.patches.d[threadID];
  
  if (threadID < streaming.update.patchGroupsCount)
  {
    uint oldLevel = 0;
    uint oldResidentID = 0;
    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
      Group group = Group_in(geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupIndex]).d;
      oldResidentID = group.residentID;
      oldLevel = group.lodLevel;
    }
    
    geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupIndex] = spatch.groupAddress;
    
    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
    #if STREAMING_DEBUG_ADDRESSES
      streaming.resident.groups.d[oldResidentID].group = Group_in(STREAMING_INVALID_ADDRESS_START);
    #endif      
    }
    else
    {
      uint loadGroupIndex = threadID - streaming.update.patchUnloadGroupsCount;

      Group group = Group_in(spatch.groupAddress).d;
    
      uint groupResidentID = group.residentID;
      StreamingGroup residentGroup;
      residentGroup.clusterCount = group.clusterCount;
      residentGroup.age = 0;
      residentGroup.group = Group_in(spatch.groupAddress);
    #if STREAMING_DEBUG_ADDRESSES
      if (uint64_t(streaming.resident.groups.d[groupResidentID].group) < STREAMING_INVALID_ADDRESS_START)
        streamingRW.request.errorUpdate = groupResidentID;
    #endif
      
      // update description in residency table
      streaming.resident.groups.d[groupResidentID] = residentGroup;

      // insert ourselves into the list of all active groups
      streaming.resident.activeGroups.d[streaming.update.loadActiveGroupsOffset + loadGroupIndex] = groupResidentID;
      
      // We might have a bit of divergence here, but shouldn't be a mission critical issue
      
      // All new groups need to build new clusters.
      // These are built into scratch space first, and then moved to final locations.
      
      uint newBuildOffset = group.streamingNewBuildOffset;
      for (uint c = 0; c < group.clusterCount; c++)
      {
        uint clusterResidentID = group.clusterResidentID + c;
        
        Cluster_in clusterRef = Cluster_in(spatch.groupAddress + Group_size + Cluster_size * c);
        streaming.resident.clusters.d[clusterResidentID] = uint64_t(clusterRef);
        
      #if TARGETS_RAY_TRACING
        Cluster cluster = clusterRef.d;
      
        ClasBuildInfo buildInfo;
        buildInfo.clusterID    = clusterResidentID;
        buildInfo.clusterFlags = 0;
        
        buildInfo.packed = 0;
        buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_triangleCount, cluster.triangleCountMinusOne+1);
        buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_vertexCount, cluster.vertexCountMinusOne+1);
        buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_indexType, 1);
        
        buildInfo.baseGeometryIndexAndFlags = ClasGeometryFlag_OPAQUE_BIT_NV;
        
        buildInfo.indexBufferStride                 = uint16_t(1);
        buildInfo.vertexBufferStride                = uint16_t(4 * 4);
        buildInfo.geometryIndexAndFlagsBufferStride = uint16_t(0);
        buildInfo.opacityMicromapIndexBufferStride  = uint16_t(0);
    
        buildInfo.vertexBuffer = uint64_t(cluster.vertices);
        buildInfo.indexBuffer  = uint64_t(cluster.localTriangles);
        
        buildInfo.geometryIndexAndFlagsBuffer = 0;
        buildInfo.opacityMicromapArray        = 0;
        buildInfo.opacityMicromapIndexBuffer  = 0;
        
        streaming.update.newClasBuilds.d[newBuildOffset + c]      = buildInfo;
        streaming.update.newClasResidentIDs.d[newBuildOffset + c] = clusterResidentID;
      #endif
      }
    }
  }
#if STREAMING_GEOMETRY_LOD_LEVEL_TRACKING
  if (threadID < streaming.update.patchGeometriesCount)
  {
    StreamingGeometryPatch sgpatch = streaming.update.geometryPatches.d[threadID];
    geometries[sgpatch.geometryID].lodsCompletedMask = sgpatch.lodsCompletedMask;
  }
#endif
}

