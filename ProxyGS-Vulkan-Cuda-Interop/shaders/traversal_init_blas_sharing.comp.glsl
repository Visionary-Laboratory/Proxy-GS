/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

/*
  
  Shader Description
  ==================
  
  Only for ray tracing and USE_BLAS_SHARING
  
  This compute shader initializes the traversal queue with the 
  root nodes of the lod hierarchy of rendered instances.
  
  Not all instances will require this, as some instances
  may use the BLAS of another instance.
  
  Compared to the regular `traversal_init.comp.glsl`, some work
  was already done in `instance_classify_lod.comp.glsl`

  A thread represents one instance.
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

#extension GL_NV_shader_subgroup_partitioned : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
  FrameConstants viewLast;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};


////////////////////////////////////////////

layout(local_size_x=TRAVERSAL_INIT_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////

void main()
{
  uint instanceID   = gl_GlobalInvocationID.x;
  uint instanceLoad = min(build.numRenderInstances-1, instanceID);
  bool isValid      = instanceID == instanceLoad;

#if USE_SORTING
  instanceLoad = build.instanceSortValues.d[instanceLoad];
  instanceID   = instanceLoad;
#endif

  RenderInstance instance = instances[instanceLoad];
  uint geometryID = instance.geometryID;
  Geometry geometry = geometries[geometryID];
  
  // by default all instances use the lowest detail blas as fallback
  uint blasBuildIndex = BLAS_BUILD_INDEX_LOWDETAIL;  
  
  // intance lod range
  // computed in `instance_classify_lod.comp.glsl`
  InstanceBuildInfo instanceInfo = build.instanceBuildInfos.d[instanceLoad];
  uint instanceLevelMin          = instanceInfo.lodLevelMin;
  uint instanceLevelMax          = instanceInfo.lodLevelMax;
  
  // geometry's pick for the shared blas lod level
  // computed in `geometry_blas_sharing.comp.glsl`
  uint shareLevelMin   = build.geometryBuildInfos.d[geometryID].shareLevelMin;
  uint shareLevelMax   = build.geometryBuildInfos.d[geometryID].shareLevelMax;
  uint shareInstanceID = build.geometryBuildInfos.d[geometryID].shareInstanceID;
  
  // When we need to build a BLAS for this instance, we need to add it's root node
  // to the traversal queue. Building the BLAS however isn't always required,
  // which the following logic shows.
  
  bool traverseRootNode = false;
  if (isValid)
  {
    if (instanceLevelMin == uint(instanceInfo.geometryLodLevelMax)) {
      // no need, lowest detail BLAS is used
      traverseRootNode = false;
    }
    else if (shareInstanceID == instanceID) {
      // one of the instances becomes the one shared by all others
      
      // we want to traverse this instance
      traverseRootNode = true;

    #if USE_RENDER_STATS
      build.instanceBuildInfos.d[instanceLoad].instanceUseCount = build.geometryBuildInfos.d[geometryID].shareUseCount + 1;
    #endif
    }
    else if (instanceLevelMin >= shareLevelMax) {
      // don't add, use shareInstance's blas instead
      blasBuildIndex = build.geometryBuildInfos.d[geometryID].shareInstanceID | BLAS_BUILD_INDEX_SHARE_BIT;
      
      traverseRootNode = false;
    }
    else {
      // we are not sharing anything, build a regular per-instance traversal
      traverseRootNode = true;
    }
  }
    
  uvec4 voteNodes = subgroupBallot(traverseRootNode);
  
  uint offsetNodes = 0;
  if (subgroupElect())
  {
    offsetNodes = atomicAdd(buildRW.traversalTaskCounter, int(subgroupBallotBitCount(voteNodes)));
  }
  
  offsetNodes = subgroupBroadcastFirst(offsetNodes);  
  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);
      
  if (traverseRootNode && offsetNodes < build.maxTraversalInfos)
  {
    uint rootNodePacked = geometry.nodes.d[0].packed;

    TraversalInfo traversalInfo;
    traversalInfo.instanceID = instanceID;
    traversalInfo.packedNode = rootNodePacked;

    build.traversalNodeInfos.d[offsetNodes] = packTraversalInfo(traversalInfo);
  }

  if (isValid) {  
    build.instanceBuildInfos.d[instanceID].clusterReferencesCount = 0;
    build.instanceBuildInfos.d[instanceID].blasBuildIndex         = blasBuildIndex;
  }
}