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
  
  This compute shader initializes the traversal queue with the 
  root nodes of the lod hierarchy of rendered instances.

  A thread represents one instance.

  NOT compatible with USE_BLAS_SHARING, see `traversal_init_blas_sharing.comp.glsl`
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
  
  uint blasBuildIndex = BLAS_BUILD_INDEX_LOWDETAIL;
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
  bool inFrustum = intersectFrustum(geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax) && intersectHiz(clipMin, clipMax)));
  
  uint visibilityState = isVisible ? INSTANCE_VISIBLE_BIT : 0;
  
  bool isRenderable = isValid
  #if USE_CULLING && TARGETS_RASTERIZATION
    && isVisible
  #endif
    ;
    
  bool traverseRootNode = isRenderable;

  if (isRenderable)
  {
    // We test if we are only using the furthest lod.
    // If that is true, then we can skip lod traversal completely and
    // straight enqueue the lowest detail cluster directly.    
    
    uint rootNodePacked = geometry.nodes.d[0].packed;
    
    uint childOffset        = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);
    uint childCountMinusOne = PACKED_GET(rootNodePacked, Node_packed_nodeChildCountMinusOne);
    
    // test if the second to last lod needs to be traversed
    uint childNodeIndex     = (childCountMinusOne > 1 ? (childCountMinusOne - 1) : 0);
    Node childNode          = geometry.nodes.d[childOffset + childNodeIndex];
    TraversalMetric traversalMetric = childNode.traversalMetric;
  
    mat4  worldMatrix  = instances[instanceID].worldMatrix;
    float uniformScale = computeUniformScale(worldMatrix);
    float errorScale   = 1.0;
  #if USE_CULLING && TARGETS_RAY_TRACING
    if (visibilityState == 0) errorScale = build.culledErrorScale;
  #endif
  
    mat4 transform = build.traversalViewMatrix * worldMatrix;
  
    // if there is no need to traverse the pen ultimate lod level,
    // then just insert the last lod level node's cluster directly
    if (!testForTraversal(mat4x3(transform), uniformScale, traversalMetric, errorScale))
    {
    
    #if TARGETS_RAY_TRACING
      // we don't need to add a cluster because we always add it
      // implictly through the use of the low detail BLAS.
      
      #if USE_RENDER_STATS
        atomicAdd(readback.numRenderedTriangles, geometry.lowDetailTriangles);
        atomicAdd(readback.numRenderedClusters, 1);
      #endif
      
    #elif TARGETS_RASTERIZATION
      // lowest detail lod is guaranteed to have only one cluster
      
      uvec4 voteClusters = subgroupBallot(true); 
      
      uint offsetClusters = 0;
      if (subgroupElect())
      {
        offsetClusters = atomicAdd(buildRW.renderClusterCounter, int(subgroupBallotBitCount(voteClusters)));
      }
  
      offsetClusters = subgroupBroadcastFirst(offsetClusters);  
      offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);
      
      if (offsetClusters < build.maxRenderClusters)
      {
        ClusterInfo clusterInfo;
        clusterInfo.instanceID = instanceID;
        clusterInfo.clusterID  = geometry.lowDetailClusterID;
        build.renderClusterInfos.d[offsetClusters] = clusterInfo;
      }
    #endif
      
      // we can skip adding the node for traversal
      traverseRootNode = false;
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

#if TARGETS_RAY_TRACING
  if (isValid) {
    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);  
    build.instanceBuildInfos.d[instanceID].clusterReferencesCount = 0;
    build.instanceBuildInfos.d[instanceID].blasBuildIndex         = blasBuildIndex;
    build.tlasInstances.d[instanceID].blasReference               = geometry.lowDetailBlasAddress;
  }
#endif
}