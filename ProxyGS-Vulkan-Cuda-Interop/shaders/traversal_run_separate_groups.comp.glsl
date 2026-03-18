
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
  
  Only for USE_SEPARATE_GROUPS
  
  This compute shader implements the traversal of cluster groups
  in the scene. Cluster groups iterate over their children
  and test the traversal metric of their generating groups
  in the opposite direction. Depending on the result
  it will then enqueue the clusters for rendering.
  
  `traversal_run.comp.glsl` is run before and outputs
    - `build.traversalGroupInfos` all traversed cluster groups that fulfill the metric.
    - `build.traversalGroupCounter` number of the groups (may exceed recorded maximum).
    - `build.indirectDispatchGroups.gridX` the dimensions of this kernel's dispatch based on above
  
  The cluster groups fill the list of to be rendered
  clusters.
    - `build.renderClusterInfos` stores all clusters that are to be rendered as linear array
    - `build.renderClusterCounter` is used to append the clusters

  one thread represents one cluster group.
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
#extension GL_KHR_memory_scope_semantics : require

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

#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};
#endif

////////////////////////////////////////////

layout(local_size_x=TRAVERSAL_GROUPS_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////

// work around compiler bug on older drivers not properly handling coherent & volatile
#define USE_ATOMIC_LOAD_STORE 1

////////////////////////////////////////////


#if USE_CULLING && TARGETS_RASTERIZATION

// simplified occlusion culling based on last frame's depth buffer
bool queryWasVisible(mat4 instanceTransform, BBox bbox)
{
  vec3 bboxMin = bbox.lo;
  vec3 bboxMax = bbox.hi;
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
  bool useOcclusion = build.useHizOcclusion != 0;
  
  bool inFrustum = intersectFrustum(bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && 
    (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax) && intersectHiz(clipMin, clipMax)));
  
  return isVisible;
}

#endif

void main()
{
  uint threadReadIndex = gl_GlobalInvocationID.x;  
  if (threadReadIndex >= min(build.traversalGroupCounter, build.maxTraversalInfos)) return;
  
  // load group and test its clusters
  
  // pull required input item from subgroupTasks
  TraversalInfo traversalInfo = unpackTraversalInfo(build.traversalGroupInfos.d[threadReadIndex]);
  
  uint groupIndex        = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
  uint groupClusterCount = PACKED_GET(traversalInfo.packedNode, Node_packed_groupClusterCountMinusOne) + 1;

  uint geometryID   = instances[traversalInfo.instanceID].geometryID;
  Geometry geometry = geometries[geometryID];

  // retrieve traversal & culling related information from the child node or cluster
  TraversalMetric traversalMetric;
#if USE_CULLING && TARGETS_RASTERIZATION
  BBox bbox;
#endif

  mat4 worldMatrix   = instances[traversalInfo.instanceID].worldMatrix;
  float uniformScale = computeUniformScale(worldMatrix);
  float errorScale   = 1.0;
#if USE_CULLING && TARGETS_RAY_TRACING
  uint visibilityState = build.instanceVisibility.d[traversalInfo.instanceID];
  // instance is not primary visible, apply different error scale
  if (visibilityState == 0) errorScale = build.culledErrorScale;
#endif
  mat4x3 traversalMatrix = mat4x3(build.traversalViewMatrix * worldMatrix);

#if USE_STREAMING
  // The later `if (traverseNode)` branch ensured that this pointer is valid
  // and we never traverse to a group that isn't resident.
  Group group = Group_in(geometry.streamingGroupAddresses.d[groupIndex]).d;
  streaming.resident.groups.d[group.residentID].age = 0;
#else
  // can directly access the group
  Group group = geometry.preloadedGroups.d[groupIndex];
#endif

  for (uint clusterIndex = 0; clusterIndex < groupClusterCount; clusterIndex++)
  {
    bool forceCluster = false;
    bool isValid = true;
    
    {
    #if USE_CULLING && TARGETS_RASTERIZATION
      bbox        = group.clusterBboxes.d[clusterIndex];
    #endif
      
      // The continous lod algorithm optimizes to get the lowest detail we can get away with.
      
      // We render a cluster if its own group was traversed because it had an error
      // greater than the threshold (it is "coarse enough"). This is fulfilled when reach
      // the code here.
      //
      // However, multiple cluster groups of previous lod levels (higher detail) may cover this 
      // same region. Therefore we must ensure that it's really this cluster to be drawn (it is "fine enough").
      //
      // This is achieved by looking at the cluster's generating group. The generating group
      // contained the geometry that this cluster was simplified from and is from the previous,
      // lower, lod level with a lower error.
      //
      // If that group wasn't traversed then we know we must be drawn, because we have the highest
      // detail required. You will see a bit later down that we use the negated results 
      // of `testForTraversal` for clusters.
      //
      // If this cluster is from the highest detail level, then there is no generating group
      // as encoded by `SHADERIO_ORIGINAL_MESH_GROUP`.
      // In streaming, it may also occur that the generating group isn't loaded, that also
      // means this cluster is the highest detail available.
      
      uint32_t clusterGeneratingGroup = group.clusterGeneratingGroups.d[clusterIndex];
    #if USE_STREAMING
      if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP
          && geometry.streamingGroupAddresses.d[clusterGeneratingGroup] < STREAMING_INVALID_ADDRESS_START)
      {
        // streaming must check if the other group actually is resident, if not then we always draw this group
        // as we know no other lod variant was loaded.
        traversalMetric = Group_in(geometry.streamingGroupAddresses.d[clusterGeneratingGroup]).d.traversalMetric;
      }
    #else
      if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP)
      {
        traversalMetric = geometry.preloadedGroups.d[clusterGeneratingGroup].traversalMetric;
      }
    #endif
      else {
        // the generating group doesn't exist, draw this group
        
        // this should always evaluate true
        traversalMetric = group.traversalMetric;
        forceCluster    = true;
      }
      // prepare to append this cluster for rendering, if metric evaluates properly
      
      // TraversalInfo aliases with ClusterInfo, packeNode == clusterID
      traversalInfo.packedNode = group.clusterResidentID + clusterIndex;
    }

    // perform traversal & culling logic  
  #if USE_CULLING && TARGETS_RASTERIZATION
    isValid            = isValid && queryWasVisible(worldMatrix, bbox);
  #endif
    bool traverse      = testForTraversal(traversalMatrix, uniformScale, traversalMetric, errorScale);
    bool renderCluster = isValid && (!traverse || forceCluster);  // clusters use negated test or are forced
    
    // nodes will enqueue their children again (producer)
    // groups will write out the clusters for rendering
    
    // we use subgroup intrinsics to avoid doing per-thread
    // atomics to get the storage offsets
    
    uvec4 voteClusters  = subgroupBallot(renderCluster);
    uint countClusters  = subgroupBallotBitCount(voteClusters);
    uint offsetClusters = 0;
    
    if (subgroupElect())
    {
      offsetClusters = atomicAdd(buildRW.renderClusterCounter, countClusters);
    }

    offsetClusters = subgroupBroadcastFirst(offsetClusters);
    offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);
    
    renderCluster = renderCluster && offsetClusters < build.maxRenderClusters;

  #if TARGETS_RAY_TRACING
    if (renderCluster)
    {
      // For ray tracing count how many clusters we later add to each instance/blas.
      // this will help us determine the list length for each blas.
      // The `blas_setup_insertion.comp.glsl` kernel then sub-allocates space for the lists
      // based on this counter.
      atomicAdd(build.instanceBuildInfos.d[traversalInfo.instanceID].clusterReferencesCount, 1);
      // the render list we write below is filled in an unsorted manner with clusters
      // from different instances. We later use the `blas_insert_clusters.comp.glsl` kernel to build
      // the list for each blas.
    }
  #endif
    
    if (renderCluster)
    {  
      // given TraversalInfo and ClusterInfo were chosen to alias in memory and be a single u64
      // we do just have to adjust the output addresses.
      
      uint writeIndex          = offsetClusters;
      uint64s_coh writePointer = uint64s_coh(uint64_t(build.renderClusterInfos));
      
    #if USE_ATOMIC_LOAD_STORE
      atomicStore(writePointer.d[writeIndex], packTraversalInfo(traversalInfo), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    #else
      writePointer.d[writeIndex] = packTraversalInfo(traversalInfo);
    #endif
      memoryBarrierBuffer();
    }
  }
}