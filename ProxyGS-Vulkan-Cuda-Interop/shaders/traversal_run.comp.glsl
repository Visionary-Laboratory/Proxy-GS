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

  This compute shader handles the scene's lod hiearchy traversal for all
  instances.

  The shader is configured to be run using persistent threads. A fixed amount of 
  threads implement a producer/consumer queuing mechanism to handle the hierarchical
  traversal.

  The producer/consumer queue is implemented by the following variables:
  
    - `build.traversalNodeInfos` stores the items that are processed as linear array
    - `build.traversalNodeWriteCounter` is used to produce new items into the array
    - `build.traversalNodeReadCounter` is used to consume from the array
    - `build.traversalTaskCounter` tracks the total number of tasks in-flight.
      It wil be incremented when new tasks are enqueued, and decremented when they are consumed.
      When it reaches zero we will have no more work left to process and the kernel can complete.

  The queue is seeded within `traversal_init.comp.glsl` with the root's of visible instances.
  
  Furthermore all clusters that are to be rendered are output via:
    - `build.renderClusterInfos` stores all clusters that are to be rendered as linear array
    - `build.renderClusterCounter` is used to append the clusters

 
  The traversal logic attempts to consume items and then tests if their children
  need further processing: further node traversal, or enqueuing into the list of rendered
  clusters.
  
  `shaderio::Node` is the same as `nvclusterlod::Node`
  
  The lod hierarchy is organized in such fashion that it represents different lod levels
  in parallel. It is not strictly spatial. It is spatial over each lod's cluster groups
  and then all lod levels' node trees are then linked together at the root of the hierarchy.
  
  This allows parallel testing of all lod levels at the same time. The evaluation of
  the lod metric is carefully chosen so that only "one" version of a lod cluster can win.
  This means we cannot accidentally mix clusters of different lod levels that represent
  the same piece of a mesh.
  
  If the input item was a node, the evaluation of the traversal metric means
  we may need to traverse deeper (higher-detail is desired and possible) and 
  produce new traversal work by adding the node's children into `build.traversalNodeInfos`. 
  This provides the upper range of the cut through the lod's directed-acyclic-graph (DAG).
  
  If the input item was a group, then the its children are clusters and the evaluation
  of their traversal metric yields the opposite, lower range, of the cut through the DAG.
  Meaning we are detailed enough.
  These clusters are appended to `build.renderClusterInfos`.
  
  The combination of both evaluations ensures we don't accidentally create overlapping lod clusters.
  
  Please refer to [A Deep Dive into Nanite Virtualized Geometry, Karis et al. 2021](https://www.advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf)
  for more details.
  
  
  USE_SEPARATE_GROUPS
  
  When active, two kernels are used for the traversal.
  The hierarchical node traversal is handled within this kernel, 
  but the leaves (cluster groups and their clusters)
  are processed in `traversal_run_separate_groups.comp.glsl`.
  
  This reduces divergence and can speed things up overall.

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

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  volatile SceneBuilding buildRW;  
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

layout(local_size_x=TRAVERSAL_RUN_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////

// work around compiler bug on older drivers not properly handling coherent & volatile
#define USE_ATOMIC_LOAD_STORE 1

////////////////////////////////////////////

// Computes the number of children for an incoming node / group task.
// These children are then processed within `processSubTask` a few lines down
uint setupTask(inout TraversalInfo traversalInfo, uint readIndex, uint pass)
{
  uint subCount = 0;
  
  bool isNode = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) == 0;
  if (isNode) {
    subCount = PACKED_GET(traversalInfo.packedNode, Node_packed_nodeChildCountMinusOne);
  }
  else {
    subCount = PACKED_GET(traversalInfo.packedNode, Node_packed_groupClusterCountMinusOne);
  }

  return subCount + 1;
}

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

void processSubTask(const TraversalInfo subgroupTasks, uint taskID, uint taskSubID, bool isValid, uint threadReadIndex, uint pass)
{
  // This function handles the primary traversal work operating on a single child of the
  // input item.
  //
  // If the input was a inner node (isNode), then we will test if we need to descend the hierarchy
  // further for this child node.
  // 
  // If the input was a leaf (!isNode) then we will test the lod metric and see if we can append
  // the child cluster for rendering.
  //
  //
  // Each thread is a child (`taskSubID`) of an incoming traversal task (`taskID`).
  // All tasks are stored in registers across the subgroup within `subgroupTasks`.
  // we access what we need via shuffle.
  //
  // The last `taskSubID` may be repeated when `isValid == false`, to allow safe memory access
  // for reads.
  // `threadReadIndex` and `pass` are only meant to aid debugging


  // pull required input item from subgroupTasks
  TraversalInfo traversalInfo;
  traversalInfo.instanceID               = subgroupShuffle(subgroupTasks.instanceID, taskID);
  traversalInfo.packedNode               = subgroupShuffle(subgroupTasks.packedNode, taskID);

  // identify type of item: node or group
  bool isNode  = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) == 0;
#if USE_SEPARATE_GROUPS
  // in this mode this kernel only processes nodes
  isNode = true;
#endif

  uint geometryID   = instances[traversalInfo.instanceID].geometryID;
  Geometry geometry = geometries[geometryID];

  // retrieve traversal & culling related information from the child node or cluster
  TraversalMetric traversalMetric;
#if USE_CULLING && TARGETS_RASTERIZATION
  BBox bbox;
#endif

  bool forceCluster = false;

  if (isNode)
  {
    uint childIndex     = taskSubID;
    uint childNodeIndex = PACKED_GET(traversalInfo.packedNode, Node_packed_nodeChildOffset) + childIndex;
    Node childNode      = geometry.nodes.d[childNodeIndex];
    traversalMetric     = childNode.traversalMetric;
  #if USE_CULLING && TARGETS_RASTERIZATION
    bbox                = geometry.nodeBboxes.d[childNodeIndex];
  #endif
    // prepare to enqueue this child node later, if metric evaluates properly
    traversalInfo.packedNode = childNode.packed;
  }
#if !USE_SEPARATE_GROUPS
  else {
    // a cluster group may be processed as well, in that case
    // we test the children clusters.
  
    uint clusterIndex = taskSubID;
    uint groupIndex   = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
  #if USE_STREAMING
    // The later `if (traverseNode)` branch ensured that this pointer is valid
    // and we never traverse to a group that isn't resident.
    Group group = Group_in(geometry.streamingGroupAddresses.d[groupIndex]).d;
    streaming.resident.groups.d[group.residentID].age = 0;
  #else
    // can directly access the group
    Group group = geometry.preloadedGroups.d[groupIndex];
  #endif

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
#endif
  
  // perform traversal & culling logic
  
  mat4 worldMatrix   = instances[traversalInfo.instanceID].worldMatrix;
  float uniformScale = computeUniformScale(worldMatrix);
  float errorScale   = 1.0;
#if USE_CULLING && TARGETS_RASTERIZATION
  isValid            = isValid && queryWasVisible(worldMatrix, bbox);
#elif USE_CULLING && TARGETS_RAY_TRACING
  uint visibilityState = build.instanceVisibility.d[traversalInfo.instanceID];
  // instance is not primary visible, apply different error scale
  if (visibilityState == 0) errorScale = build.culledErrorScale;
#endif
  bool traverse      = testForTraversal(mat4x3(build.traversalViewMatrix * worldMatrix), uniformScale, traversalMetric, errorScale);
  bool traverseNode  = isValid && isNode && traverse;    // nodes test if we can descend
  bool renderCluster = isValid && !isNode && (!traverse || forceCluster);  // clusters use negated test or are forced

#if USE_STREAMING
  if (traverseNode)
  {
    // when streaming we might need to bail out here, if the child group isn't resident
    bool isGroup    = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) != 0;
    uint groupIndex = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
  
  #if USE_STREAMING
    // streamingGroupAddresses[groupIndex] encodes two things:
    //
    //   if the address is <  STREAMING_INVALID_ADDRESS_START:
    //      then it's valid and the group is resident and we can dereference it.
    //   if the address is >= STREAMING_INVALID_ADDRESS_START:
    //      then it then the group is not resident and the lower bits encode the frame 
    //      index when the group was requested last.
    if (isGroup)
    {
      // This address read is cached, which is fine as during traversal we only care if an address was invalid and
      // traversal alone cannot change the residency.
      // Even if we manipulate it later via the atomicMax, it still remains "invalid" given the number will remain higher.
      uint64_t groupAddress = geometry.streamingGroupAddresses.d[groupIndex];
      
      if (groupAddress >= STREAMING_INVALID_ADDRESS_START) {
        // not streamed in yet, cannot process this group.
        traverseNode = false;
        
        // This operation is uncached, since we use it to test if a request was made within this frame to the same address already.
        // requestFrameIndex is always >= STREAMING_INVALID_ADDRESS_START
        uint64_t lastRequestFrameIndex = atomicMax(geometry.streamingGroupAddresses.d[groupIndex], streaming.request.frameIndex);
        
        // we haven't made the request this frame, so trigger it
        bool triggerRequest = lastRequestFrameIndex != streaming.request.frameIndex;
        
        uvec4 voteRequested  = subgroupBallot(triggerRequest);
        uint  countRequested = subgroupBallotBitCount(voteRequested);
        uint offsetRequested = 0;
        if (subgroupElect()) {
          offsetRequested = atomicAdd(streamingRW.request.loadCounter, countRequested);
        }
        offsetRequested = subgroupBroadcastFirst(offsetRequested);
        offsetRequested += subgroupBallotExclusiveBitCount(voteRequested);
        
        if (triggerRequest && offsetRequested <= streaming.request.maxLoads) {
          // while streaming data is based on geometry 
          streaming.request.loadGeometryGroups.d[offsetRequested] = uvec2(geometryID, groupIndex);
        }
      }
    }
  #endif
  }
#endif

#if USE_SEPARATE_GROUPS
  // In this mode we enqueue groups separately from nodes.
  // we hijack the "cluster enqueue" mechanism for this.
  //
  // `renderCluster` now means `traverseGroup`
  
  renderCluster = isValid && traverseNode && PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) != 0;
  if (renderCluster)
    traverseNode = false;
#endif
  
  // nodes will enqueue their children again (producer)
  // groups will write out the clusters for rendering
  
  // we use subgroup intrinsics to avoid doing per-thread
  // atomics to get the storage offsets
  
  uvec4 voteNodes = subgroupBallot(traverseNode);
  uint countNodes = subgroupBallotBitCount(voteNodes);
  
  uvec4 voteClusters = subgroupBallot(renderCluster);
  uint countClusters = subgroupBallotBitCount(voteClusters);
  
  uint offsetNodes    = 0;
  uint offsetClusters = 0;
  
  if (subgroupElect())
  {
    // increase global task counter
    atomicAdd(buildRW.traversalTaskCounter, int(countNodes));
    // get memory offsets
    offsetNodes    = atomicAdd(buildRW.traversalInfoWriteCounter, countNodes);
  #if USE_SEPARATE_GROUPS
    offsetClusters = atomicAdd(buildRW.traversalGroupCounter, countClusters);
  #else
    offsetClusters = atomicAdd(buildRW.renderClusterCounter, countClusters);
  #endif
  }
  memoryBarrierBuffer();
  
  offsetNodes = subgroupBroadcastFirst(offsetNodes);
  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);
  offsetClusters = subgroupBroadcastFirst(offsetClusters);
  offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);
  
  // verify if we actually have output space left
  
  traverseNode  = traverseNode && offsetNodes < build.maxTraversalInfos;
  
#if USE_SEPARATE_GROUPS
  // `renderCluster` means `traverseGroup` here
  renderCluster = renderCluster && offsetClusters < build.maxTraversalInfos;
#else
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
#endif
  
  // by design a thread cannot be a node and a cluster at same time.
    
  bool doStore = traverseNode || renderCluster;
  
  if (doStore)
  {  
    // given TraversalInfo and ClusterInfo were chosen to alias in memory and be a single u64
    // we do just have to adjust the output addresses.
    
    uint writeIndex          = traverseNode ? offsetNodes : offsetClusters;
    uint64s_coh writePointer = uint64s_coh(traverseNode ? uint64_t(build.traversalNodeInfos) 
    #if USE_SEPARATE_GROUPS
    : uint64_t(build.traversalGroupInfos)
    #else
    : uint64_t(build.renderClusterInfos)
    #endif
    );
    
  #if USE_ATOMIC_LOAD_STORE
    atomicStore(writePointer.d[writeIndex], packTraversalInfo(traversalInfo), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
  #else
    writePointer.d[writeIndex] = packTraversalInfo(traversalInfo);
  #endif
    
    memoryBarrierBuffer();
  }
}

////////////////////////////////////////////

// Following section contains a mechanism to distribute
// work across the warp for better efficiency.
// Rather than processing children in per-thread loops
// we process the sum of all children iteratively across the warp.
//
// imagine three threads with 4,2,1 children
// looping individually means we may get poor SIMT utilization.
// 
//  T0 T1 T2
// ----------
//  A0 B0 C0
//  A1 B1 
//  A2
//  A3
//
//  packing across warp
//
//  TO T1 T2 T3 T4 T5 T6
// ---------------------
//  A0 A1 A2 A3 B0 B1 C0

struct TaskInfo {
  uint taskID;
};

shared TaskInfo s_tasks[TRAVERSAL_RUN_WORKGROUP];


void processAllSubTasks(inout TraversalInfo traversalInfo, bool threadRunnable, int threadSubCount, uint threadReadIndex, uint pass)
{
  // Distribute new work across subgroup.
  // Each task may have a variable number of threads to be run.
  // We pack them tightly over a minimum amount of warp iterations.
  //
  // `threadReadIndex` and `pass` are only meant to aid debugging
  
  // This algorithm is described in detail in `vk_tessellated_clusters/shaders/render_raster_clusters_batched.mesh.glsl`
  
  int endOffset    = subgroupInclusiveAdd(threadSubCount);
  int startOffset  = endOffset - threadSubCount;
  int totalThreads = subgroupShuffle(endOffset, SUBGROUP_SIZE-1);
  int totalRuns    = (totalThreads + SUBGROUP_SIZE-1) / SUBGROUP_SIZE;
  
  const uint subgroupOffset = gl_SubgroupID * gl_SubgroupSize;

  bool hasTask     = threadSubCount > 0;
  uvec4 taskVote   = subgroupBallot(hasTask);
  uint taskCount   = subgroupBallotBitCount(taskVote);
  uint taskOffset  = subgroupBallotExclusiveBitCount(taskVote);
  
  if (hasTask) {
    s_tasks[subgroupOffset + taskOffset].taskID = gl_SubgroupInvocationID;
  }
  
  memoryBarrierShared();
  
  uint sumEnqueues = 0;
  
  // technique computes a total number of virtual worker threads
  // then iterates over all those. Within each iteration we check
  // which original task a thread belongs to.
  // After that we compute the relative offset to that task's start
  // which yields the child index / taskSubID.
  
  int taskBase = -1;
  for (int r = 0; r < totalRuns; r++)
  {
  
    int tFirst = r * SUBGROUP_SIZE;
    int t      = tFirst + int(gl_SubgroupInvocationID);
    
    int  relStart      = startOffset - tFirst;
    // set bit where task starts if within current run
    uint startBits     = subgroupOr(threadRunnable && relStart >= 0 && relStart < 32 ? (1 << relStart) : 0);
    
    int  task          = bitCount(startBits & gl_SubgroupLeMask.x) + taskBase;
    uint taskID        = s_tasks[subgroupOffset + task].taskID;
    
    uint taskSubID     = t - subgroupShuffle(startOffset, taskID);
    uint taskSubCount  = subgroupShuffle(threadSubCount, taskID);
  #if 0
    // only relevant for debugging
    uint taskReadIndex = subgroupShuffle(threadReadIndex, taskID); 
  #else
    uint taskReadIndex = 0;
  #endif
    taskBase           = subgroupShuffle(task, 31); // for next iteration
    
    bool taskValid     = taskSubID < taskSubCount;
    
    // do work
    processSubTask(traversalInfo, taskID, min(taskSubID,taskSubCount-1), taskValid, taskReadIndex, pass);
  }
}

////////////////////////////////////////////

void run()
{
  // This implements a persistent kernel that implements
  // a producer/consumer loop.
  //
  // special thanks to Robert Toth for the core setup.

  // the read index for the global array of `build.traversalNodeInfos`
  uint threadReadIndex = ~0;
  
  for(uint pass = 0; ; pass++)
  {
    // try to consume
  
    // if entire subgroup has no work, acquire new work
  
    if (subgroupAll(threadReadIndex == ~0)) {
      // pull new work
      if (subgroupElect()){
        threadReadIndex = atomicAdd(buildRW.traversalInfoReadCounter, SUBGROUP_SIZE);
      }
      threadReadIndex = subgroupBroadcastFirst(threadReadIndex) + gl_SubgroupInvocationID;
      threadReadIndex = threadReadIndex >= build.maxTraversalInfos ? ~0 : threadReadIndex;
      
      // if all read offsets are out of bounds, we are done for sure
      
      if (subgroupAll(threadReadIndex == ~0)){
        break;
      }
    }
  
    // let's attempt to fetch some valid work from the current state of `threadReadIndex`
  
    bool threadRunnable = false;
    TraversalInfo nodeTraversalInfo;
    
    while(true)
    {   
      if (threadReadIndex != ~0)
      {
        memoryBarrierBuffer();
        // get traversal info
      #if USE_ATOMIC_LOAD_STORE
        uint64_t rawValue = atomicLoad(build.traversalNodeInfos.d[threadReadIndex], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
      #else
        uint64_t rawValue = build.traversalNodeInfos.d[threadReadIndex];
      #endif
        nodeTraversalInfo = unpackTraversalInfo(rawValue);
        
        // reading is ahead of writing, might not have finished writing and value is still the cleared value
        threadRunnable    = nodeTraversalInfo.instanceID != ~0u && nodeTraversalInfo.packedNode != ~0u;
      }
      
      if (subgroupAny(threadRunnable))
        break;
      
      // Entire warp saw no valid work.
      // We always race ahead with reads compared to writes, but we may also
      // simply have no actual tasks left.
      
      memoryBarrierBuffer();
    #if USE_ATOMIC_LOAD_STORE
      bool isEmpty = atomicLoad(buildRW.traversalTaskCounter, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) == 0;
    #else
      bool isEmpty = buildRW.traversalTaskCounter == 0;
    #endif
      if (subgroupAny(isEmpty))
      {
        return;
      }
    }
    
    // some threads have data ready to consume
    
    if (subgroupAny(threadRunnable))
    {
      // each thread sets up a task with a variable number of children
      // this can be child nodes for an incoming node
      // or the clusters of an incoming group
      
      int threadSubCount = 0;
      
      if (threadRunnable)
      {
        threadSubCount = int(setupTask(nodeTraversalInfo, threadReadIndex, pass));
      }
      
      // Now process all tasks, we do this in a packed fashion, so that
      // we attempt to fill the warp densely. This results in processing over work
      // in multiple iterations within the warp. As a result tasks may
      // straddle the warp.
      
      // we currently mix node/group tasks across the warp
      // this should be mostly okayish as both require traversal logic to be run
      // which depends on the same input data types.
      
      processAllSubTasks(nodeTraversalInfo, threadRunnable, threadSubCount, threadReadIndex, pass);
      
      // All processed items need to decrement the global task counter
      // and reset their complete state.
      
      uint numRunnable = subgroupBallotBitCount(subgroupBallot(threadRunnable));
      
      if (subgroupElect()) {
        atomicAdd(buildRW.traversalTaskCounter, -int(numRunnable));
      }
      
      if (threadRunnable) {
        // reset read index to invalid
        threadReadIndex = ~0;
      }
    }
  }
}

void main()
{
  run();
  
#if USE_SEPARATE_GROUPS
  if (gl_GlobalInvocationID.x == 0) {
    // this sets up the grid for `traversal_run_separate_groups.comp.glsl`
  
    uint groupCount = atomicAdd(buildRW.traversalGroupCounter,0);
    groupCount = min(groupCount,build.maxTraversalInfos);
    buildRW.indirectDispatchGroups.gridX = (groupCount + TRAVERSAL_GROUPS_WORKGROUP - 1) / TRAVERSAL_GROUPS_WORKGROUP;
  }
#endif
}