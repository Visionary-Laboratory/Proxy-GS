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
  
  This compute shader does basic operations on a single thread.
  For example clamping atomic counters back to their limits or
  setting up indirect dispatches or draws etc.
  
  BUILD_SETUP_... are enums for the various operations

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
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint setup;
} push;

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
  SceneBuilding buildRW;  
};

////////////////////////////////////////////

layout(local_size_x=1) in;

////////////////////////////////////////////

void main()
{  
  // special operations for setting up indirect dispatches
  // or clamping other operations to actual limits
  
  if (push.setup == BUILD_SETUP_TRAVERSAL_RUN)
  {
    // during traversal_init we might overshoot the traversalTaskCounter  
    int traversalTaskCounter = min(buildRW.traversalTaskCounter, int(build.maxTraversalInfos));
    buildRW.traversalTaskCounter = traversalTaskCounter;
    // also set up the initial writeCounter to be equal, so that new jobs are enqueued after it
    buildRW.traversalInfoWriteCounter = uint(traversalTaskCounter);
  }
#if TARGETS_RASTERIZATION
  else if (push.setup == BUILD_SETUP_DRAW)
  {
    // during traversal_run we might overshoot visibleClusterCounter  
    uint renderClusterCounter = buildRW.renderClusterCounter;
    
    // set drawindirect for actual rendered clusters
    uint numRenderedClusters = min(renderClusterCounter, build.maxRenderClusters);
    
    buildRW.indirectDrawClusters.count = numRenderedClusters;
    buildRW.indirectDrawClusters.first = 0;

    // keep originals for array size warnings 
    readback.numRenderClusters    = renderClusterCounter;
  #if USE_SEPARATE_GROUPS
    readback.numTraversalInfos    = max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter);
  #else
    readback.numTraversalInfos    = buildRW.traversalInfoWriteCounter;
  #endif

  #if USE_RENDER_STATS
    readback.numRenderedClusters  = numRenderedClusters;
  #endif
  }
#endif
#if TARGETS_RAY_TRACING
  else if (push.setup == BUILD_SETUP_BLAS_INSERTION)
  {
    // during traversal_run we might overshoot visibleClusterCounter  
    uint renderClusterCounter = buildRW.renderClusterCounter;
    
    // set drawindirect for actual rendered clusters
    uint numRenderedClusters = min(renderClusterCounter, build.maxRenderClusters);
    
    buildRW.renderClusterCounter = numRenderedClusters;
    buildRW.indirectDispatchBlasInsertion.gridX = (numRenderedClusters + BLAS_INSERT_CLUSTERS_WORKGROUP-1) / BLAS_INSERT_CLUSTERS_WORKGROUP;
    buildRW.indirectDispatchBlasInsertion.gridY = 1;
    buildRW.indirectDispatchBlasInsertion.gridZ = 1;

    // keep originals for array size warnings 
    readback.numRenderClusters    = renderClusterCounter;
  #if USE_SEPARATE_GROUPS
    readback.numTraversalInfos    = max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter);
  #else
    readback.numTraversalInfos    = buildRW.traversalInfoWriteCounter;
  #endif

  #if USE_STATS
    readback.numRenderedClusters  += numRenderedClusters; // + because we sometimes use atomicAdd on this directly
  #endif
  }
#endif
}