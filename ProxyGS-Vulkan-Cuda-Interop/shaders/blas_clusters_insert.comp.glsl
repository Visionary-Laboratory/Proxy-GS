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
  
  This compute shader inserts the CLAS clusters that should be rendered
  into the cluster references list for each instance's BLAS.

  A single thread represents one CLAS
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

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
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

layout(local_size_x=BLAS_INSERT_CLUSTERS_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint renderClusterIndex = gl_GlobalInvocationID.x;
  
  if (renderClusterIndex < build.renderClusterCounter)
  {
    ClusterInfo cluster       = build.renderClusterInfos.d[renderClusterIndex];
    uint instanceID           = cluster.instanceID;
    uint clusterID            = cluster.clusterID;
  #if USE_STREAMING
    uint64_t clusterAddress   = streaming.resident.clasAddresses.d[clusterID];
  #else
    Geometry geometry         = geometries[instances[instanceID].geometryID];
    uint64_t clusterAddress   = geometry.preloadedClusterClasAddresses.d[clusterID];
  #endif
    
    uint buildIndex = build.instanceBuildInfos.d[instanceID].blasBuildIndex;
  #if USE_BLAS_SHARING && USE_RENDER_STATS
    uint instanceUseCount = build.instanceBuildInfos.d[instanceID].instanceUseCount;
  #else
    uint instanceUseCount = 1;
  #endif
    
    uint idx = atomicAdd(build.blasBuildInfos.d[buildIndex].clusterReferencesCount,1);
    uint64s_inout clusterReferences = uint64s_inout(build.blasBuildInfos.d[buildIndex].clusterReferences);
    clusterReferences.d[idx] = clusterAddress;
    
  #if USE_RENDER_STATS
    #if USE_STREAMING
      uint numTriangles = Cluster_in(streaming.resident.clusters.d[clusterID]).d.triangleCountMinusOne + 1;
    #else
      uint numTriangles = geometry.preloadedClusters.d[clusterID].triangleCountMinusOne + 1;
    #endif
    atomicAdd(readback.numRenderedTriangles, numTriangles * instanceUseCount);
    #if USE_BLAS_SHARING
      atomicAdd(readback.numRenderedClusters, instanceUseCount - 1);
    #endif
  #endif
  }
}