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
  
  This compute shader assigns the blasReference address to each
  tlas instance description prior updating the tlas and after
  the per-frame blas were built.

  A single thread represents one instance
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

layout(local_size_x=INSTANCES_ASSIGN_BLAS_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint instanceID = gl_GlobalInvocationID.x;
  
  if (instanceID < build.numRenderInstances)
  {
    InstanceBuildInfo buildInfo = build.instanceBuildInfos.d[instanceID];
    uint buildIndex             = buildInfo.blasBuildIndex;
    
    bool doStats = true;
    
  #if USE_BLAS_SHARING    
    // we might reference another instance's blas
    if (buildIndex != BLAS_BUILD_INDEX_LOWDETAIL && (buildIndex & BLAS_BUILD_INDEX_SHARE_BIT) != 0)
    {
      uint shareInstanceID = buildIndex & ~(BLAS_BUILD_INDEX_SHARE_BIT);
      buildInfo  = build.instanceBuildInfos.d[shareInstanceID];
      buildIndex = buildInfo.blasBuildIndex;
      
      // don't add to build stats if we are referencing another instance
      doStats = false;
    }
  #endif
    
    // By default tlasInstances are set to low detail blas,
    // override when applicable.
    
    if (buildInfo.clusterReferencesCount > 0)
    {
    #if USE_MEMORY_STATS
      if (doStats)
      {
        atomicAdd(readback.blasActualSizes, uint64_t(build.blasBuildSizes.d[buildIndex]));
      }
    #endif
      
      build.tlasInstances.d[instanceID].blasReference = build.blasBuildAddresses.d[buildIndex];
    }
  }
  
#if 1
  // stats
  if (instanceID == 0)
  {
    readback.numBlasBuilds = build.blasBuildCounter;
  }
#endif
}