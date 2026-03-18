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

  This compute shader sets up the per BLAS cluster references list start pointer.
  It does so by simply adding up the per-blas references counts that were filled during 
  `traversal_run.comp.glsl`.
  These count values are also reset, so that the `blas_clusters_insert.comp.glsl` kernel
  can increment them again when filling the lists.
  
  A single thread represents one BLAS
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

////////////////////////////////////////////

layout(local_size_x=BLAS_SETUP_INSERTION_WORKGROUP) in;

////////////////////////////////////////////


void main()
{
  uint instanceID = gl_GlobalInvocationID.x;
  
  if (instanceID < build.numRenderInstances)
  {
    uint referencesCount  = build.instanceBuildInfos.d[instanceID].clusterReferencesCount;
    if (referencesCount > 0)
    {
      uint referencesOffset = atomicAdd(buildRW.blasClasCounter, referencesCount);
      uint buildOffset      = atomicAdd(buildRW.blasBuildCounter, 1);
      
      // reset count for insertion pass
      build.blasBuildInfos.d[buildOffset].clusterReferencesCount  = 0;
      build.blasBuildInfos.d[buildOffset].clusterReferencesStride = 8;
      build.blasBuildInfos.d[buildOffset].clusterReferences       = uint64_t(build.blasClusterAddresses) + uint64_t(referencesOffset * 8);
      
      build.instanceBuildInfos.d[instanceID].blasBuildIndex = buildOffset;
    }
  }
}