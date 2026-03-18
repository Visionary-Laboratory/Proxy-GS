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
  
  This compute shader computes the distance of the instance to the camera.

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

layout(local_size_x=TRAVERSAL_PRESORT_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint instanceID = gl_GlobalInvocationID.x;
  uint instanceLoad = min(build.numRenderInstances-1, instanceID);
  
  RenderInstance instance = instances[instanceLoad];
  Geometry geometry = geometries[instance.geometryID];
  
  mat4 worldToObject = inverse(instance.worldMatrix);
  
  vec3 oPos = (worldToObject * vec4(view.viewPos.xyz,1)).xyz;
  
  bool isInside = all(equal(greaterThanEqual(oPos, geometry.bbox.lo),lessThanEqual(oPos, geometry.bbox.hi)));
  
  vec3 oPosClamp = isInside ? (geometry.bbox.lo + geometry.bbox.hi) * 0.5 :
    clamp(oPos, geometry.bbox.lo, geometry.bbox.hi);
  
  vec4 wPos = instance.worldMatrix * vec4(oPosClamp, 1);
  
  if (instanceID == instanceLoad) {
    build.instanceSortValues.d[instanceID] = instanceID;
    build.instanceSortKeys.d[instanceID]   = floatBitsToUint(distance(wPos.xyz, view.viewPos.xyz));
  }
}