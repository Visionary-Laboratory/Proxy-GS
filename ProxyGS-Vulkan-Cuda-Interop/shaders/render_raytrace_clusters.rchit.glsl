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
  
  This hit shader handles the shading of clusters in
  ray tracing. 
  
  Note the use of a new input: `gl_ClusterIDNV`
  
*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_tracing_position_fetch : require

#extension GL_EXT_spirv_intrinsics : require

// at the time of writing, no GLSL extension was available, we leverage 
// GL_EXT_spirv_intrinsics to hook up the new builtin.
#extension GL_EXT_spirv_intrinsics : require

// Note that `VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV::allowClusterAccelerationStructure` must
// be set to `VK_TRUE` to make this valid.
spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

// While not required in this sample, as we use dedicated hit-shader for clusters,
// `int gl_ClusterIDNoneNV = -1;` can be used to dynamically detect regular hits.


#include "shaderio.h"
#include "octant_encoding.h"

/////////////////////////////////

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

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
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

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;


/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

/////////////////////////////////

layout(location = 0) rayPayloadInEXT RayPayload rayHit;
layout(location = 1) rayPayloadEXT float rayHitAO;

/////////////////////////////////


#define SUPPORTS_RT 1

#if USE_DLSS
#include "dlss_util.h"
#endif

#include "render_shading.glsl"

/////////////////////////////////

void main()
{
  // get IDs
  uint clusterID  = gl_ClusterIDNV_;
  uint instanceID = gl_InstanceID;
  uint triangleID = gl_PrimitiveID;

  RenderInstance instance = instances[instanceID];
  Geometry geometry       = geometries[instance.geometryID];

  // Fetch cluster header
#if USE_STREAMING
  // dereference the cluster from the resident cluster table
  Cluster cluster = Cluster_in(streaming.resident.clusters.d[clusterID]).d;
#else
  // access the cluster data directly from the preloaded array 
  Cluster cluster = geometry.preloadedClusters.d[clusterID];
#endif

  vec4s_in  oVertices      = vec4s_in(cluster.vertices);
  uint8s_in localTriangles = uint8s_in(cluster.localTriangles);
  
  uvec3 triangleIndices    = uvec3(localTriangles.d[triangleID * 3 + 0],
                                   localTriangles.d[triangleID * 3 + 1],
                                   localTriangles.d[triangleID * 3 + 2]);

  vec3 baryWeight = vec3((1.f - barycentrics[0] - barycentrics[1]), barycentrics[0], barycentrics[1]);

  vec3 oPos = baryWeight.x * gl_HitTriangleVertexPositionsEXT[0] +
              baryWeight.y * gl_HitTriangleVertexPositionsEXT[1] + 
              baryWeight.z * gl_HitTriangleVertexPositionsEXT[2];    
  vec3 wPos = vec3(gl_ObjectToWorldEXT * vec4(oPos, 1.0));

  vec3 oNormal;
  bool backFacing = false;
#if ALLOW_VERTEX_NORMALS
  if(view.facetShading != 0 || view.flipWinding == 2)
#endif
  {
    // Otherwise compute geometric normal
    vec3 e0 = gl_HitTriangleVertexPositionsEXT[1] - gl_HitTriangleVertexPositionsEXT[0];
    vec3 e1 = gl_HitTriangleVertexPositionsEXT[2] - gl_HitTriangleVertexPositionsEXT[0];
    oNormal    = normalize(cross(e0, e1));
    
    backFacing = dot(oNormal, gl_ObjectRayDirectionEXT) > 0;
  }
#if ALLOW_VERTEX_NORMALS
  if(view.facetShading == 0)
  {
    oNormal = baryWeight.x * oct32_to_vec(floatBitsToUint(oVertices.d[triangleIndices.x].w)) + 
              baryWeight.y * oct32_to_vec(floatBitsToUint(oVertices.d[triangleIndices.y].w)) + 
              baryWeight.z * oct32_to_vec(floatBitsToUint(oVertices.d[triangleIndices.z].w));
  }
#endif

  mat3 worldMatrixIT = transpose(inverse(mat3(instance.worldMatrix)));

  vec3 wNormal = normalize(vec3(worldMatrixIT * oNormal));
  if(view.flipWinding == 1 || (view.flipWinding == 2 && backFacing))
  {
    wNormal = -wNormal;
  }
  
  uint visData = clusterID;
  if (view.visualize == VISUALIZE_LOD || view.visualize == VISUALIZE_GROUP)
  {
    if (view.visualize == VISUALIZE_LOD)
    {
      visData = floatBitsToUint(float(cluster.lodLevel) * instances[instanceID].maxLodLevelRcp);
    }
    else {
      visData = cluster.groupID;
    }
  }
  else if (view.visualize == VISUALIZE_TRIANGLE)
  {
    visData = clusterID * 256 + uint(triangleID);
  }

  vec4 shaded;
#if ALLOW_SHADING
  {
    float ambientOcclusion =
        ambientOcclusion(wPos, wNormal, view.ambientOcclusionSamples, view.ambientOcclusionRadius * view.sceneSize);

    float sunContribution  = 1.0;
    vec3  directionToLight = view.skyParams.sunDirection;
    if(view.doShadow == 1)
      sunContribution = traceShadowRay(wPos, wNormal, directionToLight);

    shaded = shading(instanceID, wPos, wNormal, visData, sunContribution, ambientOcclusion
    #if USE_DLSS
      , rayHit.dlssAlbedo, rayHit.dlssSpecular, rayHit.dlssNormalRoughness
    #endif
    );
  }
#else
  shaded = vec4(visualizeColor(visData), 1.0);
#endif

#if DEBUG_VISUALIZATION

  if(view.doWireframe != 0 || (view.visFilterInstanceID == instanceID && view.visFilterClusterID == clusterID))
  {
    vec3 derivativeTargetX = gl_WorldToObjectEXT * vec4(gl_WorldRayOriginEXT + rayHit.color.xyz, 1);
    vec3 derivativeDirX    = derivativeTargetX.xyz - gl_ObjectRayOriginEXT;
    vec3 derivativeX = intersectRayTriangle(gl_ObjectRayOriginEXT, derivativeDirX, gl_HitTriangleVertexPositionsEXT[0], gl_HitTriangleVertexPositionsEXT[1], gl_HitTriangleVertexPositionsEXT[2]);
    derivativeX = abs(derivativeX - baryWeight);


    vec3 derivativeTargetY = gl_WorldToObjectEXT * vec4(gl_WorldRayOriginEXT + rayHit.differentialY.xyz, 1);
    vec3 derivativeDirY    = derivativeTargetY.xyz - gl_ObjectRayOriginEXT;
    vec3 derivativeY = intersectRayTriangle(gl_ObjectRayOriginEXT, derivativeDirY, gl_HitTriangleVertexPositionsEXT[0], gl_HitTriangleVertexPositionsEXT[1], gl_HitTriangleVertexPositionsEXT[2]);
    derivativeY = abs(derivativeY - baryWeight);

    vec3 derivative = max(derivativeX, derivativeY);

    rayHit.color.xyz = addWireframe(shaded.xyz, baryWeight, true, derivative, view.wireColor);
  }
  else
#endif
  {
    rayHit.color.xyz = shaded.xyz;
  }

  if(gl_LaunchIDEXT.xy == view.mousePosition)
  {
    vec4  projected            = (view.viewProjMatrix * vec4(wPos, 1.f));
    float depth                = projected.z / projected.w;
    readback.clusterTriangleId = packPickingValue((clusterID << 8) | triangleID, depth);
    readback.instanceId        = packPickingValue(instanceID, depth);
  }

  rayHit.hitT = gl_HitTEXT;
}