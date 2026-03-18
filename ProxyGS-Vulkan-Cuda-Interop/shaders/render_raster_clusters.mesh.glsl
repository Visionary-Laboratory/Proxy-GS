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

  This mesh shader renders a single cluster.

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

#extension GL_NV_mesh_shader : require
#extension GL_EXT_control_flow_attributes : require

#include "shaderio.h"
#include "octant_encoding.h"

layout(push_constant) uniform pushData
{
  uint instanceID;
}
push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar,binding=BINDINGS_READBACK_SSBO,set=0) buffer readbackBuffer
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

layout(location = 0) out Interpolants
{
#if ALLOW_SHADING
  vec3      wPos;
#if ALLOW_VERTEX_NORMALS
  vec3      wNormal;
#endif
#endif
  flat uint clusterID;
  flat uint instanceID;
}
OUT[];

////////////////////////////////////////////

#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE 32
#endif

layout(local_size_x = MESHSHADER_WORKGROUP_SIZE) in;
layout(max_vertices = CLUSTER_VERTEX_COUNT, max_primitives = CLUSTER_TRIANGLE_COUNT) out;
layout(triangles) out;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);

////////////////////////////////////////////

void main()
{
  ClusterInfo cinfo = build.renderClusterInfos.d[gl_WorkGroupID.x];

  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;

  RenderInstance instance = instances[instanceID];
  Geometry geometry       = geometries[instance.geometryID];

#if USE_STREAMING
  Cluster cluster = Cluster_in(streaming.resident.clusters.d[clusterID]).d;
#else
  Cluster cluster = geometry.preloadedClusters.d[clusterID];
#endif

  uint vertMax = cluster.vertexCountMinusOne;
  uint triMax  = cluster.triangleCountMinusOne;

  if (gl_LocalInvocationID.x == 0) {
    gl_PrimitiveCountNV = triMax + 1;
  #if USE_RENDER_STATS
    atomicAdd(readback.numRenderedTriangles, uint(triMax + 1));
  #endif
  }

  vec4s_in  oVertices      = vec4s_in(cluster.vertices);
  uint8s_in localTriangles = uint8s_in(cluster.localTriangles);

  mat4 worldMatrix   = instance.worldMatrix;
  mat3 worldMatrixIT = transpose(inverse(mat3(worldMatrix)));


  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert        = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint vertLoad    = min(vert, vertMax);
    
    vec4 oVertex = oVertices.d[vertLoad];

    vec3 oPos = oVertex.xyz;    
    vec4 wPos = worldMatrix * vec4(oPos, 1.0f);
  #if ALLOW_VERTEX_NORMALS
    vec3 oNormal = oct32_to_vec(floatBitsToUint(oVertex.w));
  #endif

    if(vert <= vertMax)
    {
      gl_MeshVerticesNV[vert].gl_Position = view.viewProjMatrix * wPos;
    #if ALLOW_SHADING
      OUT[vert].wPos                      = wPos.xyz;
    #if ALLOW_VERTEX_NORMALS
      OUT[vert].wNormal                   = normalize(worldMatrixIT * oNormal);
    #endif
    #endif
      OUT[vert].clusterID                 = clusterID;
      OUT[vert].instanceID                = instanceID;
    }
  }

  [[unroll]] for(uint i = 0; i < uint(MESHLET_TRIANGLE_ITERATIONS); i++)
  {
    uint tri     = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint triLoad = min(tri, triMax);

    uvec3 indices = uvec3(localTriangles.d[triLoad * 3 + 0],
                          localTriangles.d[triLoad * 3 + 1],
                          localTriangles.d[triLoad * 3 + 2]);

    if(tri <= triMax)
    {
      gl_PrimitiveIndicesNV[tri * 3 + 0] = indices.x;
      gl_PrimitiveIndicesNV[tri * 3 + 1] = indices.y;
      gl_PrimitiveIndicesNV[tri * 3 + 2] = indices.z;
      gl_MeshPrimitivesNV[tri].gl_PrimitiveID = int(tri);
    }
  }
}