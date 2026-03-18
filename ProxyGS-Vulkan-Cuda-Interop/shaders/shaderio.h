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

#ifndef _SHADERIO_H_
#define _SHADERIO_H_

#include "shaderio_core.h"
#include "shaderio_scene.h"
#include "shaderio_streaming.h"
#include "shaderio_building.h"
#include "nvshaders/sky_io.h.slang"

/////////////////////////////////////////

#define ALLOW_SHADING 1
#define ALLOW_VERTEX_NORMALS 1

/////////////////////////////////////////

#define VISUALIZE_MATERIAL 0
#define VISUALIZE_GREY 1
#define VISUALIZE_CLUSTER 2
#define VISUALIZE_GROUP 3
#define VISUALIZE_LOD 4
#define VISUALIZE_TRIANGLE 5
#define VISUALIZE_BLAS 6

#define BBOXES_PER_MESHLET 8

/////////////////////////////////////////

#define BINDINGS_FRAME_UBO 0
#define BINDINGS_READBACK_SSBO 1
#define BINDINGS_GEOMETRIES_SSBO 2
#define BINDINGS_RENDERINSTANCES_SSBO 3
#define BINDINGS_SCENEBUILDING_SSBO 4
#define BINDINGS_SCENEBUILDING_UBO 5
#define BINDINGS_HIZ_TEX 6
#define BINDINGS_STREAMING_UBO 7
#define BINDINGS_STREAMING_SSBO 8
#define BINDINGS_TLAS 9
#define BINDINGS_RAYTRACING_DEPTH 10
// DLSS buffers start here as well
#define BINDINGS_RENDER_TARGET 11

/////////////////////////////////////////

#define BUILD_SETUP_TRAVERSAL_RUN 1
#define BUILD_SETUP_DRAW 2
#define BUILD_SETUP_BLAS_INSERTION 3

/////////////////////////////////////////

#define STREAM_SETUP_COMPACTION_OLD_NO_UNLOADS 0
#define STREAM_SETUP_COMPACTION_STATUS 1
#define STREAM_SETUP_ALLOCATOR_FREEINSERT 2
#define STREAM_SETUP_ALLOCATOR_STATUS 3

/////////////////////////////////////////

#define TRAVERSAL_PRESORT_WORKGROUP 128
#define TRAVERSAL_INIT_WORKGROUP 64
#define TRAVERSAL_RUN_WORKGROUP 64
#define TRAVERSAL_GROUPS_WORKGROUP 64
#define BLAS_SETUP_INSERTION_WORKGROUP 64
#define BLAS_INSERT_CLUSTERS_WORKGROUP 64
#define INSTANCES_ASSIGN_BLAS_WORKGROUP 64
#define INSTANCES_CLASSIFY_LOD_WORKGROUP 64
#define GEOMETRY_BLAS_SHARING_WORKGROUP 64

// must be power of 2
#define STREAM_UPDATE_SCENE_WORKGROUP 64
#define STREAM_AGEFILTER_GROUPS_WORKGROUP 128
#define STREAM_COMPACTION_NEW_CLAS_WORKGROUP 128
#define STREAM_COMPACTION_OLD_CLAS_WORKGROUP 64
#define STREAM_ALLOCATOR_LOAD_GROUPS_WORKGROUP 64
#define STREAM_ALLOCATOR_UNLOAD_GROUPS_WORKGROUP 64
#define STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP 64
#define STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP 64
#define STREAM_ALLOCATOR_SETUP_INSERTION_WORKGROUP 64

/////////////////////////////////////////

#ifndef USE_RENDER_STATS
#define USE_RENDER_STATS 1
#endif

#ifndef USE_MEMORY_STATS
#define USE_MEMORY_STATS 1
#endif

#ifndef USE_CULLING
#define USE_CULLING 1
#endif

#ifndef USE_INSTANCE_SORTING
#define USE_INSTANCE_SORTING 1
#endif

#ifndef USE_BLAS_SHARING
#define USE_BLAS_SHARING 1
#endif

#ifndef USE_STREAMING
#define USE_STREAMING 1
#endif

#ifndef MAX_VISIBLE_CLUSTERS
#define MAX_VISIBLE_CLUSTERS 1024
#endif

#ifndef TARGETS_RASTERIZATION
#define TARGETS_RASTERIZATION 0
#endif

#define TARGETS_RAY_TRACING (!(TARGETS_RASTERIZATION))

/////////////////////////////////////////

#ifdef __cplusplus
namespace shaderio {
using namespace glm;

#else

#ifndef USE_DLSS
#define USE_DLSS 1
#endif

struct RayPayload
{
  // Ray gen writes the direction through the pixel at x+1 for ray differentials.
  // Closest hit returns the shaded color there.
  vec3  color;
  float hitT;
#if DEBUG_VISUALIZATION
  // Ray direction through the pixel at y+1 for ray differentials
  vec4 differentialY;
#endif
#if USE_DLSS
  vec4 dlssNormalRoughness;
  vec4 dlssAlbedo;
  vec3 dlssSpecular;
#endif
};

#endif

struct FrameConstants
{
  mat4 projMatrix;
  mat4 projMatrixI;

  mat4 viewProjMatrix;
  mat4 viewProjMatrixI;
  mat4 viewMatrix;
  mat4 viewMatrixI;
  vec4 viewPos;
  vec4 viewDir;
  vec4 viewPlane;

  mat4 skyProjMatrixI;

  // for motion vectors
  mat4 viewProjMatrixPrev;

  ivec2 viewport;
  vec2  viewportf;

  vec2 viewPixelSize;
  vec2 viewClipSize;

  vec2 jitter;
  vec2 _pad;

  vec3  wLightPos;
  float lightMixer;

  vec3  wUpDir;
  float sceneSize;

  vec4 wMirrorBox;

  uint  flipWinding;
  uint  useMirrorBox;
  uint  visualize;
  float fov;

  float   nearPlane;
  float   farPlane;
  float   ambientOcclusionRadius;
  int32_t ambientOcclusionSamples;

  vec4 hizSizeFactors;
  vec4 nearSizeFactors;

  float hizSizeMax;
  int   facetShading;
  int   supersample;
  uint  colorXor;

  uint  dbgUint;
  float dbgFloat;
  uint  frame;
  uint  doShadow;

  vec4 bgColor;

  uvec2 mousePosition;
  float wireThickness;
  float wireSmoothing;

  vec3 wireColor;
  uint wireStipple;

  vec3  wireBackfaceColor;
  float wireStippleRepeats;

  float wireStippleLength;
  uint  doWireframe;
  uint  visFilterInstanceID;
  uint  visFilterClusterID;

  SkySimpleParameters skyParams;
};

struct Readback
{
  uint     numRenderClusters;
  uint     numTraversalInfos;
  uint     numRenderedClusters;
  uint     numBlasBuilds;
  uint64_t numRenderedTriangles;

  uint64_t blasActualSizes;

#ifdef __cplusplus
  uint32_t clusterTriangleId;
  uint32_t _packedDepth0;

  uint32_t instanceId;
  uint32_t _packedDepth1;
#else
  uint64_t clusterTriangleId;
  uint64_t instanceId;
#endif

  uint64_t debugU64;

  int  debugI;
  uint debugUI;
  uint debugF;

  uint debugA[64];
  uint debugB[64];
  uint debugC[64];
};


#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_H_
