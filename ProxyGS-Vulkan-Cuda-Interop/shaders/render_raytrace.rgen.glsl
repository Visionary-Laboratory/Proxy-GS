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

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"
#if USE_DLSS
#include "dlss_util.h"
#endif

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba8)   uniform image2D imgColor;
#if USE_DLSS
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssAlbedo, rgba8)             uniform image2D imgDlssAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecAlbedo, rgba16f)       uniform image2D imgDlssSpecAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssNormalRoughness, rgba16f)  uniform image2D imgDlssNormalRoughness;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssMotion, rg16f)             uniform image2D imgDlssMotion;
#endif

//////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT RayPayload rayHit;

//////////////////////////////////////////////////////////////

float hitBbox(vec3 rayPos, vec3 rayDir, vec3 bboxMin, vec3 bboxMax)
{
  vec3  invDir = 1.0 / (rayDir);
  vec3  tbot   = invDir * (bboxMin - rayPos);
  vec3  ttop   = invDir * (bboxMax - rayPos);
  vec3  tmin   = min(ttop, tbot);
  vec3  tmax   = max(ttop, tbot);
  float t0     = max(tmin.x, max(tmin.y, tmin.z));
  float t1     = min(tmax.x, min(tmax.y, tmax.z));
  
  return t1 > max(t0, 0.0) ? t0 : -1.0;
}

void main()
{
  // for writing debugging values to stats.debug etc.
  bool center = gl_LaunchIDEXT.xy == (gl_LaunchSizeEXT.xy / 2);
  vec2 pixelOffset = vec2(0.5);
  
#if USE_DLSS
  pixelOffset += view.jitter;
#endif

  ivec2 screen = ivec2(gl_LaunchIDEXT.xy);
  vec2  uv     = (vec2(gl_LaunchIDEXT.xy) + pixelOffset) / vec2(gl_LaunchSizeEXT.xy);


  vec2 d = uv * 2.0 - 1.0;


  vec4 origin    = view.viewMatrixI * vec4(0, 0, 0, 1);
  vec4 target    = normalize(view.projMatrixI * vec4(d.x, d.y, 1, 1));
  vec4 direction = normalize(view.viewMatrixI * vec4(target.xyz, 0));

  float tMin = view.nearPlane;
  float tMax = view.farPlane;

#if DEBUG_VISUALIZATION
  vec2 uvOffset            = (vec2(gl_LaunchIDEXT.xy) + vec2(1.5, 1.5)) / vec2(gl_LaunchSizeEXT.xy);
  vec2 dOffset             = uvOffset * 2.0 - 1.0;
  vec4 targetOffsetX       = normalize(view.projMatrixI * vec4(dOffset.x, d.y, 1, 1));
  vec4 targetOffsetY       = normalize(view.projMatrixI * vec4(d.x, dOffset.y, 1, 1));
  vec4 directionOffsetX    = normalize(view.viewMatrixI * vec4(targetOffsetX.xyz, 0));
  vec4 directionOffsetY    = normalize(view.viewMatrixI * vec4(targetOffsetY.xyz, 0));
  rayHit.color.xyz         = directionOffsetX.xyz;
  rayHit.differentialY.xyz = directionOffsetY.xyz;
#endif

  vec3  mirrorCenter = view.wMirrorBox.xyz;
  float mirrorSize   = view.wMirrorBox.w;
  vec3 mirrorHitPoint;
  vec3 mirrorNormal;
  bool mirrorHit = false;
  float mirrorT;
  
  if (view.useMirrorBox != 0)
  {
    mirrorT = hitBbox(origin.xyz, direction.xyz, mirrorCenter - mirrorSize, mirrorCenter + mirrorSize);
    mirrorHit = mirrorT > 0;
    if (mirrorHit)
    {
      mirrorHitPoint = origin.xyz + direction.xyz * mirrorT;    
      mirrorNormal   = mirrorHitPoint - mirrorCenter;
      vec3 signs     = sign(mirrorNormal);
      mirrorNormal   = abs(mirrorNormal);
      mirrorNormal   = vec3(equal(mirrorNormal, vec3(max(max(mirrorNormal.x, mirrorNormal.y),mirrorNormal.z)))) * signs;
      
      tMax = mirrorT;
    }
  }

  traceRayEXT(asScene, view.flipWinding == 2 ? 0 : gl_RayFlagsCullBackFacingTrianglesEXT, 0xff, 0, 0,  // hit offset, hit stride
              0,                                                           // miss offset
              origin.xyz, tMin, direction.xyz, tMax,
              0  // rayPayloadNV location qualifier
  );
  
  if (view.useMirrorBox != 0)
  {
    if (rayHit.hitT == 0 && mirrorHit)
    {
      traceRayEXT(asScene, view.flipWinding == 2 ? 0 : gl_RayFlagsCullBackFacingTrianglesEXT, 0xff, 0, 0,  // hit offset, hit stride
                0,                                                           // miss offset
                mirrorHitPoint, 0, reflect(direction.xyz, mirrorNormal), view.farPlane + mirrorT,
                0  // rayPayloadNV location qualifier
      );
      
      rayHit.color.xyz *= max(0,dot(mirrorNormal,-direction.xyz)) * 0.5 + 0.5;
      rayHit.hitT = mirrorT;
    }
  }
  
  bool  hitValid = rayHit.hitT != 0;  
  vec3  hitPos   = origin.xyz + direction.xyz * (hitValid ? rayHit.hitT : view.farPlane * 0.99f);
  float hitDepth = 1;
  
  if (hitValid) {
    vec4 screenPos = view.viewProjMatrix * vec4(hitPos, 1);
    hitDepth = screenPos.z / screenPos.w;
  }
  
  imageStore(imgColor, screen, vec4(rayHit.color.xyz, 1));
  imageStore(imgRaytracingDepth, screen, vec4(hitDepth, 0.f, 0.f, 0.f));
  
#if USE_DLSS
  vec2 motionVec = vec2(0);
  
  motionVec = calculateMotionVector(hitPos, view.viewProjMatrixPrev,
                                    view.viewProjMatrix, view.viewportf);

  imageStore(imgDlssMotion, screen, vec4(motionVec.x, motionVec.y, 0, 0));
  imageStore(imgDlssAlbedo, screen, rayHit.dlssAlbedo);
  imageStore(imgDlssSpecAlbedo, screen, vec4(rayHit.dlssSpecular, 1.0f));
  imageStore(imgDlssNormalRoughness, screen, rayHit.dlssNormalRoughness);
#endif
  
}