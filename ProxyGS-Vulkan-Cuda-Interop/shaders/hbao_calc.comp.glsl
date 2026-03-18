/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/* 
Based on DeinterleavedTexturing sample by Louis Bavoil
https://github.com/NVIDIAGameWorks/D3DSamples/tree/master/samples/DeinterleavedTexturing

*/

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require

#include "hbao.h"

layout(push_constant) uniform pushData {
  NVHBAOMainPush  push;
};

#define M_PI 3.14159265f

// tweakables
const float  NUM_STEPS = 12;
const float  NUM_DIRECTIONS = NVHBAO_NUM_DIRECTIONS; // texRandom/g_Jitter initialization depends on this

layout(binding=NVHBAO_MAIN_TEX_DEPTHARRAY)   uniform sampler2DArray texLinearDepth;
layout(binding=NVHBAO_MAIN_TEX_VIEWNORMAL)   uniform sampler2D      texViewNormal;


#if NVHBAO_SKIP_INTERPASS
  #if NVHBAO_BLUR
    layout(binding=NVHBAO_MAIN_IMG_RESULT,rg16f) uniform image2D imgOutput;
  #else
    layout(binding=NVHBAO_MAIN_IMG_RESULT,r8)    uniform image2D imgOutput;
  #endif
  void outputColor(ivec2 icoord, vec4 color)
  {
    icoord = icoord * 4 + ivec2(push.layer & 3, push.layer / 4);
    if (icoord.x < control.FullResolution.x && icoord.y < control.FullResolution.y){
      imageStore(imgOutput, icoord, color);
    }
  }
#else
  #if NVHBAO_BLUR
    layout(binding=NVHBAO_MAIN_IMG_RESULTARRAY,rg16f) uniform image2DArray imgOutput;
  #else
    layout(binding=NVHBAO_MAIN_IMG_RESULTARRAY,r8)    uniform image2DArray imgOutput;
  #endif
  void outputColor(ivec2 icoord, vec4 color)
  {
    imageStore(imgOutput, ivec3(icoord, push.layer), color);
  }
#endif


vec2 g_Float2Offset = control.float2Offsets[push.layer].xy;
vec4 g_Jitter       = control.jitters[push.layer];

vec3 getQuarterCoord(vec2 UV){
  return vec3(UV,float(push.layer));
}


//----------------------------------------------------------------------------------

vec3 UVToView(vec2 uv, float eye_z)
{
  return vec3((uv * control.projInfo.xy + control.projInfo.zw) * (control.projOrtho != 0 ? 1. : eye_z), eye_z);
}

vec3 FetchQuarterResViewPos(vec2 UV)
{
  float ViewDepth = textureLod(texLinearDepth,getQuarterCoord(UV),0).x;
  return UVToView(UV, ViewDepth);
}

//----------------------------------------------------------------------------------
float Falloff(float DistanceSquare)
{
  // 1 scalar mad instruction
  return DistanceSquare * control.NegInvR2 + 1.0;
}

//----------------------------------------------------------------------------------
// P = view-space position at the kernel center
// N = view-space normal at the kernel center
// S = view-space position of the current sample
//----------------------------------------------------------------------------------
float ComputeAO(vec3 P, vec3 N, vec3 S)
{
  vec3 V = S - P;
  float VdotV = dot(V, V);
  float NdotV = dot(N, V) * 1.0/sqrt(VdotV);

  // Use saturate(x) instead of max(x,0.f) because that is faster on Kepler
  return clamp(NdotV - control.NDotVBias,0,1) * clamp(Falloff(VdotV),0,1);
}

//----------------------------------------------------------------------------------
vec2 RotateDirection(vec2 Dir, vec2 CosSin)
{
  return vec2(Dir.x*CosSin.x - Dir.y*CosSin.y,
              Dir.x*CosSin.y + Dir.y*CosSin.x);
}

//----------------------------------------------------------------------------------
vec4 GetJitter()
{
  // Get the current jitter vector from the per-pass constant buffer
  return g_Jitter;
}

//----------------------------------------------------------------------------------
float ComputeCoarseAO(vec2 FullResUV, float RadiusPixels, vec4 Rand, vec3 ViewPosition, vec3 ViewNormal)
{
  RadiusPixels /= 4.0;

  // Divide by NUM_STEPS+1 so that the farthest samples are not fully attenuated
  float StepSizePixels = RadiusPixels / (NUM_STEPS + 1);

  const float Alpha = 2.0 * M_PI / NUM_DIRECTIONS;
  float AO = 0;

  [[unroll]]
  for (float DirectionIndex = 0; DirectionIndex < NUM_DIRECTIONS; ++DirectionIndex)
  {
    float Angle = Alpha * DirectionIndex;

    // Compute normalized 2D direction
    vec2 Direction = RotateDirection(vec2(cos(Angle), sin(Angle)), Rand.xy);

    // Jitter starting sample within the first step
    float RayPixels = (Rand.z * StepSizePixels + 1.0);

    for (float StepIndex = 0; StepIndex < NUM_STEPS; ++StepIndex)
    {
      vec2 SnappedUV = round(RayPixels * Direction) * control.InvQuarterResolution + FullResUV;
      vec3 S = FetchQuarterResViewPos(SnappedUV);

      RayPixels += StepSizePixels;

      AO += ComputeAO(ViewPosition, ViewNormal, S);
    }
  }

  AO *= control.AOMultiplier / (NUM_DIRECTIONS * NUM_STEPS);
  return clamp(1.0 - AO * 2.0,0,1);
}

//----------------------------------------------------------------------------------
void main()
{
  ivec2 intCoord;
  vec2  texCoord;
  
  if (setupCoordQuarter(intCoord, texCoord)) return;
  
  vec2 base = vec2(intCoord.xy) * 4.0 + g_Float2Offset;
  vec2 uv = base * (control.InvQuarterResolution / 4.0);

  vec3 ViewPosition = FetchQuarterResViewPos(uv);
  vec4 NormalAndAO =  texelFetch( texViewNormal, ivec2(base), 0);
  vec3 ViewNormal =  -(NormalAndAO.xyz * 2.0 - 1.0);

  // Compute projection of disk of radius control.R into screen space
  float RadiusPixels = control.RadiusToScreen / (control.projOrtho != 0 ? 1.0 : ViewPosition.z);

  // Get jitter vector for the current full-res pixel
  vec4 Rand = GetJitter();

  float AO = ComputeCoarseAO(uv, RadiusPixels, Rand, ViewPosition, ViewNormal);

#if NVHBAO_BLUR
  outputColor(intCoord, vec4(pow(AO, control.PowExponent), ViewPosition.z, 0, 0));
#else
  outputColor(intCoord, vec4(pow(AO, control.PowExponent)));
#endif
  
}
