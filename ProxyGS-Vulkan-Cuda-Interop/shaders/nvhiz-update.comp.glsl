/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#ifndef NV_HIZ_MAX_LEVELS
#define NV_HIZ_MAX_LEVELS   16
#endif

#ifndef NV_HIZ_MSAA_SAMPLES
#define NV_HIZ_MSAA_SAMPLES 0
#endif

#ifndef NV_HIZ_IS_FIRST
#define NV_HIZ_IS_FIRST 1
#endif

#ifndef NV_HIZ_FORMAT
#define NV_HIZ_FORMAT r32f
#endif

#ifndef NV_HIZ_OUTPUT_NEAR
#define NV_HIZ_OUTPUT_NEAR 1
#endif

#ifndef NV_HIZ_LEVELS 
#define NV_HIZ_LEVELS 3
#endif

#ifndef NV_HIZ_NEAR_LEVEL
#define NV_HIZ_NEAR_LEVEL 0
#endif

#ifndef NV_HIZ_FAR_LEVEL
#define NV_HIZ_FAR_LEVEL 0
#endif

#ifndef NV_HIZ_REVERSED_Z
#define NV_HIZ_REVERSED_Z 0
#endif

#ifndef NV_HIZ_USE_STEREO 
#define NV_HIZ_USE_STEREO 0
#endif

#if NV_HIZ_LEVELS > 1
  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_shuffle : require
#endif

#if NV_HIZ_REVERSED_Z
  #define minOp max
  #define maxOp min
#else
  #define minOp min
  #define maxOp max
#endif

layout(local_size_x=32,local_size_y=2) in;

layout(push_constant) uniform passUniforms {
  // keep in sync with nvhiz_vk.cpp
  ivec4 srcSize;
  int   writeLod;
  int   startLod;
  int   layer;
  int   _pad0;
  bvec4  levelActive;
};

#if NV_HIZ_USE_STEREO
  #define samplerTypeMS sampler2DMSArray
  #define samplerType   sampler2DArray
  #define imageType     image2DArray
  #define IACCESS(v,l)  ivec3(v,l)
#else
  #define samplerTypeMS sampler2DMS
  #define samplerType   sampler2D
  #define imageType     image2D
  #define IACCESS(v,l)  v
#endif

#if NV_HIZ_IS_FIRST && NV_HIZ_MSAA_SAMPLES
  layout(binding=0) uniform samplerTypeMS texDepth;
#else
  layout(binding=0) uniform samplerType   texDepth;
#endif
  layout(binding=1) uniform samplerType   texNear;
  
  layout(binding=2,NV_HIZ_FORMAT) uniform imageType imgNear;
  layout(binding=3,NV_HIZ_FORMAT) uniform imageType imgLevels[NV_HIZ_MAX_LEVELS];

void main()
{
  ivec2 base = ivec2(gl_WorkGroupID.xy) * 8;
  ivec2 subset = ivec2(int(gl_LocalInvocationID.x) & 1, int(gl_LocalInvocationID.x) / 2);
  subset += gl_LocalInvocationID.x >= 16 ? ivec2(2,-8) : ivec2(0,0);
  subset += ivec2(gl_LocalInvocationID.y * 4,0);
  
#if NV_HIZ_LEVELS > 1
  uint laneID = gl_SubgroupInvocationID;
#endif

  //ivec2 outcoord = base + 7 - subset;
  ivec2 outcoord = base + subset;
  ivec2 coord = outcoord * 2;
  
  float flayer = float(layer);
  
#if NV_HIZ_IS_FIRST && NV_HIZ_MSAA_SAMPLES
  #if NV_HIZ_REVERSED_Z
  float zMin = 0;
  float zMax = 1;
  #else
  float zMin = 1;
  float zMax = 0;
  #endif
  for (int i = 0; i < NV_HIZ_MSAA_SAMPLES; i++){
    vec4 zRead = vec4(texelFetch(texDepth, IACCESS(min(coord + ivec2(0,0), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(1,0), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(0,1), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(1,1), srcSize.zw), layer), i).r);
    zMin = minOp(zMin, minOp(minOp(minOp(zRead.x, zRead.y),zRead.z),zRead.w));
    zMax = maxOp(zMax, maxOp(maxOp(maxOp(zRead.x, zRead.y),zRead.z),zRead.w));
  }
#else
  #if NV_HIZ_IS_FIRST
    #define texRead texDepth
  #else
    #define texRead texNear
  #endif

  coord = min(coord, srcSize.zw);
  vec4 zRead = vec4(texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(0,0)).r,
                    texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(1,0)).r,
                    texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(0,1)).r,
                    texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(1,1)).r);
  
  float zMax = maxOp(maxOp(maxOp(zRead.x, zRead.y),zRead.z),zRead.w);
  float zMin = minOp(minOp(minOp(zRead.x, zRead.y),zRead.z),zRead.w);
#endif

  //zMax = float(gl_ThreadInWarpNV) / 32.0;
#if !(NV_HIZ_IS_FIRST && NV_HIZ_FAR_LEVEL > 0)
  imageStore(imgLevels[writeLod + 0], IACCESS(outcoord,layer), vec4(zMax));
#endif
  
#if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL == 0
  imageStore(imgNear, IACCESS(outcoord,layer), vec4(zMin));
#endif

#if NV_HIZ_LEVELS > 1
  vec4 zRead0 = vec4( zMax,
                      subgroupShuffle(zMax, laneID + 1),
                      subgroupShuffle(zMax, laneID + 2),
                      subgroupShuffle(zMax, laneID + 3));
  

#if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL >= 1
  vec4 zRead1 = vec4( zMin,
                      subgroupShuffle(zMin, laneID + 1),
                      subgroupShuffle(zMin, laneID + 2),
                      subgroupShuffle(zMin, laneID + 3));
#endif

  if ((levelActive.y || levelActive.z) && (laneID & 3) == 0)
  {
    outcoord /= 2;
    zMax = maxOp(maxOp(maxOp(zRead0.x, zRead0.y),zRead0.z),zRead0.w);
  #if !(NV_HIZ_IS_FIRST && NV_HIZ_FAR_LEVEL > 1)
    imageStore(imgLevels[writeLod + 1], IACCESS(outcoord, layer), vec4(zMax));
  #endif
  #if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL >= 1
    zMin = minOp(minOp(minOp(zRead1.x, zRead1.y),zRead1.z),zRead1.w);
    #if NV_HIZ_NEAR_LEVEL == 1
    imageStore(imgNear, IACCESS(outcoord, layer), vec4(zMin));
    #endif
  #endif
    
  #if NV_HIZ_LEVELS > 2
    if (levelActive.z) {
      outcoord /= 2;
      zRead0 = vec4(  zMax,
                      subgroupShuffle(zMax, laneID + 4),
                      subgroupShuffle(zMax, laneID + 16),
                      subgroupShuffle(zMax, laneID + 20));
    #if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL == 2
      zRead1 = vec4(  zMin,
                      subgroupShuffle(zMin, laneID + 4),
                      subgroupShuffle(zMin, laneID + 16),
                      subgroupShuffle(zMin, laneID + 20));
    #endif
      if ((laneID == 0) || (laneID == 8)) {
        zMax = maxOp(maxOp(maxOp(zRead0.x, zRead0.y),zRead0.z),zRead0.w);
        imageStore(imgLevels[writeLod + 2], IACCESS(outcoord, layer), vec4(zMax));
      #if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL == 2
        zMin = minOp(minOp(minOp(zRead1.x, zRead1.y),zRead1.z),zRead1.w);
        imageStore(imgNear, IACCESS(outcoord, layer), vec4(zMin));
      #endif
      }
    }
  #endif
  }
#endif
}
