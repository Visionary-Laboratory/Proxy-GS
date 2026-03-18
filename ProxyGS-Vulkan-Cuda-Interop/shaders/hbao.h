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

#ifndef NVHBAO_H_
#define NVHBAO_H_

#define NVHBAO_RANDOMTEX_SIZE 4
#define NVHBAO_NUM_DIRECTIONS 8

#define NVHBAO_MAIN_UBO 0
#define NVHBAO_MAIN_TEX_DEPTH 1
#define NVHBAO_MAIN_TEX_LINDEPTH 2
#define NVHBAO_MAIN_TEX_VIEWNORMAL 3
#define NVHBAO_MAIN_TEX_DEPTHARRAY 4
#define NVHBAO_MAIN_TEX_RESULTARRAY 5
#define NVHBAO_MAIN_TEX_RESULT 6
#define NVHBAO_MAIN_TEX_BLUR 7
#define NVHBAO_MAIN_IMG_LINDEPTH 8
#define NVHBAO_MAIN_IMG_VIEWNORMAL 9
#define NVHBAO_MAIN_IMG_DEPTHARRAY 10
#define NVHBAO_MAIN_IMG_RESULTARRAY 11
#define NVHBAO_MAIN_IMG_RESULT 12
#define NVHBAO_MAIN_IMG_BLUR 13
#define NVHBAO_MAIN_IMG_OUT 14

#ifndef NVHBAO_BLUR
#define NVHBAO_BLUR 1
#endif

// 1 is slower
#ifndef NVHBAO_SKIP_INTERPASS
#define NVHBAO_SKIP_INTERPASS 0
#endif

#ifdef __cplusplus
namespace glsl {
using namespace glm;
#endif

struct NVHBAOData
{
  float RadiusToScreen;  // radius
  float R2;              // 1/radius
  float NegInvR2;        // radius * radius
  float NDotVBias;

  vec2 InvFullResolution;
  vec2 InvQuarterResolution;

  ivec2 SourceResolutionScale;
  float AOMultiplier;
  float PowExponent;

  vec4  projReconstruct;
  vec4  projInfo;
  int   projOrtho;
  int   _pad0;
  ivec2 _pad1;

  ivec2 FullResolution;
  ivec2 QuarterResolution;

  mat4 InvProjMatrix;

  vec4 float2Offsets[NVHBAO_RANDOMTEX_SIZE * NVHBAO_RANDOMTEX_SIZE];
  vec4 jitters[NVHBAO_RANDOMTEX_SIZE * NVHBAO_RANDOMTEX_SIZE];
};

// keep all these equal size
struct NVHBAOMainPush
{
  int   layer;
  int   _pad0;
  ivec2 _pad1;
};

struct NVHBAOBlurPush
{
  vec2  invResolutionDirection;
  float sharpness;
  float _pad;
};

#ifdef __cplusplus
}
#else

layout(std140, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData control;
};

#ifndef NVHABO_GFX

layout(local_size_x = 32, local_size_y = 2) in;

bool setupCoord(inout ivec2 coord, inout vec2 texCoord, ivec2 res, vec2 invRes)
{
  ivec2 base   = ivec2(gl_WorkGroupID.xy) * 8;
  ivec2 subset = ivec2(int(gl_LocalInvocationID.x) & 1, int(gl_LocalInvocationID.x) / 2);
  subset += gl_LocalInvocationID.x >= 16 ? ivec2(2, -8) : ivec2(0, 0);
  subset += ivec2(gl_LocalInvocationID.y * 4, 0);

  coord = base + subset;

  if(coord.x >= res.x || coord.y >= res.y)
    return true;

  texCoord = (vec2(coord) + vec2(0.5)) * invRes;

  return false;
}

bool setupCoordFull(inout ivec2 coord, inout vec2 texCoord)
{
  return setupCoord(coord, texCoord, control.FullResolution, control.InvFullResolution);
}

bool setupCoordQuarter(inout ivec2 coord, inout vec2 texCoord)
{
  return setupCoord(coord, texCoord, control.QuarterResolution, control.InvQuarterResolution);
}

#endif

#endif
#endif
