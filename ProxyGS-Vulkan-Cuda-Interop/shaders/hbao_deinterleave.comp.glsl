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

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require

#include "hbao.h"

layout(binding=NVHBAO_MAIN_TEX_LINDEPTH)         uniform sampler2D      texLinearDepth;
layout(binding=NVHBAO_MAIN_IMG_DEPTHARRAY,r32f)  uniform image2DArray   imgDepthArray;

//----------------------------------------------------------------------------------

void outputColor(ivec2 intCoord, int layer, float value)
{
  imageStore(imgDepthArray, ivec3(intCoord,layer), vec4(value,0,0,0));
}

void main()
{
  ivec2 intCoord;
  vec2  texCoord;
  
  if (setupCoordQuarter(intCoord, texCoord)) return;

  vec2 uv = vec2(intCoord) * 4.0 + 0.5;
  uv *= control.InvFullResolution;  
  
  vec4 S0 = textureGather      (texLinearDepth, uv, 0);
  vec4 S1 = textureGatherOffset(texLinearDepth, uv, ivec2(2,0), 0);
  vec4 S2 = textureGatherOffset(texLinearDepth, uv, ivec2(0,2), 0);
  vec4 S3 = textureGatherOffset(texLinearDepth, uv, ivec2(2,2), 0);
 
  outputColor(intCoord, 0, S0.w);
  outputColor(intCoord, 1, S0.z);
  outputColor(intCoord, 2, S1.w);
  outputColor(intCoord, 3, S1.z);
  outputColor(intCoord, 4, S0.x);
  outputColor(intCoord, 5, S0.y);
  outputColor(intCoord, 6, S1.x);
  outputColor(intCoord, 7, S1.y);
  
  outputColor(intCoord, 0 + 8, S2.w);
  outputColor(intCoord, 1 + 8, S2.z);
  outputColor(intCoord, 2 + 8, S3.w);
  outputColor(intCoord, 3 + 8, S3.z);
  outputColor(intCoord, 4 + 8, S2.x);
  outputColor(intCoord, 5 + 8, S2.y);
  outputColor(intCoord, 6 + 8, S3.x);
  outputColor(intCoord, 7 + 8, S3.y);
}
