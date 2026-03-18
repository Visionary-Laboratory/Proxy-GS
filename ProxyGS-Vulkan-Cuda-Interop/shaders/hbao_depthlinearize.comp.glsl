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

layout(binding=NVHBAO_MAIN_TEX_DEPTH)           uniform sampler2D inputTexture;
layout(binding=NVHBAO_MAIN_IMG_LINDEPTH, r32f)  uniform image2D   imgLinearDepth;
#if NVHBAO_SKIP_INTERPASS
  layout(binding=NVHBAO_MAIN_IMG_DEPTHARRAY, r32f)  uniform image2DArray  imgLinearDepthArray;
#endif


float reconstructCSZ(float d, vec4 clipInfo) {
#if 1
  vec4 ndc = vec4(0,0,d,1);
  vec4 unproj = control.InvProjMatrix * ndc;
  return unproj.z / unproj.w;
#else
   // clipInfo = z_n * z_f,  z_n - z_f,  z_f, perspective = 1 : 0

  if (clipInfo[3] != 0) {
    return (clipInfo[0] / (clipInfo[1] * d + clipInfo[2]));
  }
  else {
    return (clipInfo[1]+clipInfo[2] - d * clipInfo[1]);
  }
#endif
  
}
/*
    if (in_perspective == 1.0) // perspective
    {
        ze = (zNear * zFar) / (zFar - zb * (zFar - zNear)); 
    }
    else // orthographic proj 
    {
        ze  = zNear + zb  * (zFar - zNear);
    }
*/
void main() 
{
  ivec2 intCoord;
  vec2  texCoord;
  
  if (setupCoordFull(intCoord, texCoord)) return;

  float depth = textureLod(inputTexture, texCoord.xy, 0).x;
  float linDepth = reconstructCSZ(depth, control.projReconstruct);
  imageStore(imgLinearDepth, intCoord, vec4(linDepth,0,0,0));
#if NVHBAO_SKIP_INTERPASS
  ivec2 FullResPos = intCoord;
  ivec2 Offset = FullResPos & 3;
  int SliceId = Offset.y * 4 + Offset.x;
  ivec2 QuarterResPos = FullResPos >> 2;
  imageStore(imgLinearDepthArray, ivec3(QuarterResPos, SliceId), vec4(linDepth,0,0,0));
#endif
}
