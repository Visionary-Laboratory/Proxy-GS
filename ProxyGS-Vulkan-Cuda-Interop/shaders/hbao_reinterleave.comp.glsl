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

layout(binding=NVHBAO_MAIN_TEX_RESULTARRAY)    uniform sampler2DArray texResultsArray;
#if NVHBAO_BLUR
layout(binding=NVHBAO_MAIN_IMG_RESULT, rg16f)  uniform image2D imgResult;
#else
layout(binding=NVHBAO_MAIN_IMG_RESULT, r8)     uniform image2D imgResult;
#endif

//----------------------------------------------------------------------------------

void main() {
  ivec2 intCoord;
  vec2  texCoord;
  
  if (setupCoordFull(intCoord, texCoord)) return;

  ivec2 FullResPos = intCoord;
  ivec2 Offset = FullResPos & 3;
  int SliceId = Offset.y * 4 + Offset.x;
  ivec2 QuarterResPos = FullResPos >> 2;
  
#if NVHBAO_BLUR
  imageStore(imgResult, intCoord, vec4(texelFetch( texResultsArray, ivec3(QuarterResPos, SliceId), 0).xy,0,0));
#else
  imageStore(imgResult, intCoord, vec4(texelFetch( texResultsArray, ivec3(QuarterResPos, SliceId), 0).x));
#endif
}
