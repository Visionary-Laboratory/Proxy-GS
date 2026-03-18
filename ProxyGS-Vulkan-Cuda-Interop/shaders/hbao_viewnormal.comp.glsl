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

layout(binding=NVHBAO_MAIN_TEX_LINDEPTH)          uniform sampler2D texLinearDepth;
layout(binding=NVHBAO_MAIN_IMG_VIEWNORMAL,rgba8)  uniform image2D   imgViewNormal;

//----------------------------------------------------------------------------------

vec3 UVToView(vec2 uv, float eye_z)
{
  return vec3((uv * control.projInfo.xy + control.projInfo.zw) * (control.projOrtho != 0 ? 1. : eye_z), eye_z);
}

vec3 FetchViewPos(vec2 UV)
{
  float ViewDepth = textureLod(texLinearDepth,UV,0).x;
  return UVToView(UV, ViewDepth);
}

vec3 MinDiff(vec3 P, vec3 Pr, vec3 Pl)
{
  vec3 V1 = Pr - P;
  vec3 V2 = P - Pl;
  return (dot(V1,V1) < dot(V2,V2)) ? V1 : V2;
}

vec3 ReconstructNormal(vec2 UV, vec3 P)
{
  vec3 Pr = FetchViewPos(UV + vec2(control.InvFullResolution.x, 0));
  vec3 Pl = FetchViewPos(UV + vec2(-control.InvFullResolution.x, 0));
  vec3 Pt = FetchViewPos(UV + vec2(0, control.InvFullResolution.y));
  vec3 Pb = FetchViewPos(UV + vec2(0, -control.InvFullResolution.y));
  return normalize(cross(MinDiff(P, Pr, Pl), MinDiff(P, Pt, Pb)));
}

//----------------------------------------------------------------------------------

void main() {
  ivec2 intCoord;
  vec2  texCoord;
  
  if (setupCoordFull(intCoord, texCoord)) return;

  vec3 P  = FetchViewPos(texCoord);
  vec3 N  = ReconstructNormal(texCoord, P);
  
  imageStore(imgViewNormal, intCoord, vec4(N*0.5 + 0.5,0));
}
