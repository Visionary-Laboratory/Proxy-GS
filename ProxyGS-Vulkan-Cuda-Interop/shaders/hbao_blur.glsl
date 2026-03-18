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

layout(push_constant) uniform pushData {
  NVHBAOBlurPush  blur;
};


const float KERNEL_RADIUS = 3;

//-------------------------------------------------------------------------

float BlurFunction(vec2 uv, float r, float center_c, float center_d, inout float w_total)
{
  vec2  aoz = texture(texSource, uv).xy;
  float c = aoz.x;
  float d = aoz.y;
  
  const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
  const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);
  
  float ddiff = (d - center_d) * blur.sharpness;
  float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
  w_total += w;

  return c*w;
}

vec2 BlurRun(vec2 texCoord)
{
  vec2  aoz = texture(texSource, texCoord).xy;
  float center_c = aoz.x;
  float center_d = aoz.y;
  
  float c_total = center_c;
  float w_total = 1.0;
  
  [[unroll]]
  for (float r = 1; r <= KERNEL_RADIUS; ++r)
  {
    vec2 uv = texCoord + blur.invResolutionDirection * r;
    c_total += BlurFunction(uv, r, center_c, center_d, w_total);  
  }
  
  [[unroll]]
  for (float r = 1; r <= KERNEL_RADIUS; ++r)
  {
    vec2 uv = texCoord - blur.invResolutionDirection * r;
    c_total += BlurFunction(uv, r, center_c, center_d, w_total);  
  }
  
  return vec2(c_total/w_total, center_d);
  //return vec2(aoz);
}
