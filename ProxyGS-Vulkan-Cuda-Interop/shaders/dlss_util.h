/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/


#ifndef SHADERIO_DLSS_UTIL_H
#define SHADERIO_DLSS_UTIL_H

#define SHADERIO_eDlssRenderImage 0
#define SHADERIO_eDlssAlbedo 1
#define SHADERIO_eDlssSpecAlbedo 2
#define SHADERIO_eDlssNormalRoughness 3
#define SHADERIO_eDlssMotion 4

#ifdef __cplusplus
namespace shaderio {
using namespace glm;

// #DLSS - Halton halton low discrepancy sequence, from https://www.shadertoy.com/view/wdXSW8
inline vec2 halton(int index)
{
  const vec2 coprimes = vec2(2.0F, 3.0F);
  vec2       s        = vec2(index, index);
  vec4       a        = vec4(1, 1, 0, 0);
  while(s.x > 0. && s.y > 0.)
  {
    a.x = a.x / coprimes.x;
    a.y = a.y / coprimes.y;
    a.z += a.x * fmod(s.x, coprimes.x);
    a.w += a.y * fmod(s.y, coprimes.y);
    s.x = floorf(s.x / coprimes.x);
    s.y = floorf(s.y / coprimes.y);
  }
  return vec2(a.z, a.w);
}

inline vec2 dlssJitter(uint32_t frameIndex)
{
  return halton(frameIndex) - vec2(0.5, 0.5);
}

}  // namespace shaderio

#else

// Specular albedo for DLSS
vec3 EnvBRDFApprox2(vec3 SpecularColor, float alpha, float NoV)
{
  NoV = abs(NoV);

  // [Ray Tracing Gems, Chapter 32]
  vec4 X;
  X.x = 1.0;
  X.y = NoV;
  X.z = NoV * NoV;
  X.w = NoV * X.z;

  vec4 Y;
  Y.x = 1.0;
  Y.y = alpha;
  Y.z = alpha * alpha;
  Y.w = alpha * Y.z;

  mat2 M1 = mat2(0.99044, -1.28514, 1.29678, -0.755907);
  mat3 M2 = mat3(1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627);
  mat2 M3 = mat2(0.0365463, 3.32707, 9.0632, -9.04756);
  mat3 M4 = mat3(1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123);

  // Bias and scale calculations
  float bias  = dot(M1 * X.xy, Y.xy) / dot(M2 * X.xyz, Y.xyz);
  float scale = dot(M3 * X.xy, Y.xy) / dot(M4 * X.xzw, Y.xyz);

  // This is a hack for specular reflectance of 0
  bias *= clamp(SpecularColor.g * 50.0, 0, 1);

  return SpecularColor * max(0.0, scale) + max(0.0, bias);
}

// Function to calculate 2D motion vectors for DLSS denoising
inline vec2 calculateMotionVector(vec3 worldPos,    // Current world-space hit position
                                  mat4 prevMVP,     // Previous frame's Model-View-Projection matrix
                                  mat4 currentMVP,  // Current frame's Model-View-Projection matrix
                                  vec2 resolution)  // Render target resolution
{
  // Transform current world position to clip space for current frame
  vec4 currentClipPos = currentMVP * vec4(worldPos, 1.0f);
  currentClipPos /= currentClipPos.w;

  // Transform current world position to clip space for previous frame
  vec4 prevClipPos = prevMVP * vec4(worldPos, 1.0f);
  prevClipPos /= prevClipPos.w;

  // Convert clip space coordinates to screen space (0 to 1 range)
  vec2 currentScreenPos = vec2(currentClipPos.xy) * 0.5f + 0.5f;
  vec2 prevScreenPos    = vec2(prevClipPos.xy) * 0.5f + 0.5f;

  // Calculate motion vector in screen space
  vec2 motionVector = prevScreenPos - currentScreenPos;

  // Scale motion vector to pixel space
  motionVector *= resolution;

  return motionVector;
}
#endif

#endif  // DLSS_UTIL_H
