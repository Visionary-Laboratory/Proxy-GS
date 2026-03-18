/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _OCTANT_ENCODING_H_
#define _OCTANT_ENCODING_H_

#ifdef __cplusplus
namespace shaderio {
#define OCT_INLINE inline
#define OCT_FLOOR glm::floor
#define OCT_CLAMP glm::clamp
#define OCT_ABS glm::abs
OCT_INLINE uint32_t pack_oct32(vec2 v)
{
  union
  {
    int16_t  snorm[2];
    uint32_t packed;
  };
  snorm[0] = static_cast<int16_t>(glm::clamp(int32_t(std::round(v.x * float(0x7FFF))), -0x7FFF, 0x7FFF));
  snorm[1] = static_cast<int16_t>(glm::clamp(int32_t(std::round(v.y * float(0x7FFF))), -0x7FFF, 0x7FFF));
  return packed;
}
OCT_INLINE vec2 unpack_oct32(uint32_t v)
{
  union
  {
    int16_t  snorm[2];
    uint32_t packed;
  };
  packed = v;
  return vec2(float(snorm[0]) / float(0x7FFF), float(snorm[1]) / float(0x7FFF));
}
#else
#define OCT_INLINE
#define OCT_FLOOR floor
#define OCT_CLAMP clamp
#define OCT_ABS abs
uint pack_oct32(vec2 v)
{
  return packSnorm2x16(v);
}
vec2 unpack_oct32(uint v)
{
  return unpackSnorm2x16(v);
}
#endif

// oct functions from http://jcgt.org/published/0003/02/01/paper.pdf
OCT_INLINE vec2 oct_signNotZero(vec2 v)
{
  return vec2((v.x >= 0.0f) ? +1.0f : -1.0f, (v.y >= 0.0f) ? +1.0 : -1.0f);
}
OCT_INLINE vec3 oct_to_vec(vec2 e)
{
  vec3 v = vec3(e.x, e.y, 1.0f - OCT_ABS(e.x) - OCT_ABS(e.y));
  if(v.z < 0.0f)
  {
    vec2 os = oct_signNotZero(e);
    v.x     = (1.0f - OCT_ABS(e.y)) * os.x;
    v.y     = (1.0f - OCT_ABS(e.x)) * os.y;
  }
  return normalize(v);
}

OCT_INLINE vec3 oct32_to_vec(uint32_t v)
{
  return oct_to_vec(unpack_oct32(v));
}

OCT_INLINE vec2 vec_to_oct(vec3 v)
{
  // Project the sphere onto the octahedron, and then onto the xy plane
  vec2 p = vec2(v.x, v.y) * (1.0f / (OCT_ABS(v.x) + OCT_ABS(v.y) + OCT_ABS(v.z)));
  // Reflect the folds of the lower hemisphere over the diagonals
  return (v.z <= 0.0f) ? (vec2(1.0f - OCT_ABS(p.y), 1.0f - OCT_ABS(p.x)) * oct_signNotZero(p)) : p;
}

OCT_INLINE vec2 vec_to_oct_precise(vec3 v, int bits)
{
  vec2 s = vec_to_oct(v);  // Remap to the square
                           // Each snorm's max value interpreted as an integer,
                           // e.g., 127.0 for snorm8
  float M = float(1 << ((bits / 2) - 1)) - 1.0f;
  // Remap components to snorm(n/2) precision...with floor instead
  // of round (see equation 1)
  s                        = OCT_FLOOR(OCT_CLAMP(s, -1.0f, +1.0f) * M) * (1.0f / M);
  vec2  bestRepresentation = s;
  float highestCosine      = dot(oct_to_vec(s), v);
  // Test all combinations of floor and ceil and keep the best.
  // Note that at +/- 1, this will exit the square... but that
  // will be a worse encoding and never win.
  for(int i = 0; i <= 1; ++i)
  {
    for(int j = 0; j <= 1; ++j)
    {
      // This branch will be evaluated at compile time
      if((i != 0) || (j != 0))
      {
        // Offset the bit pattern (which is stored in floating
        // point!) to effectively change the rounding mode
        // (when i or j is 0: floor, when it is one: ceiling)
        vec2  candidate = vec2(i, j) * (1 / M) + s;
        float cosine    = dot(oct_to_vec(candidate), v);
        if(cosine > highestCosine)
        {
          bestRepresentation = candidate;
          highestCosine      = cosine;
        }
      }
    }
  }
  return bestRepresentation;
}

OCT_INLINE uint vec_to_oct32(vec3 v)
{
  return pack_oct32(vec_to_oct_precise(v, 32));
}

#undef OCT_ABS
#undef OCT_FLOOR
#undef OCT_CLAMP
#undef OCT_INLINE

#ifdef __cplusplus
}
#endif

#endif