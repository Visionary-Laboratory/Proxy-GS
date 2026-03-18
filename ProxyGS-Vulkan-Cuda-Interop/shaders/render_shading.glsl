/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/


#extension GL_EXT_fragment_shader_barycentric : enable

// Approximates the batlow color ramp from the scientific color ramps package.
// Input will be clamped to [0, 1]; output is sRGB.
vec3 batlow(float t)
{
  t             = clamp(t, 0.0f, 1.0f);
  const vec3 c5 = vec3(10.741, -0.934, -16.125);
  const vec3 c4 = vec3(-28.888, 2.021, 34.529);
  const vec3 c3 = vec3(24.263, -0.335, -20.561);
  const vec3 c2 = vec3(-6.069, -1.511, 2.47);
  const vec3 c1 = vec3(0.928, 1.455, 0.327);
  const vec3 c0 = vec3(0.007, 0.103, 0.341);

  const vec3 result = ((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0;

  return min(result, vec3(1.0f));
}

vec3 hue2rgb(float hue)
{
  hue= fract(hue);
  return clamp(vec3(
    abs(hue*6.0-3.0)-1.0,
    2.0-abs(hue*6.0-2.0),
    2.0-abs(hue*6.0-4.0)
  ), vec3(0), vec3(1));
}

vec3 lodMix(float v)
{
  float low = 0.15;
  if (v == 0) {
    return vec3(1);
  }
  else if (v < low) {
    return mix(vec3(1), hue2rgb(0.5), v / low);
  }
  else {
    v = (v - low) / (1.0 - low);
    return hue2rgb(0.5 - v * 0.5);
  }
}

vec3 colorizeID(uint clusterID)
{
  return vec3(unpackUnorm4x8(murmurHash(clusterID ^ view.colorXor)).xyz * 0.5 + 0.3);
}

vec3 visualizeColor(uint visData, uint instanceID)
{
  if(view.visualize == VISUALIZE_CLUSTER || view.visualize == VISUALIZE_GROUP || view.visualize == VISUALIZE_TRIANGLE)
  {
    return colorizeID(visData).xyz;
  }
  else if (view.visualize == VISUALIZE_LOD) 
  {
    return vec3(lodMix(uintBitsToFloat(visData))) * 0.7 + 0.2;
    //return vec3(pow(lodMix(uintBitsToFloat(visData)),vec3(8.0))) * 0.8 + 0.1;
  }
  else if (view.visualize == VISUALIZE_MATERIAL)
  {
    return pow(unpackUnorm4x8(instances[instanceID].packedColor).xyz * 0.95 + 0.05, vec3(1.0/2.2));
  }
#if TARGETS_RAY_TRACING
  else if (view.visualize == VISUALIZE_BLAS) {
    uint blasBuildIndex = build.instanceBuildInfos.d[instanceID].blasBuildIndex;
    
    if (blasBuildIndex == BLAS_BUILD_INDEX_LOWDETAIL)
    {
      return vec3(0.9, 0.1, 0.1);
    }
  #if USE_BLAS_SHARING
    else if ((blasBuildIndex & BLAS_BUILD_INDEX_SHARE_BIT) != 0)
    {
      instanceID = blasBuildIndex & ~(BLAS_BUILD_INDEX_SHARE_BIT);
    }
  #endif
    return colorizeID(instanceID).xyz;
  }
#endif
  else
  {
    return vec3(0.95);
  }
}

vec4 shading(uint instanceID, vec3 wPos, vec3 wNormal, uint visData, float overheadLight, float ambientOcclusion
#if USE_DLSS
  , out vec4 dlssAlbedo, out vec3 dlssSpecular, out vec4 dlssNormalRoughness
#endif
)
{
  const vec3 sunColor           = vec3(0.99f, 1.f, 0.71f);
  const vec3 skyColor           = view.skyParams.skyColor;
  const vec3 groundColor        = view.skyParams.groundColor;
  const float materialRoughness = 0;
  const vec3  materialAlbedo    = visualizeColor(visData, instanceID);

  vec4 color   = vec4(0.f);
  vec3 normal  = normalize(wNormal.xyz);
  vec3 wEyePos = vec3(view.viewMatrixI[3].x, view.viewMatrixI[3].y, view.viewMatrixI[3].z);
  vec3 eyeDir  = normalize(wEyePos.xyz - wPos.xyz);
  
#if USE_DLSS
  dlssAlbedo = vec4(materialAlbedo,0);
  dlssNormalRoughness = vec4(wNormal.xyz, materialRoughness);
  dlssSpecular = EnvBRDFApprox2(vec3(1), materialRoughness, dot(wNormal, eyeDir));
#endif

  // Ambient
  float ambientIntensity = 1.f;
  vec3  ambientLighting  = ambientOcclusion * materialAlbedo* ambientIntensity
                         * mix(groundColor, skyColor, dot(normal, view.wUpDir.xyz) * 0.5 + 0.5) ;

  // Light mixer
  float lightMixer             = view.lightMixer;
  float flashlightIntensity    = 1.0f - lightMixer;
  float overheadLightIntensity = lightMixer;

  // Flashlight
  vec3  flashlightLighting  = vec3(0.f);
  {
    // Use a flashlight intensity similar to the sky color for average luminance consistency
    flashlightIntensity *= max(skyColor.x, max(skyColor.y, skyColor.z));
    vec3  lightDir     = normalize(view.wLightPos.xyz - wPos.xyz);
    vec3  reflDir      = normalize(-reflect(lightDir, normal));
    float bsdf         = abs(dot(normal, lightDir)) + pow(max(0, dot(reflDir, eyeDir)), 16) * 0.3;
    flashlightLighting = flashlightIntensity * materialAlbedo * bsdf;
  }

  // Overhead light
  vec3 overheadLightColor = view.skyParams.sunColor * view.skyParams.sunIntensity;
  vec3 overheadLighting   = vec3(overheadLightIntensity * overheadLight * overheadLightColor);
  {
    vec3 lightDir = normalize(view.skyParams.sunDirection);
    vec3 reflDir  = normalize(-reflect(lightDir, normal));
    float diffuse    = max(0, dot(normal, lightDir));
    float specular   = pow(max(0, dot(reflDir, eyeDir)), 16) * 0.3;
    float bsdf       = diffuse + specular;
    overheadLighting = overheadLighting * materialAlbedo * bsdf;
  }

  color.xyz = overheadLighting + flashlightLighting + ambientLighting;
  color.w   = 1.0;
  return color;
}

uint64_t packPickingValue(uint32_t v, float z)
{
  z         = 1.f - clamp(z, 0.f, 1.f);
  uint bits = floatBitsToUint(z);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  uint64_t value = (uint64_t(bits) << 32) | uint64_t(v);
  return value;
}


// Return the width [0..1] for which the line should be displayed or not
float getLineWidth(in vec3 deltas, in float thickness, in float smoothing, in vec3 barys)
{
  barys         = smoothstep(deltas * (thickness), deltas * (thickness + smoothing), barys);
  float minBary = min(barys.x, min(barys.y, barys.z));
  return 1.0 - minBary;
}

// Position along the edge [0..1]
float edgePosition(vec3 barycentrics)
{
  return max(barycentrics.z, max(barycentrics.y, barycentrics.x));
}

// Return 0 or 1 if edgePos should be diplayed or not
float stipple(in float stippleRepeats, in float stippleLength, in float edgePos)
{
  float offset = 1.0 / stippleRepeats;
  offset *= 0.5 * stippleLength;
  float pattern = fract((edgePos + offset) * stippleRepeats);
  return 1.0 - step(stippleLength, pattern);
}


vec3 addWireframe(vec3 color, vec3 barycentrics, bool frontFacing, vec3 barycentricsDerivatives, vec3 wireColor)
{
  float oThickness    = view.wireThickness * 0.5;
  float thickness     = oThickness * 0.5;  // Thickness for both side of the edge, must be divided by 2
  float smoothing     = oThickness * view.wireSmoothing;  // Could be thickness
  bool  enableStipple = (view.wireStipple == 1);

  // Uniform position on the edge [0, 1]
  float edgePos = edgePosition(barycentrics);

  if(!frontFacing)
  {
    enableStipple = true;  // Forcing backface to always stipple the line
    wireColor     = view.wireBackfaceColor;
  }


  // fwidth ? return the sum of the absolute value of derivatives in x and y
  //          which makes the width in screen space
  vec3 deltas = barycentricsDerivatives;  //fwidthFine(barycentrics);

  // Get the wireframe line width
  float lineWidth = getLineWidth(deltas, thickness, smoothing, barycentrics);

  // [optional]
  if(enableStipple)
  {
    float stippleFact = stipple(view.wireStippleRepeats, view.wireStippleLength, edgePos);
    lineWidth *= stippleFact;  // 0 or 1
  }

  // Final color
  return mix(color, wireColor, lineWidth);
}

#if SUPPORTS_RT == 1

uint wangHash(uint seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}

uint xxhash32(uint3 p)
{
  const uvec4 primes = uint4(2246822519U, 3266489917U, 668265263U, 374761393U);
  uint        h32;
  h32 = p.z + primes.w + p.x * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 += p.y * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 = primes.x * (h32 ^ (h32 >> 15));
  h32 = primes.y * (h32 ^ (h32 >> 13));
  return h32 ^ (h32 >> 16);
}

//-----------------------------------------------------------------------
// https://www.pcg-random.org/
//-----------------------------------------------------------------------
uint pcg(inout uint state)
{
  uint prev = state * 747796405u + 2891336453u;
  uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
  state     = prev;
  return (word >> 22u) ^ word;
}

//-----------------------------------------------------------------------
// Generate a random float in [0, 1) given the previous RNG state
//-----------------------------------------------------------------------
float rand(inout uint seed)
{
  uint r = pcg(seed);
  return float(r) * (1.F / float(0xffffffffu));
}
// Generate an arbitrary orthonormal basis from a normal vector
void computeDefaultBasis(const vec3 z, out vec3 x, out vec3 y)
{
  const float yz = -z.y * z.z;
  y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

  x = cross(y, z);
}
#ifndef M_PI
#define M_PI 3.141592653589
#endif

vec3 offsetRay(vec3 p, vec3 dir, vec3 geonrm)
{
  vec3 n = sign(dot(dir, geonrm)) * geonrm;

  // Smallest epsilon that can be added without losing precision is 1.19209e-07, but we play safe
  const float epsilon = 1.0f / 65536.0f;  // Safe epsilon

  float magnitude = length(p);
  float offset    = epsilon * magnitude;
  // multiply the direction vector by the smallest offset
  vec3 offsetVector = n * offset;
  // add the offset vector to the starting point
  vec3 offsetPoint = p + offsetVector;
  return offsetPoint;
}


float ambientOcclusion(vec3 wPos, vec3 wNormal, uint32_t sampleCount, float radius)
{
  if (sampleCount == 0) return 0.7f;
#if USE_DLSS
  uint32_t seed = xxhash32(uvec3(gl_LaunchIDEXT.x, gl_LaunchIDEXT.y, view.frame));
#else
  uint32_t seed = wangHash(gl_LaunchIDEXT.x) ^ wangHash(gl_LaunchIDEXT.y);
#endif
  
  vec3     z    = wNormal;
  vec3     x, y;
  computeDefaultBasis(z, x, y);

  uint32_t occlusion = 0u;

  for(uint32_t i = 0; i < sampleCount; i++)
  {
    float r1 = 2 * M_PI * rand(seed);
    float r2 = rand(seed);
    float sq = sqrt(1.0 - r2);

    vec3 wDirection  = vec3(cos(r1) * sq, sin(r1) * sq, sqrt(r2));
    wDirection       = wDirection.x * x + wDirection.y * y + wDirection.z * z;
    rayHitAO         = 1.f;
    uint mask        = 0xFF;
    traceRayEXT(asScene, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                mask /*0xFF*/, 0, 0, 1, offsetRay(wPos, wDirection, wNormal), 1e-4f, wDirection, radius, 1);
    if(rayHitAO > 0.f)
    {
      occlusion++;
    }
  }
  float linearAo = float(sampleCount - occlusion) / float(sampleCount);
  return max(0.2f, linearAo* linearAo);
}


// Returns 0.0 if there is a hit along the light direction and 1.0, if nothing was hit
float traceShadowRay(vec3 wPos, vec3 wNormal, vec3 wDirection)
{
  rayHitAO         = 1.f;
  uint  mask       = 0xFF;
  uint  flags      = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  float minT       = 0.001f;
  float maxT       = 10000000.0f;
  traceRayEXT(asScene, flags, mask, 0, 0, 1, offsetRay(wPos, wDirection, wNormal), minT, wDirection, maxT, 1);

  return (rayHitAO > 0.f) ? 0.0F : 1.0f;
}

float determinant(vec3 a, vec3 b, vec3 c)
{
  return dot(cross(a, b), c);
}

vec3 intersectRayTriangle(vec3 origin, vec3 direction, vec3 v0, vec3 v1, vec3 v2)
{
  // Edge vectors
  vec3 e1 = v1 - v0;
  vec3 e2 = v2 - v0;

  // Plane normal (cross product of edge vectors)
  vec3 planeNormal = cross(e1, e2);

  // Check for parallelism (ray parallel to the plane)
  float nDotDir = dot(planeNormal, direction);
  //if (abs(nDotDir) < 0.001f) {
  //  return vec3(-1.0); // Return negative values to indicate no intersection
  //}

  // Distance from ray origin to the plane
  float t = dot(planeNormal, v0 - origin) / nDotDir;

  // Check if intersection is behind the ray origin (negative t means no intersection)
  //if (t <= 0.0) {
  //  return vec3(-1.0); // Return negative values to indicate no intersection
  //}

  // Intersection point
  vec3 p = origin + t * direction;

  // Compute barycentric coordinates using determinant
  vec3  temp = p - v0;
  float det  = determinant(e1, e2, planeNormal);
  float u    = dot(cross(temp, e2), planeNormal) / det;
  float v    = dot(cross(e1, temp), planeNormal) / det;
  float w    = 1.0 - u - v;

  return vec3(w, u, v);
}

ivec2 objectToPixel(vec3 objectPos)
{
  vec3 wObjectPos = gl_ObjectToWorldEXT * vec4(objectPos, 1.f);

  vec4 pPos = view.viewProjMatrix * vec4(wObjectPos, 1.f);

  pPos /= pPos.w;
  pPos.xy = pPos.xy * vec2(0.5f) + vec2(0.5f);
  pPos.xy *= vec2(gl_LaunchSizeEXT.xy);
  return ivec2(pPos.xy);
}



#endif  // SUPPORTS_RT