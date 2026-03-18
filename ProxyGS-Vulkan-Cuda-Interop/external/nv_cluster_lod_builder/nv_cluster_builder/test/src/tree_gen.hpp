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
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// AI generated...

#include <cmath>
#include <functional>
#include <random>
#include <test_util.hpp>
#include <vector>

static constexpr float g_twoPi = 6.28318530718f;

inline vec3f evaluateBezier(const vec3f& p0, const vec3f& p1, const vec3f& p2, float t)
{
  float u  = 1 - t;
  float tt = t * t;
  float uu = u * u;

  vec3f point = uu * p0;    // Quadratic term
  point += 2 * u * t * p1;  // Linear term
  point += tt * p2;         // Constant term

  return point;
}

inline GeometryMesh makeTriangleStrip(std::function<vec3f(float)> path, uint32_t segments, float width)
{
  GeometryMesh mesh;

  float e = 0.001f;
  for(uint32_t i = 0; i <= segments; ++i)
  {
    float t = float(i) / float(segments);

    vec3f position = path(t);
    vec3f dp1      = path(t + e) - path(t);                        // First derivative (tangent vector)
    vec3f dp2      = path(t + 2 * e) - 2 * path(t + e) + path(t);  // Second derivative

    vec3f normal = cross(dp1, dp2);
    normal       = normalize(normal) * width;  // Scale normal to desired strip width

    vec3f leftPoint  = position - normal * 0.5f;
    vec3f rightPoint = position + normal * 0.5f;

    mesh.positions.push_back(leftPoint);
    mesh.positions.push_back(rightPoint);

    if(i > 0)  // Add triangle indices after the first segment
    {
      size_t idx = mesh.positions.size();
      mesh.triangles.push_back({idx - 2, idx - 3, idx - 1});
      mesh.triangles.push_back({idx - 2, idx - 4, idx - 3});
    }
  }

  return mesh;
}

inline GeometryMesh makeBranch(std::function<vec3f(float)> path, uint32_t segments, uint32_t segmentsCircular, float radius)
{
  GeometryMesh mesh;

  float e = 0.001f;
  for(uint32_t j = 0; j <= segments; ++j)
  {
    float t        = float(j) / float(segments);
    vec3f position = path(t);
    vec3f tangent  = normalize(path(t + e) - path(t));                     // Compute tangent vector
    vec3f normal = normalize(vec3f(-tangent[1], tangent[0], tangent[2]));  // Arbitrary normal perpendicular to tangent
    vec3f binormal = normalize(cross(tangent, normal));                    // Compute binormal for perpendicularity

    size_t baseIndex = mesh.positions.size();

    float segmentRadius = powf(1.0f - t, 0.1f) * radius;
    // Generate vertices for a ring around the path position
    for(uint32_t i = 0; i < segmentsCircular; ++i)
    {
      float angle = (g_twoPi * float(i)) / float(segmentsCircular);  // Corrected calculation of angle using segmentsCircular
      vec3f offset = segmentRadius * (cosf(angle) * normal + sinf(angle) * binormal);
      vec3f vertex = position + offset;

      mesh.positions.push_back(vertex);
    }

    // Add triangle indices for the cylinder body
    if(j > 0)
    {
      for(uint32_t i = 0; i < segmentsCircular; ++i)
      {
        uint32_t next = (i + 1) % segmentsCircular;  // Corrected modulo operation to properly handle adjacency

        mesh.triangles.push_back({baseIndex + i, baseIndex + next, baseIndex + i - segmentsCircular});
        mesh.triangles.push_back({baseIndex + next, baseIndex + next - segmentsCircular, baseIndex + i - segmentsCircular});
      }
    }
  }

  return mesh;
}

inline GeometryMesh makeCone(std::function<vec3f(float)> path, float t, float radius, uint32_t segments)
{
  GeometryMesh mesh;

  vec3f position = path(t);
  vec3f tangent  = normalize(path(t + 0.01f) - path(t));                   // Compute tangent vector
  vec3f normal   = normalize(vec3f(-tangent[1], tangent[0], tangent[2]));  // Arbitrary normal perpendicular to tangent
  vec3f binormal = cross(tangent, normal);                                 // Compute binormal for perpendicularity

  // Generate vertices for the base ring
  size_t baseIndex = mesh.positions.size();
  for(uint32_t j = 0; j <= segments; ++j)
  {
    float angle  = (g_twoPi * float(j)) / float(segments);
    vec3f offset = radius * (cosf(angle) * normal + sinf(angle) * binormal);
    vec3f vertex = position + offset + tangent * radius;
    mesh.positions.push_back(vertex);
  }

  // Add the tip of the cone
  vec3f    tip      = position;  // Position for cone tip
  uint32_t tipIndex = uint32_t(mesh.positions.size());
  mesh.positions.push_back(tip);

  // Add triangle indices for the cone
  for(uint32_t i = 0; i < segments; ++i)
  {
    uint32_t next = (i + 1) % segments;
    mesh.triangles.push_back({baseIndex + i, baseIndex + next, tipIndex});
  }

  return mesh;
}

inline GeometryMesh mergeMeshes(const GeometryMesh& mesh1, const GeometryMesh& mesh2)
{
  GeometryMesh mergedMesh;

  // Combine positions
  mergedMesh.positions = mesh1.positions;
  mergedMesh.positions.insert(mergedMesh.positions.end(), mesh2.positions.begin(), mesh2.positions.end());

  // Combine triangles, adjusting the indices of the second mesh
  mergedMesh.triangles = mesh1.triangles;
  size_t offset        = mesh1.positions.size();  // Offset for indices of mesh2
  mergedMesh.triangles.reserve(mesh1.triangles.size() + mesh2.triangles.size());
  for(const auto& triangle : mesh2.triangles)
  {
    mergedMesh.triangles.push_back({triangle[0] + offset, triangle[1] + offset, triangle[2] + offset});
  }

  return mergedMesh;
}

inline float unitRand()
{
  static std::mt19937                          gen(0);
  static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  return dis(gen);
}

inline uint32_t intRand(uint32_t min, uint32_t max)
{
  static std::mt19937                     gen(0);
  std::uniform_int_distribution<uint32_t> dis(min, max);
  return dis(gen);
}

inline std::function<vec3f(float t)> branchPath(vec3f base, float sideScale, float height)
{
  float angle = unitRand() * g_twoPi;
  vec2f side  = vec2f{cosf(angle), sinf(angle)} * sideScale;
  return [base, side, height](float t) {
    return evaluateBezier(base, base + vec3f{side[0], height * 0.3f, side[1]}, base + vec3f{side[0], height, side[1]}, t);
  };
}

inline void generateTree(GeometryMesh& treeMesh, vec3f base, float side, float height, uint32_t depth)
{
  // Generate a branch path
  auto path = branchPath(base, side * (depth > 3 ? 0.3f : 1.0f), height);

  // Create geometry for the branch
  if(depth == 0)
  {
    treeMesh = mergeMeshes(treeMesh, makeTriangleStrip(path, 3u + uint32_t(height * 3.0f), height * 0.3f));

    // Add cones along the triangle strip
    uint32_t cones = intRand(2, 5);
    for(uint32_t i = 1; i < cones; ++i)
    {
      treeMesh = mergeMeshes(treeMesh, makeCone(path, (float(i) - unitRand() * 0.5f) / float(cones + 1), height * 0.2f, 5));
    }
  }
  else
  {
    treeMesh = mergeMeshes(treeMesh, makeBranch(path, depth * 2 + 4, depth + 4, height * 0.05f));

    uint32_t branches = depth == 2 ? intRand(4, 16) : intRand(2, 5);
    for(uint32_t i = 1; i <= branches; ++i)
    {
      generateTree(treeMesh, path((float(i) + unitRand() * 0.5f) / float(branches + 1)), side * (0.5f + 0.4f * unitRand()),
                   height * (0.4f + 0.5f * unitRand()), depth - intRand(1, std::max(1u, depth / 2u)));
    }
  }
}

inline GeometryMesh generateTree(uint32_t levels = 4)
{
  GeometryMesh treeMesh;
  generateTree(treeMesh, {0.0f, 0.0f, 0.0f}, 4.0f, 6.0f, levels);
  treeMesh.name = "tree_l" + std::to_string(levels);
  return treeMesh;
}
