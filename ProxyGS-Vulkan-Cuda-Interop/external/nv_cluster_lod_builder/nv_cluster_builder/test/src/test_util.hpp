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

#include <algorithm>
#include <clusterizer.hpp>   // internal, for unit testing
#include <clusters_cpp.hpp>  // internal include from lib, for vec3 etc.
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <nvcluster/nvcluster_storage.hpp>
#include <ostream>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

using nvcluster::AABB;
using nvcluster::Range;
using nvcluster::vec2f;
using nvcluster::vec3f;
using nvcluster::vec3u;

// Computes the axis-aligned bounding box of a triangle with the given indices.
inline AABB aabb(vec3u triangle, std::span<const vec3f> positions)
{
  using namespace nvcluster;
  return {min(min(positions[triangle[0]], positions[triangle[1]]), positions[triangle[2]]),
          max(max(positions[triangle[0]], positions[triangle[1]]), positions[triangle[2]])};
}

template <std::ranges::input_range Range>
bool allUnique(const Range& range)
{
  std::unordered_set unique(std::begin(range), std::end(range));
  return unique.size() == std::ranges::size(range);
}

template <std::ranges::input_range Range>
bool contains(const Range& range, std::ranges::range_value_t<Range> value)
{
  return std::ranges::find(range, value) != std::end(range);
}

// Shortcut for passing Range offset and count to std::span::subspan(), which
// returns a span pointing to a possibly smaller range of values.
template <class Items>
constexpr auto subspan(Items& items, nvcluster_Range range)
{
  assert(range.count == 0 || range.offset < std::ranges::size(items));
  assert(range.count == 0 || range.offset + range.count <= std::ranges::size(items));
  return std::span(items).subspan(range.offset, range.count);
}

// Simple mesh struct. Triangle indices and vertex positions. Plus a name for
// better context when tests fail.
struct GeometryMesh
{
  std::string        name;
  std::vector<vec3u> triangles;
  std::vector<vec3f> positions;

  // Dump the mesh to a .obj file for testing
  void write(std::ostream& os) const
  {
    os << "g mesh\n";
    for(auto& p : positions)
      os << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
    for(auto& t : triangles)
      os << "f " << t[0] + 1 << " " << t[1] + 1 << " " << t[2] + 1 << "\n";
  };
  void write(const std::filesystem::path& path) const
  {
    std::ofstream ofile(path);
    write(ofile);
  }
};

inline nvcluster::MeshConnections makeMeshConnections(bool parallelize, const GeometryMesh& mesh)
{
  return nvcluster::makeMeshConnections(parallelize,
                                        nvcluster::ItemVertices(reinterpret_cast<const uint32_t*>(mesh.triangles.data()),
                                                                uint32_t(mesh.triangles.size()), 3u),
                                        uint32_t(mesh.positions.size()));
}

inline void check(nvcluster_Result result)
{
  if(result != nvcluster_Result::NVCLUSTER_SUCCESS)
    throw std::runtime_error(nvclusterResultString(result));
}

// nvcluster_Context wrapper handles ownership, lifetime, doesn't leak when
// tests return etc.
struct ScopedContext
{
  ScopedContext(const nvcluster_ContextCreateInfo& createInfo = {})
  {
    check(nvclusterCreateContext(&createInfo, &context));
  }
  ~ScopedContext() { std::ignore = nvclusterDestroyContext(context); }
  ScopedContext(const ScopedContext& other)            = delete;
  ScopedContext& operator=(const ScopedContext& other) = delete;
  operator nvcluster_Context() const { return context; }
  nvcluster_Context context = nullptr;
};

// Shortcut to build clusters from various forms of inputs
struct ClusterStorage : nvcluster::ClusterStorage
{
  // External API
  ClusterStorage(const nvcluster_Config& config, const nvcluster_Input& input)
  {
    check(generateClusters(ScopedContext(), config, input, *this));
  }

  // Internal interface, for unit testing
  ClusterStorage(const nvcluster::Input& input)
  {
    if(input.segments.size() != 1)
      throw std::runtime_error("segmented clustering not implemented in this test");
    nvcluster_Counts requiredCounts;
    check(nvclusterGetRequirements(ScopedContext(), &input.config, uint32_t(input.boundingBoxes.size()), &requiredCounts));
    clusterItemRanges.resize(requiredCounts.clusterCount);
    items.resize(input.boundingBoxes.size());
    nvcluster_OutputClusters output{.clusterItemRanges = clusterItemRanges.data(),
                                    .items             = items.data(),
                                    .clusterCount      = uint32_t(clusterItemRanges.size()),
                                    .itemCount         = uint32_t(items.size())};
    nvcluster_Range          outputSegment{};
    check(clusterize(true, input, nvcluster::OutputClusters(output, &outputSegment, 1)));
    if(outputSegment.offset != 0 || size_t(outputSegment.count) != output.clusterCount)
      throw std::runtime_error("expected one segment with everything");
    clusterItemRanges.resize(output.clusterCount);
  }
};

// Returns the number of unique vertices per cluster to verify the vertex limit
// feature
inline std::vector<uint32_t> countClusterVertices(const nvcluster::ClusterStorage& clustering, const GeometryMesh& mesh)
{
  std::vector<uint32_t> result;
  result.reserve(clustering.clusterItemRanges.size());
  for(nvcluster_Range r : clustering.clusterItemRanges)
  {
    std::span<const uint32_t>    cluster = subspan(clustering.items, r);
    std::unordered_set<uint32_t> uniqueVertices;
    for(auto i : cluster)
    {
      uniqueVertices.insert(mesh.triangles[i][0]);
      uniqueVertices.insert(mesh.triangles[i][1]);
      uniqueVertices.insert(mesh.triangles[i][2]);
    }
    result.push_back(uint32_t(uniqueVertices.size()));
  }
  return result;
}

// Icosahedron data.
namespace icosahedron {
constexpr float              X         = .525731112119133606f;
constexpr float              Z         = .850650808352039932f;
static std::array<vec3f, 12> positions = {{{-X, 0.0, Z},
                                           {X, 0.0, Z},
                                           {-X, 0.0, -Z},
                                           {X, 0.0, -Z},
                                           {0.0, Z, X},
                                           {0.0, Z, -X},
                                           {0.0, -Z, X},
                                           {0.0, -Z, -X},
                                           {Z, X, 0.0},
                                           {-Z, X, 0.0},
                                           {Z, -X, 0.0},
                                           {-Z, -X, 0.0}}};
static std::array<vec3u, 20> triangles = {{{0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
                                           {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
                                           {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
                                           {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}}};
}  // namespace icosahedron

// Type of a function to call when creating a triangle. Takes 3 positions as
// inputs.
using triangle_callback = std::function<void(vec3f, vec3f, vec3f)>;

// Recursively subdivides a triangle on a sphere by a factor of 2^depth.
// Calls the callback function on each new triangle.
inline void subdivide(vec3f v0, vec3f v1, vec3f v2, int depth, triangle_callback& callback)
{
  if(depth == 0)
  {
    callback(v0, v1, v2);
  }
  else
  {
    vec3f v01 = normalize(v0 + v1);
    vec3f v12 = normalize(v1 + v2);
    vec3f v20 = normalize(v2 + v0);
    subdivide(v0, v01, v20, depth - 1, callback);
    subdivide(v1, v12, v01, depth - 1, callback);
    subdivide(v2, v20, v12, depth - 1, callback);
    subdivide(v01, v12, v20, depth - 1, callback);
  }
}

// Makes an icosphere with 20 * (4^depth) triangles.
inline void makeIcosphere(int depth, triangle_callback& callback)
{
  for(size_t i = 0; i < icosahedron::triangles.size(); i++)
  {
    const vec3f v0 = icosahedron::positions[icosahedron::triangles[i][0]];
    const vec3f v1 = icosahedron::positions[icosahedron::triangles[i][1]];
    const vec3f v2 = icosahedron::positions[icosahedron::triangles[i][2]];
    subdivide(v0, v1, v2, depth, callback);
  }
}

inline GeometryMesh makeIcosphere(int subdivision)
{
  std::unordered_map<vec3f, uint32_t> vertexCache;
  std::vector<vec3u>                  triangles;
  // Our triangle callback function tries to place each of the vertices in the
  // vertex cache; each of the `it` iterators point to the existing value if
  // the vertex was already in the cache, or to a new value at the end of the
  // cache if it's a new vertex.
  triangle_callback callback = [&vertexCache, &triangles](vec3f v0, vec3f v1, vec3f v2) {
    auto [it0, new0] = vertexCache.try_emplace(v0, static_cast<uint32_t>(vertexCache.size()));
    auto [it1, new1] = vertexCache.try_emplace(v1, static_cast<uint32_t>(vertexCache.size()));
    auto [it2, new2] = vertexCache.try_emplace(v2, static_cast<uint32_t>(vertexCache.size()));
    triangles.push_back({it0->second, it1->second, it2->second});
  };
  makeIcosphere(subdivision, callback);
  std::vector<vec3f> positions(vertexCache.size());
  for(const auto& [position, index] : vertexCache)
  {
    positions[index] = position;
  }
  [[maybe_unused]] size_t edgeCount = (triangles.size() * 3) / 2;  // 3 edges per triangle and each is shared between two trianlges exactly once
  [[maybe_unused]] size_t vertexCount = 2 + edgeCount - triangles.size();  // Euler's polyhedron formula
  assert(positions.size() == vertexCount);  // Double check vertex cache deduplicates vertices as expected
  return {"icosphere" + std::to_string(subdivision) + "(" + std::to_string(triangles.size()) + " tris)", triangles, positions};
}
