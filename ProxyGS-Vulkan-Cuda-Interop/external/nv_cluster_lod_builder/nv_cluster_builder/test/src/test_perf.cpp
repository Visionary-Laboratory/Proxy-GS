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

#include <gtest/gtest.h>
#include <nanobench.h>
#include <test_util.hpp>
#include <tree_gen.hpp>

namespace nb = ankerl::nanobench;

struct SpatialDesc
{
  SpatialDesc(const GeometryMesh& mesh)
  {
    boundingBoxes.resize(mesh.triangles.size());
    std::ranges::transform(mesh.triangles, boundingBoxes.begin(), [&](vec3u tri) { return aabb(tri, mesh.positions); });
    centroids.resize(boundingBoxes.size());
    std::ranges::transform(boundingBoxes, centroids.begin(), [](AABB b) { return b.center(); });
  }
  std::vector<AABB>  boundingBoxes;
  std::vector<vec3f> centroids;
  nvcluster_Input    clusterInput(const GeometryMesh* mesh = nullptr) const
  {
    return nvcluster_Input{
        .itemBoundingBoxes     = reinterpret_cast<const nvcluster_AABB*>(boundingBoxes.data()),
        .itemCentroids         = reinterpret_cast<const nvcluster_Vec3f*>(centroids.data()),
        .itemCount             = uint32_t(boundingBoxes.size()),
        .itemConnectionRanges  = nullptr,
        .connectionTargetItems = nullptr,
        .connectionWeights     = nullptr,
        .connectionVertexBits  = nullptr,
        .connectionCount       = 0,
        .itemVertices          = mesh ? reinterpret_cast<const uint32_t*>(mesh->triangles.data()) : nullptr,
        .vertexCount           = mesh ? uint32_t(mesh->triangles.size()) : 0,
    };
  }
};

TEST(Perf, All)
{
#if !defined(NDEBUG)
  GTEST_SKIP() << "Skipping performance tests in debug mode";
#else
  GeometryMesh sphere = makeIcosphere(4);
  SpatialDesc  sphereDesc(sphere);
  GeometryMesh tree = generateTree(3);
  SpatialDesc  treeDesc(tree);
  auto         sphereSingleTri = sphere;
  sphereSingleTri.triangles.resize(1);
  nb::Bench()
      .minEpochTime(std::chrono::milliseconds(500))
      .minEpochIterations(10)
      .warmup(1)
      .run("makeMeshConnections", [&] { nb::doNotOptimizeAway(makeMeshConnections(false, sphere)); })
      .run("makeMeshConnections parallel", [&] { nb::doNotOptimizeAway(makeMeshConnections(true, sphere)); })
      .run("makeMeshConnections parallel single tri",
           [&] { nb::doNotOptimizeAway(makeMeshConnections(true, sphereSingleTri)); })
      .run("cluster sphere limit t=[28,32]",
           [&] {
             nb::doNotOptimizeAway(ClusterStorage(
                 nvcluster_Config{
                     .minClusterSize        = 28,
                     .maxClusterSize        = 32,
                     .maxClusterVertices    = ~0u,
                     .costUnderfill         = 0.0f,
                     .costOverlap           = 0.0f,
                     .costUnderfillVertices = 0.0f,
                     .itemVertexCount       = 3,
                     .preSplitThreshold     = 0,
                 },
                 sphereDesc.clusterInput()));
           })
      .run("cluster sphere limit t=[28,32], v=32*3",
           [&] {
             nb::doNotOptimizeAway(ClusterStorage(
                 nvcluster_Config{
                     .minClusterSize        = 28,
                     .maxClusterSize        = 32,
                     .maxClusterVertices    = 32 * 3,
                     .costUnderfill         = 0.0f,
                     .costOverlap           = 0.0f,
                     .costUnderfillVertices = 0.0f,
                     .itemVertexCount       = 3,
                     .preSplitThreshold     = 0,
                 },
                 sphereDesc.clusterInput(&sphere)));
           })
      .run("cluster sphere limit t=[28,32], v=16",
           [&] {
             nb::doNotOptimizeAway(ClusterStorage(
                 nvcluster_Config{
                     .minClusterSize        = 28,
                     .maxClusterSize        = 32,
                     .maxClusterVertices    = 16,
                     .costUnderfill         = 0.0f,
                     .costOverlap           = 0.0f,
                     .costUnderfillVertices = 0.0f,
                     .itemVertexCount       = 3,
                     .preSplitThreshold     = 0,
                 },
                 sphereDesc.clusterInput(&sphere)));
           })
      .run("cluster sphere limit t=[28,32], v=16, costs",
           [&] {
             nb::doNotOptimizeAway(ClusterStorage(
                 nvcluster_Config{
                     .minClusterSize        = 28,
                     .maxClusterSize        = 32,
                     .maxClusterVertices    = 16,
                     .costUnderfill         = 0.1f,
                     .costOverlap           = 0.1f,
                     .costUnderfillVertices = 0.1f,
                     .itemVertexCount       = 3,
                     .preSplitThreshold     = 0,
                 },
                 sphereDesc.clusterInput(&sphere)));
           })
      .run("cluster tree limit t=[28,32]",
           [&] {
             nb::doNotOptimizeAway(ClusterStorage(
                 nvcluster_Config{
                     .minClusterSize        = 28,
                     .maxClusterSize        = 32,
                     .maxClusterVertices    = ~0u,
                     .costUnderfill         = 0.0f,
                     .costOverlap           = 0.0f,
                     .costUnderfillVertices = 0.0f,
                     .itemVertexCount       = 3,
                     .preSplitThreshold     = 0,
                 },
                 treeDesc.clusterInput()));
           })
      .run("cluster tree limit t=[28,32], v=16",
           [&] {
             nb::doNotOptimizeAway(ClusterStorage(
                 nvcluster_Config{
                     .minClusterSize        = 28,
                     .maxClusterSize        = 32,
                     .maxClusterVertices    = 16,
                     .costUnderfill         = 0.0f,
                     .costOverlap           = 0.0f,
                     .costUnderfillVertices = 0.0f,
                     .itemVertexCount       = 3,
                     .preSplitThreshold     = 0,
                 },
                 treeDesc.clusterInput(&tree)));
           })
      .run("cluster tree limit t=[28,32], v=16, costs", [&] {
        nb::doNotOptimizeAway(ClusterStorage(
            nvcluster_Config{
                .minClusterSize        = 28,
                .maxClusterSize        = 32,
                .maxClusterVertices    = 16,
                .costUnderfill         = 0.1f,
                .costOverlap           = 0.1f,
                .costUnderfillVertices = 0.1f,
                .itemVertexCount       = 3,
                .preSplitThreshold     = 0,
            },
            treeDesc.clusterInput(&tree)));
      });
#endif
}
