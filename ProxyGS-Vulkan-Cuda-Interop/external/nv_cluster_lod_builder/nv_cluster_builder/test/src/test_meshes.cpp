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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <numeric>
#include <nvcluster/nvcluster.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <ranges>
#include <stdexcept>
#include <test_util.hpp>
#include <tree_gen.hpp>
#include <unordered_map>
#include <unordered_set>

#ifdef TEST_MESHES
#include <cgltf.h>

std::vector<GeometryMesh> gltfMeshes(std::filesystem::path path)
{
  std::vector<GeometryMesh> meshes;
  cgltf_options             options = {};
  cgltf_data*               data    = nullptr;
  cgltf_result              result  = cgltf_parse_file(&options, path.string().c_str(), &data);

  if(result != cgltf_result_success || !data)
  {
    throw std::runtime_error("Failed to load glTF file: " + path.string());
  }

  result = cgltf_load_buffers(&options, data, path.string().c_str() /* surprise! does not expect parent path*/);
  if(result != cgltf_result_success)
  {
    cgltf_free(data);
    throw std::runtime_error("Failed to load buffers for glTF file: " + path.string());
  }

  for(size_t i = 0; i < data->meshes_count; ++i)
  {
    const cgltf_mesh& cgltfMesh = data->meshes[i];
    for(size_t j = 0; j < cgltfMesh.primitives_count; ++j)
    {
      const cgltf_primitive& primitive = cgltfMesh.primitives[j];
      if(primitive.type != cgltf_primitive_type_triangles)
      {
        continue;
      }

      std::span<const cgltf_attribute> attributes(primitive.attributes, primitive.attributes_count);
      auto                             positionsIt = std::ranges::find_if(attributes, [](const cgltf_attribute& a) {
        return a.type == cgltf_attribute_type_position;
      });
      if(positionsIt == attributes.end())
      {
        throw std::runtime_error("Failed to find position attribute");
      }

      std::string meshName = cgltfMesh.name ? std::string(cgltfMesh.name) : std::to_string(i);
      std::string materialName = primitive.material && primitive.material->name ? std::string(primitive.material->name) : "null";

      GeometryMesh mesh;
      mesh.name = path.stem().string() + "_" + meshName + "_p" + std::to_string(j) + "_" + materialName;
      mesh.positions.resize(positionsIt->data->count);
      mesh.triangles.resize(primitive.indices->count / 3);
      for(uint32_t k = 0; k < uint32_t(mesh.positions.size()); ++k)
      {
        if(!cgltf_accessor_read_float(positionsIt->data, k, reinterpret_cast<float*>(mesh.positions.data() + k), 3))
        {
          throw std::runtime_error("Failed to read positions");
        }
      }
      for(uint32_t k = 0; k < uint32_t(mesh.triangles.size() * 3); ++k)
      {
        if(!cgltf_accessor_read_uint(primitive.indices, k, reinterpret_cast<uint32_t*>(mesh.triangles.data()) + k, 1))
        {
          throw std::runtime_error("Failed to read positions");
        }
      }
      meshes.push_back(std::move(mesh));
    }
  }
  cgltf_free(data);
  return meshes;
}

#endif

void testVertexUnderfill1(const GeometryMesh& mesh)
{
  // Compute spatial inputs
  std::vector<AABB> boundingBoxes(mesh.triangles.size());
  std::ranges::transform(mesh.triangles, boundingBoxes.begin(), [&](vec3u tri) { return aabb(tri, mesh.positions); });
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](AABB b) { return b.center(); });

  // Compute connectivity inputs
  nvcluster::MeshConnections connections = makeMeshConnections(true, mesh);

  // Test a few maxTriangles and maxVertices values
  auto maxVertexTests = std::to_array<uint32_t>({0xffffffffu, 13, 32, 64});
  for(auto maxTriangles : std::to_array<uint32_t>({32, 64, 128}))
  {
    uint32_t noMaximumClusterCount = 0;
    std::array<uint32_t, maxVertexTests.size()> overflowWithoutLimit = {};  // Counts of clusters that would overflow the vertex limit if one was not set
    std::array<uint32_t, maxVertexTests.size()> totalClusters = {};  // Total cluster count for each maxVertices test
    for(size_t maxVerticesIndex = 0; maxVerticesIndex < maxVertexTests.size(); ++maxVerticesIndex)
    {
      uint32_t       noPenaltyClusterCount = 0;  // Cluster count for the first costUnderfillVertices iteration (0.0f)
      const uint32_t maxVertices           = maxVertexTests[maxVerticesIndex];
      for(auto costUnderfillVertices : std::to_array<float>({0.0f, 0.1f, 1.0f}))
      {
        if(maxVertices == 0xffffffffu && costUnderfillVertices != 0.0f)
          continue;

        // Cluster the mesh
        SCOPED_TRACE("Max triangles " + std::to_string(maxTriangles) + " vertices " + std::to_string(maxVertices)
                     + " vertex underfill cost " + std::to_string(costUnderfillVertices));
        std::vector<nvcluster::Range> oneSegment{{0, uint32_t(boundingBoxes.size())}};
        ClusterStorage                clustering(nvcluster::Input{
            nvcluster_Config{
                               .minClusterSize        = 1,
                               .maxClusterSize        = maxTriangles,
                               .maxClusterVertices    = maxVertices,
                               .costUnderfill         = 0.1f,
                               .costOverlap           = 0.1f,
                               .costUnderfillVertices = costUnderfillVertices,
                               .itemVertexCount       = 3,
                               .preSplitThreshold     = 0,
            },
            boundingBoxes,
            centroids,
            oneSegment,
            connections.connectionRanges,
            connections.connectionItems,
            {},
            connections.connectionVertexBits,
        });

        // Process the results, i.e. unique vertex counts per cluster
        std::vector<uint32_t> clusterVertexCounts = countClusterVertices(clustering, mesh);
        uint32_t              lessThan25          = 0;
        for(uint32_t count : clusterVertexCounts)
        {
          // Should never overflow what we passed to maxClusterVertices
          EXPECT_LE(count, maxVertices);

          // Record would-have-overflowed counts to make sure we're not just
          // getting lucky with maxTriangles
          if(maxVertices == 0xffffffffu)
          {
            for(size_t i = 0; i < maxVertexTests.size(); ++i)
            {
              if(count > maxVertexTests[i])
                ++overflowWithoutLimit[i];
              ++totalClusters[i];
            }
          }

          // Count clusters with under 25% vertices
          if(count < maxVertices / 4)
            ++lessThan25;
        }

        const uint32_t clusterCount = uint32_t(clustering.clusterItemRanges.size());
        if(maxVertices == 0xffffffffu)
        {
          // Set noMaximumClusterCount for later
          assert(noMaximumClusterCount == 0);
          noMaximumClusterCount = clusterCount;
        }
        else if(costUnderfillVertices == 0.0f)
        {
          // Set noPenaltyClusterCount for later
          assert(noPenaltyClusterCount == 0);
          assert(noMaximumClusterCount != 0);
          noPenaltyClusterCount = clusterCount;
          EXPECT_GE(clusterCount, uint32_t(float(noMaximumClusterCount) * 0.9f)) << "More clusters is expected if "
                                                                                    "there's a vertex limit";

          // Verify by comparing to the previous iteration that had no vertex
          // limit. It's not uncommon to double the number of clusters when the
          // max triangles is similar or less
          if(maxTriangles <= maxVertices)
          {
            EXPECT_LT(clusterCount, uint32_t(float(noMaximumClusterCount) * 2.0f))
                << "Should not be generating absurd "
                   "numbers of clusters with a vertex limit";
          }

          EXPECT_LT(lessThan25, uint32_t(clustering.clusterItemRanges.size() / 8 + 2))
              << "Should not be generating absurd "
                 "numbers of tiny clusters with a vertex limit";
        }
        else
        {
          assert(noMaximumClusterCount != 0);
          assert(noPenaltyClusterCount != 0);
          EXPECT_GE(clusterCount, uint32_t(float(noMaximumClusterCount) * 0.9f)) << "More clusters is expected if "
                                                                                    "there's a vertex limit";

          EXPECT_LE(lessThan25, uint32_t(clustering.clusterItemRanges.size() / 8 + 2))
              << "Should not be generating absurd "
                 "numbers of tiny clusters with a vertex limit and cost";

          // If there would have been some overflowing clusters and we are
          // likely vertex limited...
          if(overflowWithoutLimit[maxVerticesIndex] > 10 && maxTriangles > (maxVertices * 2u) / 3u)
          {
#if 0
            // Verify the vertex underfill cost reduces the number of clusters by
            // comparing to the initial iterations that had no vertex underfill
            // cost.
            // TODO: would like to see better numbers here
            EXPECT_LT(clusterCount, (noPenaltyClusterCount * 99u) / 100u)
                << "Should be generating less clusters with a vertex "
                   "underfill cost when vertex limited";
#else
            EXPECT_LE(clusterCount, noPenaltyClusterCount) << "Should at least not be generating more clusters with a "
                                                              "vertex underfill cost";
#endif
          }
        }
      }
    }

    // The tree mesh is expected to make clusters with more vertices than
    // indices. Validate that there are some overflows if a vertex limit is not
    // set for vertex limits similar or less than triangle limits.
    for(size_t i = 1; i < overflowWithoutLimit.size(); ++i)
    {
      SCOPED_TRACE("Max triangles " + std::to_string(maxTriangles) + ", vertices " + std::to_string(maxVertexTests[i]));
      if(maxTriangles > maxVertexTests[i] / 2u)
      {
        EXPECT_GT(overflowWithoutLimit[i], 0);
      }
      //printf("Overflow clusters %u/%u = %.1f%%\n", overflowWithoutLimit[i], totalClusters[i],
      //       100.0f * float(overflowWithoutLimit[i]) / float(totalClusters[i]));
    }
  }
}

static AABB operator*(const AABB& aabb, float scale)
{
  return {aabb.min * scale, aabb.max * scale};
}

// Returns the average underfill cost metric ratio (lower is better, reducing
// the number of clusters with underfilled vertices)
float testVertexUnderfill2(const GeometryMesh& mesh, float connectionWeight = 0.0f, float scale = 1.0f)
{
  uint32_t totalUnderfillMetricWithLimit             = 0;
  uint32_t totalUnderfillMetricWithLimitAndUnderfill = 0;

  // Compute spatial inputs
  std::vector<AABB> boundingBoxes(mesh.triangles.size());
  std::ranges::transform(mesh.triangles, boundingBoxes.begin(),
                         [&](vec3u tri) { return aabb(tri, mesh.positions) * scale; });
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](AABB b) { return b.center(); });

  // Compute connectivity inputs
  nvcluster::MeshConnections connections = makeMeshConnections(true, mesh);

  // Optional weights
  std::vector<float> connectionWeights;
  if(connectionWeight != 0.0f)
    connectionWeights.resize(connections.connectionItems.size(), connectionWeight);

  // Verify connections
  for(size_t i = 0; i < connections.connectionRanges.size(); ++i)
  {
    EXPECT_FALSE(contains(subspan(connections.connectionItems, connections.connectionRanges[i]), uint32_t(i)))
        << "should not connect to self";
    EXPECT_TRUE(allUnique(subspan(connections.connectionItems, connections.connectionRanges[i])))
        << "should be no duplicate connections";
    if(i < 10)
    {
      for(uint32_t j : subspan(connections.connectionItems, connections.connectionRanges[i]))
      {
        bool sharesAtLeastOne = false;
        for(uint32_t a = 0; a < 3; ++a)
        {
          for(uint32_t b = 0; b < 3; ++b)
          {
            sharesAtLeastOne = sharesAtLeastOne || mesh.triangles[j][a] == mesh.triangles[i][b];
          }
        }
        EXPECT_TRUE(sharesAtLeastOne) << "connected triangles must share a vertex";
      }
    }
  }

  // Iterate over a few different vertex limits
  for(auto maxVertices : std::to_array<uint32_t>({4, 7, 64}))
  {
    // Skip tests where there would only be a single cluster
    if(maxVertices >= mesh.positions.size())
      continue;

    const uint32_t maxTriangles = 64;

    SCOPED_TRACE("Max vertices: " + std::to_string(maxVertices));

    // Compute the number of clusters with each vertex count (histogram)
    std::vector<uint32_t> vertexCountHistogramWithoutLimit;
    std::vector<uint32_t> vertexCountHistogramWithLimit;
    std::vector<uint32_t> vertexCountHistogramWithLimitAndUnderfill;

    // Number of clusters that naturally exceed the vertex limit if one
    // isn't set. If this is not positive, the vertex limit wouldn't do
    // anything anyway.
    uint32_t overflowWithoutLimit = 0;

    // Total clusters with/without vertex limit
    uint32_t clusterCountWithoutLimit          = 0;
    uint32_t clusterCountWithLimit             = 0;
    uint32_t clusterCountWithLimitAndUnderfill = 0;

    // Total SAH cost of all clusters with/without vertex limit
    float sahCostWithoutLimit          = 0.0f;
    float sahCostWithLimit             = 0.0f;
    float sahCostWithLimitAndUnderfill = 0.0f;

    // Cluster the geometry three times:
    // 1. without any vertex limit
    // 2. with just a vertex limit
    // 3. with a vertex limit and underfill cost
    for(int i = 0; i < 3; ++i)
    {
      bool             vertexLimit           = i == 1 || i == 2;
      float            costUnderfillVertices = i == 2 ? 0.5f : 0.0f;
      nvcluster_Config config{
          .minClusterSize        = 1,
          .maxClusterSize        = maxTriangles,
          .maxClusterVertices    = vertexLimit ? maxVertices : ~0u,
          .costUnderfill         = 0.0f,
          .costOverlap           = 0.0f,
          .costUnderfillVertices = costUnderfillVertices,
          .itemVertexCount       = 3,
          .preSplitThreshold     = 0,
      };
      std::vector<nvcluster::Range> oneSegment{{0, uint32_t(boundingBoxes.size())}};
      ClusterStorage                clustering(nvcluster::Input{
          config,
          boundingBoxes,
          centroids,
          oneSegment,
          connections.connectionRanges,
          connections.connectionItems,
          connectionWeights,  // may be empty
          connections.connectionVertexBits,
      });

      // Count unique vertices per cluster
      auto& clusterCount = i == 0 ? clusterCountWithoutLimit : (i == 1 ? clusterCountWithLimit : clusterCountWithLimitAndUnderfill);
      auto& vertexCountHistogram = i == 0 ? vertexCountHistogramWithoutLimit :
                                            (i == 1 ? vertexCountHistogramWithLimit : vertexCountHistogramWithLimitAndUnderfill);
      auto& sahCost = i == 0 ? sahCostWithoutLimit : (i == 1 ? sahCostWithLimit : sahCostWithLimitAndUnderfill);
      clusterCount  = uint32_t(clustering.clusterItemRanges.size());
      vertexCountHistogram.resize(config.itemVertexCount * maxTriangles + 1, 0u);
      for(nvcluster_Range r : clustering.clusterItemRanges)
      {
        EXPECT_LE(r.count, maxTriangles) << "Max triangles per cluster exceeded";
        std::span<const uint32_t>    cluster = subspan(clustering.items, r);
        std::unordered_set<uint32_t> uniqueVertices;
        AABB                         aabb = AABB::empty();
        for(auto t : cluster)
        {
          uniqueVertices.insert(mesh.triangles[t][0]);
          uniqueVertices.insert(mesh.triangles[t][1]);
          uniqueVertices.insert(mesh.triangles[t][2]);
          aabb += boundingBoxes[t];
        }
        uint32_t uniqueVertexCount = uint32_t(uniqueVertices.size());
        if(!vertexLimit && uniqueVertexCount > maxVertices)
        {
          ++overflowWithoutLimit;
        }
        else if(vertexLimit)
        {
          EXPECT_LE(uniqueVertexCount, maxVertices) << "Vertex limit should be a hard limit";
        }
        vertexCountHistogram[uniqueVertexCount]++;
        sahCost += aabb.half_area() * float(cluster.size());
      }
    }

    SCOPED_TRACE("Clusters before, after vertex limit and with cost: " + std::to_string(clusterCountWithoutLimit)
                 + " -> " + std::to_string(clusterCountWithLimit) + " -> " + std::to_string(clusterCountWithLimitAndUnderfill));

    // Clusters with less than 3 vertices doesn't make sense
    EXPECT_EQ(vertexCountHistogramWithoutLimit[0], 0);
    EXPECT_EQ(vertexCountHistogramWithoutLimit[1], 0);
    EXPECT_EQ(vertexCountHistogramWithoutLimit[2], 0);
    EXPECT_EQ(vertexCountHistogramWithLimit[0], 0);
    EXPECT_EQ(vertexCountHistogramWithLimit[1], 0);
    EXPECT_EQ(vertexCountHistogramWithLimit[2], 0);
    EXPECT_EQ(vertexCountHistogramWithLimitAndUnderfill[0], 0);
    EXPECT_EQ(vertexCountHistogramWithLimitAndUnderfill[1], 0);
    EXPECT_EQ(vertexCountHistogramWithLimitAndUnderfill[2], 0);

    // Ignore further validation when the vertex limit doesn't have any
    // effect on the number of clusters
    if(clusterCountWithoutLimit == clusterCountWithLimit)
      continue;

    // Vertex limit makes more clusters
    EXPECT_LE(sahCostWithLimit, sahCostWithoutLimit) << "Splitting more nodes should only decrease the SAH";
    EXPECT_GE(clusterCountWithLimit, clusterCountWithoutLimit) << "Splitting more nodes will probably just increase "
                                                                  "the number of clusters";

    // Sanity check not too many clusters are produced when max triangles is
    // similar.
    // NOTE: more than double is expected if the vertex limit is far below the triangle limit
    if(maxTriangles <= maxVertices)
    {
      EXPECT_LE(clusterCountWithLimit, (clusterCountWithoutLimit - overflowWithoutLimit) + (overflowWithoutLimit * 24u) / 10u)
          << "Should not have more than double the number of overflow clusters with a vertex limit for regular "
             "meshes (with a 20% fudge factor)";
    }

    // Check for abnormally tiny clusters
    if(maxVertices > 10)
    {
      uint32_t tinyClusterCount =
          std::accumulate(vertexCountHistogramWithLimit.begin(), vertexCountHistogramWithLimit.begin() + 10, 0u);
      uint32_t largeClusterCount = std::accumulate(vertexCountHistogramWithLimit.begin() + maxVertices - 10,
                                                   vertexCountHistogramWithLimit.begin() + maxVertices, 0u);
      EXPECT_LE(tinyClusterCount, largeClusterCount + 2 /* ignore tiny meshes where noise dominates */)
          << "Shouldn't have an abnormal number of tiny clusters "
             "compared to large clusters";
    }

    // Check that the underfill cost reduces the number of clusters. Skip this
    // check for tiny vertex limits.
    if(maxVertices > 10)
    {
#if 0
      EXPECT_LT(clusterCountWithLimitAndUnderfill, (clusterCountWithLimit * 99u) / 100u)
          << "Underfill cost should reduce the number of clusters";
#else
      EXPECT_LE(clusterCountWithLimitAndUnderfill, (clusterCountWithLimit * 105u) / 100u)
          << "Underfill cost should at least not increase the number of clusters much";
#endif
    }

    // Compute a linear cost metric for under-filling cluster vertices. This
    // is trivially the sum of the number of missing vertices from each
    // cluster. Also compute the expected score, if cluster sizes were a
    // uniform distribution. We should do no worse than random.
    uint32_t underfillScore = 0;
    assert(vertexCountHistogramWithLimit.size() >= size_t(maxVertices) + 1);
    for(uint32_t i = 0; i <= maxVertices; ++i)
    {
      underfillScore += (maxVertices - i) * vertexCountHistogramWithLimit[i];
    }
    // Sum of counting numbers including n is n(n+1)/2; subtract 3 for
    // 0-vertices, 1-vertex and 2-vertex cases and divide by the total
    // possible outcomes.
    float    averageScore           = float((maxVertices - 3u) * (maxVertices - 2u)) / (2.0f * float(maxVertices - 3u));
    uint32_t underfillScoreIfRandom = uint32_t(ceilf(averageScore * float(clusterCountWithLimit)));
    //EXPECT_LE(underfillScore, underfillScoreIfRandom) << "Vertex limit tends to make smaller clusters than "
    //                                                     "randomly splitting";
    EXPECT_LE(underfillScore, underfillScoreIfRandom + underfillScoreIfRandom / 20)
        << "Vertex limit tends to make much smaller clusters than randomly splitting (plus a 5% fudge factor)";

    // Compute the same cost metric with the underfill cost
    uint32_t underfillScoreWithUnderfill = 0;
    assert(vertexCountHistogramWithLimitAndUnderfill.size() >= size_t(maxVertices) + 1);
    for(uint32_t i = 0; i <= maxVertices; ++i)
    {
      underfillScoreWithUnderfill += (maxVertices - i) * vertexCountHistogramWithLimitAndUnderfill[i];
    }

    if(maxVertices > 10)
    {
#if 0
      EXPECT_LT(underfillScoreWithUnderfill, (underfillScore * 99u) / 100u)
          << "Making clusters with a vertex underfill cost should "
            "reduce the overall underfill cost metric";
#else
      EXPECT_LT(underfillScoreWithUnderfill, (underfillScore * 130u) / 100u)
          << "Making clusters with a vertex underfill cost should "
             "at least not increase the overall underfill cost metric a lot";
#endif
    }

    totalUnderfillMetricWithLimit += underfillScore;
    totalUnderfillMetricWithLimitAndUnderfill += underfillScoreWithUnderfill;
  }

  return float(totalUnderfillMetricWithLimitAndUnderfill) / float(totalUnderfillMetricWithLimit);
}

float testItemUnderfill(const GeometryMesh& mesh, uint32_t maxTriangles)
{
  auto                                            itemUnderfillCosts = std::to_array({0.0f, 0.1f, 1.0f});
  std::array<uint32_t, itemUnderfillCosts.size()> underfillScores    = {};

  // Compute spatial inputs
  std::vector<AABB> boundingBoxes(mesh.triangles.size());
  std::ranges::transform(mesh.triangles, boundingBoxes.begin(), [&](vec3u tri) { return aabb(tri, mesh.positions); });
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](AABB b) { return b.center(); });

  for(size_t costIndex = 0; costIndex < itemUnderfillCosts.size(); ++costIndex)
  {
    const float      itemUnderfillCost = itemUnderfillCosts[costIndex];
    uint32_t&        underfillScore    = underfillScores[costIndex];
    nvcluster_Config config{
        .minClusterSize        = 1,
        .maxClusterSize        = maxTriangles,
        .maxClusterVertices    = ~0u,
        .costUnderfill         = itemUnderfillCost,
        .costOverlap           = 0.0f,
        .costUnderfillVertices = 0.0f,
        .itemVertexCount       = 3,
        .preSplitThreshold     = 0,
    };
    std::vector<nvcluster::Range> oneSegment{{0, uint32_t(boundingBoxes.size())}};
    ClusterStorage                clustering(nvcluster::Input{
        config,
        boundingBoxes,
        centroids,
        oneSegment,
    });

    // Compute the histogram of triangle counts
    std::vector<uint32_t> itemCountHistogram(maxTriangles + 1, 0);
    for(const auto& clusterRange : clustering.clusterItemRanges)
    {
      if(clusterRange.count > maxTriangles)  // ASSERT_LE can't be used here
        throw std::runtime_error("Cluster size exceeds maximum");
      itemCountHistogram[clusterRange.count]++;
    }

    // Compute a somewhat arbitrary score to measure underutilized space per
    // cluster
    underfillScore = 0;
    for(uint32_t i = 0; i <= maxTriangles; ++i)
    {
      underfillScore += (maxTriangles - i) * itemCountHistogram[i];
    }
  }

  // Return the average underfill score compared to the no-cost case
  return float(underfillScores[1] + underfillScores[2]) / float(underfillScores[0] * 2);
}

TEST(Underfill, Isosphere)
{
  GeometryMesh mesh = makeIcosphere(2);
  // testVertexUnderfill2() is not run because an icosphere's vertex efficiency is too high to be useful
  float itemScoreRatio = testItemUnderfill(mesh, 64);
  printf("Item underfill score ratio: %f\n", itemScoreRatio);
}

TEST(Underfill, GeneratedTree)
{
  GeometryMesh mesh = generateTree(2);
  testVertexUnderfill1(mesh);
  float vertexScoreRatio = testVertexUnderfill2(mesh);
  float itemScoreRatio   = testItemUnderfill(mesh, 64);
  printf("Vertex underfill score ratio: %f, item underfill score ratio: %f\n", vertexScoreRatio, itemScoreRatio);
}

// TODO: these trip up the underfill cost calculation, even with
// COMPUTE_AVERAGE_CUT_VERTICES
#if 0
TEST(Underfill, GeneratedCylinder1)
{
  GeometryMesh mesh = makeBranch(branchPath({0.0f}, 0.0f, 5.0f), 512, 5, 0.3f);
  testVertexUnderfill1(mesh);
  float vertexScoreRatio = testVertexUnderfill2(mesh);
  float itemScoreRatio   = testItemUnderfill(mesh, 64);
  printf("Vertex underfill score ratio: %f, item underfill score ratio: %f\n", vertexScoreRatio, itemScoreRatio);
}

TEST(Underfill, GeneratedCylinder2)
{
  GeometryMesh mesh = makeBranch(branchPath({0.0f}, 1.0f, 5.0f), 512, 5, 0.03f);
  testVertexUnderfill1(mesh);
  float vertexScoreRatio = testVertexUnderfill2(mesh);
  float itemScoreRatio   = testItemUnderfill(mesh, 64);
  printf("Vertex underfill score ratio: %f, item underfill score ratio: %f\n", vertexScoreRatio, itemScoreRatio);
}
#endif

TEST(Underfill, GeneratedLeaf)
{
  GeometryMesh mesh = makeTriangleStrip(branchPath({0.0f}, 30.0f, 500.0f), 512, 1.0f);
  testVertexUnderfill1(mesh);
  float vertexScoreRatio = testVertexUnderfill2(mesh);
  float itemScoreRatio   = testItemUnderfill(mesh, 64);
  printf("Vertex underfill score ratio: %f, item underfill score ratio: %f\n", vertexScoreRatio, itemScoreRatio);
}

TEST(Underfill, Meshes)
{
#ifndef TEST_MESHES
  GTEST_SKIP() << "Configure with -DNVCLUSTER_TEST_MESHES=ON to enable";
#else
  float totalItemUnderfillScoreRatio   = 0.0f;
  float totalItemUnderfillTests        = 0.0f;
  float totalVertexUnderfillScoreRatio = 0.0f;
  float totalVertexUnderfillTests      = 0.0f;

  std::vector<std::filesystem::path> gltfFiles;
  for(const auto& entry : std::filesystem::directory_iterator("."))
  {
    if(entry.path().extension() == ".gltf")
    {
      gltfFiles.push_back(entry.path());
    }
  }

  if(gltfFiles.empty())
  {
    GTEST_SKIP() << "No meshes to test in current directory";
  }

  for(const auto& gltfFile : gltfFiles)
  {
    for(GeometryMesh mesh : gltfMeshes(gltfFile))
    {
#if 0
      if(mesh.name != "test:mesh(meshname):prim(0):mat(materialname)")
        continue;
#endif

      EXPECT_GT(mesh.triangles.size(), 0);
      EXPECT_GT(mesh.positions.size(), 0);

      totalItemUnderfillScoreRatio += testItemUnderfill(mesh, 64);
      totalItemUnderfillTests += 1.0f;

      printf("%s\n", mesh.name.c_str());
      {
        SCOPED_TRACE(mesh.name);
        totalVertexUnderfillScoreRatio += testVertexUnderfill2(mesh);
        totalVertexUnderfillTests += 1.0f;
      }
      {
        //SCOPED_TRACE(mesh.name + " (with weights)");
        //testVertexUnderfill2(mesh, 0.1f);
      }
      {
        //SCOPED_TRACE(mesh.name + " (with weights and scaled)");
        //testVertexUnderfill2(mesh, 0.1f, 0.001f);
      }
    }
  }
  printf("Average item underfill score ratio when cost is added (lower is better): %f\n",
         totalItemUnderfillScoreRatio / totalItemUnderfillTests);
  printf("Average vertex underfill score ratio when cost is added (lower is better): %f\n",
         totalVertexUnderfillScoreRatio / totalVertexUnderfillTests);
#endif
}
