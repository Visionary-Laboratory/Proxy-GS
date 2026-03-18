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
#include <array>
#include <cmath>
#include <execution>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nvcluster/nvcluster.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <ranges>
#include <span>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <test_util.hpp>
#include <underfill_cost.hpp>
#include <unordered_map>
#include <unordered_set>

#if defined(NVCLUSTER_BUILDER_COMPILING)
#error NVCLUSTER_BUILDER_COMPILING must only be defined when building the library
#endif

// Sort items within each cluster by their index and then the clusters based on
// the first item. This makes verifying clustering results easier.
void sortClusters(nvcluster::ClusterStorage& clustering)
{
  for(const nvcluster_Range& cluster : clustering.clusterItemRanges)
  {
    std::sort(clustering.items.begin() + cluster.offset,                   // Start of elements to sort
              clustering.items.begin() + cluster.offset + cluster.count);  // Exclusive end of elements to sort
  }

  std::sort(clustering.clusterItemRanges.begin(), clustering.clusterItemRanges.end(),
            [&clustering](nvcluster_Range& clusterA, nvcluster_Range& clusterB) {
              return clustering.items[clusterA.offset] < clustering.items[clusterB.offset];
            });
}

// Sorts clusters within segments, but not the segments themselves
void sortClusters(nvcluster::SegmentedClusterStorage& clustering)
{
  for(const nvcluster_Range& cluster : clustering.clusterItemRanges)
  {
    std::sort(clustering.items.begin() + cluster.offset,                   // Start of elements to sort
              clustering.items.begin() + cluster.offset + cluster.count);  // Exclusive end of elements to sort
  }

  for(auto& segment : clustering.segmentClusterRanges)
  {
    std::sort(clustering.clusterItemRanges.begin() + segment.offset,                  // Start
              clustering.clusterItemRanges.begin() + segment.offset + segment.count,  // Exclusive end
              [&clustering](nvcluster_Range& clusterA, nvcluster_Range& clusterB) {
                return clustering.items[clusterA.offset] < clustering.items[clusterB.offset];
              });
  }
}

TEST(Config, ExactCount)
{
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 10, .maxClusterSize = 10};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);
    EXPECT_EQ(counts.clusterCount, 9);
  }
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 10, .maxClusterSize = 10};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 91, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);
    EXPECT_EQ(counts.clusterCount, 10);
  }
}

TEST(Config, RangeCount)
{
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 5, .maxClusterSize = 10};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);
    EXPECT_EQ(counts.clusterCount, 18);
  }
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 5, .maxClusterSize = 10};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 91, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);
    EXPECT_EQ(counts.clusterCount, 19);
  }
}

TEST(Config, PresplitCount)
{
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 5, .maxClusterSize = 10, .preSplitThreshold = 89};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);

    // 18 full clusters plus two possibly non-full due to pre-split
    EXPECT_EQ(counts.clusterCount, 20);
  }
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 5, .maxClusterSize = 10, .preSplitThreshold = 10};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);

    // 18 full clusters plus 9 possibly non-full due to pre-split
    EXPECT_EQ(counts.clusterCount, 27);
  }
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 10, .maxClusterSize = 10, .preSplitThreshold = 89};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);

    // 9 full clusters plus two possibly non-full due to pre-split
    EXPECT_EQ(counts.clusterCount, 11);
  }
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 10, .maxClusterSize = 10, .preSplitThreshold = 5};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);

    // 9 full clusters plus 18 possibly non-full due to pre-split
    EXPECT_EQ(counts.clusterCount, 27);
  }
}

TEST(Config, VertexLimitedCount)
{
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 5, .maxClusterSize = 10, .maxClusterVertices = 11, .itemVertexCount = 3};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);

    // There is no minClusterVertices and minClusterSize is ignored when
    // maxClusterVertices is set
    EXPECT_EQ(counts.clusterCount, 90);
  }
  {
    nvcluster_Counts counts;
    nvcluster_Config config{.minClusterSize = 5, .maxClusterSize = 10, .maxClusterVertices = 12, .itemVertexCount = 3};
    EXPECT_EQ(nvclusterGetRequirements(ScopedContext(), &config, 90, &counts), nvcluster_Result::NVCLUSTER_SUCCESS);

    // There is no minClusterVertices and minClusterSize is ignored when
    // maxClusterVertices is set
    EXPECT_EQ(counts.clusterCount, 90);
  }
}

// This is the test that appears in the README, so any changes to this test
// also should likely be reflected in the README file.
TEST(Clusters, Simple2x2)
{
  // Test items
  // 0 and 2 are close and should be in a cluster
  // 1 and 3 are close and should be in a cluster
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},
      {{0, 100, 0}, {1, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}},
      {{1, 100, 0}, {2, 101, 1}},
  };

  // Generate centroids
  std::vector<vec3f> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = boundingBoxes[i].center();
  }

  // Input structs
  nvcluster_Input  input{.itemBoundingBoxes = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
                         .itemCentroids     = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
                         .itemCount         = static_cast<uint32_t>(boundingBoxes.size())};
  nvcluster_Config config{
      .minClusterSize    = 2,
      .maxClusterSize    = 2,
      .costUnderfill     = 0.0f,  // zero to one (exclusive)
      .costOverlap       = 0.0f,  // zero to one (exclusive)
      .preSplitThreshold = 0,     // median-split bigger nodes (0=disable)
  };

  // Clustering
  ScopedContext             context;
  nvcluster::ClusterStorage clustering;
  nvcluster_Result          result = nvcluster::generateClusters(context.context, config, input, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);

  sortClusters(clustering);

  // Verify
  ASSERT_EQ(clustering.clusterItemRanges.size(), 2);
  ASSERT_EQ(clustering.items.size(), 4);
  const nvcluster_Range cluster0 = clustering.clusterItemRanges[0];
  ASSERT_EQ(cluster0.count, 2);
  EXPECT_EQ(clustering.items[cluster0.offset], 0);
  EXPECT_EQ(clustering.items[cluster0.offset + 1], 2);
  const nvcluster_Range cluster1 = clustering.clusterItemRanges[1];
  ASSERT_EQ(cluster1.count, 2);
  EXPECT_EQ(clustering.items[cluster1.offset], 1);
  EXPECT_EQ(clustering.items[cluster1.offset + 1], 3);

  for(size_t clusterIndex = 0; clusterIndex < clustering.clusterItemRanges.size(); ++clusterIndex)
  {
    const nvcluster_Range& range = clustering.clusterItemRanges[clusterIndex];
    for(uint32_t clusterItemIndex = 0; clusterItemIndex < range.count; ++clusterItemIndex)
    {
      uint32_t clusterItem = clustering.items[range.offset + clusterItemIndex];
      std::ignore          = clusterItem;
    }
  }
};

/*
 * Tests that weights affect the clusterizer's result.
 *
 * In the following diagram, v0 ... v3 are bounding boxes,
 * the edges are connections, and the `w` labels are weights:
 *
 *  v0 <- w1 -> v2
 *   ^           |
 *   |           |
 *   |           |
 * w1000       w1000
 *   |           |
 *   |           |
 *   v           v
 *  v1 <- w1 -> v3
 */
TEST(Clusters, Simple2x2Weights)
{
  // Test items
  // 0 and 2 are close and would normally be in a cluster
  // 1 and 3 are close and would normally be in a cluster
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},
      {{0, 100, 0}, {1, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}},
      {{1, 100, 0}, {2, 101, 1}},
  };

  // Adjacency/connections to override normal spatial clustering and instead
  // make clusters of {0, 1}, {2, 3}.
  std::vector<nvcluster_Range> connectionRanges{{0, 2}, {2, 2}, {4, 2}, {6, 2}};  // 2 connections each
  std::vector<uint32_t>        connectionItems{
      1, 2,  // item 0 connections
      0, 3,  // item 1 connections
      0, 3,  // item 2 connections
      1, 2,  // item 3 connections
  };
  std::vector<float> connectionWeights{
      1000.0f, 1.0f,     // weight from 0 to 1 and 2 respectively
      1000.0f, 1.0f,     // weight from 1 to 0 and 3 respectively
      1.0f,    1000.0f,  // weight from 2 to 0 and 3 respectively
      1.0f,    1000.0f,  // weight from 3 to 1 and 2 respectively
  };

  // Generate centroids
  std::vector<vec3f> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = boundingBoxes[i].center();
  }

  // Input structs
  nvcluster_Input input{
      .itemBoundingBoxes     = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
      .itemCentroids         = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
      .itemCount             = static_cast<uint32_t>(boundingBoxes.size()),
      .itemConnectionRanges  = connectionRanges.data(),
      .connectionTargetItems = connectionItems.data(),
      .connectionWeights     = connectionWeights.data(),
      .connectionCount       = static_cast<uint32_t>(connectionItems.size()),
  };
  nvcluster_Config config{
      .minClusterSize    = 2,
      .maxClusterSize    = 2,
      .costUnderfill     = 0.0f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = 0,
  };

  // Clustering
  ScopedContext             context;
  nvcluster::ClusterStorage clustering;
  nvcluster_Result          result = nvcluster::generateClusters(context.context, config, input, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);

  sortClusters(clustering);

  // Verify
  ASSERT_EQ(clustering.clusterItemRanges.size(), 2);
  ASSERT_EQ(clustering.items.size(), 4);
  const nvcluster_Range cluster0 = clustering.clusterItemRanges[0];
  ASSERT_EQ(cluster0.count, 2);
  EXPECT_EQ(clustering.items[cluster0.offset], 0);
  EXPECT_EQ(clustering.items[cluster0.offset + 1], 1);
  const nvcluster_Range cluster1 = clustering.clusterItemRanges[1];
  ASSERT_EQ(cluster1.count, 2);
  EXPECT_EQ(clustering.items[cluster1.offset], 2);
  EXPECT_EQ(clustering.items[cluster1.offset + 1], 3);
};

TEST(Clusters, EmptyConnections)
{
  // Test items
  // 0 and 2 are close and would normally be in a cluster
  // 1 and 3 are close and would normally be in a cluster
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},
      {{0, 100, 0}, {1, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}},
      {{1, 100, 0}, {2, 101, 1}},
  };

  // Every item has no connections
  std::vector<nvcluster_Range> connectionRanges{{0, 0}, {0, 0}, {0, 0}, {0, 0}};
  std::vector<uint32_t>        connectionItems{};
  std::vector<float>           connectionWeights{};

  // Generate centroids
  std::vector<vec3f> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = boundingBoxes[i].center();
  }

  // Input structs
  nvcluster_Input input{
      .itemBoundingBoxes     = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
      .itemCentroids         = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
      .itemCount             = static_cast<uint32_t>(boundingBoxes.size()),
      .itemConnectionRanges  = connectionRanges.data(),
      .connectionTargetItems = connectionItems.data(),
      .connectionWeights     = connectionWeights.data(),
      .connectionCount       = static_cast<uint32_t>(connectionItems.size()),
  };
  nvcluster_Config config{
      .minClusterSize    = 2,
      .maxClusterSize    = 2,
      .costUnderfill     = 0.0f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = 0,
  };

  // Clustering
  ScopedContext             context;
  nvcluster::ClusterStorage clustering;
  nvcluster_Result          result = nvcluster::generateClusters(context.context, config, input, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);
  ASSERT_EQ(clustering.clusterItemRanges.size(), 2);
  ASSERT_EQ(clustering.items.size(), 4);
};

TEST(Clusters, EmptyInput)
{
  // Test clustering with zero items
  std::vector<nvcluster::AABB> boundingBoxes;
  std::vector<vec3f>           centroids;
  nvcluster_Input              input{
                   .itemBoundingBoxes     = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
                   .itemCentroids         = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
                   .itemCount             = 0,
                   .itemConnectionRanges  = nullptr,
                   .connectionTargetItems = nullptr,
                   .connectionWeights     = nullptr,
                   .connectionCount       = 0,
  };
  nvcluster_Config config{
      .minClusterSize    = 1,
      .maxClusterSize    = 2,
      .costUnderfill     = 0.0f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = 0,
  };

  ScopedContext             context;
  nvcluster::ClusterStorage clustering;
  nvcluster_Result          result = nvcluster::generateClusters(context.context, config, input, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);
  ASSERT_EQ(clustering.clusterItemRanges.size(), 0);
  ASSERT_EQ(clustering.items.size(), 0);
}

TEST(Clusters, EmptySegmentedInput)
{
  // Test clustering with zero items and zero segments
  std::vector<nvcluster::AABB> boundingBoxes;
  std::vector<vec3f>           centroids;
  std::vector<nvcluster_Range> segments;  // empty

  nvcluster_Input input{
      .itemBoundingBoxes     = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
      .itemCentroids         = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
      .itemCount             = 0,
      .itemConnectionRanges  = nullptr,
      .connectionTargetItems = nullptr,
      .connectionWeights     = nullptr,
      .connectionCount       = 0,
  };
  nvcluster_Config config{
      .minClusterSize    = 1,
      .maxClusterSize    = 2,
      .costUnderfill     = 0.0f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = 0,
  };

  ScopedContext                      context;
  nvcluster::SegmentedClusterStorage clustering;
  nvcluster_Result                   result =
      nvcluster::generateSegmentedClusters(context.context, config, input,
                                           {segments.data(), static_cast<uint32_t>(segments.size())}, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);
  ASSERT_EQ(clustering.clusterItemRanges.size(), 0);
  ASSERT_EQ(clustering.items.size(), 0);
}

TEST(Clusters, OneItemEmptySegment)
{
  // Test with one item, but the only segment is empty (does not reference the item)
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},
  };
  std::vector<vec3f> centroids{
      boundingBoxes[0].center(),
  };
  // Segment does not reference the item (offset=0, count=0)
  std::vector<nvcluster_Range> segments{
      {0, 0},
  };

  nvcluster_Input input{
      .itemBoundingBoxes     = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
      .itemCentroids         = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
      .itemCount             = static_cast<uint32_t>(boundingBoxes.size()),
      .itemConnectionRanges  = nullptr,
      .connectionTargetItems = nullptr,
      .connectionWeights     = nullptr,
      .connectionCount       = 0,
  };
  nvcluster_Config config{
      .minClusterSize    = 1,
      .maxClusterSize    = 2,
      .costUnderfill     = 0.0f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = 0,
  };

  ScopedContext                      context;
  nvcluster::SegmentedClusterStorage clustering;
  nvcluster_Result                   result =
      nvcluster::generateSegmentedClusters(context.context, config, input,
                                           {segments.data(), static_cast<uint32_t>(segments.size())}, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);
  ASSERT_EQ(clustering.clusterItemRanges.size(), 0);
  ASSERT_EQ(clustering.items.size(), 1);  // unreferenced items still appear in the output
}

TEST(Clusters, Segmented2x2)
{
  // Test items
  // 0 and 2 are close and should be in a cluster
  // 1 and 3 are close and should be in a cluster
  // Repeated 3 times for each segment with a slight x offset
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}}, {{0, 100, 0}, {1, 101, 1}}, {{1, 0, 0}, {2, 1, 1}}, {{1, 100, 0}, {2, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}}, {{1, 100, 0}, {2, 101, 1}}, {{2, 0, 0}, {3, 1, 1}}, {{2, 100, 0}, {3, 101, 1}},
      {{2, 0, 0}, {3, 1, 1}}, {{2, 100, 0}, {3, 101, 1}}, {{3, 0, 0}, {4, 1, 1}}, {{3, 100, 0}, {4, 101, 1}},
  };

  // Segments
  // segment 0 should contain items 4 to 8
  // segment 1 should contain items 8 to 12
  // segment 2 should contain items 0 to 4
  std::vector<nvcluster_Range> segments{
      {4, 4},
      {8, 4},
      {0, 4},
  };

  // Generate centroids
  std::vector<vec3f> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = boundingBoxes[i].center();
  }

  // Input structs
  nvcluster_Input  input{.itemBoundingBoxes = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
                         .itemCentroids     = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
                         .itemCount         = static_cast<uint32_t>(boundingBoxes.size())};
  nvcluster_Config config{
      .minClusterSize    = 2,
      .maxClusterSize    = 2,
      .costUnderfill     = 0.0f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = 0,
  };

  // Clustering
  ScopedContext                      context;
  nvcluster::SegmentedClusterStorage clustering;
  nvcluster_Result                   result =
      nvcluster::generateSegmentedClusters(context.context, config, input,
                                           {segments.data(), static_cast<uint32_t>(segments.size())}, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);

  // Sort everything to validate items in clusters
  sortClusters(clustering);

  // Verify segment order remains consistent
  for(size_t segmentIndex = 0; segmentIndex < 3; ++segmentIndex)
  {
    const nvcluster_Range& segment      = clustering.segmentClusterRanges[segmentIndex];
    const nvcluster_Range& firstCluster = clustering.clusterItemRanges[segment.offset + 0];
    const uint32_t         firstItem    = clustering.items[firstCluster.offset + 0];
    EXPECT_EQ(firstItem, segments[segmentIndex].offset);
  }

  // Verify cluster in each segment and items in each cluster
  ASSERT_EQ(clustering.items.size(), 2 * 2 * 3);
  ASSERT_EQ(clustering.clusterItemRanges.size(), 2 * 3);
  ASSERT_EQ(clustering.segmentClusterRanges.size(), 3);
  for(size_t segmentIndex = 0; segmentIndex < 3; ++segmentIndex)
  {
    const uint32_t expectedFirstItem = segments[segmentIndex].offset;

    const nvcluster_Range& segment = clustering.segmentClusterRanges[segmentIndex];
    ASSERT_EQ(segment.count, 2);
    const nvcluster_Range& cluster0 = clustering.clusterItemRanges[segment.offset + 0];
    ASSERT_EQ(cluster0.count, 2);
    EXPECT_EQ(clustering.items[cluster0.offset + 0], expectedFirstItem + 0);
    EXPECT_EQ(clustering.items[cluster0.offset + 1], expectedFirstItem + 2);
    const nvcluster_Range& cluster1 = clustering.clusterItemRanges[segment.offset + 1];
    ASSERT_EQ(cluster1.count, 2);
    EXPECT_EQ(clustering.items[cluster1.offset + 0], segments[segmentIndex].offset + 1);
    EXPECT_EQ(clustering.items[cluster1.offset + 1], segments[segmentIndex].offset + 3);
  }
};

// Tests that the clusterizer respects minClusterSize and maxClusterSize.
TEST(Clusters, MinMaxItemSizes)
{
  ScopedContext context;

  const GeometryMesh mesh = makeIcosphere(3);

  std::vector<nvcluster::AABB> boundingBoxes(mesh.triangles.size());
  for(size_t i = 0; i < mesh.triangles.size(); i++)
  {
    boundingBoxes[i] = aabb(mesh.triangles[i], mesh.positions);
  }

  std::vector<vec3f> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = boundingBoxes[i].center();
  }

  nvcluster_Input input{.itemBoundingBoxes = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
                        .itemCentroids     = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
                        .itemCount         = static_cast<uint32_t>(boundingBoxes.size())};
  for(uint32_t sizeMax = 1; sizeMax < 10; ++sizeMax)
  {
    SCOPED_TRACE("Exact size: " + std::to_string(sizeMax));
    nvcluster_Config config{
        .minClusterSize    = sizeMax,
        .maxClusterSize    = sizeMax,
        .costUnderfill     = 0.0f,
        .costOverlap       = 0.0f,
        .preSplitThreshold = 0,
    };
    nvcluster::ClusterStorage clustering;
    nvcluster_Result          result = nvcluster::generateClusters(context.context, config, input, clustering);
    ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);

    // Validate all items exist and are unique
    std::set<uint32_t> uniqueItems(clustering.items.begin(), clustering.items.end());
    EXPECT_EQ(uniqueItems.size(), clustering.items.size());
    EXPECT_EQ(*std::ranges::max_element(uniqueItems), clustering.items.size() - 1);

    // We requested that all clusters have `sizeMax` triangles. When
    // mesh.triangle.size() isn't a multiple of `sizeMax`, though, there'll be
    // one cluster with the remaining triangles. So the minimum cluster size
    // should be
    uint32_t expectedMin = uint32_t(mesh.triangles.size()) % sizeMax;
    if(expectedMin == 0)
      expectedMin = sizeMax;
    // And the largest cluster should have size `sizeMax`.
    // Let's test that's true:
    uint32_t trueMinSize = ~0U, trueMaxSize = 0;
    for(const nvcluster_Range& cluster : clustering.clusterItemRanges)
    {
      trueMinSize = std::min(trueMinSize, cluster.count);
      trueMaxSize = std::max(trueMaxSize, cluster.count);
    }

    EXPECT_EQ(expectedMin, trueMinSize);
    EXPECT_EQ(sizeMax, trueMaxSize);
  }
}

// Test that preSplitThreshold works.
// Our mesh here is an icosahedron with 1280 triangles, and we set the
// pre-split threshold to 1000, so the code should start by splitting into
// two sets of elements.
TEST(Clusters, PreSplit)
{
  ScopedContext context;

  const uint32_t     preSplitThreshold = 1000;
  const GeometryMesh mesh              = makeIcosphere(3);
  // Make sure we'll pre-split at least once:
  EXPECT_GT(mesh.triangles.size(), preSplitThreshold);

  std::vector<nvcluster::AABB> boundingBoxes(mesh.triangles.size());
  for(size_t i = 0; i < mesh.triangles.size(); i++)
  {
    boundingBoxes[i] = aabb(mesh.triangles[i], mesh.positions);
  }

  std::vector<vec3f> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = boundingBoxes[i].center();
  }

  nvcluster_Input  input{.itemBoundingBoxes = reinterpret_cast<nvcluster_AABB*>(boundingBoxes.data()),
                         .itemCentroids     = reinterpret_cast<nvcluster_Vec3f*>(centroids.data()),
                         .itemCount         = static_cast<uint32_t>(boundingBoxes.size())};
  nvcluster_Config config = {
      .minClusterSize    = 100,
      .maxClusterSize    = 100,
      .costUnderfill     = 0.0f,
      .costOverlap       = 0.0f,
      .preSplitThreshold = preSplitThreshold,
  };
  nvcluster::ClusterStorage clustering;
  nvcluster_Result          result = nvcluster::generateClusters(context.context, config, input, clustering);
  ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);

  // Validate all items exist and are unique
  std::set<uint32_t> uniqueItems(clustering.items.begin(), clustering.items.end());
  EXPECT_EQ(uniqueItems.size(), clustering.items.size());
  EXPECT_EQ(*std::ranges::max_element(uniqueItems), clustering.items.size() - 1);

  // Validate all items are covered by a range exactly once
  std::vector<uint32_t> itemClusterCounts(clustering.items.size(), 0);
  for(const nvcluster_Range& range : clustering.clusterItemRanges)
  {
    for(uint32_t i = range.offset; i < range.offset + range.count; i++)
    {
      itemClusterCounts[i]++;
    }
  }
  // Is every element in `itemClusterCounts` equal to 1?
  EXPECT_EQ(std::set(itemClusterCounts.begin(), itemClusterCounts.end()), std::set<uint32_t>{1});

  // Validate most sizes are the maximum
  std::unordered_map<uint32_t, uint32_t> clusterSizeCounts;  // cluster size -> number of clusters with that size
  for(const nvcluster_Range& range : clustering.clusterItemRanges)
  {
    clusterSizeCounts[range.count]++;
  }

  // This number of clusters had the maximum size:
  const uint32_t maxSizedCount = clusterSizeCounts[config.maxClusterSize];
  // This number of clusters were undersized:
  const uint32_t undersizedCount = static_cast<uint32_t>(clustering.clusterItemRanges.size()) - maxSizedCount;
  // There should be at most this number of undersized clusters.
  // That is, there are ceil(mesh.triangles.size() / preSplitThreshold)
  // sets after pre-splitting. Each set should generate at most 1 undersized cluster.
  const uint32_t expectedUndersized = uint32_t(mesh.triangles.size() + preSplitThreshold - 1) / preSplitThreshold;
  EXPECT_LE(undersizedCount, expectedUndersized);
  EXPECT_GE(maxSizedCount, clustering.clusterItemRanges.size() - expectedUndersized);
}

extern "C" int runCTest(void);

TEST(Clusters, CAPITest)
{
  EXPECT_EQ(runCTest(), 1);
}

namespace nvcluster {
// Default to sqrt(vertexCount) for averageCutVertices - modelled for a triangle grid, but fails for cylinders
inline Underfill generalUnderfillCount(const Input& input, uint32_t itemCount, uint32_t vertexCount)
{
  return generalUnderfillCount(input, itemCount, vertexCount, sqrtf(float(vertexCount)));
}
}  // namespace nvcluster

TEST(GeneralUnderfill, VertexCount)
{
  nvcluster_Config config{};
  config.maxClusterSize     = 128;
  config.maxClusterVertices = 3;  // set later
  config.itemVertexCount    = 3;  // 3 = triangles
  std::vector<nvcluster_Range> segments{{}};
  nvcluster::Input             input{config, {}, nvcluster_Segments{segments.data(), 1}};

  // Underfill of 3 vertices with a max of 3 should be zero
  nvcluster::Underfill underfill = nvcluster::generalUnderfillCount(input, 13, 3);
  EXPECT_TRUE(underfill.vertexLimited);
  EXPECT_EQ(underfill.underfillCount, 0);

  // Two 5 vertices across two clsuters of 3 should be an underfill between 0 and 1
  underfill = nvcluster::generalUnderfillCount(input, 13, 5);
  EXPECT_LE(underfill.underfillCount, 1);

  // Same for 4 vertices some may be duplicated when the node is split.
  underfill = nvcluster::generalUnderfillCount(input, 13, 4);
  EXPECT_LE(underfill.underfillCount, 1);

  // 7 implies 3 clusters but still with a low underfill because there are now
  // more chances for duplicated vertices.
  underfill = nvcluster::generalUnderfillCount(input, 13, 7);
  EXPECT_LE(underfill.underfillCount, 1);

  // Double check the triangle/item limited check works
  underfill = nvcluster::generalUnderfillCount(input, 200, 3);
  EXPECT_FALSE(underfill.vertexLimited);
  EXPECT_EQ(underfill.underfillCount, config.maxClusterSize * 2 - 200);

  for(config.maxClusterVertices = 10; config.maxClusterVertices < 20; ++config.maxClusterVertices)
  {
    SCOPED_TRACE("maxClusterVertices " + std::to_string(config.maxClusterVertices));

    // Check same vertex counts as the max always produce zero underfill
    underfill = nvcluster::generalUnderfillCount(input, 100, config.maxClusterVertices);
    EXPECT_TRUE(underfill.vertexLimited);
    EXPECT_EQ(underfill.underfillCount, 0);

    // Check that a few less produces some underfill but that it is low
    underfill = nvcluster::generalUnderfillCount(input, 100, config.maxClusterVertices * 3 - 3);
    EXPECT_NEAR(float(underfill.underfillCount), float(config.maxClusterVertices) * 0.28f, 1.2f);
  }
}

TEST(GeneralUnderfill, VertexGrid)
{
  nvcluster_Config config{};
  config.maxClusterSize     = 9999;
  config.maxClusterVertices = 0;  // set later
  config.itemVertexCount    = 3;  // 3 = triangles
  std::vector<nvcluster_Range> segments{{}};
  nvcluster::Input             input{config, {}, nvcluster_Segments{segments.data(), 1}};
  for(uint32_t ch = 2; ch < 20; ++ch)
  {
    uint32_t cw           = ch;  // square clusters
    uint32_t clusterVerts = (cw + 1) * (ch + 1);
    for(uint32_t h = ch; h < 100; ++h)
    {
      uint32_t w = h;  // square initial grids
      SCOPED_TRACE("Grid " + std::to_string(w) + "x" + std::to_string(h) + " to clusters " + std::to_string(cw) + "x"
                   + std::to_string(ch));

      uint32_t verts = (w + 1) * (h + 1);
      uint32_t tris  = w * h * 2;

      config.maxClusterVertices = clusterVerts;

      nvcluster::Underfill underfill = nvcluster::generalUnderfillCount(input, tris, verts);
      EXPECT_TRUE(underfill.vertexLimited);
      EXPECT_LT(underfill.underfillCount, config.maxClusterVertices) << "Underfill should never equal the maximum, or "
                                                                        "be above";

      if(w == cw && h == ch)
      {
        // We are perfectly at the maximum already. Underfill should be zero
        EXPECT_EQ(clusterVerts, verts);
        EXPECT_EQ(underfill.underfillCount, 0);
      }
      else if(w % cw == 0 && h % ch == 0)
      {
        // Grid is an exact modulo of the cluster size. Underfill should be zero
        EXPECT_EQ(underfill.underfillCount, 0);
      }
      else if(w == cw * 2 - 1 && h == ch * 2 - 1 && cw > 10)
      {
        // Grid is slightly smaller than a multiple of the clusters. Expect a small underfill.
        EXPECT_LT(underfill.underfillCount, (clusterVerts * 3u) / 8u);
      }
      else if(w == cw * 2 + 1 && h == ch * 2 + 1 && cw > 12)
      {
        // Grid is slightly bigger than the clusters. Expect a large underfill.
        EXPECT_GT(underfill.underfillCount, (clusterVerts * 5u) / 8u);
      }
    }
  }
}

TEST(Clusters, SimpleLineMaxVertex)
{
  // A straight row of boxes
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},  //
      {{1, 0, 0}, {2, 1, 1}},  //
      {{2, 0, 0}, {3, 1, 1}},  //
      {{3, 0, 0}, {4, 1, 1}},  //
      {{4, 0, 0}, {5, 1, 1}},  //
      {{5, 0, 0}, {6, 1, 1}},  //
  };
  uint32_t           itemCount = uint32_t(boundingBoxes.size());
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](nvcluster::AABB b) { return b.center(); });

  // Neightboring items have sharedCount connections between them
  for(uint32_t sharedCount = 1; sharedCount <= 4; ++sharedCount)
  {
    uint32_t itemVertexCount = sharedCount * 2;
    SCOPED_TRACE("Item Vertex Count: " + std::to_string(itemVertexCount));

    // Populate ranges and vertex bits
    std::vector<nvcluster::Range> connectionRanges{
        {0, itemVertexCount / 2},           {sharedCount * 1, itemVertexCount}, {sharedCount * 3, itemVertexCount},
        {sharedCount * 5, itemVertexCount}, {sharedCount * 7, itemVertexCount}, {sharedCount * 9, itemVertexCount / 2},
    };
    std::vector<uint32_t> connectionItems(connectionRanges.back().end(), 0xffffffffu);
    std::vector<float>    connectionWeights(connectionItems.size(), 0.0f);
    std::vector<uint8_t>  connectionVertexBits(connectionItems.size(), 0u);
    for(uint32_t i = 0; i < sharedCount; ++i)
    {
      connectionItems[connectionRanges[0].offset + i]               = 1;
      connectionItems[connectionRanges[1].offset + i]               = 0;
      connectionItems[connectionRanges[1].offset + sharedCount + i] = 2;
      connectionItems[connectionRanges[2].offset + i]               = 1;
      connectionItems[connectionRanges[2].offset + sharedCount + i] = 3;
      connectionItems[connectionRanges[3].offset + i]               = 2;
      connectionItems[connectionRanges[3].offset + sharedCount + i] = 4;
      connectionItems[connectionRanges[4].offset + i]               = 3;
      connectionItems[connectionRanges[4].offset + sharedCount + i] = 5;
      connectionItems[connectionRanges[5].offset + i]               = 4;

      uint8_t backwardIndex                                              = uint8_t(1 << (i * 2 + 0));
      uint8_t forwardIndex                                               = uint8_t(1 << (i * 2 + 1));
      connectionVertexBits[connectionRanges[0].offset + i]               = forwardIndex;
      connectionVertexBits[connectionRanges[1].offset + i]               = backwardIndex;
      connectionVertexBits[connectionRanges[1].offset + sharedCount + i] = forwardIndex;
      connectionVertexBits[connectionRanges[2].offset + i]               = backwardIndex;
      connectionVertexBits[connectionRanges[2].offset + sharedCount + i] = forwardIndex;
      connectionVertexBits[connectionRanges[3].offset + i]               = backwardIndex;
      connectionVertexBits[connectionRanges[3].offset + sharedCount + i] = forwardIndex;
      connectionVertexBits[connectionRanges[4].offset + i]               = backwardIndex;
      connectionVertexBits[connectionRanges[4].offset + sharedCount + i] = forwardIndex;
      connectionVertexBits[connectionRanges[5].offset + i]               = backwardIndex;
    }

    // Iterate max vertex limit from the minimum to half the total items
    for(uint32_t maxVerticesPerCluster = itemVertexCount; maxVerticesPerCluster <= itemVertexCount * (itemCount / 2); ++maxVerticesPerCluster)
    {
      SCOPED_TRACE("Max Vertices Per Cluster: " + std::to_string(maxVerticesPerCluster));

      // Build clusters
      std::vector<nvcluster::Range> oneSegment{{0, uint32_t(boundingBoxes.size())}};
      ClusterStorage                clustering(nvcluster::Input{
          nvcluster_Config{
                             .minClusterSize        = 1,
                             .maxClusterSize        = 6,
                             .maxClusterVertices    = maxVerticesPerCluster,
                             .costUnderfill         = 0.0f,
                             .costOverlap           = 0.0f,
                             .costUnderfillVertices = 1.0f,
                             .itemVertexCount       = itemVertexCount,
                             .preSplitThreshold     = 0,
          },
          boundingBoxes,
          centroids,
          oneSegment,
          connectionRanges,
          connectionItems,
          connectionWeights,
          connectionVertexBits,
      });
      sortClusters(clustering);

      // Cast to a Range with a .end()
      auto clusteringItemRanges = std::span(reinterpret_cast<const nvcluster::Range*>(clustering.clusterItemRanges.data()),
                                            clustering.clusterItemRanges.size());

      // Verify splits are all on the X axis. Both SAH and the vertex underfill
      // cost should prevent anything else
      EXPECT_EQ(clustering.items[clustering.clusterItemRanges.front().offset], 0) << "First item is not first in the "
                                                                                     "first cluster";
      for(size_t i = 0; i < clusteringItemRanges.size() - 1; ++i)
      {
        EXPECT_EQ(clustering.items[clusteringItemRanges[i].end() - 1] + 1, clustering.items[clusteringItemRanges[i + 1].offset])
            << "Cluster borders don't match consecutive items";
      }
      EXPECT_EQ(clustering.items[clusteringItemRanges.back().end() - 1], clustering.items.size() - 1)
          << "Last item is not last in the last cluster";

      // Having verified all items are sequential, we can compute the number of
      // unique vertices more easily.
      uint32_t expectedMaxItemsPerCluster = maxVerticesPerCluster / sharedCount - 1;
      uint32_t expectedMaxUniqueVertices  = expectedMaxItemsPerCluster * sharedCount + 1;
      for(const auto& range : clusteringItemRanges)
        EXPECT_LE(range.count * sharedCount + 1, expectedMaxUniqueVertices);
    }
  }
}

TEST(Clusters, AutoMaxVertex)
{
  // A grid of 3x3 quads
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},  //
      {{1, 0, 0}, {2, 1, 1}},  //
      {{2, 0, 0}, {3, 1, 1}},  //
      {{0, 1, 0}, {1, 2, 1}},  //
      {{1, 1, 0}, {2, 2, 1}},  //
      {{2, 1, 0}, {3, 2, 1}},  //
      {{0, 2, 0}, {1, 3, 1}},  //
      {{1, 2, 0}, {2, 3, 1}},  //
      {{2, 2, 0}, {3, 3, 1}},  //
  };
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](nvcluster::AABB b) { return b.center(); });

  // Quads each have 4 vertices
  // 0  - 1  - 2  - 3
  // 4  - 5  - 6  - 7
  // 8  - 9  - 10 - 11
  // 12 - 13 - 14 - 15
  std::vector<uint32_t> itemVertices{
      0,  1,  4,  5,   //
      1,  2,  5,  6,   //
      2,  3,  6,  7,   //
      4,  5,  8,  9,   //
      5,  6,  9,  10,  //
      6,  7,  10, 11,  //
      8,  9,  12, 13,  //
      9,  10, 13, 14,  //
      10, 11, 14, 15,  //
  };

  // Unit test the internal connection computation directly
  for(int iteration = 0; iteration < 10; ++iteration)
  {
    nvcluster::MeshConnections connections =
        nvcluster::makeMeshConnections(true, nvcluster::ItemVertices(itemVertices.data(), 9u, 4u), 16u);
    ASSERT_EQ(connections.connectionRanges.size(), 9);
    ASSERT_EQ(connections.connectionItems.size(), 3 * 4 + 5 * 4 + 8);
    ASSERT_EQ(connections.connectionVertexBits.size(), 3 * 4 + 5 * 4 + 8);
    auto item0Connections = subspan(connections.connectionItems, connections.connectionRanges[0]);
    auto item3Connections = subspan(connections.connectionItems, connections.connectionRanges[3]);
    auto item6Connections = subspan(connections.connectionItems, connections.connectionRanges[6]);
    EXPECT_THAT(item0Connections, testing::UnorderedElementsAre(1, 3, 4));
    EXPECT_THAT(item3Connections, testing::UnorderedElementsAre(0, 1, 4, 6, 7));
    EXPECT_THAT(item6Connections, testing::UnorderedElementsAre(3, 4, 7));
    auto item0VertexBits = subspan(connections.connectionVertexBits, connections.connectionRanges[0]);
    auto item3VertexBits = subspan(connections.connectionVertexBits, connections.connectionRanges[3]);
    auto item6VertexBits = subspan(connections.connectionVertexBits, connections.connectionRanges[6]);
    auto atValueInOther  = [](auto& target, auto& other, auto value) {
      for(size_t i = 0; i < target.size(); ++i)
      {
        if(target[i] == value)
        {
          return other[i];
        }
      }
      return std::ranges::range_value_t<decltype(other)>{};
    };
    uint8_t item0To1Bits = atValueInOther(item0Connections, item0VertexBits, 1u);
    uint8_t item0To3Bits = atValueInOther(item0Connections, item0VertexBits, 3u);
    uint8_t item0To4Bits = atValueInOther(item0Connections, item0VertexBits, 4u);
    EXPECT_EQ(std::popcount(item0To1Bits), 2);
    EXPECT_EQ(std::popcount(item0To3Bits), 2);
    EXPECT_EQ(std::popcount(item0To4Bits), 1);
    EXPECT_EQ(std::popcount(uint32_t(item0To1Bits & ~(item0To3Bits | item0To4Bits))), 1)
        << "one bit should be a unique connection from item 1 to item 1";
    EXPECT_EQ(std::popcount(uint32_t(item0To3Bits & ~(item0To1Bits | item0To4Bits))), 1)
        << "one bit should be a unique connection from item 1 to item 3";
    EXPECT_EQ(std::popcount(uint32_t(item0To4Bits & ~(item0To1Bits | item0To3Bits))), 0) << "no unique bits connecting "
                                                                                            "item 1 to item 4";
    uint8_t item3To0Bits = atValueInOther(item3Connections, item3VertexBits, 0u);
    uint8_t item3To1Bits = atValueInOther(item3Connections, item3VertexBits, 1u);
    uint8_t item3To4Bits = atValueInOther(item3Connections, item3VertexBits, 4u);
    uint8_t item3To6Bits = atValueInOther(item3Connections, item3VertexBits, 6u);
    uint8_t item3To7Bits = atValueInOther(item3Connections, item3VertexBits, 7u);
    EXPECT_EQ(std::popcount(item3To0Bits), 2);
    EXPECT_EQ(std::popcount(item3To1Bits), 1);
    EXPECT_EQ(std::popcount(item3To4Bits), 2);
    EXPECT_EQ(std::popcount(item3To6Bits), 2);
    EXPECT_EQ(std::popcount(item3To7Bits), 1);
    EXPECT_EQ(std::popcount(uint32_t(item3To0Bits & ~(item3To1Bits | item3To4Bits | item3To6Bits | item3To7Bits))), 1)
        << "one bit should be a unique connection from item 3 to item 0";
    EXPECT_EQ(std::popcount(uint32_t(item3To1Bits & ~(item3To0Bits | item3To4Bits | item3To6Bits | item3To7Bits))), 0)
        << "no unique bits connecting item 3 to item 1";
    EXPECT_EQ(std::popcount(uint32_t(item3To4Bits & ~(item3To0Bits | item3To1Bits | item3To6Bits | item3To7Bits))), 0)
        << "no unique bits connecting item 3 to item 4";
    EXPECT_EQ(std::popcount(uint32_t(item3To6Bits & ~(item3To0Bits | item3To1Bits | item3To4Bits | item3To7Bits))), 1)
        << "one bit should be a unique connection from item 3 to item 6";
    EXPECT_EQ(std::popcount(uint32_t(item3To7Bits & ~(item3To0Bits | item3To1Bits | item3To4Bits | item3To6Bits))), 0)
        << "no unique bits connecting item 3 to item 7";
    uint8_t item6To3Bits = atValueInOther(item6Connections, item6VertexBits, 3u);
    uint8_t item6To4Bits = atValueInOther(item6Connections, item6VertexBits, 4u);
    uint8_t item6To7Bits = atValueInOther(item6Connections, item6VertexBits, 7u);
    EXPECT_EQ(std::popcount(item6To3Bits), 2);
    EXPECT_EQ(std::popcount(item6To4Bits), 1);
    EXPECT_EQ(std::popcount(item6To7Bits), 2);
    EXPECT_EQ(std::popcount(uint32_t(item6To3Bits & ~(item6To4Bits | item6To7Bits))), 1)
        << "one bit should be a unique connection from item 6 to item 3";
    EXPECT_EQ(std::popcount(uint32_t(item6To4Bits & ~(item6To3Bits | item6To7Bits))), 0) << "no unique bits connecting "
                                                                                            "item 6 to item 4";
    EXPECT_EQ(std::popcount(uint32_t(item6To7Bits & ~(item6To3Bits | item6To4Bits))), 1)
        << "one bit should be a unique connection from item 6 to item 7";
  }

  // Build clusters
  for(uint32_t maxVertices = 6; maxVertices <= 16; ++maxVertices)
  {
    nvcluster_Config config{
        .minClusterSize        = 1,
        .maxClusterSize        = 4,
        .maxClusterVertices    = 6,
        .costUnderfill         = 0.0f,
        .costOverlap           = 0.0f,
        .costUnderfillVertices = 0.0f,
        .itemVertexCount       = 4,
        .preSplitThreshold     = 0,
    };
    nvcluster_Input input{
        .itemBoundingBoxes = reinterpret_cast<const nvcluster_AABB*>(boundingBoxes.data()),
        .itemCentroids     = reinterpret_cast<const nvcluster_Vec3f*>(centroids.data()),
        .itemCount         = uint32_t(boundingBoxes.size()),
        .itemVertices      = itemVertices.data(),
        .vertexCount       = *std::max_element(itemVertices.begin(), itemVertices.end()) + 1,
    };

    nvcluster::ClusterStorage clustering;
    nvcluster_Result          result = nvcluster::generateClusters(ScopedContext(), config, input, clustering);
    ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);

    for(const nvcluster_Range& cluster : clustering.clusterItemRanges)
    {
      EXPECT_LE(cluster.count, maxVertices);
    }
  }
}


TEST(Clusters, AutoMaxVertexDisconnected)
{
  // A grid of 3x3 quads
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},  //
      {{1, 0, 0}, {2, 1, 1}},  //
      {{2, 0, 0}, {3, 1, 1}},  //
      {{0, 1, 0}, {1, 2, 1}},  //
      {{1, 1, 0}, {2, 2, 1}},  //
      {{2, 1, 0}, {3, 2, 1}},  //
      {{0, 2, 0}, {1, 3, 1}},  //
      {{1, 2, 0}, {2, 3, 1}},  //
      {{2, 2, 0}, {3, 3, 1}},  //
  };
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](nvcluster::AABB b) { return b.center(); });

  // Quads each have 4 vertices, but they're all unique
  std::vector<uint32_t> itemVertices(boundingBoxes.size() * 4, 0);
  std::iota(itemVertices.begin(), itemVertices.end(), 0);

  // Build clusters
  for(uint32_t maxVertices = 4; maxVertices <= 4; ++maxVertices)
  {
    nvcluster_Config config{
        .minClusterSize        = 1,
        .maxClusterSize        = 9,
        .maxClusterVertices    = maxVertices,
        .costUnderfill         = 0.0f,
        .costOverlap           = 0.0f,
        .costUnderfillVertices = 0.0f,
        .itemVertexCount       = 4,
        .preSplitThreshold     = 0,
    };
    nvcluster_Input input{
        .itemBoundingBoxes = reinterpret_cast<const nvcluster_AABB*>(boundingBoxes.data()),
        .itemCentroids     = reinterpret_cast<const nvcluster_Vec3f*>(centroids.data()),
        .itemCount         = uint32_t(boundingBoxes.size()),
        .itemVertices      = itemVertices.data(),
        .vertexCount       = *std::max_element(itemVertices.begin(), itemVertices.end()) + 1,
    };
    nvcluster::ClusterStorage clustering;
    nvcluster_Result          result = nvcluster::generateClusters(ScopedContext(), config, input, clustering);
    ASSERT_EQ(result, nvcluster_Result::NVCLUSTER_SUCCESS);
    for(const nvcluster_Range& clusterItemRange : clustering.clusterItemRanges)
    {
      EXPECT_LE(clusterItemRange.count * config.itemVertexCount, maxVertices);
    }
  }
}

TEST(Clusters, Simple2x2MaxVertex)
{
  // Test items
  // 0 and 2 are close and should be in a cluster
  // 1 and 3 are close and should be in a cluster
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},
      {{0, 100, 0}, {1, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}},
      {{1, 100, 0}, {2, 101, 1}},
  };

  // Connections are modelled as shared quad vertices. Everything connects to
  // everything, but "through" shared vertices. Like a dual multigraph where
  // edges are unique to shared vertices. Note that in this test a connection
  // per vertex is used. In practice, apps should consolidate all vertex bits
  // into triangle connections.
  std::vector<nvcluster::Range> connectionRanges{
      {0, 5},
      {5, 5},
      {10, 5},
      {15, 5},
  };
  std::vector<uint32_t> connectionItems{
      1, 1, 2, 2, 3,  //
      0, 0, 2, 3, 3,  //
      0, 0, 1, 3, 3,  //
      0, 1, 1, 2, 2,
  };
  std::vector<uint8_t> connectionVertexBits{
      1 << 1, 1 << 3, 1 << 2, 1 << 3, 1 << 2,  //
      1 << 0, 1 << 2, 1 << 2, 1 << 2, 1 << 3,  //
      1 << 0, 1 << 1, 1 << 1, 1 << 1, 1 << 3,  //
      1 << 0, 1 << 0, 1 << 1, 1 << 0, 1 << 2,
  };

  // Generate centroids
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](nvcluster::AABB b) { return b.center(); });

  // Build clusters
  for(float costUnderfillVertices : {0.0f, 1.0f})
  {
    std::vector<nvcluster::Range> oneSegment{{0, uint32_t(boundingBoxes.size())}};
    ClusterStorage                clustering(nvcluster::Input{
        nvcluster_Config{
                           .minClusterSize = 1,
                           .maxClusterSize = 4,  // Should not result in 4 or even 3, because that would exceed 6 vertices per cluster
                           .maxClusterVertices    = 6,  // 6 vertices per cluster
                           .costUnderfill         = 0.0f,
                           .costOverlap           = 0.0f,
                           .costUnderfillVertices = costUnderfillVertices,
                           .itemVertexCount       = 4,  // 4 vertices per quad
                           .preSplitThreshold     = 0,
        },
        boundingBoxes,
        centroids,
        oneSegment,
        connectionRanges,
        connectionItems,
        {},
        connectionVertexBits,
    });
    sortClusters(clustering);

    // Verify
    ASSERT_EQ(clustering.items.size(), 4);
    EXPECT_GE(clustering.clusterItemRanges.size(), 2);
    for(size_t i = 0; i < clustering.clusterItemRanges.size(); ++i)
      EXPECT_LE(clustering.clusterItemRanges[i].count, 2);

    if(costUnderfillVertices > 0.0f)
    {
      // Check the underfill cost is working to make full clusters rather
      // splitting nodes just based on the SAH cost. These may fail without but
      // there's no guarantee.
      ASSERT_EQ(clustering.clusterItemRanges.size(), 2);
      ASSERT_EQ(clustering.items.size(), 4);
      const nvcluster_Range cluster0 = clustering.clusterItemRanges[0];
      ASSERT_EQ(cluster0.count, 2);
      EXPECT_EQ(clustering.items[cluster0.offset], 0);
      EXPECT_EQ(clustering.items[cluster0.offset + 1], 2);
      const nvcluster_Range cluster1 = clustering.clusterItemRanges[1];
      ASSERT_EQ(cluster1.count, 2);
      EXPECT_EQ(clustering.items[cluster1.offset], 1);
      EXPECT_EQ(clustering.items[cluster1.offset + 1], 3);
    }
  }
};

TEST(Clusters, ZeroMaxVertex)
{
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}}, {{0, 100, 0}, {1, 101, 1}}, {{1, 0, 0}, {2, 1, 1}}, {{1, 100, 0}, {2, 101, 1}}};
  std::vector<vec3f> centroids(boundingBoxes.size());
  std::ranges::transform(boundingBoxes, centroids.begin(), [](nvcluster::AABB b) { return b.center(); });

  // Build clusters
  ClusterStorage clustering(
      nvcluster_Config{
          .minClusterSize        = 1,
          .maxClusterSize        = 4,
          .maxClusterVertices    = 0,  // Implies ~0u
          .costUnderfill         = 0.0f,
          .costOverlap           = 0.0f,
          .costUnderfillVertices = 0.0f,
          .itemVertexCount       = 4,
          .preSplitThreshold     = 0,
      },
      nvcluster_Input{
          .itemBoundingBoxes = reinterpret_cast<const nvcluster_AABB*>(boundingBoxes.data()),
          .itemCentroids     = reinterpret_cast<const nvcluster_Vec3f*>(centroids.data()),
          .itemCount         = uint32_t(boundingBoxes.size()),
          .itemVertices      = nullptr,
          .vertexCount       = 0,
      });

  // Main point of this test is not to throw an exception when building clusters
  // above
  ASSERT_EQ(clustering.clusterItemRanges.size(), 1);
}

TEST(Utils, MeshConnectionsQuad)
{
  GeometryMesh               mesh{{}, {{0, 1, 2}, {0, 2, 3}}, {{}, {}, {}, {}}};
  nvcluster::MeshConnections connections = makeMeshConnections(true, mesh);
  ASSERT_EQ(connections.connectionRanges.size(), 2);
  ASSERT_EQ(connections.connectionItems.size(), 2);
  ASSERT_EQ(connections.connectionVertexBits.size(), 2);
  ASSERT_EQ(connections.connectionRanges[0].count, 1);
  ASSERT_EQ(connections.connectionRanges[1].count, 1);
  ASSERT_EQ(connections.connectionItems[connections.connectionRanges[0].offset], 1);
  ASSERT_EQ(connections.connectionItems[connections.connectionRanges[1].offset], 0);
  EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[0].offset]), 2);
  EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[1].offset]), 2);
}

TEST(Utils, MeshConnectionsEdge)
{
  GeometryMesh mesh{{}, {{0, 1, 2}, {3, 4, 5}, {2, 6, 7}, {6, 7, 8}}, {{}, {}, {}, {}, {}, {}, {}, {}, {}}};
  nvcluster::MeshConnections connections = makeMeshConnections(true, mesh);
  ASSERT_EQ(connections.connectionRanges.size(), 4);
  ASSERT_EQ(connections.connectionItems.size(), 4);
  ASSERT_EQ(connections.connectionVertexBits.size(), 4);
  ASSERT_EQ(connections.connectionRanges[0].count, 1);
  ASSERT_EQ(connections.connectionRanges[1].count, 0);
  ASSERT_EQ(connections.connectionRanges[2].count, 2);
  ASSERT_EQ(connections.connectionRanges[3].count, 1);
  ASSERT_EQ(connections.connectionItems[connections.connectionRanges[0].offset], 2);
  ASSERT_EQ(connections.connectionItems[connections.connectionRanges[3].offset], 2);
  EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[0].offset]), 1);
  EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[3].offset]), 2);
  if(connections.connectionItems[connections.connectionRanges[2].offset + 0] == 0)
  {
    EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[2].offset + 0]), 1);
    EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[2].offset + 1]), 2);
  }
  else
  {
    EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[2].offset + 0]), 2);
    EXPECT_EQ(std::popcount(connections.connectionVertexBits[connections.connectionRanges[2].offset + 1]), 1);
  }
}

TEST(MaxVertex, Icosphere)
{
  for(int icosphereLevels = 0; icosphereLevels <= 3; ++icosphereLevels)
  {
    // Create test geometry
    GeometryMesh mesh = makeIcosphere(icosphereLevels);
    SCOPED_TRACE(mesh.name);
    std::vector<nvcluster::AABB> boundingBoxes(mesh.triangles.size());
    std::ranges::transform(mesh.triangles, boundingBoxes.begin(), [&](vec3u tri) { return aabb(tri, mesh.positions); });
    std::vector<vec3f> centroids(boundingBoxes.size());
    std::ranges::transform(boundingBoxes, centroids.begin(), [](nvcluster::AABB b) { return b.center(); });

    // Compute triangle connections with vertex bits
    nvcluster::MeshConnections connections = makeMeshConnections(true, mesh);

    // Iterate through a range of maxVertices
    std::vector<float> connectionWeights(connections.connectionItems.size(), 0.0f);
    for(uint32_t maxVertices = 3; maxVertices <= 16; ++maxVertices)
    {
      SCOPED_TRACE("Max vertices: " + std::to_string(maxVertices));

      // Build clusters
      nvcluster_Config config{
          .minClusterSize        = 1,
          .maxClusterSize        = 100,
          .maxClusterVertices    = maxVertices,
          .costUnderfill         = 0.0f,
          .costOverlap           = 0.0f,
          .costUnderfillVertices = 1.0f,
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
          maxVertices % 2 == 0 ? std::span<const float>(connectionWeights) : std::span<const float>(),  // exercise optional weights
          connections.connectionVertexBits,
      });

      // Compute vertex counts to verify the limit was not hit
      std::vector<uint32_t> clusterVertexCounts = countClusterVertices(clustering, mesh);
      std::vector<uint32_t> vertexCountCounts(maxVertices, 0);
      uint32_t              overHalfEmpty = 0;
      uint32_t              overHalfFull  = 0;
      for(uint32_t count : clusterVertexCounts)
      {
        EXPECT_LE(count, maxVertices);
        vertexCountCounts[count - 1]++;
        if(count < (maxVertices + 1) / 2)
          overHalfEmpty++;
        if(count > maxVertices / 2)
          overHalfFull++;
      }
      EXPECT_LE(overHalfEmpty, overHalfFull);
    }
  }
}

TEST(VertexLimit, ClustersFromVertices)
{
  // Test known values for perfect grids of triangles
  using namespace nvcluster;
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, 4.0f), 256.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, 9.0f), 64.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, 25.0f), 16.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, 81.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, 289.0f), 1.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, 4.0f), 64.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, 9.0f), 16.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, 25.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, 81.0f), 1.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(25.0f, 4.0f), 16.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(25.0f, 9.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(25.0f, 25.0f), 1.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(9.0f, 4.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(9.0f, 9.0f), 1.0f);

  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, sqrtf(289.0f), 4.0f), 256.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, sqrtf(289.0f), 9.0f), 64.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, sqrtf(289.0f), 25.0f), 16.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, sqrtf(289.0f), 81.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(289.0f, sqrtf(289.0f), 289.0f), 1.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, sqrtf(81.0f), 4.0f), 64.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, sqrtf(81.0f), 9.0f), 16.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, sqrtf(81.0f), 25.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(81.0f, sqrtf(81.0f), 81.0f), 1.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(25.0f, sqrtf(25.0f), 4.0f), 16.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(25.0f, sqrtf(25.0f), 9.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(25.0f, sqrtf(25.0f), 25.0f), 1.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(9.0f, sqrtf(9.0f), 4.0f), 4.0f);
  EXPECT_EQ(guessRequiredClustersForVertexLimit(9.0f, sqrtf(9.0f), 9.0f), 1.0f);
}

TEST(VertexLimit, VerticesFromClusters)
{
  // Inverse of RequiredClusters
  using namespace nvcluster;
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, 256.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, 64.0f), 9.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, 16.0f), 25.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, 4.0f), 81.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, 1.0f), 289.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, 64.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, 16.0f), 9.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, 4.0f), 25.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, 1.0f), 81.0f);
  EXPECT_EQ(guessVerticesAfterClustering(25.0f, 16.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(25.0f, 4.0f), 9.0f);
  EXPECT_EQ(guessVerticesAfterClustering(25.0f, 1.0f), 25.0f);
  EXPECT_EQ(guessVerticesAfterClustering(9.0f, 4.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(9.0f, 1.0f), 9.0f);

  EXPECT_EQ(guessVerticesAfterClustering(289.0f, sqrtf(289.0f), 256.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, sqrtf(289.0f), 64.0f), 9.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, sqrtf(289.0f), 16.0f), 25.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, sqrtf(289.0f), 4.0f), 81.0f);
  EXPECT_EQ(guessVerticesAfterClustering(289.0f, sqrtf(289.0f), 1.0f), 289.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, sqrtf(81.0f), 64.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, sqrtf(81.0f), 16.0f), 9.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, sqrtf(81.0f), 4.0f), 25.0f);
  EXPECT_EQ(guessVerticesAfterClustering(81.0f, sqrtf(81.0f), 1.0f), 81.0f);
  EXPECT_EQ(guessVerticesAfterClustering(25.0f, sqrtf(25.0f), 16.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(25.0f, sqrtf(25.0f), 4.0f), 9.0f);
  EXPECT_EQ(guessVerticesAfterClustering(25.0f, sqrtf(25.0f), 1.0f), 25.0f);
  EXPECT_EQ(guessVerticesAfterClustering(9.0f, sqrtf(9.0f), 4.0f), 4.0f);
  EXPECT_EQ(guessVerticesAfterClustering(9.0f, sqrtf(9.0f), 1.0f), 9.0f);
}

TEST(VertexLimit, Underfill)
{
  for(float n = 289.0f; n > 200.0f; n -= 1.0f)
  {
    const float targetVertices     = 81.0f;
    float       c                  = nvcluster::guessRequiredClustersForVertexLimit(n, targetVertices);
    float       clusterCount       = ceilf(c);
    float       verticesPerCluster = nvcluster::guessVerticesAfterClustering(n, clusterCount);
    float       availableVertices  = clusterCount * targetVertices;
    float       underfill          = availableVertices - verticesPerCluster * clusterCount;
    EXPECT_GE(underfill, 0.0f);
    EXPECT_LE(underfill, targetVertices);
    if(n == 289.0f)
    {
      EXPECT_EQ(underfill, 0.0f);
    }
    if(n > 222.0f)
    {
      EXPECT_NEAR(underfill, (289.0f - n) * 1.07f, 1.0f);
    }
    //printf("n: %f, c: %f, clusterCount: %f, verticesPerCluster: %f, availableVertices: %f, underfill: %f\n", n, c, clusterCount, verticesPerCluster, availableVertices, underfill);
  }
}

TEST(ParallelAlgorithms, ForEachIsParallel)
{
  constexpr size_t          N = 10;
  std::vector<int>          data(N, 0);
  std::set<std::thread::id> thread_ids;
  std::mutex                mtx;
  std::vector<int>          one({0});
  std::for_each(std::execution::seq, one.begin(), one.end(), [&](int&) {
    std::for_each(std::execution::par_unseq, data.begin(), data.end(), [&](int& x) {
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++x;
        std::lock_guard<std::mutex> lock(mtx);
        thread_ids.insert(std::this_thread::get_id());
      }
    });
  });

  // At least two unique thread IDs should be seen for parallel execution
  EXPECT_GE(thread_ids.size(), 2u);
}

TEST(ParallelAlgorithms, ExclusiveScanIsParallel)
{
  constexpr size_t          N = 10;
  std::vector<int>          data(N, 1);
  std::vector<int>          output(N, 0);
  std::set<std::thread::id> thread_ids;
  std::mutex                mtx;

  std::exclusive_scan(std::execution::par_unseq, data.begin(), data.end(), output.begin(), 0, [&](int a, int b) {
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::lock_guard<std::mutex> lock(mtx);
      thread_ids.insert(std::this_thread::get_id());
    }
    return a + b;
  });

  EXPECT_GE(thread_ids.size(), 2u);
}

TEST(ParallelAlgorithms, TransformExclusiveScanIsParallel)
{
  constexpr size_t          N = 10;
  std::vector<int>          data(N, 1);
  std::vector<int>          indirect(N, 1);
  std::vector<int>          output(N, 0);
  std::set<std::thread::id> thread_ids;
  std::mutex                mtx;
  std::iota(indirect.begin(), indirect.end(), 0);

  std::transform_exclusive_scan(std::execution::par_unseq, indirect.begin(), indirect.end(), output.begin(), 0,
                                std::plus<int>(), [&](int x) {
                                  {
                                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                                    std::lock_guard<std::mutex> lock(mtx);
                                    thread_ids.insert(std::this_thread::get_id());
                                  }
                                  return data[x];
                                });

  EXPECT_GE(thread_ids.size(), 2u);
}

TEST(ParallelAlgorithms, TransformReduceIsParallel)
{
  constexpr size_t          N = 10;
  std::vector<int>          data1(N, 1);
  std::vector<int>          data2(N, 2);
  std::set<std::thread::id> thread_ids;
  std::mutex                mtx;

  (void)std::transform_reduce(
      std::execution::par_unseq, data1.begin(), data1.end(), data2.begin(), 0,
      [&](int a, int b) {
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          std::lock_guard<std::mutex> lock(mtx);
          thread_ids.insert(std::this_thread::get_id());
        }
        return a + b;
      },
      [&](int a, int b) {
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          std::lock_guard<std::mutex> lock(mtx);
          thread_ids.insert(std::this_thread::get_id());
        }
        return a * b;
      });

  EXPECT_GE(thread_ids.size(), 2u);
}
