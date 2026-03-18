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
#pragma once

#include <clusters_cpp.hpp>
#include <connections.hpp>
#include <nvcluster/nvcluster.h>
#include <span>

namespace nvcluster {

inline MeshConnections makeMeshConnections(bool parallelize, const nvcluster_Config& inputConfig, const nvcluster_Input& input)
{
  return makeMeshConnections(parallelize, ItemVertices(input.itemVertices, input.itemCount, inputConfig.itemVertexCount),
                             input.vertexCount);
}

struct Input
{
  Input(const nvcluster_Config& inputConfig, const nvcluster_Input& input, const nvcluster_Segments& inputSegments)
      : Input(inputConfig,
              std::span(reinterpret_cast<const AABB*>(input.itemBoundingBoxes), input.itemCount),
              std::span(reinterpret_cast<const vec3f*>(input.itemCentroids), input.itemCount),
              std::span(reinterpret_cast<const Range*>(inputSegments.segmentItemRanges), inputSegments.segmentCount),
              maybeNull(reinterpret_cast<const Range*>(input.itemConnectionRanges), input.itemCount),
              maybeNull(input.connectionTargetItems, input.connectionCount),
              maybeNull(input.connectionWeights, input.connectionCount),
              maybeNull(input.connectionVertexBits, input.connectionCount))
  {
  }

  Input(const nvcluster_Config& inputConfig, const nvcluster_Input& input, const nvcluster_Segments& inputSegments, const MeshConnections& meshConnections)
      : Input(inputConfig,
              std::span(reinterpret_cast<const AABB*>(input.itemBoundingBoxes), input.itemCount),
              std::span(reinterpret_cast<const vec3f*>(input.itemCentroids), input.itemCount),
              std::span(reinterpret_cast<const Range*>(inputSegments.segmentItemRanges), inputSegments.segmentCount),
              meshConnections.connectionRanges,
              meshConnections.connectionItems,
              {},  // incompatible with auto-computed connections
              meshConnections.connectionVertexBits)
  {
  }

  Input(const nvcluster_Config&               config_,
        std::span<const AABB>                 boundingBoxes_,
        std::span<const vec3f>                centroids_,
        std::span<const Range>                segments_,
        std::span<const Range>                itemConnectionRanges_  = {},
        std::span<const uint32_t>             connectionTargetItems_ = {},
        std::span<const float>                connectionWeights_     = {},
        std::span<const nvcluster_VertexBits> connectionVertexBits_  = {})
      : config(config_)
      , boundingBoxes(boundingBoxes_)
      , centroids(centroids_)
      , segments(segments_)
      , itemConnectionRanges(itemConnectionRanges_)
      , connectionTargetItems(connectionTargetItems_)
      , connectionWeights(connectionWeights_)
      , connectionVertexBits(connectionVertexBits_)
  {
    // NOTE: validation is done by the C API and none here to avoid throwing
    // more exceptions than the standard library already does, e.g. bad_alloc
  }

  // Minimal spatial-only input
  const nvcluster_Config& config;
  std::span<const AABB>   boundingBoxes;
  std::span<const vec3f>  centroids;

  // Clusterize within each range of items
  std::span<const Range> segments;

  // Optional connections (may be empty)
  std::span<const Range>                itemConnectionRanges;
  std::span<const uint32_t>             connectionTargetItems;
  std::span<const float>                connectionWeights;
  std::span<const nvcluster_VertexBits> connectionVertexBits;

private:
  template <class T>
  std::span<const T> maybeNull(const T* ptr, uint32_t size)
  {
    return ptr ? std::span<const T>{ptr, size} : std::span<const T>{};
  }
};

struct OutputClusters
{
  OutputClusters(nvcluster_OutputClusters& output, nvcluster_Range* outputSegments, uint32_t outputSegmentCount)
      : clusterItemRanges(reinterpret_cast<Range*>(output.clusterItemRanges), output.clusterCount)
      , items(reinterpret_cast<uint32_t*>(output.items), output.itemCount)
      , segments(reinterpret_cast<Range*>(outputSegments), outputSegmentCount)
      , clusterCount(output.clusterCount)
      , itemCount(output.itemCount)
  {
  }
  std::span<Range>    clusterItemRanges;
  std::span<uint32_t> items;
  std::span<Range>    segments;
  uint32_t&           clusterCount;  // output count reference
  uint32_t&           itemCount;     // output count reference
};

NVCLUSTER_API [[nodiscard]] nvcluster_Result clusterize(bool parallelize, const Input& input, const OutputClusters& clusters);

}  // namespace nvcluster
