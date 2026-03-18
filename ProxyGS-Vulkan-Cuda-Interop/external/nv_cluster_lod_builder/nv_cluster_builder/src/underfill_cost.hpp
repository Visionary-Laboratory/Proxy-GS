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
/// @file Heuristics to compute item/triangle and vertex "underfill costs",
/// which encourage the clusterizer to form bigger clusters, still within the
/// maximums. Separated from the source file for unit testing.
#pragma once

#include <clusterizer.hpp>
#include <clusters_cpp.hpp>

namespace nvcluster {

// Switch to compute a connectedness metric that indicates how many vertices
// will be duplicated after cutting the node, rather than assume the square root
// of vertices will be cut. Takes a bit longer but can help with long skinny
// geometry like triangle strips.
static constexpr bool COMPUTE_AVERAGE_CUT_VERTICES = true;

struct Underfill
{
  uint32_t underfillCount = 0;

  // underfillCount is unique vertices if true, otherwise count is items
  bool vertexLimited = false;
};

inline float guessRequiredClustersForVertexLimit(float currentVertices, float targetVertices)
{
  // s=\frac{nv-2\sqrt{n}v+2\sqrt{\left(\sqrt{n}-1\right)^{4}v}+n-2\sqrt{n}+v+1}{\left(v-1\right)^{2}}
  float sqrtN         = sqrtf(currentVertices);
  float sqrtNMinus1_4 = powf(sqrtN - 1.0f, 4.0f);
  float numerator     = currentVertices * targetVertices            //
                    - 2.0f * sqrtN * targetVertices                 //
                    + 2.0f * sqrtf(sqrtNMinus1_4 * targetVertices)  //
                    + currentVertices                               //
                    - 2.0f * sqrtN                                  //
                    + targetVertices                                //
                    + 1.0f;
  float denominator = (targetVertices - 1.0f) * (targetVertices - 1.0f);
  return numerator / denominator;
}

inline float guessRequiredClustersForVertexLimit(float currentVertices, float averageCutVertices, float targetVertices)
{
  // (2 sqrt((a - 1)^2 (a^2 - 2 a v + n (v - 1) + v)) + 2 a^2 - 2 a (v + 1) + n (v - 1) + v + 1)/(v - 1)^2
  float a  = averageCutVertices;
  float v  = targetVertices;
  float n  = currentVertices;
  float t1 = a * a - 2.0f * a * v + n * (v - 1.0f) + v;
  if(t1 < 0.0f)  // candidate split with less than the average cut vertices (e.g. first or last few)
    return 1.0f;
  float t2 = 2.0f * (a - 1.0f) * sqrtf(t1) + 2.0f * a * a - 2.0f * a * (v + 1.0f) + n * (v - 1.0f) + v + 1.0f;
  return t2 / ((v - 1.0f) * (v - 1.0f));
}

// Inverse of guessVertexLimitRequiredClusters()
inline float guessVerticesAfterClustering(float currentVertices, float clusters)
{
  // v\left(n,s\right)=\frac{\left(\sqrt{n}+\sqrt{s}-1\right)^{2}}{s}
  float t = sqrtf(currentVertices) + sqrtf(clusters) - 1.0f;
  return (t * t) / clusters;
}

// Inverse of guessVertexLimitRequiredClusters()
inline float guessVerticesAfterClustering(float currentVertices, float averageCutVertices, float clusters)
{
  // v(n, s) = (2 (a - 1) sqrt(s) - 2 a + n + s + 1)/s
  float t = 2.0f * (averageCutVertices - 1.0f) * sqrtf(clusters) - 2.0f * averageCutVertices + currentVertices + clusters + 1.0f;
  return t / clusters;
}

// Returns the number of items remaining to fill the last bucket
inline uint32_t underfillCount(uint32_t bucketSize, uint32_t itemCount)
{
  return div_ceil(itemCount, bucketSize) * bucketSize - itemCount;
}

// Computes the expected number of vertices less than the maximum in the
// remaining cluster. This is entirely modelled off connections from shared
// vertices between a rectangular grid of triangles.
// TODO: remove AABB
inline Underfill generalUnderfillCount(const Input& input, uint32_t itemCount, uint32_t vertexCount, float averageCutVertices)
{
  float requiredClustersItems = float(itemCount) / float(input.config.maxClusterSize);
  float requiredClustersVertices =
      COMPUTE_AVERAGE_CUT_VERTICES ?
          guessRequiredClustersForVertexLimit(float(vertexCount), averageCutVertices, float(input.config.maxClusterVertices)) :
          guessRequiredClustersForVertexLimit(float(vertexCount), float(input.config.maxClusterVertices));

  if(requiredClustersItems > requiredClustersVertices)
  {
    // Item limited
    return {underfillCount(input.config.maxClusterSize, itemCount), false};
  }
  else
  {
    // Vertex limited
    float clusterCount       = ceilf(requiredClustersVertices - 1e-6f);
    float verticesPerCluster = COMPUTE_AVERAGE_CUT_VERTICES ?
                                   guessVerticesAfterClustering(float(vertexCount), averageCutVertices, clusterCount) :
                                   guessVerticesAfterClustering(float(vertexCount), clusterCount);
    float availableVertices  = clusterCount * float(input.config.maxClusterVertices);
    float underfill          = availableVertices - verticesPerCluster * clusterCount + 0.5f;
    assert(verticesPerCluster > 1.0f);
    assert(underfill >= 0.0f);
    return {uint32_t(underfill), true};
  }
}

}  // namespace nvcluster
