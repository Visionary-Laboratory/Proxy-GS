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
#include <atomic>
#include <bit>
#include <clusterizer.hpp>
#include <execution>
#include <nvcluster/nvcluster.h>
#include <parallel.hpp>
#include <ranges>
#include <span>
#include <stddef.h>
#include <thread>
#include <underfill_cost.hpp>
#include <vector>

#define PRINT_PERF 0

// Workaround for libc++ std::execution
#include <parallel_execution_libcxx.hpp>

// Scoped profiler for quick and coarse results
// https://stackoverflow.com/questions/31391914/timing-in-an-elegant-way-in-c
#if PRINT_PERF
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#endif

namespace nvcluster {

class Stopwatch
{
#if PRINT_PERF
public:
  Stopwatch(std::string name)
      : m_name(std::move(name))
      , m_beg(std::chrono::high_resolution_clock::now())
  {
  }
  ~Stopwatch()
  {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " ms\n";
  }

private:
  std::string                                                 m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
#else
public:
  template <class T>
  Stopwatch(T&&)
  {
  }
#endif
};

// "Functor" to sort item indices based on their centroid coordinates along each
// axis
template <uint32_t Axis>
struct CentroidCompare
{
  CentroidCompare(std::span<const vec3f> centroids)
      : m_centroids(centroids)
  {
  }
  inline bool operator()(const uint32_t& itemA, const uint32_t& itemB) const
  {
    // For architectural meshes, centroids may form a grid, where many
    // coordinates would be equal. In these cases we still want ordering on the
    // remaining axes so a "composite key" is used.
    constexpr int A0 = Axis;
    constexpr int A1 = (Axis + 1) % 3;
    constexpr int A2 = (Axis + 2) % 3;
    const vec3f&  c0 = m_centroids[itemA];
    const vec3f&  c1 = m_centroids[itemB];
#if 1
    // Compares lexicographically
    return std::tuple(c0[A0], c0[A1], c0[A2]) < std::tuple(c1[A0], c1[A1], c1[A2]);
#else
    // Equivalent to
    return (c0[A0] < c1[A0]) || (c0[A0] == c1[A0] && c0[A1] < c1[A1])
           || (c0[A0] == c1[A0] && c0[A1] == c1[A1] && c0[A2] < c1[A2]);
#endif
  }
  std::span<const vec3f> m_centroids;
};

// Classic surface area heuristic cost
inline float sahCost(const AABB& aabb, const uint32_t& itemCount)
{
  return aabb.half_area() * float(itemCount);
}

// Candidate split position within a node
struct Split
{
  uint32_t    axis          = std::numeric_limits<uint32_t>::max();
  uint32_t    position      = std::numeric_limits<uint32_t>::max();  // index of first item in the right node
  float       cost          = std::numeric_limits<float>::max();
  bool        leftComplete  = false;  // the left child is a valid cluster
  bool        rightComplete = false;  // the right child is a valid cluster
  inline bool valid() { return axis != std::numeric_limits<uint32_t>::max(); }
  bool        operator<(const Split& other) const { return cost < other.cost; }
};

// An explicit min function to pass to std::reduce
inline Split minSplitCost(Split a, Split b)
{
  return std::min(a, b);
}

struct NodeRange : Range
{
  // Track which segment nodes belong to
  uint32_t segment;
};

// Temporary storage and parameter pack for functions that split nodes.
struct SplitNodeTemp
{
  // The subrange of global nodeItems this structure of sub-arrays represents
  Range range;

  // Item indices, sorted by centroids on each axis and partitioned when
  // splitting into nodes, per axis.
  std::array<std::span<uint32_t>, 3> nodeItems;

  // Identification of the side on which each item lies when splitting a node
  // (left = 1, right = 0)
  // WARNING: these are global (not node-relative) and subrange() is a
  // passthrough. Threads processing different nodes perform scattered reads and
  // writes concurrently.
  std::span<uint8_t> partitionSides;

  // Left-cumulative bounding boxes of node items
  std::span<AABB> leftAabbs;

  // Right-cumulative bounding boxes of node items
  std::span<AABB> rightAabbs;

  // Delta (from left to right) of cut edge weights within the node at each position
  std::span<float> deltaWeights;

  // Sum of edge weights cut at each position within the node
  std::span<float> splitWeights;

  // Running total unique vertices from the left and from the right. Als
  // temporary memory as the prefix sum to compute them is not in-place.
  std::span<uint32_t> leftUniqueVertices;
  std::span<uint32_t> rightUniqueVertices;
  std::span<uint32_t> tmpUniqueVertices;

  // A duplicate of connectionTargetItems with indices into the global
  // nodeItems rather than item indices.
  // WARNING: these are global (not node-relative) and subrange() is a passthrough
  std::array<std::span<const uint32_t>, 3> connectionItemsInNodes;

  // Returns the subrange for each array in this structure of arrays, except for
  // partitionSides and connectionItemsInNodes
  template <bool HasConnectionWeights, bool HasVertexLimit>
  SplitNodeTemp subspan(Range subrange) const
  {
    return {
        .range = {range.offset + subrange.offset, subrange.count},
        .nodeItems =
            {
                nodeItems[0].subspan(subrange.offset, subrange.count),
                nodeItems[1].subspan(subrange.offset, subrange.count),
                nodeItems[2].subspan(subrange.offset, subrange.count),
            },
        .partitionSides = partitionSides /* NOTE: global, no subspan() */,
        .leftAabbs      = leftAabbs.subspan(subrange.offset, subrange.count),
        .rightAabbs     = rightAabbs.subspan(subrange.offset, subrange.count),
        .deltaWeights = !HasConnectionWeights ? std::span<float>{} : deltaWeights.subspan(subrange.offset, subrange.count),
        .splitWeights = !HasConnectionWeights ? std::span<float>{} : splitWeights.subspan(subrange.offset, subrange.count),
        .leftUniqueVertices     = !HasVertexLimit || leftUniqueVertices.empty() ?
                                      std::span<uint32_t>{} :
                                      leftUniqueVertices.subspan(subrange.offset, subrange.count),
        .rightUniqueVertices    = !HasVertexLimit || rightUniqueVertices.empty() ?
                                      std::span<uint32_t>{} :
                                      rightUniqueVertices.subspan(subrange.offset, subrange.count),
        .tmpUniqueVertices      = !HasVertexLimit || tmpUniqueVertices.empty() ?
                                      std::span<uint32_t>{} :
                                      tmpUniqueVertices.subspan(subrange.offset, subrange.count),
        .connectionItemsInNodes = !(HasConnectionWeights || HasVertexLimit) ?
                                      std::array<std::span<const uint32_t>, 3>{} :
                                      std::array<std::span<const uint32_t>, 3>{
                                          connectionItemsInNodes[0] /* NOTE: global, no subspan() */,
                                          connectionItemsInNodes[1] /* NOTE: global, no subspan() */,
                                          connectionItemsInNodes[2] /* NOTE: global, no subspan() */,
                                      }
    };
  }
};

/**
 * @brief Compute an aggregate cost of splitting a node just before splitIndex,
 * balancing multiple factors. The indexing convention is that the split happens
 * before the index, so index 0 is not considered.
 * @tparam Parallelize optionally parallelize internally or externally
 * @tparam HasConnectionWeights compile-time switch to process and add cut costs
 * to split candidates
 * @tparam HasVertexLimit compile-time switch to compute vertex counts for split
 * candidates
 * @tparam AlignBoth enforce Input::Config::minClusterSize and
 * Input::Config::maxClusterSize from both ends of the node or just one
 * @param input global input structure originally passed to create()
 * @param node node attribute subranges and scratch data to compute split costs
 * @param nodeAabb bounding box of all items in the node
 * @param leftAabb bounding box of items left of the split
 * @param rightAabb bounding box of items right of the split
 * @param splitIndex index of the first item to the right of the split
 */
template <bool Parallelize, bool HasConnectionWeights, bool HasVertexLimit, bool AlignBoth>
inline float splitCost(const Input&         input,
                       const SplitNodeTemp& node,
                       const AABB&          nodeAabb,
                       const AABB&          leftAabb,
                       const AABB&          rightAabb,
                       float                averageCutVertices,
                       uint32_t             splitIndex)
{
  // Make leaves adhere to the min and max leaf size rule
  uint32_t acceptableRemainder = input.config.maxClusterSize - input.config.minClusterSize;
  uint32_t leftSize            = splitIndex;  // Rename to be more definite about what is included on either side
  uint32_t rightSize           = node.range.count - leftSize;
  bool leftAlign = leftSize % input.config.minClusterSize <= (leftSize / input.config.minClusterSize) * acceptableRemainder;
  bool rightAlign = rightSize % input.config.minClusterSize <= (rightSize / input.config.minClusterSize) * acceptableRemainder;

  float cost = std::numeric_limits<float>::max();

  // Ignore input.config.minClusterSize if HasVertexLimit
  bool aligned = AlignBoth ? leftAlign && rightAlign : leftAlign || rightAlign;
  if(HasVertexLimit || aligned)
  {
    float leftCost  = sahCost(leftAabb, leftSize);
    float rightCost = sahCost(rightAabb, rightSize);
    cost            = leftCost + rightCost;

#if 1
    // Increase cost for under-filled nodes (e.g. we want more clusters with
    // maxClusterSize than minClusterSize). Don't apply the cost if we're
    // already under the limit, which can happen when we're vertex limited
    // instead.
    if(HasVertexLimit && input.config.costUnderfillVertices != 0.0f)
    {
      // General underfill count which may change depending on whether each node
      // is item or vertex limited
      // NOTE: averageCutVertices is ignored if COMPUTE_AVERAGE_CUT_VERTICES is false
      Underfill leftUnderfill = generalUnderfillCount(input, leftSize, node.leftUniqueVertices[splitIndex], averageCutVertices);
      Underfill rightUnderfill = generalUnderfillCount(input, rightSize, node.rightUniqueVertices[splitIndex], averageCutVertices);
      cost += sahCost(nodeAabb, leftUnderfill.underfillCount)
              * (leftUnderfill.vertexLimited ? input.config.costUnderfillVertices : input.config.costUnderfill);
      cost += sahCost(nodeAabb, rightUnderfill.underfillCount)
              * (rightUnderfill.vertexLimited ? input.config.costUnderfillVertices : input.config.costUnderfill);
    }
    else
    {
      // Item underfill count. Simple compared to including unique vertex counts
      uint32_t leftItemUnderfill  = underfillCount(input.config.maxClusterSize, leftSize);
      uint32_t rightItemUnderfill = underfillCount(input.config.maxClusterSize, rightSize);
      cost += sahCost(nodeAabb, leftItemUnderfill + rightItemUnderfill) * input.config.costUnderfill;
    }
#endif

#if 1
    // Increase cost for lef/right bounding box overlap
    AABB intersection = leftAabb.intersect(rightAabb);
    cost += sahCost(intersection, node.range.count) * input.config.costOverlap;
#endif

#if 1
    // Increase cost for weighted edge connections outside the
    // node and crossing the split plane.
    if constexpr(HasConnectionWeights)
    {
      // Convert "ratio cut" values to SAH relative costs. Costs are all based on
      // SAH, a measure of trace cost for the bounding surface area. Shoehorning a
      // min-cut cost into the mix is problematic because the same metric needs to
      // work at various scales of node sizes and item counts. SAH scales with the
      // the number of items. The cut cost likely scales with the square root of the
      // number of items, assuming they form surfaces in the 3D space, but ratio cut
      // also normalizes by the item count. This attempts to undo that. A good way
      // to verify is to plot maximum costs against varying node sizes.
      float normalizeCutWeights = float(node.range.count * node.range.count);

      // "Ratio cut" - divide by the number of items in each
      // node to avoid degenerate cuts of just the first/last
      // items.
      float cutCost      = node.splitWeights[splitIndex];
      float ratioCutCost = cutCost / float(leftSize) + cutCost / float(rightSize);
      cost += ratioCutCost * normalizeCutWeights;
      assert(cutCost < 1e12);  // smoke check for weight overflow
    }
#endif
  }
  return cost;
}

struct DeltaSplitConnectionAttribs
{
  // Difference between the sum connection weights cut before and after the
  // item. This is computed by summing positive weights for connections within
  // the node to the right of the item and negative for connections to the left.
  // Think rising and falling edge weights. Later, the prefix sum of the deltas
  // will then provide the *cut cost* in O(1) for any split position.
  float leftCutWeight = 0.0f;

  // Number of unique vertices the given item adds if included in either side of
  // the split: left or right. E.g. leftUniqueVertices is the number of new
  // unique vertices the given item adds when added to all items before it in
  // the node.
  uint32_t leftUniqueVertices  = 0;
  uint32_t rightUniqueVertices = 0;
};

/**
 * @brief Combined iteration over item connections, producing multiple values
 * depending on compile-time options. See DeltaSplitConnectionAttribs.
 * @tparam HasConnectionWeights compile-time switch to process and add cut costs
 * to split candidates
 * @tparam HasVertexLimit compile-time switch to compute vertex counts for split
 * candidates
 * @param input global input structure originally passed to create()
 * @param tmp scratch data
 * @param node range of items in the node being split, only needed for its size
 * @param leftAabb bounding box of items left of the split
 * @param rightAabb bounding box of items right of the split
 * @param splitIndex index of the first item to the right of the split
 */
template <bool HasConnectionWeights, bool HasVertexLimit, uint32_t Axis>
DeltaSplitConnectionAttribs deltaSplitConnectionAttribs(const Input& input, const SplitNodeTemp& node, uint32_t nodeItemIndex, uint32_t itemIndex)
{
  DeltaSplitConnectionAttribs result{};
  nvcluster_VertexBits        vertexBitsLeft  = 0u;
  nvcluster_VertexBits        vertexBitsRight = 0u;

  // Make subspans of just the item's connections
  const Range&              itemConnectionsRange = input.itemConnectionRanges[itemIndex];
  std::span<const uint32_t> itemConnectionsInNodes =
      node.connectionItemsInNodes[Axis].subspan(itemConnectionsRange.offset, itemConnectionsRange.count);
  std::span<const float>                itemConnectionWeights;
  std::span<const nvcluster_VertexBits> itemConnectionVertexBits;
  if constexpr(HasConnectionWeights)
    itemConnectionWeights = input.connectionWeights.subspan(itemConnectionsRange.offset, itemConnectionsRange.count);
  if constexpr(HasVertexLimit)
    itemConnectionVertexBits = input.connectionVertexBits.subspan(itemConnectionsRange.offset, itemConnectionsRange.count);

  // Iterate over the connections and accumulate the weights: positive for
  // connections beginning at splitIndex (i.e. to the right) and negative for
  // ending connections (i.e. to the left)
  for(size_t i = 0; i < itemConnectionsInNodes.size(); ++i)
  {
    // Find the index of the connected item within the connections of the node
    // to split. This unsigned value wraps if less than node.offset.
    uint32_t connectedItemIndexInNode = itemConnectionsInNodes[i] - node.range.offset;

    // Skip connections to items not in stored in the connections for the split node
    if(connectedItemIndexInNode >= node.range.count)
      continue;

    // Add the weights for new connections added by this node, going left to
    // right, and subtract the weights for ending connections. The structure
    // explicitly stores bidirectional connections so the other ends of these
    // will be added at their respective indices during iteration.
    if constexpr(HasConnectionWeights)
    {
      result.leftCutWeight += (nodeItemIndex < connectedItemIndexInNode) ? itemConnectionWeights[i] : -itemConnectionWeights[i];
    }

    // Add the vertex bits connecting items to other items within the node to
    // the left or right.
    if constexpr(HasVertexLimit)
    {
      if(connectedItemIndexInNode > nodeItemIndex)
      {
        vertexBitsRight |= itemConnectionVertexBits[i];
      }
      else
      {
        assert(connectedItemIndexInNode < nodeItemIndex);  // check there are no self connections
        vertexBitsLeft |= itemConnectionVertexBits[i];
      }
    }
  }

  // Count the unique vertex bits this item has to the left and right.
  if constexpr(HasVertexLimit)
  {
    result.leftUniqueVertices  = input.config.itemVertexCount - uint32_t(std::popcount(vertexBitsLeft));
    result.rightUniqueVertices = input.config.itemVertexCount - uint32_t(std::popcount(vertexBitsRight));
  }
  return result;
}

// Standalone check to see if the number of unique vertices exceeds the user
// defined limit. Use only in case the input is a single valid cluster.
template <uint32_t Axis>
inline bool vertexCountOverflows(const Input& input, const SplitNodeTemp& node)
{
  // Early out if even the maximum possible unique vertices must be less than
  // the limit
  if(node.range.count * input.config.itemVertexCount <= input.config.maxClusterVertices)
  {
    return false;
  }

  uint32_t runningTotalUniqueVertices = 0;
  for(uint32_t itemIndex = 0; itemIndex < node.range.count; ++itemIndex)
  {
    runningTotalUniqueVertices += deltaSplitConnectionAttribs<false /* don't need weights */, true /* has vertex limit */, Axis>(
                                      input, node, itemIndex, node.nodeItems[Axis][itemIndex])
                                      .leftUniqueVertices;
    if(runningTotalUniqueVertices > input.config.maxClusterVertices)
      return true;
  }
  return false;
}

inline bool vertexCountOverflows(uint32_t axis, const Input& input, const SplitNodeTemp& node)
{
  if(axis == 0)
  {
    return vertexCountOverflows<0>(input, node);
  }
  else if(axis == 1)
  {
    return vertexCountOverflows<1>(input, node);
  }
  else
  {
    assert(axis == 2);
    return vertexCountOverflows<2>(input, node);
  }
}

/**
 * @brief Returns the minimum cost split position for one axis. Takes some
 * temporary arrays as arguments to reuse allocations between passes.
 * @tparam Parallelize optionally parallelize internally or externally
 * @tparam HasConnectionWeights compile-time switch to process and add cut costs
 * to split candidates
 * @tparam HasVertexLimit compile-time switch to compute vertex counts for split
 * candidates
 * @tparam Axis consider splits along (x, y, z), (0, 1, 2)
 * @tparam AlignBoth enforce Input::Config::minClusterSize and
 * Input::Config::maxClusterSize from both ends of the node or just one
 * @param input global input structure originally passed to create()
 * @param node node attribute subranges and scratch data to compute split costs
 * @param split input and output current best split position
 *
 * Note: std algorithm and structures of arrays are used frequently due to the
 * convenience of std execution to parallelize steps.
 */
template <bool Parallelize, bool HasConnectionWeights, bool HasVertexLimit, uint32_t Axis, bool AlignBoth>
void findBestSplit(const Input& input, const SplitNodeTemp& node, Split& split)
{
  float averageCutVertices = 0.0f;

  // Populate the left to right bounding boxes, growing with each item.
  // std::transform_exclusive_scan is a prefix sum, e.g. {1, 1, 1} becomes {0,
  // 1, 2} but here the plus operator unions bounding boxes. The AABB '+' plus
  // operator unions the bounds.
  std::transform_exclusive_scan(exec<Parallelize>, node.nodeItems[Axis].begin(), node.nodeItems[Axis].end(),
                                node.leftAabbs.begin(), AABB::empty(), std::plus<AABB>(),
                                [&bboxes = input.boundingBoxes](const uint32_t& i) { return bboxes[i]; });

  // Populate the right to left bounding boxes, growing with each item.
  // std::transform_inclusive_scan is a prefix post-sum, e.g. {1, 1, 1} becomes
  // {1, 2, 3}. Note the use of rbegin()/rend() to reverse iterate. The AABB '+'
  // plus operator unions the bounds.
  std::transform_inclusive_scan(exec<Parallelize>, node.nodeItems[Axis].rbegin(), node.nodeItems[Axis].rend(),
                                node.rightAabbs.rbegin(), std::plus<AABB>(),
                                [&bboxes = input.boundingBoxes](const uint32_t& i) { return bboxes[i]; });

  // Build cumulative adjacency weights if adjacency was provided
  bool hasVertexUnderfillCost = HasVertexLimit && input.config.costUnderfillVertices != 0.0f;
  if(HasConnectionWeights || hasVertexUnderfillCost)
  {
    // Compute DeltaSplitConnectionAttribs for each item in the node
    std::for_each(exec<Parallelize>, node.nodeItems[Axis].begin(), node.nodeItems[Axis].end(), [&](uint32_t& itemIndex) {
      // DANGER: pointer arithmetic to get index in the node
      // being split :(
      uint32_t nodeItemIndex = uint32_t(&itemIndex - &node.nodeItems[Axis][0]);

      // Iterates over item connections
      DeltaSplitConnectionAttribs attribs =
          deltaSplitConnectionAttribs<HasConnectionWeights, HasVertexLimit, Axis>(input, node, nodeItemIndex, itemIndex);

      // Record the difference in total cut connecting weights if
      // the item were included in the left node vs not.
      if constexpr(HasConnectionWeights)
      {
        node.deltaWeights[nodeItemIndex] = attribs.leftCutWeight;
      }

      // Record new unique vertex counts
      // NOTE: Looks like a typo, but we write (left, right) to
      // (tmp, left), using them as temporary arrays since the
      // scans below are not done in-place.
      if constexpr(HasVertexLimit)
      {
        node.tmpUniqueVertices[nodeItemIndex]  = attribs.leftUniqueVertices;
        node.leftUniqueVertices[nodeItemIndex] = attribs.rightUniqueVertices;
      }
    });

    if constexpr(HasConnectionWeights)
    {
      // Prefix sum scan the delta weights to find the total cut cost at each
      // position
      std::exclusive_scan(exec<Parallelize>, node.deltaWeights.begin(), node.deltaWeights.end(), node.splitWeights.begin(), 0.0f);
    }

    // Prefix sum scan the unique vertex count deltas to find the running total
    // unique vertices from either end of the node
    if constexpr(HasVertexLimit)
    {
      std::inclusive_scan(exec<Parallelize>, node.leftUniqueVertices.rbegin(), node.leftUniqueVertices.rend(),
                          node.rightUniqueVertices.rbegin());
      std::exclusive_scan(exec<Parallelize>, node.tmpUniqueVertices.begin(), node.tmpUniqueVertices.end(),
                          node.leftUniqueVertices.begin(), 0u);
      assert(node.rightUniqueVertices[0] == node.leftUniqueVertices.back() + node.tmpUniqueVertices.back());  // check for asymmetry in connection vertex bits

      // Compute the average number of vertices that would need to be duplicated
      // if splitting the mesh at each candidate
      if constexpr(COMPUTE_AVERAGE_CUT_VERTICES)
      {
#if 0
        // Ignore cuts towards the ends of the node as they may be degenerate
        // and not representative of typical cuts.
        ptrdiff_t ignoreEnds = (node.leftUniqueVertices.size() - 1u) / 8u;
#else
        ptrdiff_t ignoreEnds = 0u;
#endif
        uint32_t nodeUniqueVertexCount = node.rightUniqueVertices[0];
        averageCutVertices =
            std::transform_reduce(exec<Parallelize>, node.leftUniqueVertices.begin() + 1 + ignoreEnds,
                                  node.leftUniqueVertices.end() - ignoreEnds,
                                  node.rightUniqueVertices.begin() + 1 + ignoreEnds, 0.0f, std::plus<float>(),
                                  [nodeUniqueVertexCount](uint32_t leftVertexCount, uint32_t rightVertexCount) {
                                    return float(leftVertexCount + rightVertexCount - nodeUniqueVertexCount);
                                  })
            / float(node.range.count - 1u - 2u * ignoreEnds);
      }
    }
  }

  // Replace the best split by computing costs at each candidate and reducing.
  // Inputs are the left and right bounding box, skipping index 0 (meaning the
  // left child would be empty).
  // std::transform_reduce combines pairs of results until there is one
  // remaining. It is initialized with the current best split and min_split() is
  // given to keep the split candidate with minimum cost, possibly with parallel
  // execution.
  split = std::transform_reduce(
      exec<Parallelize>,                                 // possibly parallel
      node.leftAabbs.begin() + 1, node.leftAabbs.end(),  // input left bounding box
      node.rightAabbs.begin() + 1,                       // input right bounding box
      split,                                             // current best split
      minSplitCost,                                      // reduce by taking the minimum cost
      [&](const AABB& leftAabb, const AABB& rightAabb) -> Split {
        uint32_t splitPosition = (uint32_t)(&leftAabb - &node.leftAabbs.front());  // DANGER: pointer arithmetic to get index :(
        const AABB& nodeAabb = node.rightAabbs[0];  // Last rightAabb from the right contains the entire node AABB
        float       cost     = splitCost<Parallelize, HasConnectionWeights, HasVertexLimit, AlignBoth>(
            input, node, nodeAabb, leftAabb, rightAabb, averageCutVertices, splitPosition);
        return Split{.axis = Axis, .position = splitPosition, .cost = cost, .leftComplete = false, .rightComplete = false};
      });

  //printf("chose axis %u pos %u\n", split.axis, split.position);
  if(split.axis == Axis)
  {
    split.leftComplete  = (split.position <= input.config.maxClusterSize);
    split.rightComplete = (node.range.count - split.position <= input.config.maxClusterSize);

    // Mark children as incomplete if over the vertex limit. If already
    // computing unique vertex counts for every split candidate, we can check if
    // the left/right nodes are under the limit here. Otherwise an explicit
    // check is done for the final best split in splitNode().
    if constexpr(HasVertexLimit)
    {
      if(hasVertexUnderfillCost && split.leftComplete && node.leftUniqueVertices[split.position] > input.config.maxClusterVertices)
      {
        assert(split.position > 1);  // single item nodes must always be complete
        split.leftComplete = false;
      }
      if(hasVertexUnderfillCost && split.rightComplete && node.rightUniqueVertices[split.position] > input.config.maxClusterVertices)
      {
        assert(node.range.count - split.position > 1);  // single item nodes must always be complete
        split.rightComplete = false;
      }
    }
  }
}

// The input connections are a list of other item indices connected to each
// item. The algorithm needs the indices of those items in the global node items
// list. This function computes the indices with a scatter/gather operation.
template <bool Parallelize>
void buildConnectionsInNodes(const Input& input,  // Input items and connections
                             std::span<const uint32_t> nodeItems,  // List of item indices along an axis, sorted by centroid within each node range
                             std::span<uint32_t> connectionItemsInNodes,  // Output list of indices of connected items in 'nodeItems'
                             std::vector<uint32_t>& itemIndexToNodesIndex)  // Temporary storage for the scatter/gather operation
{
  // Scatter the item indices in nodes to the itemIndexToNodesIndex array
  parallel_batches<Parallelize, 4096>(nodeItems.size(),
                                      [&](size_t i) { itemIndexToNodesIndex[nodeItems[i]] = uint32_t(i); });

  // Gather the connection node indices to the connectionItemsInNodes array
  parallel_batches<Parallelize, 4096>(input.connectionTargetItems.size(), [&](size_t i) {
    connectionItemsInNodes[i] = itemIndexToNodesIndex[input.connectionTargetItems[i]];
  });
}

// Splits the per-axis sorted item lists at the chosen split position along the given axis,
// so that the items on the left side of the split are moved to the front of each list, while preserving the item ordering within each partition.
template <bool Parallelize>
inline void partitionAtSplit(const std::array<std::span<uint32_t>, 3>& nodeItems, uint32_t axis, uint32_t splitPosition, std::span<uint8_t> partitionSides)
{

  // Mark each item as being on the left or right side of the split
  // This is trivially done by traversing the list of items sorted along the chosen axis, and marking their side depending
  // on whether their index in the list is below or above the split position.
  // This approach uses more memory than checking the centroid but is faster.
  auto leftItems  = nodeItems[axis].subspan(0, splitPosition);
  auto rightItems = nodeItems[axis].subspan(splitPosition);
  std::for_each(exec<Parallelize>, leftItems.begin(), leftItems.end(),
                [&partitionSides](uint32_t i) { partitionSides[i] = 1u; });
  std::for_each(exec<Parallelize>, rightItems.begin(), rightItems.end(),
                [&partitionSides](uint32_t i) { partitionSides[i] = 0u; });

  if(axis != 0)
  {
    std::stable_partition(exec<Parallelize>, nodeItems[0].begin(), nodeItems[0].end(),
                          [&partitionSides](const uint32_t& i) { return partitionSides[i]; });
  }
  if(axis != 1)
  {
    std::stable_partition(exec<Parallelize>, nodeItems[1].begin(), nodeItems[1].end(),
                          [&partitionSides](const uint32_t& i) { return partitionSides[i]; });
  }
  if(axis != 2)
  {
    std::stable_partition(exec<Parallelize>, nodeItems[2].begin(), nodeItems[2].end(),
                          [&partitionSides](const uint32_t& i) { return partitionSides[i]; });
  }
}

// Take a node defined by its sorted lists of item indices and recursively splits it along its longest axis until
// the number of items in each node is less than or equal to maxItemsPerNode.
template <bool Parallelize>
static void splitAtMedianUntil(const Input& spatial,  // Original definition of the input spatial items
                               std::vector<uint8_t>& partitionSides,  // Partition identifier (left = 1, right = 0) for each item, used to partition the sorted item lists
                               const std::array<std::span<uint32_t>, 3>& nodeItems,  // Sorted item indices along each axis for the current node
                               size_t maxItemsPerNode,  // Maximum number of items allowed per node, used to stop the recursion
                               uint32_t nodeStartIndex,  // Starting index of the node in the complete sorted lists of items
                               uint32_t segmentIndex,  // Index of the segment this node was split from
                               std::vector<NodeRange>& nodeItemRanges  // Output ranges of the nodes (in the sorted item lists) created by the recursive split
)
{
  uint32_t nodeItemCount = uint32_t(nodeItems[0].size());

  // If the current node is smaller than the maximum allowed item count, return its range
  if(nodeItemCount < maxItemsPerNode)
  {
    nodeItemRanges.push_back(NodeRange{nodeStartIndex, nodeItemCount, segmentIndex});
    return;
  }

  // Compute the AABB of the centroids of the items referenced by the node. Since the items are sorted along each axis,
  // the bounds on each coordinate are trivial to compute using the first and last items in the sorted list for that axis.
  // This does not provide the exact AABB for the node (ideallly we should combine the AABBs of each item), but this centroid-based
  // approximation is trivial and sufficient for the purpose of pre-splitting large inputs.
  AABB aabb{{spatial.centroids[nodeItems[0].front()][0], spatial.centroids[nodeItems[1].front()][1],
             spatial.centroids[nodeItems[2].front()][2]},
            {spatial.centroids[nodeItems[0].back()][0], spatial.centroids[nodeItems[1].back()][1],
             spatial.centroids[nodeItems[2].back()][2]}};

  // Deduce the splitting axis from the longest side of the AABB
  vec3f    size = aabb.size();
  uint32_t axis = size[0] > size[1] && size[0] > size[2] ? 0u : (size[1] > size[2] ? 1u : 2u);

  // Split the sorted item vectors at the median, preserving the order along each axis
  uint32_t splitPosition = nodeItemCount / 2;
  partitionAtSplit<Parallelize>(nodeItems, axis, splitPosition, partitionSides);

  // Extract the left and right halves of the sorted item lists
  auto left  = std::to_array({
      nodeItems[0].subspan(0, splitPosition),
      nodeItems[1].subspan(0, splitPosition),
      nodeItems[2].subspan(0, splitPosition),
  });
  auto right = std::to_array({
      nodeItems[0].subspan(splitPosition),
      nodeItems[1].subspan(splitPosition),
      nodeItems[2].subspan(splitPosition),
  });
  // Continue the split recursively on the left and right halves
  splitAtMedianUntil<Parallelize>(spatial, partitionSides, left, maxItemsPerNode, nodeStartIndex, segmentIndex, nodeItemRanges);
  splitAtMedianUntil<Parallelize>(spatial, partitionSides, right, maxItemsPerNode, nodeStartIndex + splitPosition,
                                  segmentIndex, nodeItemRanges);
}

// Find the lowest cost split on any axis, perform the split (partition
// sortedItems, maintaining order) and write two child nodes to the output.
template <bool Parallelize, bool HasConnectionWeights, bool HasVertexLimit>
Split splitNode(const Input&         input,  // Input items and connections
                const SplitNodeTemp& node)   // Node to split and attribute subranges
{
  // Find a split candidate by looking for the best split along each axis
  Split split;
  findBestSplit<Parallelize, HasConnectionWeights, HasVertexLimit, 0, true>(input, node, split);
  findBestSplit<Parallelize, HasConnectionWeights, HasVertexLimit, 1, true>(input, node, split);
  findBestSplit<Parallelize, HasConnectionWeights, HasVertexLimit, 2, true>(input, node, split);

  // Item count is too small to make clusters between min/max size. Fall
  // back to aligning splits from the left so there should be just one
  // cluster outside the range. This should be rare.
  if(!split.valid())
  {
    findBestSplit<Parallelize, HasConnectionWeights, HasVertexLimit, 0, false>(input, node, split);
    findBestSplit<Parallelize, HasConnectionWeights, HasVertexLimit, 1, false>(input, node, split);
    findBestSplit<Parallelize, HasConnectionWeights, HasVertexLimit, 2, false>(input, node, split);
  }

  // Split (before the indexed element) should be after the first and before the
  // last. Assert because we don't want to pay for this check in release.
  assert(split.position > 0 && split.position < node.range.count);

  // Split the node at the chosen axis and position
  partitionAtSplit<Parallelize>(node.nodeItems, split.axis, split.position, node.partitionSides);

  // Mark children as incomplete if over the vertex limit. If the user requested
  // a vertex limit but no vertex underfill cost, and the maximum items is
  // similar to the maximum vertices (which is common), it's faster to do an
  // explicit check for small nodes rather than compute unique vertex counts for
  // every position in every node. If a vertex underfill cost is added, this
  // check is already done in findBestSplit().
  if constexpr(HasVertexLimit)
  {
    if(input.config.costUnderfillVertices == 0.0f)
    {
      if(split.leftComplete
         && vertexCountOverflows(split.axis, input, node.subspan<HasConnectionWeights, HasVertexLimit>(Range{0, split.position})))
        split.leftComplete = false;
      if(split.rightComplete
         && vertexCountOverflows(split.axis, input,
                                 node.subspan<HasConnectionWeights, HasVertexLimit>(
                                     Range{split.position, node.range.count - split.position})))
        split.rightComplete = false;
    }
  }

  return split;
}

struct NextNodeBatch
{
  // Output nodes storage where child nodes will be added
  std::span<NodeRange> outNodes;

  // Current number of nodes in the output
  uint32_t& outNodesAlloc;

  template <bool ThreadSafe>
  void push(const NodeRange& newNode)
  {
    assert(newNode.count > 0);
    // Node is still too big. Keep splitting it by appending to the next batch.
    if constexpr(ThreadSafe)
    {
      uint32_t index = std::atomic_ref(outNodesAlloc)++;
      assert(size_t(index) < outNodes.size());
      outNodes[index] = newNode;
    }
    else
    {
      assert(outNodesAlloc < outNodes.size());
      outNodes[outNodesAlloc++] = newNode;
    }
  }
};

// Encapsulates clustering output. Since segments will contain variable-size
// clusters, clusters are first written to temporary storage, then written to
// the output in segment order. If there is only one segment, we can instead
// write directly to the output.
class CompleteClusterOutput
{
public:
  CompleteClusterOutput(const Input& input_, const OutputClusters& outputClusters_)
      : input(input_)
      , outputClusters(outputClusters_)
      , resultClusters(outputClusters.clusterItemRanges)
  {
    if(input.segments.size() > 1)
    {
      resultClusterStorage.resize(outputClusters.clusterItemRanges.size());
      resultSegments.resize(outputClusters.clusterItemRanges.size());
      resultClusters = resultClusterStorage;
    }

    // Initialize the output size. This grows with each call to push().
    outputClusters.clusterCount = 0u;

    // The itemCount must always match the input since the algorithm does not
    // move unreferenced items.
    outputClusters.itemCount = uint32_t(input.boundingBoxes.size());
  }

  ~CompleteClusterOutput() { packResults(); }

  template <bool ThreadSafe>
  void push(const NodeRange& newNode)
  {
    assert(newNode.count > 0);
    // Append the node as a cluster
    if constexpr(ThreadSafe)
    {
      uint32_t index = std::atomic_ref(outputClusters.clusterCount)++;
      assert(size_t(index) < resultClusters.size());
      resultClusters[index] = newNode;
      if(!resultSegments.empty())  // don't write segments if there is only one
        resultSegments[index] = newNode.segment;
    }
    else
    {
      assert(outputClusters.clusterCount < resultClusters.size());
      resultClusters[outputClusters.clusterCount] = newNode;
      if(!resultSegments.empty())  // don't write segments if there is only one
        resultSegments[outputClusters.clusterCount] = newNode.segment;
      outputClusters.clusterCount++;
    }
    if(newNode.count < input.config.minClusterSize)
      clusterUnderflowCount++;
  }

  uint32_t getClusterUnderflowCount() const { return clusterUnderflowCount.load(); }

private:
  void packResults()
  {
    // Group clusters by segment and write segments
    if(input.segments.empty())
    {
      assert(resultClusterStorage.empty() && resultSegments.empty());
    }
    else if(input.segments.size() == 1)
    {
      // Fast path for a single segment, clusters have already been written to the
      // output (redirected by std::span resultClusters). Just need to write the
      // single segment containing all clusters.
      outputClusters.segments[0] = {0, outputClusters.clusterCount};
      assert(resultClusterStorage.empty() && resultSegments.empty());
    }
    else
    {
      // Count output clusters per segment
      std::ranges::fill(outputClusters.segments, Range{0, 0});
      for(uint32_t i = 0; i < outputClusters.clusterCount; i++)
      {
        outputClusters.segments[resultSegments[i]].count++;
      }

      // Prefix sum segment counts into offsets
      for(uint32_t i = 1; i < input.segments.size(); i++)
      {
        outputClusters.segments[i].offset = outputClusters.segments[i - 1].offset + outputClusters.segments[i - 1].count;
      }

      // Zero segment counts
      for(uint32_t i = 0; i < input.segments.size(); i++)
      {
        outputClusters.segments[i].count = 0;
      }

      // Write output clusters
      for(uint32_t i = 0; i < outputClusters.clusterCount; i++)
      {
        auto& segment                                                = outputClusters.segments[resultSegments[i]];
        outputClusters.clusterItemRanges[segment.offset + segment.count++] = resultClusters[i];
        assert(resultClusters[i].count > 0);
      }
    }
  }

  const Input&          input;
  const OutputClusters& outputClusters;

  std::atomic<uint32_t> clusterUnderflowCount = 0;

  std::vector<Range> resultClusterStorage;

  // Either OutputClusters::clusterItemRanges or resultClusters
  std::vector<uint32_t> resultSegments;

  // Empty or resultSegments, if there are multiple segments
  std::span<Range> resultClusters;
};

// Temporary object to emit child nodes to the next batch or as final clusters
struct NodeOutput
{
  NextNodeBatch          nextBatch;
  CompleteClusterOutput& complete;

  // Process two nodes from a split node. Both are either added to the next batch
  // for processing or emitted as clusters if complete.
  template <bool ThreadSafe>
  void emitSplit(const NodeRange& node, const Split& split)
  {
    NodeRange left{node.offset, split.position, node.segment};
    if(split.leftComplete)
      complete.push<ThreadSafe>(left);
    else
      nextBatch.push<ThreadSafe>(left);

    NodeRange right{node.offset + split.position, node.count - split.position, node.segment};
    if(split.rightComplete)
      complete.push<ThreadSafe>(right);
    else
      nextBatch.push<ThreadSafe>(right);
  }
};

// Starting from a set of spatial items defined by their bounding boxes and
// centroids, and an optional adjacency graph describing the connectivity
// between them, this function groups those items into clusters using a
// recursive bisection. This is similar to building a BVH but discarding the
// hierarchy.
template <bool Parallelize, bool HasConnectionWeights, bool HasVertexLimit>
nvcluster_Result clusterize(const Input& input, const OutputClusters& clusters)
{
  Stopwatch swClusterize("clusterize");

  assert(input.boundingBoxes.size() == input.centroids.size());  // already implied by the C API
  if(input.config.minClusterSize == 0 || input.config.maxClusterSize == 0 || input.config.minClusterSize > input.config.maxClusterSize)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_INVALID_CONFIG_CLUSTER_SIZES;
  }
  if(input.boundingBoxes.size() != clusters.items.size())
  {
    return nvcluster_Result::NVCLUSTER_ERROR_INVALID_OUTPUT_ITEM_INDICES_SIZE;
  }
  if(input.segments.size() != clusters.segments.size())
  {
    return nvcluster_Result::NVCLUSTER_ERROR_SEGMENT_COUNT_MISMATCH;
  }
  if constexpr(HasConnectionWeights || HasVertexLimit)
  {
    if(input.boundingBoxes.size() != input.itemConnectionRanges.size())
    {
      return nvcluster_Result::NVCLUSTER_ERROR_SPATIAL_AND_CONNECTIONS_ITEM_COUNT_MISMATCH;
    }
    if(input.config.maxClusterVertices < input.config.itemVertexCount)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_MAX_VERTICES_LESS_THAN_ITEM_VERTICES;
    }
    if(input.config.itemVertexCount > std::numeric_limits<nvcluster_VertexBits>::digits)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_ITEM_VERTEX_COUNT_OVERFLOW;
    }
  }

  uint32_t itemCount = uint32_t(input.boundingBoxes.size());

  // Wrap the output to handle multiple segments with variable-size clusters
  CompleteClusterOutput completeNodeOutput{input, clusters};

  // Early out if there are no items to cluster
  if(itemCount == 0 || input.segments.empty())
  {
    return nvcluster_Result::NVCLUSTER_SUCCESS;
  }

  // Temporary data
  // Used to mark the items as belonging to the left (1) or right (0) side of the split
  std::vector<uint8_t> partitionSides(itemCount);
  // Bouding boxes of the left children of the currently processed node
  std::vector<AABB> leftAabbs(itemCount);
  // Bouding boxes of the right children of the currently processed node
  std::vector<AABB> rightAabbs(itemCount);
  // Difference of adjacency weights for each item due to a split at an item
  std::vector<float> deltaWeights;
  // Adjacency weights resulting from a split at an item
  std::vector<float> splitWeights;
  // Running total unique vertices from the left and from the right
  std::vector<uint32_t> leftUniqueVertices;
  std::vector<uint32_t> rightUniqueVertices;
  // In practice, std::inclusive_scan does not work in-place, so some extra space is needed
  std::vector<uint32_t> tmpUniqueVertices;
  // For each item, stores the index of its connected items in the sorted list of items
  std::vector<uint32_t> connectionItemsInNodes[3];
  // Scatter/gather temporary data for buildAdjacencyInSortedList()
  std::vector<uint32_t> itemIndexToNodesIndex[3];

  // The kD-tree will split the array of spatial items recursively along the X, Y and Z axes. In order to
  // run the splitting algorithm we first need to sort the input spatial items along each of those axis,
  // so the splitting will only require a simple partitioning.
  // In order to save memory we will use the storage area for the resulting clustered item indices as a temporary storage
  // for the indices of items sorted along the X axis.
  std::vector<uint32_t> sortedY(itemCount);
  std::vector<uint32_t> sortedZ(itemCount);
  // Initialize the array of per-axis item indices so each entry references one item
  for(uint32_t i = 0; i < uint32_t(itemCount); i++)
  {
    clusters.items[i] = i;
    sortedY[i]        = i;
    sortedZ[i]        = i;
  }

  // Sort the items along the X, Y and Z axes based on the location of their
  // centroids. As mentioned above the storage area for the output clustered
  // item indices is used as a temporary storage for the sorted indices along
  // the X axis
  auto nodeItems = std::to_array<std::span<uint32_t>>({clusters.items, sortedY, sortedZ});
  {
    Stopwatch swSort("sort");
    for(Range segment : input.segments)
    {
      std::sort(exec<Parallelize>, nodeItems[0].begin() + segment.offset,
                nodeItems[0].begin() + segment.offset + segment.count, CentroidCompare<0>(input.centroids));
      std::sort(exec<Parallelize>, nodeItems[1].begin() + segment.offset,
                nodeItems[1].begin() + segment.offset + segment.count, CentroidCompare<1>(input.centroids));
      std::sort(exec<Parallelize>, nodeItems[2].begin() + segment.offset,
                nodeItems[2].begin() + segment.offset + segment.count, CentroidCompare<2>(input.centroids));
    }
  }

  // Temporary data for connectivity costs
  if constexpr(HasConnectionWeights)
  {
    deltaWeights.resize(itemCount);
    splitWeights.resize(itemCount);
  }
  if constexpr(HasVertexLimit)
  {
    leftUniqueVertices.resize(itemCount);
    rightUniqueVertices.resize(itemCount);
    tmpUniqueVertices.resize(itemCount);
  }
  if constexpr(HasConnectionWeights || HasVertexLimit)
  {
    // Maintain graph adjacency within in each sortedItems array to avoid
    // expensive searches when computing the split costs.
    for(uint32_t axis = 0; axis < 3; ++axis)
    {
      connectionItemsInNodes[axis].resize(input.connectionTargetItems.size());
      itemIndexToNodesIndex[axis].resize(itemCount);
    }
  }

  // Create an argument pack of all temporary data allocated above. This is
  // split into subranges for nodes as they are created and split. Most notably,
  // it allows convenient temporary data reuse.
  SplitNodeTemp intermediates = {
      .range                  = Range{0, itemCount},
      .nodeItems              = nodeItems,
      .partitionSides         = partitionSides,
      .leftAabbs              = leftAabbs,
      .rightAabbs             = rightAabbs,
      .deltaWeights           = deltaWeights,
      .splitWeights           = splitWeights,
      .leftUniqueVertices     = leftUniqueVertices,
      .rightUniqueVertices    = rightUniqueVertices,
      .tmpUniqueVertices      = tmpUniqueVertices,
      .connectionItemsInNodes = {connectionItemsInNodes[0], connectionItemsInNodes[1], connectionItemsInNodes[2]},
  };

  // BVH style recursive bisection. Split nodes recursively until they have the
  // desired number of items. Unlike a BVH build, the hierarchy is not stored
  // and leaf nodes are immediately emitted as clusters. The current list of
  // nodes is double buffered nodes for each level.
  std::vector<NodeRange> nodeItemRangesCurrent, nodeItemRangesNext;
  size_t                 maxNodes = (2 * itemCount) / input.config.maxClusterSize;
  if(HasVertexLimit)
  {
    // We can guarantee only 1 underflow without additional constraints.
    // Unfortunately when constraining connectivity, there may be many more
    // nodes in the worst case
    maxNodes = std::max(maxNodes, size_t((itemCount * input.config.itemVertexCount + input.config.maxClusterVertices - 1)
                                         / input.config.maxClusterVertices));
  }
  nodeItemRangesCurrent.reserve(maxNodes);
  nodeItemRangesNext.reserve(maxNodes);

  uint32_t sanitizedPreSplitThreshold = std::max(input.config.preSplitThreshold, input.config.maxClusterSize * 2);
  if(input.config.preSplitThreshold == 0 || itemCount < sanitizedPreSplitThreshold)
  {
    // Segments become root nodes to be split
    for(uint32_t i = 0; i < input.segments.size(); i++)
    {
      nodeItemRangesCurrent.push_back(NodeRange{input.segments[i].offset, input.segments[i].count, i});
    }
  }
  else
  {
    Stopwatch swSplitMedian("splitMedian");
    // Performance optimization. If there are more than preSplitThreshold items in the root node,
    // create child nodes by performing simple median splits on the input item lists until each node contains a maximum of sanitizedPreSplitThreshold items.
    // This reduces the overally computational cost of the BVH build by applying the more accurate (and costly) node splitting algorithm only on smaller nodes.
    for(uint32_t segmentIndex = 0; segmentIndex < input.segments.size(); segmentIndex++)
    {
      Range segment      = input.segments[segmentIndex];
      auto  segmentItems = std::to_array({
          nodeItems[0].subspan(segment.offset, segment.count),
          nodeItems[1].subspan(segment.offset, segment.count),
          nodeItems[2].subspan(segment.offset, segment.count),
      });
      splitAtMedianUntil<Parallelize>(input, partitionSides, segmentItems, sanitizedPreSplitThreshold, segment.offset,
                                      segmentIndex, nodeItemRangesCurrent);
    }
  }

  // Handle the case where the root node, or nodes after median splitting, are
  // already a valid cluster
  if constexpr(HasVertexLimit)
  {
    // If connectivity is used, convert connected indices to sorted item indices
    // to get constant lookup time for items within nodes.
    // WARNING: only builds axis zero, used below
    constexpr uint32_t axis = 0;
    buildConnectionsInNodes<Parallelize>(input, nodeItems[axis], connectionItemsInNodes[axis], itemIndexToNodesIndex[axis]);
  }
  nodeItemRangesCurrent.erase(
      std::remove_if(nodeItemRangesCurrent.begin(), nodeItemRangesCurrent.end(),
                     [&](const NodeRange& node) {
                       if(node.count <= input.config.maxClusterSize
                          && (!HasVertexLimit
                              || !vertexCountOverflows<0 /* axis zero */>(
                                  input, intermediates.subspan<false /* weights not used */, true /* vertex limit */>(node))))
                       {
                         if(node.count > 0u)
                           completeNodeOutput.push<false>(node);
                         return true;
                       }
                       return false;
                     }),
      nodeItemRangesCurrent.end());

  // Bisect nodes until no internal nodes are left. The nodes array is double buffered
  // for simplicity - could also be a thread safe queue. Leaf nodes are removed
  // and written to the output.
  while(!nodeItemRangesCurrent.empty())
  {
    // If connectivity is used, convert connected indices to node item indices
    // to get constant lookup time for items within nodes.
    if constexpr(HasConnectionWeights || HasVertexLimit)
    {
      Stopwatch sw("build adjacency lookup");
      for(uint32_t axis = 0; axis < 3; ++axis)
      {
        buildConnectionsInNodes<Parallelize>(input, nodeItems[axis], connectionItemsInNodes[axis], itemIndexToNodesIndex[axis]);
      }
    }

    uint32_t nodesNextAlloc = 0;
    nodeItemRangesNext.resize(nodeItemRangesNext.capacity());  // conservative over-allocation
    NodeOutput nodeOutput{NextNodeBatch{nodeItemRangesNext, nodesNextAlloc}, completeNodeOutput};

    // Process all nodes in the current level. If there are a small number of
    // large nodes, parallelize internally, otherwise parallelize externally.
    // TODO: tune for systems with high thread counts
    size_t nodeCountThreshold =
        std::max(1u, nodeItemRangesCurrent[0].count > 100000 ? std::thread::hardware_concurrency() :
                                                               std::thread::hardware_concurrency() / 4u);

    // Add an exception for small nodes if compiled with MSVC, where parallel
    // execution doesn't perform well with small batches.
    #if _MSC_VER
    if (nodeItemRangesCurrent[0].count < 4096)
        nodeCountThreshold = 0;
    #endif

    bool parallelizeInternally = nodeItemRangesCurrent.size() <= nodeCountThreshold;
    if(!Parallelize || parallelizeInternally)
    {
      for(const NodeRange& node : nodeItemRangesCurrent)
      {
        Split split = splitNode<Parallelize, HasConnectionWeights, HasVertexLimit>(
            input, intermediates.subspan<HasConnectionWeights, HasVertexLimit>(node));
        nodeOutput.emitSplit<false /* no thread safety needed */>(node, split);
      }
    }
    else
    {
      parallel_batches<Parallelize /* true */, 1>(nodeItemRangesCurrent.size(), [&](size_t i) {
        const NodeRange& node  = nodeItemRangesCurrent[i];
        Split            split = splitNode<false /* not parallel */, HasConnectionWeights, HasVertexLimit>(
            input, intermediates.subspan<HasConnectionWeights, HasVertexLimit>(node));
        nodeOutput.emitSplit<true /* thread safe */>(node, split);
      });
    }

    // Resize down to what was used before the swap
    nodeItemRangesNext.resize(nodesNextAlloc);

    // Swap current and next nodes arrays to process the next batch. std::swap()
    // uses move semantics, so this just swaps pointers to the internal storage.
    nodeItemRangesCurrent.clear();
    std::swap(nodeItemRangesNext, nodeItemRangesCurrent);
  }

  // It is possible to have less than the minimum number of items per cluster,
  // but there should be at most one, unless pre-splitting or the vertex limit
  // is used.
  uint32_t allowedUnderflow = input.config.preSplitThreshold == 0 ? 1 : div_ceil(itemCount, sanitizedPreSplitThreshold);
  if(!HasVertexLimit && completeNodeOutput.getClusterUnderflowCount() > allowedUnderflow)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_INTERNAL_MULTIPLE_UNDERFLOW;
  }
  return nvcluster_Result::NVCLUSTER_SUCCESS;
}

template <bool Parallelize, bool HasConnectionWeights>
nvcluster_Result clusterize(const Input& input, const OutputClusters& clusters)
{
  // Vertex limit computation can be skipped if maxClusterVertices is not set
  bool hasVertexLimit = input.config.maxClusterVertices != ~0u;
  return hasVertexLimit ? clusterize<Parallelize, HasConnectionWeights, true>(input, clusters) :
                          clusterize<Parallelize, HasConnectionWeights, false>(input, clusters);
}

template <bool Parallelize>
nvcluster_Result clusterize(const Input& input, const OutputClusters& clusters)
{
  return !input.connectionWeights.empty() ? clusterize<Parallelize, true>(input, clusters) :
                                            clusterize<Parallelize, false>(input, clusters);
}

nvcluster_Result clusterize(bool parallelize, const Input& input, const OutputClusters& clusters)
{
// Set NVCLUSTER_MULTITHREADED to 1 to use parallel processing, or set it to 0
// to use a single thread for all operations, which can be easier to debug.
#if !defined(NVCLUSTER_MULTITHREADED) || NVCLUSTER_MULTITHREADED
  return parallelize ? clusterize<true>(input, clusters) : clusterize<false>(input, clusters);
#else
  (void)parallelize;
  return clusterize<false>(input, clusters);
#endif
}

}  // namespace nvcluster
