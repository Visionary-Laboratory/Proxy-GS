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

#include <atomic>
#include <connections.hpp>
#include <execution>
#include <parallel.hpp>
#include <ranges>
#include <unordered_map>
#include <vector>

// Workaround for libc++ std::execution
#include <parallel_execution_libcxx.hpp>

namespace nvcluster {

// Initial reservation for item connections
static constexpr uint32_t AVERAGE_ADJACENCY_GUESS = 12;

// Sorted map small vector size
static constexpr uint32_t SMALL_VECTOR_SIZE = 32;

// Switch to indirect indexing if vertex count is this much more than the item
// count
static constexpr uint32_t INDIRECT_INDEXING_VERTEX_RATIO_THRESHOLD = 5;

// Incomplete small vector implementation, just for use in SortedMap. Only
// beneficial on Windows.
template <typename T, size_t N>
class SmallVector
{
public:
  T*       begin() { return isStack() ? stackData() : m_heapData.data(); }
  T*       end() { return begin() + m_size; }
  const T* begin() const { return isStack() ? stackData() : m_heapData.data(); }
  const T* end() const { return begin() + m_size; }
  size_t   size() const { return m_size; }
  T*       insert(T* pos, const T& value)
  {
    if(isStack())
    {
      assert(pos >= stackData() && pos <= stackData() + m_size);
      if(m_size == N)
      {
        m_heapData.reserve(N * 2);
        m_heapData.insert(m_heapData.end(), stackData(), stackData() + N);
        pos = m_heapData.data() + (pos - stackData());
      }
      else
      {
        std::copy_backward(pos, stackData() + m_size, stackData() + m_size + 1);
        *pos = value;
        ++m_size;
        return pos;
      }
    }
    assert(pos >= m_heapData.data() && pos <= m_heapData.data() + m_size);
    pos = &*m_heapData.insert(m_heapData.begin() + (pos - m_heapData.data()), value);
    ++m_size;
    return pos;
  }
  void erase(T* pos)
  {
    if(isStack())
    {
      assert(pos >= stackData() && pos < stackData() + m_size);
      std::copy(pos + 1, stackData() + m_size, pos);
      --m_size;
    }
    else
    {
      assert(pos >= m_heapData.data() && pos < m_heapData.data() + m_size);
      m_heapData.erase(m_heapData.begin() + (pos - m_heapData.data()));
      --m_size;
      assert(m_size == m_heapData.size());
    }
  }

private:
  bool isStack() const { return m_heapData.empty(); }

  T*       stackData() { return reinterpret_cast<T*>(&m_stackData); }
  const T* stackData() const { return reinterpret_cast<const T*>(&m_stackData); }

  alignas(std::array<T, N>) std::byte m_stackData[sizeof(std::array<T, N>)] = {};

  std::vector<T> m_heapData;
  size_t         m_size = 0;
};


// A sorted vector used to map a small number of items. E.g. computing a list of
// all adjacent triangles.
template <class Key, class Value>
class SortedMap
{
public:
  Value& operator[](const Key& key)
  {
    auto it = std::ranges::lower_bound(m_data, key, {}, &std::pair<Key, Value>::first);
    if(it == m_data.end() || it->first != key)
    {
      it = m_data.insert(it, {key, Value()});
    }
    return it->second;
  }
  auto begin() const { return m_data.begin(); }
  auto end() const { return m_data.end(); }
  auto size() const { return m_data.size(); }
  void erase(const Key& key)
  {
    auto it = std::ranges::lower_bound(m_data, key, {}, &std::pair<Key, Value>::first);
    if(it != m_data.end() && it->first == key)
    {
      m_data.erase(it);
    }
  }

private:
  SmallVector<std::pair<Key, Value>, SMALL_VECTOR_SIZE> m_data;
};

// Utility to compute vertex to item back references. VertexIndirection is
// useful if items only reference a few vertices.
template <bool VertexIndirection>
struct VertexConnections
{
  template <class ParallelizeType>
  VertexConnections(ParallelizeType&&, ItemVertices itemVertices, uint32_t vertexCount)
  {
    constexpr bool Parallelize = ParallelizeType::value;

    // Compute vertex indirection - a map of unique vertex indices
    // TODO: Parallelize? concurrent map or per-thread dedupe and reduce
    if constexpr(VertexIndirection)
    {
      vertexIndirection.reserve(itemVertices.itemCount() * 2);
      for(size_t i = 0; i < itemVertices.itemCount(); ++i)
      {
        for(uint32_t vertexIndex : itemVertices.vertices(i))
          vertexIndirection.try_emplace(vertexIndex, uint32_t(vertexIndirection.size()));
      }
    }

    size_t indirectSize = VertexIndirection ? vertexIndirection.size() : size_t(vertexCount);

    // Compute range sizes
    vertexItemCounts = std::vector<uint32_t>(indirectSize, 0U);
    parallel_batches<Parallelize>(itemVertices.itemCount(), [&](size_t itemIndex) {
      for(uint32_t vertexIndex : itemVertices.vertices(itemIndex))
      {
        if constexpr(VertexIndirection)
          vertexIndex = vertexIndirection.at(vertexIndex);
        std::atomic_ref(vertexItemCounts[vertexIndex])++;
      }
    });

    // Compute range offsets
    vertexItemOffsets = std::vector<uint32_t>(indirectSize);
    std::exclusive_scan(exec<Parallelize>, vertexItemCounts.begin(), vertexItemCounts.end(), vertexItemOffsets.begin(), 0U);
    uint32_t totalVertexItems = vertexItemOffsets.back() + vertexItemCounts.back();

    // Compute vertexItems by scatter writing to vertex ranges of each item
    std::ranges::fill(vertexItemCounts, 0u);
    vertexItems = std::vector<uint32_t>(totalVertexItems);
    parallel_batches<Parallelize>(itemVertices.itemCount(), [&](size_t itemIndex) {
      for(uint32_t vertexIndex : itemVertices.vertices(itemIndex))
      {
        if constexpr(VertexIndirection)
          vertexIndex = vertexIndirection.at(vertexIndex);
        vertexItems[vertexItemOffsets[vertexIndex] + std::atomic_ref(vertexItemCounts[vertexIndex])++] = uint32_t(itemIndex);
      }
    });
  }

  // Return a list of items connected to a vertex
  std::span<const uint32_t> items(uint32_t vertexIndex) const
  {
    if constexpr(VertexIndirection)
      vertexIndex = vertexIndirection.at(vertexIndex);
    return std::span(vertexItems).subspan(vertexItemOffsets[vertexIndex], vertexItemCounts[vertexIndex]);
  };

  // Compute items that each item connects to. It returns a
  // 'map[otherItem] = vertexBits'
  SortedMap<uint32_t, uint8_t> itemConnectionVertexBits(uint32_t itemIndex, std::span<const uint32_t> vertices) const
  {
    // Create output
    SortedMap<uint32_t, uint8_t> connections;

    // Scatter write connections and unique vertex bits (unique to just this item)
    for(size_t i = 0; i < vertices.size(); ++i)
    {
      for(uint32_t other : items(vertices[i]))
        connections[other] |= uint8_t(1) << i;
    }

    // Remove self
    connections.erase(itemIndex);
    return connections;
  }

  // Map of unique vertex indices to indices in counts and offsets arrays
  std::unordered_map<uint32_t, uint32_t> vertexIndirection;

  std::vector<uint32_t> vertexItemCounts;   // aka. vertex valence
  std::vector<uint32_t> vertexItemOffsets;  // prefix sum of counts
  std::vector<uint32_t> vertexItems;        // linearized ranges of items (e.g. triangles)
};

template <bool VertexIndirection>
MeshConnections makeMeshConnectionsParallel(ItemVertices itemVertices, uint32_t vertexCount)
{
  // Compute lists of items sharing each vertex
  VertexConnections<VertexIndirection> vertexItems(std::true_type{}, itemVertices, vertexCount);

  // Build ranges of the results of itemConnectionVertexBits()
  // There's a few ways to linearize in parallel:
  // 1. Compute and count each range, allocate total, recompute and fill
  //    (computes twice)
  // 2. Compute and store each range in the heap, allocate total, fill (uses
  //    more memory)
  // 3. Compute each range, hold a lock to allocate, fill directly (holds
  //    locks, reallocates, computes once, reduced memory)
  // We'll go with option 1. *shrug*
  // TODO: combine counts/offsets into result ranges or split result ranges
  std::vector<uint32_t> connectionItemCounts(itemVertices.itemCount(), 0);
  std::vector<uint32_t> connectionItemOffsets(itemVertices.itemCount());
  parallel_batches<true, 1>(itemVertices.itemCount(), [&](size_t itemIndex) {
    std::span<const uint32_t>    vertices  = itemVertices.vertices(itemIndex);
    SortedMap<uint32_t, uint8_t> adjacency = vertexItems.itemConnectionVertexBits(uint32_t(itemIndex), vertices);
    connectionItemCounts[itemIndex]        = uint32_t(adjacency.size());
  });
  std::exclusive_scan(exec<true>, connectionItemCounts.begin(), connectionItemCounts.end(), connectionItemOffsets.begin(), 0U);

  MeshConnections result;
  result.connectionRanges.resize(itemVertices.itemCount());
  result.connectionItems.resize(connectionItemOffsets.back() + connectionItemCounts.back());
  result.connectionVertexBits.resize(connectionItemOffsets.back() + connectionItemCounts.back());
  parallel_batches<true, 1>(itemVertices.itemCount(), [&](size_t itemIndex) {
    std::span<const uint32_t>    vertices  = itemVertices.vertices(itemIndex);
    SortedMap<uint32_t, uint8_t> adjacency = vertexItems.itemConnectionVertexBits(uint32_t(itemIndex), vertices);
    Range                        range     = {connectionItemOffsets[itemIndex], 0};
    for(auto [item, bits] : adjacency)
    {
      result.connectionItems[range.offset + range.count]      = item;
      result.connectionVertexBits[range.offset + range.count] = bits;
      range.count++;
    }
    result.connectionRanges[itemIndex] = range;
  });
  return result;
}

// Faster specialization for single-threaded execution that does not compute
// adjacency twice.
template <bool VertexIndirection>
MeshConnections makeMeshConnectionsSequential(ItemVertices itemVertices, uint32_t vertexCount)
{
  // Compute lists of items sharing each vertex
  VertexConnections<VertexIndirection> vertexItems(std::false_type{}, itemVertices, vertexCount);

  // Build ranges of the results of itemConnectionVertexBits()
  MeshConnections result;
  result.connectionRanges.resize(itemVertices.itemCount());
  result.connectionItems.reserve(itemVertices.itemCount() * AVERAGE_ADJACENCY_GUESS);
  result.connectionVertexBits.reserve(itemVertices.itemCount() * AVERAGE_ADJACENCY_GUESS);
  for(size_t itemIndex = 0; itemIndex < itemVertices.itemCount(); ++itemIndex)
  {
    std::span<const uint32_t>    vertices  = itemVertices.vertices(itemIndex);
    SortedMap<uint32_t, uint8_t> adjacency = vertexItems.itemConnectionVertexBits(uint32_t(itemIndex), vertices);
    result.connectionRanges[itemIndex]     = {uint32_t(result.connectionItems.size()), uint32_t(adjacency.size())};
    for(auto [item, bits] : adjacency)
    {
      result.connectionItems.push_back(item);
      result.connectionVertexBits.push_back(bits);
    }
  }
  return result;
}

// Switch to vertex indirection if vertex count is much more than the item count
template <bool Parallelize>
MeshConnections makeMeshConnections(ItemVertices itemVertices, uint32_t vertexCount)
{
  if(vertexCount > itemVertices.itemCount() * itemVertices.itemVertexCount() * INDIRECT_INDEXING_VERTEX_RATIO_THRESHOLD)
  {
    if constexpr(Parallelize)
      return makeMeshConnectionsParallel<true>(itemVertices, vertexCount);
    else
      return makeMeshConnectionsSequential<true>(itemVertices, vertexCount);
  }
  else
  {
    if constexpr(Parallelize)
      return makeMeshConnectionsParallel<false>(itemVertices, vertexCount);
    else
      return makeMeshConnectionsSequential<false>(itemVertices, vertexCount);
  }
}

// Expand dynamic parallel flag to compile time permutations
MeshConnections makeMeshConnections(bool parallelize, ItemVertices itemVertices, uint32_t vertexCount)
{
#if !defined(NVCLUSTER_MULTITHREADED) || NVCLUSTER_MULTITHREADED
  return parallelize ? makeMeshConnections<true>(itemVertices, vertexCount) : makeMeshConnections<false>(itemVertices, vertexCount);
#else
  (void)parallelize;
  return makeMeshConnections<false>(itemVertices, vertexCount);
#endif
}

}  // namespace nvcluster
