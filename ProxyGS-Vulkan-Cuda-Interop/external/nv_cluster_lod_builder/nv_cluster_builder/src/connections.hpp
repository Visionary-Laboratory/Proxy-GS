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
#include <clusters_cpp.hpp>
#include <inttypes.h>
#include <span>
#include <unordered_map>
#include <vector>

namespace nvcluster {

// A 2D uint32_t array pointer, used to interpret nvcluster_Input::itemVertices.
// Rename to UintSpan2D if used multiple times. Could replace with
// mdspan<uint32_t, 2>.
struct ItemVertices
{
public:
  ItemVertices(const uint32_t* itemVertices, uint32_t itemCount, uint32_t itemVertexCount)
      : m_itemVertices(itemVertices)
      , m_itemCount(itemCount)
      , m_itemVertexCount(itemVertexCount)
  {
  }
  uint32_t                  itemCount() const { return m_itemCount; }              // mdspan::extent(0)
  uint32_t                  itemVertexCount() const { return m_itemVertexCount; }  // mdspan::extent(1)
  std::span<const uint32_t> vertices(size_t itemIndex) const                       // ~submdspan
  {
    return std::span(m_itemVertices, m_itemCount * m_itemVertexCount).subspan(itemIndex * m_itemVertexCount, m_itemVertexCount);
  }

private:
  const uint32_t* m_itemVertices;
  uint32_t        m_itemCount;
  uint32_t        m_itemVertexCount;
};

// Utility to generate item connections and vertex bits to use the vertex limit
// feature.
struct MeshConnections
{
  std::vector<Range>    connectionRanges;
  std::vector<uint32_t> connectionItems;
  std::vector<uint8_t>  connectionVertexBits;
};

NVCLUSTER_API MeshConnections makeMeshConnections(bool parallelize, ItemVertices itemVertices, uint32_t vertexCount);

}  // namespace nvcluster
