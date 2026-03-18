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

#ifndef NVCLUSTER_CLUSTERS_H
#define NVCLUSTER_CLUSTERS_H

#define NVCLUSTER_VERSION 2

#include <float.h>
#include <limits.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(NVCLUSTER_BUILDER_SHARED)
#if defined(_MSC_VER)
// msvc
#if defined(NVCLUSTER_BUILDER_COMPILING)
#define NVCLUSTER_API __declspec(dllexport)
#else
#define NVCLUSTER_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
// gcc/clang
#define NVCLUSTER_API __attribute__((visibility("default")))
#else
// Unsupported. If hit, use cmake GenerateExportHeader
#pragma warning Unsupported compiler
#define NVCLUSTER_API
#endif
#else  // defined(NVCLUSTER_BUILDER_SHARED)
// static lib, no export needed
#define NVCLUSTER_API
#endif

#ifdef __cplusplus
#define NVCLUSTER_DEFAULT(x) = x
#else
#define NVCLUSTER_DEFAULT(x)
#endif

// Binary-stable bool for C
typedef uint8_t nvcluster_Bool;
#define NVCLUSTER_TRUE (nvcluster_Bool)1u
#define NVCLUSTER_FALSE (nvcluster_Bool)0u

typedef enum nvcluster_Result
{
  NVCLUSTER_SUCCESS,
  NVCLUSTER_ERROR_CONTEXT_VERSION_MISMATCH,
  NVCLUSTER_ERROR_INVALID_CONFIG_CLUSTER_SIZES,
  NVCLUSTER_ERROR_MISSING_SPATIAL_BOUNDING_BOXES,
  NVCLUSTER_ERROR_MISSING_SPATIAL_CENTROIDS,
  NVCLUSTER_ERROR_INVALID_OUTPUT_ITEM_INDICES_SIZE,
  NVCLUSTER_ERROR_SPATIAL_AND_CONNECTIONS_ITEM_COUNT_MISMATCH,
  NVCLUSTER_ERROR_SEGMENT_AND_ITEM_COUNT_CONTRADICTION,
  NVCLUSTER_ERROR_SEGMENT_COUNT_MISMATCH,
  NVCLUSTER_ERROR_MAX_CLUSTER_VERTICES_WITHOUT_CONNECTION_BITS,
  NVCLUSTER_ERROR_MAX_VERTICES_LESS_THAN_ITEM_VERTICES,
  NVCLUSTER_ERROR_NO_CONNECTION_ATTRIBUTES,
  NVCLUSTER_ERROR_ITEM_VERTEX_COUNT_OVERFLOW,
  NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTICES_PROVIDED,
  NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTEX_COUNT_PROVIDED,
  NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_PER_ITEM_VERTEX_COUNT,
  NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_VERTEX_COUNT,
  NVCLUSTER_ERROR_NULL_INPUT,
  NVCLUSTER_ERROR_NULL_CONTEXT,
  NVCLUSTER_ERROR_NULL_OUTPUT,

  // These likely indicate a bug with the library
  NVCLUSTER_ERROR_INTERNAL_MULTIPLE_UNDERFLOW,
} nvcluster_Result;

typedef struct nvcluster_Vec3f
{
  float x NVCLUSTER_DEFAULT(0.0f);
  float y NVCLUSTER_DEFAULT(0.0f);
  float z NVCLUSTER_DEFAULT(0.0f);
} nvcluster_Vec3f;

#define nvcluster_defaultVec3f() {0.0f, 0.0f, 0.0f}

// Axis aligned bounding box
typedef struct nvcluster_AABB
{
#ifdef __cplusplus
  float bboxMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float bboxMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
#else
  float bboxMin[3];
  float bboxMax[3];
#endif
} nvcluster_AABB;

// clang-format off
#define nvcluster_defaultAABB() {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}}
// clang-format on

// An index/cursor based subrange
typedef struct nvcluster_Range
{
  uint32_t offset NVCLUSTER_DEFAULT(0u);
  uint32_t count  NVCLUSTER_DEFAULT(0u);
} nvcluster_Range;

#define nvcluster_defaultRange() {0u, 0u}

// Clustering knobs, including cost balancing
typedef struct nvcluster_Config
{
  // Minimum number of items per output cluster.
  // Ignored when maxClusterVertices is set.
  uint32_t minClusterSize NVCLUSTER_DEFAULT(1u);

  // Maximum number of items per output cluster
  uint32_t maxClusterSize NVCLUSTER_DEFAULT(~0u);

  // Maximum number of unique vertices per output cluster, if
  // connectionVertexBits or itemVertices is not null. Setting a
  // maxClusterVertices value causes minClusterSize to be ignored.
  // A value of 0u may also be used to disable the vertex limit.
  uint32_t maxClusterVertices NVCLUSTER_DEFAULT(~0u);

  // Cost penalty for under-filling clusters, [0, 1]
  float costUnderfill NVCLUSTER_DEFAULT(0.1f);

  // Cost penalty for overlapping bounding boxes, [0, 1]
  float costOverlap NVCLUSTER_DEFAULT(0.1f);

  // Cost penalty for under-filling maxClusterVertices, [0, 1]
  // Requires maxClusterVertices to be set.
  float costUnderfillVertices NVCLUSTER_DEFAULT(0.0f);

  // I.e. 3 when clustering triangles, 4 for quads. The number of bits that may
  // be set in each connectionVertexBits, and limited by its bit width (at most
  // 8).
  uint32_t itemVertexCount NVCLUSTER_DEFAULT(3);

  // If nonzero items will be recursively partitioned at the median of the
  // longest bounding box side until subsets contain at most preSplitThreshold
  // items prior to actual clustering. This is an optimization intended to speed
  // up clustering of large sets of items, e.g. more than 100k.
  uint32_t preSplitThreshold NVCLUSTER_DEFAULT(0u);
} nvcluster_Config;

#define nvcluster_defaultConfig() {1u, ~0u, 0.1f, 0.1f, 0u}

// Type to hold unique bits for connections between items. E.g. triangle A might
// connect to triangle B, sharing two vertices. Two of the first 3 bits would be
// set for that connection. The third may be used in another of triangle A's
// connection. Triangle B would use its own bits to identify the same shared
// vertices.
typedef uint8_t nvcluster_VertexBits;

typedef struct nvcluster_Input
{
  // Required section of spatial definition of items to cluster

  // Bounding boxes of items to cluster
  const nvcluster_AABB* itemBoundingBoxes NVCLUSTER_DEFAULT(nullptr);

  // Center positions (xyz) of items to cluster
  const nvcluster_Vec3f* itemCentroids NVCLUSTER_DEFAULT(nullptr);

  // Number of elements in itemBoundingBoxes and itemCentroids
  uint32_t itemCount NVCLUSTER_DEFAULT(0u);

  // Optional section to add weighted item connectivity to optimize the spatial
  // partitioning towards "minimum cuts". I.e. try to form clusters with high
  // interconnected weights.
  //
  // While the connections are unidirectional, the algorithm requires this data
  // structure to specify two directions explicitly for each connection. I.e. if
  // item A has a connection to item B, then item B must have a connection to
  // item A. The weights must match in both connections. Vertex bits, if used,
  // will not. The terms item and connection are equivalent to node and edge in
  // graph theory.

  // Each item has Range::count connections to other items, stored at
  // Range::offset in connectionTargetItems. Connections are unidirectional but must
  // be duplicated to store bidirectional connections.
  const nvcluster_Range* itemConnectionRanges NVCLUSTER_DEFAULT(nullptr);

  // Connected item indices that itemConnectionRanges selects from (i.e. edges
  // in a graph)
  const uint32_t* connectionTargetItems NVCLUSTER_DEFAULT(nullptr);

  // Optional. Weight of each connection. The same value must be used in return
  // connections.
  const float* connectionWeights NVCLUSTER_DEFAULT(nullptr);

  // Optional. Used when maxClusterVertices is needed. Rather than consume
  // indices to triangle vertices and try to match them, this library takes
  // per-item-connection bits to identify unique vertices, e.g. triangle A
  // connects to triangle B through triangle A's vertex 0 and 1. Note that
  // values will not be symmetric in return connections as other items will
  // identify those same vertices with different bits.
  //
  // ____2_____  Example for triangle A (local vertex 0, 1, 2)
  // \   |\   |
  //  \D | \ C|  connectionTargetItems = { index of B,     C,     D};
  //   \ |A \ |  connectionVertexBits  = {      0b011, 0b110, 0b101};
  //     0___1|
  //     |   /
  //     | B/
  //     | /
  //     |/
  const nvcluster_VertexBits* connectionVertexBits NVCLUSTER_DEFAULT(nullptr);

  // Size of connectionTargetItems, connectionWeights and connectionVertexBits
  uint32_t connectionCount NVCLUSTER_DEFAULT(0u);

  // Quick alternative to use maxClusterVertices, replacing connection inputs.
  // If not null, computes connectionVertexBits internally from the 2D array of
  // vertex indices itemVertices[itemCount][nvcluster_Config::itemVertexCount].
  // For triangle vertex indices, this could be a straight cast of
  // std::span<vec3u> triangleVertices, setting itemVertexCount to 3.
  const uint32_t* itemVertices NVCLUSTER_DEFAULT(nullptr);

  // Number of unique vertices referenced by itemVertices, if used. I.e. the
  // maximum value plus one. Used for internal intermediate allocation size.
  uint32_t vertexCount NVCLUSTER_DEFAULT(0u);
} nvcluster_Input;

// Optionally divide items to cluster into segments and cluster within each
// segment in a single API call. Segments must not overlap. Unreferenced items
// will still appear in nvcluster_OutputClusters::items.
typedef struct nvcluster_Segments
{
  // Each segment defines range of items to cluster within
  const nvcluster_Range* segmentItemRanges NVCLUSTER_DEFAULT(nullptr);

  // Number of segments
  uint32_t segmentCount NVCLUSTER_DEFAULT(0u);
} nvcluster_Segments;

// Clustering output counts. For example, nvclusterGetRequirements() will first
// write the upper limit of generated clusters. This must be used to size the
// allocation given to e.g. nvclusterBuild(), which will write the
// exact cluster count written.
typedef struct nvcluster_Counts
{
  uint32_t clusterCount NVCLUSTER_DEFAULT(0u);
} nvcluster_Counts;

// Clustering output, defining selections of input items that form clusters
// created by partitioning input items spatially
typedef struct nvcluster_OutputClusters
{
  // Clusters defined by ranges of item indices, where each cluster starts at
  // range.offset in items and contains range.count items
  nvcluster_Range* clusterItemRanges NVCLUSTER_DEFAULT(nullptr);

  // Indices of the input items, referenced by clusterItemRanges
  uint32_t* items NVCLUSTER_DEFAULT(nullptr);

  // Initially the number of elements in clusterItemRanges
  // The nvclusterBuild*() replaces it with the element count written
  uint32_t clusterCount NVCLUSTER_DEFAULT(0u);

  // Initially the number of elements in items
  // The nvclusterBuild*() replaces it with the element count written
  uint32_t itemCount NVCLUSTER_DEFAULT(0u);
} nvcluster_OutputClusters;

struct nvcluster_Context_t;
typedef struct nvcluster_Context_t* nvcluster_Context;

typedef struct nvcluster_ContextCreateInfo
{
  // Version expected. nvclusterCreateContext() returns
  // nvcluster_Result::NVCLUSTER_ERROR_CONTEXT_VERSION_MISMATCH if another is found at
  // runtime.
  uint32_t version NVCLUSTER_DEFAULT(NVCLUSTER_VERSION);

  // Set to NVCLUSTER_TRUE or NVCLUSTER_FALSE to enable or disable internal
  // parallelisation using std execution policies at runtime
  nvcluster_Bool parallelize NVCLUSTER_DEFAULT(NVCLUSTER_TRUE);
} nvcluster_ContextCreateInfo;

#define nvcluster_defaultContextCreateInfo() {NVCLUSTER_VERSION, NVCLUSTER_TRUE}

// Usage:
// 1. Call nvclusterGetRequirements(...) or
//    nvclusterGetRequirementsSegmented(...) to get conservative sizes
// 2. Allocate data for nvcluster_OutputClusters
// 3. Call nvclusterBuild(...) or nvclusterBuildSegmented(...)
// 4. Resize down to what was written
//
// Alternatively use ClusterStorage or SegmentedClusterStorage, which
// encapsulates the above.
//
// The segmented output, clusterSegments must have space to store
// nvcluster_Segments::segmentCount Range objects
NVCLUSTER_API nvcluster_Result nvclusterGetRequirements(nvcluster_Context       context,
                                                        const nvcluster_Config* config,
                                                        uint32_t                itemCount,
                                                        nvcluster_Counts*       outputRequiredCounts);
NVCLUSTER_API nvcluster_Result nvclusterBuild(nvcluster_Context         context,
                                              const nvcluster_Config*   config,
                                              const nvcluster_Input*    input,
                                              nvcluster_OutputClusters* outputClusters);
NVCLUSTER_API nvcluster_Result nvclusterGetRequirementsSegmented(nvcluster_Context         context,
                                                                 const nvcluster_Config*   config,
                                                                 uint32_t                  itemCount,
                                                                 const nvcluster_Segments* segments,
                                                                 nvcluster_Counts*         outputRequiredCounts);
NVCLUSTER_API nvcluster_Result nvclusterBuildSegmented(nvcluster_Context         context,
                                                       const nvcluster_Config*   config,
                                                       const nvcluster_Input*    input,
                                                       const nvcluster_Segments* segments,
                                                       nvcluster_OutputClusters* outputClusters,
                                                       nvcluster_Range*          clusterSegments);
NVCLUSTER_API uint32_t         nvclusterVersion(void);
NVCLUSTER_API nvcluster_Result nvclusterCreateContext(const nvcluster_ContextCreateInfo* info, nvcluster_Context* context);
NVCLUSTER_API nvcluster_Result nvclusterDestroyContext(nvcluster_Context context);
NVCLUSTER_API const char*      nvclusterResultString(nvcluster_Result result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NVCLUSTER_CLUSTERS_H
