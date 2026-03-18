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
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include "shaderio_core.h"

#ifndef _SHADERIO_SCENE_H_
#define _SHADERIO_SCENE_H_

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else

#ifndef CLUSTER_VERTEX_COUNT
#define CLUSTER_VERTEX_COUNT 32
#endif

#ifndef CLUSTER_TRIANGLE_COUNT
#define CLUSTER_TRIANGLE_COUNT 32
#endif

#endif

#define SHADERIO_ORIGINAL_MESH_GROUP 0xffffffffu
#define SHADERIO_MAX_LOD_LEVELS 32


struct BBox
{
  vec3 lo;
  vec3 hi;
  // relevant to cluster's triangles
  float shortestEdge;
  float longestEdge;
};
BUFFER_REF_DECLARE_ARRAY(BBoxes_in, BBox, readonly, 16);

// A cluster contains a small number of triangles and vertices.
// It is always part of a group.
struct Cluster
{
  uint8_t triangleCountMinusOne;
  uint8_t vertexCountMinusOne;
  uint8_t lodLevel;
  uint8_t groupChildIndex;

  uint32_t groupID;

  BUFFER_REF(vec4s_in) vertices;
  BUFFER_REF(uint8s_in) localTriangles;

  uint64_t _pad;
};
BUFFER_REF_DECLARE(Cluster_in, Cluster, , 16);
BUFFER_REF_DECLARE_ARRAY(Clusters_inout, Cluster, , 16);
BUFFER_REF_DECLARE_SIZE(Cluster_size, Cluster, 32);

// A group contains multiple clusters that are the result of
// a common mesh decimation operation. Clusters within a group
// are watertight to each other. Groups are always streamed in
// completely, which simplifies the streaming management.

struct TraversalMetric
{
  // scalar by design, avoid hiccups with packing
  // order must match `nvclusterlod::Node`
  float boundingSphereX;
  float boundingSphereY;
  float boundingSphereZ;
  float boundingSphereRadius;
  float maxQuadricError;
};

struct Group
{
  uint32_t geometryID;
  uint32_t groupID;

  // streaming: global unique id given on load
  //            clusters array starts directly after group
  // preloaded: local id within geometry
  uint32_t residentID;
  uint32_t clusterResidentID;

  // when this group is first loaded, this is where the
  // temporary clas builds start.
  uint32_t streamingNewBuildOffset;

  uint16_t lodLevel;
  uint16_t clusterCount;

  TraversalMetric traversalMetric;

  BUFFER_REF(uint32s_in) clusterGeneratingGroups;
  BUFFER_REF(BBoxes_in) clusterBboxes;
};

BUFFER_REF_DECLARE(Group_in, Group, , 16);
BUFFER_REF_DECLARE_ARRAY(Groups_in, Group, , 16);
BUFFER_REF_DECLARE_SIZE(Group_size, Group, 64);

#ifdef __cplusplus
// must match `nvclusterlod::InteriorNode`
struct NodeRange
{
  uint32_t isNode : 1;
  uint32_t childOffset : 26;
  uint32_t childCountMinusOne : 5;
};

// must match `nvclusterlod::LeafNode`
struct GroupRange
{
  uint32_t isNode : 1;
  uint32_t groupIndex : 23;
  uint32_t groupClusterCountMinusOne : 8;
};
#endif

// must match `nvclusterlod::Node`
struct Node
{
#ifdef __cplusplus
  union
  {
    NodeRange  nodeRange;
    GroupRange groupRange;
  };
#else
  uint32_t packed;

#define Node_packed_isGroup 0 : 1

#define Node_packed_nodeChildOffset 1 : 26
#define Node_packed_nodeChildCountMinusOne 27 : 5

#define Node_packed_groupIndex 1 : 23
#define Node_packed_groupClusterCountMinusOne 24 : 8

#endif
  // use scalar to avoid glsl alignment hiccups
  TraversalMetric traversalMetric;
};
BUFFER_REF_DECLARE_ARRAY(Nodes_in, Node, readonly, 8);

struct LodLevel
{
  float    minBoundingSphereRadius;
  float    minMaxQuadricError;
  uint32_t groupOffset;
  uint32_t groupCount;
};
BUFFER_REF_DECLARE_ARRAY(LodLevels_inout, LodLevel, , 8);

struct Geometry
{
  uint32_t instancesOffset;
  uint32_t instancesCount;
  uint32_t lodsCompletedMask;  // bit mask for which lod-levels are fully loaded, requires STREAMING_GEOMETRY_LOD_LEVEL_TRACKING
  uint32_t lodLevelsCount;

  // lowest detail data is always available
  uint32_t lowDetailTriangles;
  uint32_t lowDetailClusterID;
  uint64_t lowDetailBlasAddress;

  // object space geometry bbox
  BBox bbox;

  BUFFER_REF(LodLevels_inout) lodLevels;

  // lod hierarchy traversal
  BUFFER_REF(Nodes_in) nodes;
  BUFFER_REF(BBoxes_in) nodeBboxes;

  // streaming (null if preloaded)
  // provides memory address of a resident group.
  //
  // Note this 64-bit value uses a special encoding.
  // only addresses < STREAMING_INVALID_ADDRESS_BEGIN can be dereferenced.
  BUFFER_REF(uint64s_inout) streamingGroupAddresses;

  // preloaded (null if streaming)
  // clusters
  BUFFER_REF(Groups_in) preloadedGroups;
  BUFFER_REF(Clusters_inout) preloadedClusters;
  // for ray tracing
  BUFFER_REF(uint64s_in) preloadedClusterClasAddresses;
  BUFFER_REF(uint32s_in) preloadedClusterClasSizes;
};
BUFFER_REF_DECLARE(Geometry_in, Geometry, readonly, 16);
BUFFER_REF_DECLARE(Geometry_inout, Geometry, , 16);

struct RenderInstance
{
  mat4 worldMatrix;

  uint32_t geometryID;
  uint32_t materialID;
  float    maxLodLevelRcp;
  uint32_t packedColor;
};
BUFFER_REF_DECLARE_ARRAY(RenderInstances_in, RenderInstance, readonly, 16);

#ifdef __cplusplus
// clusters are stored right next to group
static_assert((sizeof(Group) % sizeof(Cluster)) == 0);
}
#endif

#endif
