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

#include "shaderio_scene.h"

#ifndef _SHADERIO_STREAMING_H_
#define _SHADERIO_STREAMING_H_

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else
// only affect shaders

// light weight check for clas allocator, counts number of used bits
// during build_freegaps and compares it against persistently tracked
// allocatedSize
#define STREAMING_DEBUG_USEDBITS_COUNT 1

// heavy weight, tests all bits being 0 or 1 respectively for the space
// a group takes on load, or unload.
#define STREAMING_DEBUG_FREEGAPS_OVERLAP 0

#endif

/////////////////////////////////////////

// we tag the 64 bit VA of a cluster group with a special address range
// when it hasn't been loaded yet, values greater than this encode the
// frame index that this group has been requested last.
#define STREAMING_INVALID_ADDRESS_START (uint64_t(1) << 63)

// checks previous address when loading/unloading a group
#define STREAMING_DEBUG_ADDRESSES 0

// simplifies debugging clas allocator by doing full build of free lists
// every frame
#define STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS 0

// disables cluster move operations and blas/tlas builds as as well as actual ray tracing
#define STREAMING_DEBUG_WITHOUT_RT 0

// avoid move operation do a slow in-shader move of clas data, set to 1 only properly works with persistent allocator
#define STREAMING_DEBUG_MANUAL_MOVE 0

// Must be 32 to se we have easier processing of the bit arrays.
// the minimum allocation will use 32 bits, which can only straddle across two u32 values
#define STREAMING_ALLOCATOR_MIN_SIZE 32

// Tracks whether a geometry lod level is loaded completely
#define STREAMING_GEOMETRY_LOD_LEVEL_TRACKING 0

/////////////////////////////////////////

struct StreamingRequest
{
  uint maxLoads;
  uint maxUnloads;
  uint loadCounter;
  uint unloadCounter;
#ifdef __cplusplus
  union
  {
    uint64_t frameIndex;
    uint32_t frameIndexU32[2];
  };
#else
  uint64_t frameIndex;  // already embeds STREAMING_INVALID_ADDRESS_START
#endif

  // for compaction based clas management
  uint64_t clasCompactionUsedSize;
  uint     clasCompactionCount;
  // for persistent allocator clas management
  uint     clasAllocatedMaxSizedLeft;
  uint64_t clasAllocatedUsedSize;
  uint64_t clasAllocatedWastedSize;

  uint taskIndex;
  uint errorUpdate;
  uint errorAgeFilter;
  uint errorClasNotFound;
  uint errorClasList;
  uint errorClasAlloc;
  uint errorClasDealloc;
  int  errorClasUsedVsAlloc;

  BUFFER_REF(uvec2s_inout) loadGeometryGroups;
  BUFFER_REF(uvec2s_inout) unloadGeometryGroups;
};

/////////////////////////////////////////

struct ClasBuildInfo
{
  uint32_t clusterID;
  uint32_t clusterFlags;

#define ClasBuildInfo_packed_triangleCount 0 : 9
#define ClasBuildInfo_packed_vertexCount 9 : 9
#define ClasBuildInfo_packed_positionTruncateBitCount 18 : 6
#define ClasBuildInfo_packed_indexType 24 : 4
#define ClasBuildInfo_packed_opacityMicromapIndexType 28 : 4
  uint32_t packed;

  // struct VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV
  // {
  //   uint32_t geometryIndex : 24;
  //   uint32_t reserved : 5;
  //   uint32_t geometryFlags : 3;
  // };
  // VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV << 29
#define ClasGeometryFlag_OPAQUE_BIT_NV (4 << 29)
  uint32_t baseGeometryIndexAndFlags;

  uint16_t indexBufferStride;
  uint16_t vertexBufferStride;
  uint16_t geometryIndexAndFlagsBufferStride;
  uint16_t opacityMicromapIndexBufferStride;
  uint64_t indexBuffer;
  uint64_t vertexBuffer;
  uint64_t geometryIndexAndFlagsBuffer;
  uint64_t opacityMicromapArray;
  uint64_t opacityMicromapIndexBuffer;
};
BUFFER_REF_DECLARE_ARRAY(ClasBuildInfos_inout, ClasBuildInfo, , 16);

struct StreamingPatch
{
  uint32_t geometryID;
  uint32_t groupIndex;
  uint64_t groupAddress;
};
BUFFER_REF_DECLARE_ARRAY(StreamingPatchs_in, StreamingPatch, , 16);

struct StreamingGeometryPatch
{
  uint32_t geometryID;
  uint32_t lodsCompletedMask;
};
BUFFER_REF_DECLARE_ARRAY(StreamingGeometryPatchs_in, StreamingGeometryPatch, , 8);

struct StreamingUpdate
{
  // unload operations are before load operations
  uint patchUnloadGroupsCount;
  // total operations
  uint patchGroupsCount;

  // geometry patch count
  uint patchGeometriesCount;
  uint _pad;

  // all newly loaded groups have linear positions in the
  // compacted list of active groups starting with this value
  uint loadActiveGroupsOffset;
  uint loadActiveClustersOffset;

  uint taskIndex;
  uint frameIndex;

  // loaded come first, then unloaded
  BUFFER_REF(StreamingPatchs_in) patches;

  // newly loaded group clusters fill these
  BUFFER_REF(ClasBuildInfos_inout) newClasBuilds;
  BUFFER_REF(uint32s_inout) newClasResidentIDs;
  BUFFER_REF(uint32s_inout) newClasSizes;
  BUFFER_REF(uint64s_inout) newClasAddresses;
  uint32_t newClasCount;

  // gometry state handling
  BUFFER_REF(StreamingGeometryPatchs_in) geometryPatches;

  // compaction
  uint32_t moveClasCounter;
  uint64_t moveClasSize;
  BUFFER_REF(uint64s_inout) moveClasSrcAddresses;
  BUFFER_REF(uint64s_inout) moveClasDstAddresses;
};

/////////////////////////////////////////

struct StreamingGroup
{
  uint32_t clusterCount;
  int32_t  age;
  BUFFER_REF(Group_in) group;
};
BUFFER_REF_DECLARE_ARRAY(StreamingGroup_inout, StreamingGroup, , 16);

struct StreamingResident
{
  // Dynamic content:
  //
  // These are lists of groups that don't include
  // lowest detail, as those are always kept resident

  uint activeGroupsCount;
  uint activeClustersCount;

  BUFFER_REF(uint32s_in) activeGroups;

  // Resident content:
  //
  // Due to immutability of the residentID, these tables
  // are indexed sparsely, and they do contain all
  // content, including persistent lowest detail.

  BUFFER_REF(StreamingGroup_inout) groups;

  // only if persistent clas allocator is used
  BUFFER_REF(uvec2s_inout) groupClasSizes;

  BUFFER_REF(uint64s_inout) clusters;
  BUFFER_REF(uint64s_inout) clasAddresses;
  BUFFER_REF(uint32s_inout) clasSizes;

  uint64_t clasBaseAddress;
  uint64_t clasMaxSize;

  // single element address for persistent information storage
  // we read from these values when no load/unloads are performed per frame
  //
  // for compaction based clas management
  BUFFER_REF(uint64s_inout) clasCompactionUsedSize;
  // for persistent allocator clas management
  BUFFER_REF(uint32s_inout) clasAllocatedMaxSizedLeft;

  // aid debugging
  uint taskIndex;
  uint frameIndex;
};

/////////////////////////////////////////

struct AllocatorRange
{
  int32_t  count;
  uint32_t offset;
};
BUFFER_REF_DECLARE_ARRAY(AllocatorRange_inout, AllocatorRange, , 8);
struct AllocatorStats
{
  int64_t allocatedSize;
  int64_t wastedSize;
};
BUFFER_REF_DECLARE(AllocatorStats_inout, AllocatorStats, , 8);


struct StreamingAllocator
{
  uint freeGapsCounter;
  uint granularityByteShift;  // size of one unit in (1 << shift) bytes
  uint maxAllocationSize;     // not in bytes but units
  uint sectorCount;
  uint sectorMaxAllocationSized;
  uint sectorSizeShift;
  uint baseWastedSize;
  uint usedBitsCount;

  DispatchIndirectCommand dispatchFreeGapsInsert;

  BUFFER_REF(uint32s_inout) freeGapsPos;
  BUFFER_REF(uint16s_inout) freeGapsSize;
  BUFFER_REF(uint32s_inout) freeGapsPosBinned;
  BUFFER_REF(AllocatorRange_inout) freeSizeRanges;
  BUFFER_REF(uint32s_inout) usedBits;
  BUFFER_REF(uint32s_inout) usedSectorBits;
  BUFFER_REF(AllocatorStats_inout) stats;
};

/////////////////////////////////////////

struct SceneStreaming
{
  int32_t ageThreshold;
  uint    frameIndex;

  StreamingResident  resident;
  StreamingUpdate    update;
  StreamingRequest   request;
  StreamingAllocator clasAllocator;
};


#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_STREAMING_H_