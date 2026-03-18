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

/*
  
  Shader Description
  ==================
  
  This compute shader does a few simple operations that require only a single thread.

  STREAM_SETUP_... are enums for the various operations
  
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint setup;
} push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};

layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) coherent buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

////////////////////////////////////////////

layout(local_size_x=1) in;

////////////////////////////////////////////

void main()
{
  if (push.setup == STREAM_SETUP_COMPACTION_OLD_NO_UNLOADS)
  {
    // we will not do compaction of old when there are no unloads.
    // However appending new still depends on the/ `moveClasSize` to be configured 
    // correctly, so that we will append after it.
    
    // first streaming frame has special rule
    // (note we start at frame 1 not 0)
    if (streaming.frameIndex == 1)
    {
      // reset the persistent stored value to zero
      streaming.resident.clasCompactionUsedSize.d[0] = 0;
      streamingRW.update.moveClasSize = 0;
    }
    else {    
      streamingRW.update.moveClasSize = streaming.resident.clasCompactionUsedSize.d[0];
    }
  }
  else if (push.setup == STREAM_SETUP_COMPACTION_STATUS)
  {
    // move compaction for clas memory management
    if (streaming.update.patchGroupsCount > 0) {
      // persistently store the total compacted clas size
      streaming.resident.clasCompactionUsedSize.d[0] = streamingRW.update.moveClasSize;
      // for readback
      streamingRW.request.clasCompactionUsedSize = streamingRW.update.moveClasSize;
      streamingRW.request.clasCompactionCount    = streamingRW.update.moveClasCounter;
    }
    else {
      // no update, pull value from persistent storage
      streamingRW.request.clasCompactionUsedSize = streaming.resident.clasCompactionUsedSize.d[0];
      streamingRW.request.clasCompactionCount    = 0;
    }
  }
  else if (push.setup == STREAM_SETUP_ALLOCATOR_FREEINSERT)
  {
    uint freeGaps = streamingRW.clasAllocator.freeGapsCounter;
    uint maxFreeGaps = (streaming.clasAllocator.sectorCount << streaming.clasAllocator.sectorSizeShift);
  
    // reset to zero for `stream_allocator_setup_insertion.comp.glsl`
    streamingRW.clasAllocator.freeGapsCounter = 0;
    
    // and setup actual dispatch that inserts the freegaps into the lists 
    // within `stream_allocator_freelist_insert.comp.glsl`
    streamingRW.clasAllocator.dispatchFreeGapsInsert.gridX = (min(freeGaps,maxFreeGaps) + STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP -1) / STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP;
    streamingRW.clasAllocator.dispatchFreeGapsInsert.gridY = 1;
    streamingRW.clasAllocator.dispatchFreeGapsInsert.gridZ = 1;
  #if STREAMING_DEBUG_USEDBITS_COUNT
    // error check allocation state prior adding new groups
    uint64_t allocatedSize = streaming.clasAllocator.stats.d.allocatedSize;    
    if (streaming.clasAllocator.usedBitsCount > 0 && 
        allocatedSize != uint64_t(streaming.clasAllocator.usedBitsCount) << streaming.clasAllocator.granularityByteShift)
    {
      streamingRW.request.errorClasUsedVsAlloc = int(allocatedSize >> streaming.clasAllocator.granularityByteShift) - int(streaming.clasAllocator.usedBitsCount);
    }
  #endif
  }
  else if (push.setup == STREAM_SETUP_ALLOCATOR_STATUS)
  {
    if (streaming.frameIndex == 1)
    {
      // seed all available for first frame
      uint clasAllocatedMaxSizedLeft = streaming.clasAllocator.sectorMaxAllocationSized * streaming.clasAllocator.sectorCount;
      streaming.resident.clasAllocatedMaxSizedLeft.d[0] = clasAllocatedMaxSizedLeft;
      streamingRW.request.clasAllocatedMaxSizedLeft     = clasAllocatedMaxSizedLeft;
    #if USE_MEMORY_STATS
      streaming.clasAllocator.stats.d.allocatedSize = 0;
      streaming.clasAllocator.stats.d.wastedSize    = streaming.clasAllocator.baseWastedSize << streaming.clasAllocator.granularityByteShift;
    #endif
    }
    else {
      // persistent allocator for clas memory management
      if (streaming.update.patchGroupsCount > 0) {
        // count can be negative
        uint clasAllocatedMaxSizedLeft = uint(max(0,streaming.clasAllocator.freeSizeRanges.d[streaming.clasAllocator.maxAllocationSize-1].count));
        streaming.resident.clasAllocatedMaxSizedLeft.d[0] = clasAllocatedMaxSizedLeft;
        streamingRW.request.clasAllocatedMaxSizedLeft     = clasAllocatedMaxSizedLeft;
      }
      else {
        // no update, pull value from persistent storage
        streamingRW.request.clasAllocatedMaxSizedLeft = streaming.resident.clasAllocatedMaxSizedLeft.d[0];
      }
    }
  #if USE_MEMORY_STATS
    streamingRW.request.clasAllocatedUsedSize   = streaming.clasAllocator.stats.d.allocatedSize;
    streamingRW.request.clasAllocatedWastedSize = streaming.clasAllocator.stats.d.wastedSize;
  #endif
  }
}