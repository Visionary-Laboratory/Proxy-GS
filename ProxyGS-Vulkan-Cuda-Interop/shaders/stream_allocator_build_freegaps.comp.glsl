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
  
    Note: The sample showcases two ways to manage CLAS memory on the device.
    One using a persistent allocator system (`stream_allocator...` files),
    and one using a simple compaction scheme (`stream_compaction...` files).
    This file is part of the allocator system.
  
  This compute shader analyzes the used clas memory and builds a list
  of free memory gaps that the alloction phase then can make use of.
  To find the free gaps the memory usage is represented in a giant
  bit array where one bit represents a memory region of a certain number of bytes
  (the granularity at which the allocatoer operates on).
  
  Allocation is done in `stream_allocator_load_groups.comp.glsl` and
  tags bits as used, while freeing is performed when unloading groups in 
  `stream_allocator_unload_groups.comp.glsl` and marks bits
  as unused. The unloading must be performed prior this kernel.
  
  This compute shader's subgroups look at a range of 
  memory usage bits and scan them linearly to find and merge
  free gaps up to `maxAllocationSize`.
  
  One thread operates on one 32-bit `usedBits` value.
  
  We update how many gaps of a certain size exist by incrementing
  `streaming.clasAllocator.freeSizeRanges.d[freeGapSize-1].count`, which
  is later used to build the size-binned lists of gaps that the
  allocation process depends on.
  
  The starting positions and the sizes of free gaps are written to
  `streaming.clasAllocator.freeGapsPos` and `streaming.clasAllocator.freeGapsSize`

  We later bin the gap positions based on their sizes in the 
  `stream_allocator_freegaps_insert.comp.glsl` kernel writing out
  `streaming.clasAllocator.freeGapsPosBinned`.
  
  If allocations are required then the the follow-up operations to this kernel are
  the STREAM_SETUP_ALLOCATOR_FREEINSERT step within `stream_setup.comp.glsl`
  and then `stream_allocator_setup_insertion.comp.glsl`.
  
  TODO potential improvement: build the un-binned freegaps in per sector lists
  and only if there was a change to the sector (triggered by load or unload). 
  Then do the free gaps insertion into the global binned list on per-sector basis. 
  This would avoid looking at bits and building lists of unchanged sectors.
  
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
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_subgroup_extended_types_int64 : require
#extension GL_KHR_shader_subgroup_shuffle_relative: require

#include "shaderio.h"

////////////////////////////////////////////

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
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

////////////////////////////////////////////

layout(local_size_x=STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP) in;

////////////////////////////////////////////

#define SUBGROUP_COUNT (STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP / SUBGROUP_SIZE)


void main()
{
  // Each sugroup operates on STREAMING_ALLOCATOR_SECTOR_SIZE many u32s
  // looping over them while linearly scanning and merging free gaps into regions up to
  // maxAllocationSize
  
  const uint sectorID      = gl_WorkGroupID.x * SUBGROUP_COUNT + gl_SubgroupID;
  // each sector operates on this many 32-bit values
  const uint sectorSize32  = 1 << streaming.clasAllocator.sectorSizeShift;
  // where the sector starts in the global `usedBits` array that represents the entire memory
  const uint sectorStart32 = sectorID << streaming.clasAllocator.sectorSizeShift;
  
  // in units and not bytes (overall within this shader we only operate in units)
  const uint maxAllocationSize = streaming.clasAllocator.maxAllocationSize;

  // when no loads are performed in this frame, then we only need the statistics 
  // of the state of the free space, and not the actual gap positions
  const bool updateHasNoLoads = streaming.update.patchGroupsCount == streaming.update.patchUnloadGroupsCount 
                                && STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS == 0;
  
  if (sectorID >= streaming.clasAllocator.sectorCount) return;
  
  // Take shortcut to a simpler logic if we know the entire sector is empty
  if ((streaming.clasAllocator.usedSectorBits.d[sectorID / 32] & (1 << (sectorID & 31))) == 0)
  {  
    // The pre-computed number tells us how many max-sized allocations fit within a sector.
    uint maxGapsCount = streaming.clasAllocator.sectorMaxAllocationSized;
    // their might be a tail depending on the max allocation size and the sector size
    uint sizeLeft     = (32 << streaming.clasAllocator.sectorSizeShift) - (maxGapsCount * maxAllocationSize);
    
    // do not record gaps < STREAMING_ALLOCATOR_MIN_SIZE
    
    // first thread in subgroup reports the sizes to the atomic counters
    uint storageStart = 0;
    if (gl_SubgroupInvocationID == 0)
    {
      atomicAdd(streaming.clasAllocator.freeSizeRanges.d[maxAllocationSize-1].count, int(maxGapsCount));
      if (sizeLeft >= STREAMING_ALLOCATOR_MIN_SIZE)
      {
        atomicAdd(streaming.clasAllocator.freeSizeRanges.d[sizeLeft-1].count, int(1));
      }
      storageStart = atomicAdd(streamingRW.clasAllocator.freeGapsCounter, maxGapsCount + (sizeLeft >= STREAMING_ALLOCATOR_MIN_SIZE ? 1 : 0));
    }
    
    if (updateHasNoLoads)
    {
      // Without loads happening this frame, we don't actually need to output the detailed
      // positions of the gaps, we are just interested in the histogram. 
      // Zero things here and the subsquent fill operations won't do any work.
      maxGapsCount = 0;
      sizeLeft     = 0;
    }
    
    // distribute filling the max gaps over the entire subgroup
    storageStart = subgroupBroadcastFirst(storageStart);
    for (uint gap = gl_SubgroupInvocationID; gap < maxGapsCount; gap += SUBGROUP_SIZE)
    {
      uint freeGapPos = gap * maxAllocationSize + sectorStart32 * 32;
      streaming.clasAllocator.freeGapsPos.d[storageStart + gap]  = freeGapPos;
      streaming.clasAllocator.freeGapsSize.d[storageStart + gap] = uint16_t(maxAllocationSize);
    }
   
    // the tail is handled by the first thread alone
    if (gl_SubgroupInvocationID == 0 && sizeLeft >= STREAMING_ALLOCATOR_MIN_SIZE)
    {
      uint freeGapPos = maxGapsCount * maxAllocationSize + sectorStart32 * 32;
      streaming.clasAllocator.freeGapsPos.d[storageStart + maxGapsCount]  = freeGapPos;
      streaming.clasAllocator.freeGapsSize.d[storageStart + maxGapsCount] = uint16_t(sizeLeft);
    }
  
    return;
  }

  // Without the shortcut we actually have to look at all bits.
  
  // We distribute this loop of scanning all bits by iterating in subgroup wide operations,
  // however we need some persistent state to be brought from one iteration to the next.  
  uint previousIterationLastBit              = 1;
  uint previousIterationLastGlobalRangeStart = 0;
  
  // want to find out if the sector is acually fully empty
  // and for debugging also how many bits were set
  uint sumUsedBitsCount = 0;

  // iterate over all bits, each thread is looking at one 32 bit value and we loop over sectorSize32
  // may values with the subgroup in lock-step.
  for (uint idx32 = gl_SubgroupInvocationID; idx32 < sectorSize32; idx32 += SUBGROUP_SIZE)
  {
    bool isLastIdx = idx32 == (sectorSize32-1);
  
    uint usedBits  = streaming.clasAllocator.usedBits.d[idx32 + sectorStart32];
    uint freeBits  = ~usedBits;

    sumUsedBitsCount += bitCount(usedBits);
    
    // some simple cases, all 32-bit are used or unused
    bool allUsed  = freeBits == 0;
    bool allFree  = usedBits == 0;
    
    // find the region of free bits within
    int freeBeginBit  = freeBits != 0 ? findLSB(freeBits) : -1;
    int freeEndBit    = freeBits != 0 && !allFree ? findLSB(~(freeBits >> freeBeginBit)) - 1 + freeBeginBit : 31;
    // is our last bit free
    uint lastBit      = usedBits >> 31;

    //  We are looking for "free" regions 
    //
    //   fb freeBeginBit
    //   fe freeEndBit
    //   -  marks free bit
    //   x  marks used bit
    //  | | defines boundaries of u32 we operate on, we may access
    //      the previous u32's last bit through shuffle
    //
    //  There are four states the u32 of the thread can have:
    //
    // all bits free 
    //     | - - - - . . . - - - - |
    //    fb 0
    //                        fe 31
    //
    // all bits used
    //     | x x x x . . . x x x x |
    // fb -1
    //                        fe 31
    // begin partial free
    //     | x x x - . . . - - - - |
    //          fb 3
    //                        fe 31
    // end partial free
    //     | - - - x . . . x x x x |
    //    fb 0
    //        fe 2
    //
    // other scenarios, like multiple small gaps, are eliminated by design 
    // and can be ignored
    //
  
    uint previousLastBit = subgroupShuffleUp(lastBit, 1);
    if (gl_SubgroupInvocationID == 0) previousLastBit = previousIterationLastBit;
  
  
    // To detect the start of longer ranges that may span multiple u32s,
    // we use an exclusive max to the last begin.
    // If the u32 has a start that doesn't end within, we will pass this value to the
    // subgroup max.
    // 
    // we start a new free region in this u32 if the previous ended used and we have a begin
    //    x | - - - - . . . - - - - |
    // or if we have a new begin within, independent of previous
    //    - | x x - - . . . - - - - |
    //    x | x x - - . . . - - - - |
    //
    // in both cases the free region must contain the last bit, to allow continuation
    
    uint globalRangeStart     = ((previousLastBit == 1 && freeBeginBit == 0) || freeBeginBit > 0) && freeEndBit == 31 
                                  ? idx32 * 32 + freeBeginBit : previousIterationLastGlobalRangeStart;
    uint lastGlobalRangeStart = subgroupExclusiveMax(globalRangeStart);
    if (gl_SubgroupInvocationID == 0) lastGlobalRangeStart = previousIterationLastGlobalRangeStart;
  
    // for next subgroup loop iteration
    previousIterationLastBit = subgroupShuffle(lastBit, 31);
    previousIterationLastGlobalRangeStart = subgroupShuffle(max(lastGlobalRangeStart,globalRangeStart), 31);
    
    // Actual free range insertion is delayed until there is a transition
    // from free to used, therefore the previous last bit matters.
    
    // all used and previous also used, do nothing
    //
    //   x | x x x x . . . x x x x |
    //
    if (allUsed && previousLastBit == 1) {}
    
    // all free, leave to next, unless isLastIdx
    //
    //   ? | - - - - . . . - - - - | 
    //
    // only start region, leave to next, unless isLastIdx
    //
    //   x | x x - - . . . - - - - | 
    //
    else if ((allFree || (freeBeginBit > 0 && freeEndBit == 31 && previousLastBit == 1)) && !isLastIdx) {}
    
    // create a new region
    //
    //  finish previous
    //   - | x x x x . . . x x x x |
    //   - | - - - x . . . x x x x |
    //  if isLastIdx, finish continued allFree
    //   - | - - - - . . . - - - - | 
    //  if isLastIdx, start & finish independent allFree
    //   x | - - - - . . . - - - - |
    //
    // Note: due to allocation size minimum of 32 it cannot happen that the very last u32
    // in a sector would require two range starts (end previous, and start & within).
    //   - | x - - - . . . - - - - |
    // we also cannot start and end within
    //   x | - - - - . . . - x x x |
    // nor have to end previous and start a new range
    //   - | x x x - . . . - - - - |
    else
    {    
      uint rangeStart;
      // the previous u32 ended with a free bit, so
      // get the information where the free range started from the global range
      if (previousLastBit == 0)
      {
        // start is from previous
        rangeStart = lastGlobalRangeStart;
      }
      else {
        // we start fresh within
        // strictly speaking this can only happen on isLastIdx and with freeBeginBit == 0
        rangeStart = idx32 * 32 + freeBeginBit;
      }
      
      
      // allUsed means we end with previous bit (-1)
      // otherwise we end with first region within us
      uint rangeEnd  = idx32 * 32 + (allUsed ? -1 : freeEndBit);
      uint rangeSize = rangeEnd + 1 - rangeStart;
      
      uint maxGapsCount = rangeSize / maxAllocationSize;
      uint sizeLeft     = rangeSize - (maxGapsCount * maxAllocationSize);
      
      rangeStart += sectorStart32 * 32;
      rangeEnd   += sectorStart32 * 32;
      
      // do not record gaps < STREAMING_ALLOCATOR_MIN_SIZE

      uint gapsCount = maxGapsCount + (sizeLeft >= STREAMING_ALLOCATOR_MIN_SIZE ? 1 : 0);
      uint storageStart = atomicAdd(streamingRW.clasAllocator.freeGapsCounter, gapsCount);
      
      if (maxGapsCount > 0) {
        atomicAdd(streaming.clasAllocator.freeSizeRanges.d[maxAllocationSize-1].count, int(maxGapsCount));
      }
      if (sizeLeft >= STREAMING_ALLOCATOR_MIN_SIZE) 
      {
        atomicAdd(streaming.clasAllocator.freeSizeRanges.d[sizeLeft-1].count, int(1));
      }
      
      if (updateHasNoLoads)
      {
        // Without loads happening this frame, we don't actually need to output the detailed
        // positions of the gaps, we are just interested in the histogram. 
        // Zero things here and the subsquent fill operations won't do any work.
        maxGapsCount = 0;
        sizeLeft     = 0;
      }
      for (uint gap = 0; gap < maxGapsCount; gap++)
      {
        uint freeGapPos = rangeStart + gap * maxAllocationSize;
        streaming.clasAllocator.freeGapsPos.d[storageStart + gap]  = freeGapPos;
        streaming.clasAllocator.freeGapsSize.d[storageStart + gap] = uint16_t(maxAllocationSize);
      }
      if (sizeLeft >= STREAMING_ALLOCATOR_MIN_SIZE) 
      {
        uint freeGapPos = rangeStart + maxGapsCount * maxAllocationSize;
        streaming.clasAllocator.freeGapsPos.d[storageStart + maxGapsCount]  = freeGapPos;
        streaming.clasAllocator.freeGapsSize.d[storageStart + maxGapsCount] = uint16_t(sizeLeft);
      }
    }
  }

  if (subgroupAll(sumUsedBitsCount == 0))
  {
    // entire sector was empty
    atomicAnd(streaming.clasAllocator.usedSectorBits.d[sectorID / 32], ~(1 << (sectorID & 31)));
  }
#if STREAMING_DEBUG_USEDBITS_COUNT
  else 
  {
    // for error checking
    sumUsedBitsCount = subgroupAdd(sumUsedBitsCount);
    if (gl_SubgroupInvocationID == 0){
      atomicAdd(streamingRW.clasAllocator.usedBitsCount, sumUsedBitsCount);
    }
  }
#endif
}

