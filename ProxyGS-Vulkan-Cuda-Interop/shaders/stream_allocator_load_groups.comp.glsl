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
  
  This compute shader handles allocation of clas space
  for newly built groups. It also builds the appropriate
  move operations of newly built clas to their new locations.
  
  One thread represents one newly loaded group
  
  First we try to find a gap for each group based on its requested size (and search a bit more).
  We hope that in grand scheme of things the scene behaves well enough that we keep
  allocating and freeing similarily sized requests.
  
  If the individual group doesn't find space, we will make a request over all
  groups that didn't find space in batches up to maxAllocationSize.
  
  
  before this kernel
  - `stream_allocator_unload_groups.comp.glsl`
    marks areas as unused
  - `stream_update_scene.comp.glsl`
    sets up new clas builds
  - building of new clas into temporary scratch space
  - `stream_allocator_build_freegaps.comp.glsl`
    performs computation of free gaps based on memory usage
  - `stream_allocator_setup_insertion.comp.glsl`
    prepares binning of gaps into size-based ranges
  - `stream_allocator_freegaps_insert.comp.glsl`
    bins the free gaps into size-based regions which the
    allocation process here reads from.
  
  afterwards
  - move new clas to persistent location
  - blas reference list build etc.
  
  TODO might need to improve divergent loops
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
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_shader_subgroup_extended_types_int64 : require

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

layout(local_size_x=STREAM_ALLOCATOR_LOAD_GROUPS_WORKGROUP) in;

////////////////////////////////////////////


bool findAllocation(inout uint allocSize, inout uint allocPos, uint attempts, uint requestGrowth)
{
  const uint maxAllocationSize = streaming.clasAllocator.maxAllocationSize;

  // We start out looking for a free gap with exactly the size we asked for.
  // If not succesfull we try a few more times with bigger requests.

  uint requestSize = allocSize;
  bool found = false;
  while (!found && attempts > 0 && requestSize <= maxAllocationSize) {
    
    int32_t idx = atomicAdd(streaming.clasAllocator.freeSizeRanges.d[requestSize-1].count, -1);
    if (idx >= 1)
    {
      // there was a gap left, let's use it
      uint rangeOffset = streaming.clasAllocator.freeSizeRanges.d[requestSize-1].offset;
      allocPos = streaming.clasAllocator.freeGapsPosBinned.d[uint(idx-1) + rangeOffset];
      
      found    = true;
    }
    else 
    {
      // no gap left, try a larger one
      requestSize += requestGrowth;
      attempts--;
    }
  }
  
  if (found)
  {
    // we don't want to leave small unused space after the allocation behind,
    // so we associate the space with this allocation, despite some waste (tracked in statistics)
    allocSize = requestSize;
  }
  
  return found;
}

void main()
{
  // in allocation system units, not in bytes!
  const uint maxAllocationSize    = streaming.clasAllocator.maxAllocationSize;
  // to convert between bytes and units
  const uint granularityByteShift = streaming.clasAllocator.granularityByteShift;
  const uint granularityByteMask  = (1 << granularityByteShift) - 1;

  // loads are stored after unloads within the patch array
  const uint patchLoadGroupsCount = streaming.update.patchGroupsCount - streaming.update.patchUnloadGroupsCount;  

  const uint threadID             = gl_GlobalInvocationID.x;
  const bool valid                = threadID < patchLoadGroupsCount;
  
  // treat invalid as found, this avoids threads contributing to our
  // batched group allocation scheme.
  bool found            = !valid;
  uint newGroupByteSize = 0;
  uint allocSize        = 0;
  uint allocPos         = 0;
  uint newBuildOffset   = 0;
  uint groupResidentID  = 0;
  Group group;
  
  if (valid)
  {
    // get details the newly loaded groups
    
    // The groups' residentIDs are always stored at the end of the activeGroups.    
    groupResidentID = streaming.resident.activeGroups.d[threadID + streaming.update.loadActiveGroupsOffset];
    group           = streaming.resident.groups.d[groupResidentID].group.d;
    
    // First compute space of all newly built clas within a group
    
    // All clas are built in a canonical order that was pre-determined on the CPU
    // and offsets are enoced in the group itself.
    newBuildOffset = group.streamingNewBuildOffset;
    newGroupByteSize = 0;
    for (uint c = 0; c < group.clusterCount; c++)
    {
      uint clasSize = streaming.update.newClasSizes.d[newBuildOffset + c];
      newGroupByteSize += clasSize;
    }
    
    // The alloction system works in a certain byte granularity, convert the request
    // in the units of the system.
    allocSize = (newGroupByteSize + granularityByteMask) >> granularityByteShift;
    allocPos  = 0;
    
    // We use bit scanning to find free gaps, if allocations were < 32 bits then we could have
    // multiple tiny gaps enoced in a singl u32 and have to account for that, which complicates
    // our scan logic (stream_allocator_build_freegaps.comp.glsl). Hence easier to just
    // waste a bit.
    // In reality there will hardly ever be waste due to this, cause a group contains multiple
    // clusters and so the sum of the allocation size is almost always greater than this.
    allocSize = max(allocSize, STREAMING_ALLOCATOR_MIN_SIZE);

    // Let's look for a free space for this group.
    // We search a few times with minimum growth, hoping most
    // groups end up with similar sizes.
    found = findAllocation(allocSize, allocPos, 16, 1);
  }
  
  // if we couldn't make an allocation individually then combine multiple groups
  // up to maxAllocationSize
  uvec4 voteNotFound = subgroupBallot(!found);
  if (voteNotFound != uvec4(0))
  {  
    uint inclusiveSum = subgroupInclusiveAdd(!found ? allocSize : 0);
    uint exclusiveSum = inclusiveSum - (!found ? allocSize : 0);
    
    // we iteratively find batches that fit in the maxAllocationSize
    // example for maxAllocationSize == 8
    //
    //       invocation:  0  1  2  3  4  5  6  7  8  9 ...
    //     voteNotFound:  -  -  x  x  x  -  x  x  -  x
    //     
    //             size:  0  0  1  2  4  0  3  2  0  3
    //            i.sum:  0  0  1  3  7  7 10 12 12 15
    //            e.sum:  0  0  0  1  3  7  7 10 12 12
    //   
    // first batch iteration:
    //     voteNotFound:  -  -  x  x  x  -  x  x  -  x
    //            first:  2 (uniform)
    //          rebased:  -  -  1  3  7  7 10 12 12 15
    //         in limit:  -  -  x  x  x  -  -  -  -  -
    //             last:  4 (uniform)
    //     request size:  7 (last rebased)
    //   delta to first:  -  -  0  1  3  -  -  -  -  -
    //
    // second batch iteration:
    //     voteNotFound:  -  -  -  -  -  -  x  x  -  x
    //            first:  6 (uniform)
    //          rebased:  -  -  -  -  -  -  3  5  5  8
    //         in limit:  -  -  -  -  -  -  x  x  -  x
    //             last:  9 (uniform)
    //     request size:  8 (last rebased)
    //   delta to first:  -  -  -  -  -  -  0  3  5  5
    
    while(voteNotFound != uvec4(0))
    {
      // find where the batch starts and ends
      uint  firstInBatch = subgroupBallotFindLSB(voteNotFound);
      uint  firstBase    = subgroupShuffle(exclusiveSum, firstInBatch);
    
      uint  rebasedInclusiveSum = inclusiveSum - firstBase;
      
      uvec4 voteInLimit = subgroupBallot(rebasedInclusiveSum <= maxAllocationSize && !found);
      uint  lastInBatch = subgroupBallotFindMSB(voteInLimit);
      
      uint requestSize  = subgroupShuffle(rebasedInclusiveSum, lastInBatch);
      uint requestWaste = 0;
      bool batchFound   = false;
      
      // now that we have a bunch of groups we allocate in this batch, try make the allocation
      
      if (gl_SubgroupInvocationID == firstInBatch)
      {
        uint requestSizeOrig = requestSize;
        // search a bit again, this time less, and with larger growth
        batchFound = findAllocation(requestSize, allocPos, 8, 32);
        if (!batchFound){
          // Again no luck, now we need to fall back to max allocation size.
          
          // By design our allocation system guarantees that each group that is loaded can use a full maxAllocationSize
          // gap, so this here should never fail.
          
          int32_t idx = atomicAdd(streaming.clasAllocator.freeSizeRanges.d[maxAllocationSize-1].count, -1);
          if (idx >= 1)
          {
            uint rangeOffset = streaming.clasAllocator.freeSizeRanges.d[maxAllocationSize-1].offset;
            allocPos = streaming.clasAllocator.freeGapsPosBinned.d[uint(idx-1) + rangeOffset];
            
            // While `findAllocation` does associate the full gap size to the allocation request,
            // for worst-case slot we do want to leave gaps behind, as long as they are not small.
            
            if (requestSize + STREAMING_ALLOCATOR_MIN_SIZE >= maxAllocationSize)
            {
              requestSize = maxAllocationSize;
            }
            
            batchFound = true;
          }
        }
        // we still might allocate a bit more than neede
        requestWaste = requestSize - requestSizeOrig;
      }
      
      batchFound = subgroupAny(batchFound);
      
      if (!batchFound)
      {
        // should never happen by design
        break;
      }
      
      // Update allocPos and allocSize for all threads within this batch.
      // !found condition is required, as the batch can span threads that already have an allocation.
      if (!found && firstInBatch <= gl_SubgroupInvocationID && gl_SubgroupInvocationID <= lastInBatch)
      {
        // the first thread was doing the actual allocation, get details from it
        allocPos        = subgroupShuffle(allocPos,      firstInBatch);
        uint firstExSum = subgroupShuffle(exclusiveSum,  firstInBatch);
        requestWaste    = subgroupShuffle(requestWaste,  firstInBatch);
        
        // compute our relative position to the allocation position, given more than one group
        // might share this alloction.
        uint deltaToFirst = exclusiveSum - firstExSum;
        // apply the delta
        allocPos += deltaToFirst;
        
        // the last group in the batch will get the wasted space tail of the batch allocation.
        if (gl_SubgroupInvocationID == lastInBatch) {
          allocSize += requestWaste;
        }
        
        // this group has been served
        found = true;
      }
      
      // for next iteration remove the groups (bits) of the current batch
      voteNotFound &= ~voteInLimit;
    }
  }
  
  if (valid) 
  {
    if (!found)
    {
      // should never happen by design, we only load new groups if there is guaranteed clas allocation space left
      streamingRW.request.errorClasNotFound = 1 + threadID;
      for (uint c = 0; c < group.clusterCount; c++)
      {
        streaming.update.moveClasSrcAddresses.d[newBuildOffset + c] = 0;
        streaming.update.moveClasDstAddresses.d[newBuildOffset + c] = 0;
      }
      return;
    }
    
    // convert the allocation position from units back to bytes
    uint64_t groupBaseAddress = streaming.resident.clasBaseAddress + (uint64_t(allocPos) << granularityByteShift);

    // we keep some allocation information in the resident object table, so we can speed up
    // the unloading process, where we give back the memory range we used.
    
    uint allocByteSize  = allocSize << granularityByteShift;
    uint wastedByteSize = allocByteSize - newGroupByteSize;
    
    // store group allocation size in units, but waste in bytes
    streaming.resident.groupClasSizes.d[groupResidentID] = uvec2(allocSize, wastedByteSize);
    
    // then assign new clas address and fill the move operations
    for (uint c = 0; c < group.clusterCount; c++)
    {
      uint clusterResidentID = group.clusterResidentID + c;
      
      uint clasSize        = streaming.update.newClasSizes.d[newBuildOffset + c];
      uint64_t clasAddress = streaming.update.newClasAddresses.d[newBuildOffset + c];
      
      uint64_t clasNewAddress = groupBaseAddress;
      
      // update persistent information in resident table
      streaming.resident.clasSizes.d[clusterResidentID]     = clasSize;
      streaming.resident.clasAddresses.d[clusterResidentID] = clasNewAddress;
      
      // setup the move of the newly built clas from the scratch
      streaming.update.moveClasSrcAddresses.d[newBuildOffset + c] = clasAddress;
      // to the allocated persistent address
      streaming.update.moveClasDstAddresses.d[newBuildOffset + c] = clasNewAddress;

    #if STREAMING_DEBUG_MANUAL_MOVE
      // these moves are normally done with a `VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV`,
      // but for debbuging can be done here.
      // look for `STREAMING_DEBUG_MANUAL_MOVE` in `SceneStreaming::cmdPostTraversal`
      uint64s_in inPointer = uint64s_in(clasAddress);
      uint64s_inout outPointer = uint64s_inout(clasNewAddress);
      for (uint d = 0; d < clasSize / 8; d++){
        outPointer.d[d] = inPointer.d[d];
      }
    #endif
      
      groupBaseAddress += uint64_t(clasSize);
    }
    
  #if USE_MEMORY_STATS
    atomicAdd(streamingRW.clasAllocator.stats.d.allocatedSize, int64_t(allocByteSize));
    atomicAdd(streamingRW.clasAllocator.stats.d.wastedSize,    int64_t(wastedByteSize));
  #endif
    
    // for allocation management, tag bits as used
    //
    // allocPos and allocSize are in minimum granularity,
    // which is what we use to tag the appropriate bits.
    
    uint startPos = allocPos;
    uint endPos   = allocPos + allocSize - 1;
    
    uint startBit = (startPos) & 31;
    uint endBit   = (endPos) & 31;
    
    uint start32 = startPos / 32;
    uint end32   = endPos / 32;
    
    uint startMask = ~0;
    uint endMask   = ~0;
    
    if (startBit != 0)
    {
      startMask = ~((1u << (startBit))-1);
    }
    if (endBit != 31)
    {
      endMask =  (1u << (endBit + 1))-1;
    }
    
    bool single32 = start32 == end32;      
    if (single32)
    {
      startMask = endMask | startMask;
    }
    
    // start and end of an allocated region may end up in the same u32,
    // hence we need atomics for start and end
    
    uint oldMask = atomicOr(streaming.clasAllocator.usedBits.d[start32], startMask);
  #if STREAMING_DEBUG_FREEGAPS_OVERLAP
    // for error checking we test if the region was indeed full empty
    uint hadError = 0;
    if ((oldMask & startMask) != 0){
      hadError = startPos;
    }
  #endif
  
    if (!single32) 
    {
      // process the region that is exclusively covered by this allocation
      for (uint32_t i = start32 + 1; i < end32; i++)
      {
       #if STREAMING_DEBUG_FREEGAPS_OVERLAP
        if(streaming.clasAllocator.usedBits.d[i] != 0){
          hadError = i * 32;
        }
       #endif
        streaming.clasAllocator.usedBits.d[i] = ~0;
      }
      
      oldMask = atomicOr(streaming.clasAllocator.usedBits.d[end32], endMask);
     #if STREAMING_DEBUG_FREEGAPS_OVERLAP
      if((oldMask & endMask) != 0){
        hadError = endPos;
      }
     #endif
    }
    
  #if STREAMING_DEBUG_FREEGAPS_OVERLAP
    if (hadError != 0){
      streamingRW.request.errorClasAlloc = hadError;
    }
  #endif
    
    // Tag sector is in use.
    // An allocation can only be within one sector.
    uint32_t sectorID = start32 >> streaming.clasAllocator.sectorSizeShift;
    atomicOr(streaming.clasAllocator.usedSectorBits.d[sectorID / 32], 1 << (sectorID & 31));
  }  
}

