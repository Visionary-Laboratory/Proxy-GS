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

#pragma once

#include <queue>

#include <nvutils/logger.hpp>
#include <nvutils/id_pool.hpp>
#include <nvvk/buffer_suballocator.hpp>
#include <nvvk/command_pools.hpp>

#include "scene.hpp"
#include "resources.hpp"
#include "../shaders/shaderio_streaming.h"


namespace lodclusters {

static const uint32_t STREAMING_MAX_ACTIVE_TASKS = 3;
static const uint32_t INVALID_TASK_INDEX         = ~0;

struct StreamingConfig
{
  bool usePersistentClasAllocator = true;
  bool useAsyncTransfer           = false;
  bool useDecoupledAsyncTransfer  = false;

  uint32_t maxPerFrameLoadRequests   = 1024;
  uint32_t maxPerFrameUnloadRequests = 4096;

  uint32_t maxGroups   = 1 << 16;
  uint32_t maxClusters = 0;  // if 0 then computed from maxGroups

  size_t maxTransferMegaBytes = 32;
  size_t maxGeometryMegaBytes = 1024 * 2;
  size_t maxClasMegaBytes     = 1024 * 2;

  VkBuildAccelerationStructureFlagsKHR clasBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  uint32_t                             clasPositionTruncateBits = 0;

  // for persistent allocator
  uint32_t clasAllocatorSectorSizeShift = 10;
  // granularity of allocator in multiples of clas alignment
  uint32_t clasAllocatorGranularityShift = 0;
};

struct StreamingStats
{
  uint32_t residentGroups   = 0;
  uint32_t residentClusters = 0;
  uint32_t maxGroups        = 0;
  uint32_t maxClusters      = 0;

  uint32_t persistentGroups    = 0;
  uint32_t persistentClusters  = 0;
  uint64_t persistentDataBytes = 0;
  uint64_t persistentClasBytes = 0;

  uint64_t maxDataBytes      = 0;
  uint64_t reservedDataBytes = 0;
  uint64_t usedDataBytes     = 0;

  uint64_t reservedClasBytes = 0;
  uint64_t usedClasBytes     = 0;
  uint64_t wastedClasBytes   = 0;
  uint32_t maxSizedLeft      = 0;
  uint32_t maxSizedReserved  = 0;

  uint64_t maxTransferBytes     = 0;
  uint64_t transferBytes        = 0;
  uint32_t transferCount        = 0;
  uint32_t loadCount            = 0;
  uint32_t unloadCount          = 0;
  uint32_t uncompletedLoadCount = 0;
  uint32_t maxLoadCount         = 0;
  uint32_t maxUnloadCount       = 0;

  uint32_t couldNotAllocateGroup = 0;
  uint32_t couldNotAllocateClas  = 0;
  uint32_t couldNotTransfer      = 0;
  uint32_t couldNotStore         = 0;
};

union GeometryGroup
{
  struct
  {
    uint32_t geometryID;
    uint32_t groupID;
  };
  uint64_t key;
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingRequests
//
// Requests from the device to be handled by the streaming manager.
// Device fills it, host reacts by updating.
//
// This mostly provides the storage space for the geometry groups to be
// loaded and unloaded, both on device and the host copy.

class StreamingRequests
{
public:
  struct TaskInfo
  {
    const shaderio::StreamingRequest* shaderData;
    const GeometryGroup*              loadGeometryGroups;
    const GeometryGroup*              unloadGeometryGroups;
  };

  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);
  void deinit(Resources& res);

  size_t getOperationsSize() const;

  // for updates:
  // within same frame
  // first prepare request
  void applyTask(shaderio::StreamingRequest& shaderData, uint32_t taskIndex, uint32_t frameIndex);
  // then trigger readback for that request
  // srcBuffer & offset are for the location that stores the content of shaderio::StreamingRequest
  void cmdRunTask(VkCommandBuffer cmd, const shaderio::StreamingRequest& shaderData, VkBuffer srcBuffer, size_t srcBufferOffset);

  // later frame, get results when cmd update completed
  const TaskInfo& getCompletedTask(uint32_t taskIndex) { return m_taskInfos[taskIndex]; }

private:
  nvvk::Buffer m_requestBuffer;
  nvvk::Buffer m_requestHostBuffer;

  uint64_t m_requestSize;
  uint64_t m_shaderDataOffset;

  shaderio::StreamingRequest m_shaderData;
  TaskInfo                   m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingResident
//
// This class holds the persistent table of resident groups and clusters.
// Each group is assigned a groupResidentID, an immutable index in the group table,
// as well as a range of indices starting at clusterResidentID for the cluster table.
//
// During the initialization time we fill the table with all persistently-
// loaded geometry groups (lowest detail).
//
// see `StreamingResident::uploadInitialState`
//
// The class also manages the list of "active" groups that can be loaded and unloaded.
// These active groups are stored after the persistent ones, which are not
// part of the active list.
// This list is stored as a tightly packed array of groupResidentIDs so shaders
// can easily iterate all resident groups.
//
// When streaming system triggers the load of new groups, their indices are
// always appended at the end of the list. The compaction of the list, which
// is triggered by unloads, simply pops the last element in the freed spot.
// As small optimization, we only upload the range of indices that has changed,
// not the entire list.
//
// The object also contains the clas buffer in which resident clusters are stored.
// `initClas`/`deinitClas` are separated out, so that a pure rasterization renderer
// can avoid extra memory cost.

class StreamingResident
{
public:
  static const uint32_t INVALID_GROUP = ~0;

  struct Group
  {
    GeometryGroup             geometryGroup;
    uint32_t                  activeIndex;
    uint32_t                  groupResidentID;
    uint32_t                  clusterResidentID;
    uint16_t                  clusterCount;
    uint16_t                  lodLevel;
    uint64_t                  deviceAddress;
    nvvk::BufferSubAllocation storageHandle;
  };

  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);
  const Group* initClas(Resources&                   res,
                        const StreamingConfig&       config,
                        shaderio::StreamingResident& shaderData,
                        uint32_t&                    loGroupsCount,
                        uint32_t&                    loClustersCount,
                        uint32_t&                    loMaxGroupClustersCount);
  void         deinitClas(Resources& res);
  void         deinit(Resources& res);
  void         reset(shaderio::StreamingResident& shaderData);

  size_t getOperationsSize() const;
  size_t getClasOperationsSize() const;
  void   getStats(StreamingStats& stats) const;

  // run after initial persistent lo-detail groups were added
  void                uploadInitialState(Resources::BatchedUploader& uploader, shaderio::StreamingResident& shaderData);
  const nvvk::Buffer& getClasBuffer() const { return m_clasManageBuffer; }

  const StreamingResident::Group* findGroup(GeometryGroup geometryGroup) const;
  const StreamingResident::Group& getGroup(uint32_t groupResidentID) const { return m_groups[groupResidentID]; }

  // for updates:
  uint32_t getLoadActiveGroupsOffset() const;
  uint32_t getLoadActiveClustersOffset() const;

  // first handle adding & removing
  bool                      canAllocateGroup(uint32_t numClusters) const;
  StreamingResident::Group* addGroup(GeometryGroup geometryGroup, uint32_t clusterCount);
  void                      removeGroup(uint32_t groupResidentID);

  // then run this update, it will be based on all residency modifications up until this point
  // returns number of bytes transferred
  size_t cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex);

  // later apply when cmd update was completed
  void applyTask(shaderio::StreamingResident& shaderData, uint32_t taskIndex, uint32_t frameIndex);
  void cmdRunTask(VkCommandBuffer cmd, uint32_t taskIndex);


private:
  // tracks range of indices that were manipulated
  // since last update, triggered by unloads.
  struct UpdateRange
  {
    uint32_t lo = uint32_t(~0);
    uint32_t hi = 0;

    void update(uint32_t index)
    {
      lo = std::min(lo, index);
      hi = std::max(hi, index);
    }

    uint32_t count() const { return hi == 0 && lo == ~0 ? 0 : 1 + hi - lo; }
  };

  struct TaskInfo
  {
    VkBufferCopy                region;
    shaderio::StreamingResident shaderData;
  };

  // uint64_t is GeometryGroup::key
  std::unordered_map<uint64_t, uint32_t> m_mapGeometryGroup2Residency;

  nvutils::IDPool m_groupAllocator;
  nvutils::IDPool m_clusterAllocator;

  uint32_t m_maxClusters;
  uint32_t m_maxGroups;
  size_t   m_maxClasBytes;

  std::vector<Group> m_groups;

  // index into above
  std::vector<uint32_t> m_activeGroupIndices;

  uint32_t m_lowDetailGroupsCount;
  uint32_t m_lowDetailClustersCount;
  uint32_t m_lowDetailMaxGroupClusters;

  uint32_t m_activeGroupsCount;
  uint32_t m_activeClustersCount;

  nvvk::Buffer m_residentBuffer;
  uint64_t     m_residentGroupsOffset;
  uint64_t     m_residentClustersOffset;
  uint64_t     m_residentActiveOffset;
  uint64_t     m_residentActiveUpdateOffset;

  nvvk::Buffer      m_clasManageBuffer;
  nvvk::LargeBuffer m_clasDataBuffer;

  shaderio::StreamingResident m_shaderData;

  nvvk::BufferTyped<uint32_t> m_residentActiveHostBuffer;
  UpdateRange                 m_groupIndicesUpdateRange;

  TaskInfo m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingAllocator
//
// This class implements a persistent allocator on the GPU that
// allows to do allocation management in shaders.
//
// The compute kernels scan the memory for free gaps and make
// them available as list for different gap sizes up to
// `maxAllocationByteSize`. The memory's use is encoded in
// a giant bitfield where each bit represents `granularityByteSize`
// bytes.
//
// We use it in the sample to manage clas memory, as the host
// doesn't know the size of the clas after building and we want
// to avoid detailed readbacks and host to be involved in the
// allocation process.

class StreamingAllocator
{
public:
  void init(Resources&                    res,
            size_t                        totalMegaBytes,
            uint32_t                      maxAllocationByteSize,
            uint32_t                      granularityByteSize,
            uint32_t                      sectorSizeShift,
            shaderio::StreamingAllocator& shaderData);
  void deinit(Resources& res);

  size_t   getOperationsSize() const;
  uint32_t getMaxSized() const;

  void cmdReset(VkCommandBuffer cmd);
  void cmdBeginFrame(VkCommandBuffer cmd);

private:
  shaderio::StreamingAllocator m_shaderData;

  nvvk::Buffer m_managementBuffer;
};


//////////////////////////////////////////////////////////////////////////
//
// StreamingUpdates
//
// Provides storage for update tasks that modify the per-geometry group pointers
// and are performed on the device. These patches reflect changes made to
// the `StreamingResident` table.
//
// Furthermore we might need extra information when building new clas as part
// of loading new groups within an update task.
//
// We also track how much groups & clusters were scheduled for loading, this helps us
// estimate the clas memory space we have left when handling a new request to
// load new groups. Cause at that point in time we have to estimate using
// the worst-case size for all those "yet to be built clas".
//
// Note:
// Giving back the memory of unloading tasks must be delayed until after
// an update has completed on the device, otherwise we risk taking memory
// from a frame that was scheduled in the past of the host but might still
// be executing on the device.

class StreamingUpdates
{
public:
  struct TaskInfo
  {
    uint32_t                          loadCount;
    uint32_t                          unloadCount;
    uint32_t                          newClusterCount;
    uint32_t                          loadActiveGroupsOffset;
    uint32_t                          loadActiveClustersOffset;
    uint32_t                          geometryPatchCount;
    shaderio::StreamingPatch*         loadPatches;
    shaderio::StreamingPatch*         unloadPatches;
    nvvk::BufferSubAllocation*        unloadHandles;
    shaderio::StreamingGeometryPatch* geometryPatches;
  };

  struct NewInfo
  {
    uint32_t groups   = 0;
    uint32_t clusters = 0;
  };

  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);
  void initClas(Resources& res, const StreamingConfig& config, const SceneConfig& sceneConfig);
  void deinitClas(Resources& res);
  void deinit(Resources& res);

  size_t getOperationsSize() const;
  size_t getClasOperationsSize() const;

  void reset();

  NewInfo getFutureNew(uint64_t frameIndex) const
  {
    // first get all pending counts that we don't know in which frame they end up yet,
    // but by design are guaranteed in the future of frameIndex
    NewInfo info = m_pendingNew;

    // then all scheduled work after this frame
    for(uint32_t i = 0; i < STREAMING_MAX_ACTIVE_TASKS; i++)
    {
      if(m_scheduledNewFrame[i] > frameIndex)
      {
        info.groups += m_scheduledNew[i].groups;
        info.clusters += m_scheduledNew[i].clusters;
      }
    }
    return info;
  }

  // first run update
  TaskInfo& getNewTask(uint32_t taskIndex);
  // returns number of bytes transferred
  size_t cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex);
  // then apply if upload completed
  void applyTask(shaderio::StreamingUpdate& shaderData, uint32_t taskIndex, uint32_t frameIndex);

  // later frame, must have applied task completed
  const TaskInfo& getCompletedTask(uint32_t taskIndex) const { return m_taskInfos[taskIndex]; }

private:
  nvvk::BufferTyped<shaderio::StreamingPatch> m_patchesBuffer;
  nvvk::BufferTyped<shaderio::StreamingPatch> m_patchesHostBuffer;

  std::vector<nvvk::BufferSubAllocation> m_unloadHandles;
  TaskInfo                               m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];

  shaderio::StreamingUpdate m_shaderData;

  uint32_t m_clusterCountAlignment;
  uint32_t m_scheduleIndex;
  NewInfo  m_pendingNew;
  NewInfo  m_scheduledNew[STREAMING_MAX_ACTIVE_TASKS]      = {};
  uint64_t m_scheduledNewFrame[STREAMING_MAX_ACTIVE_TASKS] = {};

  // persistent
  nvvk::Buffer m_clasBuffer;
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingStorage
//
// Storage contains the geometric data for the active resident groups
// (persistent resident groups were stored in the `SceneStreaming` class directly).
// It also contains scratch space to handle the uploads from host to device.
// The resident group has an immutable device memory location over its lifetime.
//
// Depending on the StreamingConfig uploads may be performed on an
// asynchronous transfer queue. In that case we leverage a dedicated
// nvvk::RingCommandPool to provide command buffers.

class StreamingStorage
{
public:
  struct TaskInfo
  {
    size_t usedMemory;
    size_t baseOffset;
    size_t regionCount;
  };

  void init(Resources& res, const StreamingConfig& config);
  void deinit(Resources& res);
  void reset();

  // freeing is not done during regular transfer tasks
  void free(nvvk::BufferSubAllocation& handle);

  void   getStats(StreamingStats& stats) const;
  size_t getOperationsSize() const;
  size_t getMaxDataSize() const;

  // for transfer task:
  // first get operation
  TaskInfo& getNewTask(uint32_t taskIndex);
  // first test if space is available
  bool canTransfer(const TaskInfo& operation, size_t size) const;
  // then allocate
  bool allocate(nvvk::BufferSubAllocation& handle, GeometryGroup group, size_t sz, uint64_t& deviceAddress);
  // and get transfer space
  void* appendTransfer(TaskInfo& operation, const nvvk::BufferSubAllocation& dstHandle);
  // at end of updates trigger cmd update
  // returns number of copy operations required
  uint32_t cmdUploadTask(VkCommandBuffer cmd);

  nvvk::ManagedCommandPools m_taskCommandPool;

private:
  struct CopyInfo
  {
    VkBuffer targetBuffer;
    size_t   regionOffset;
    size_t   regionCount;
  };

  size_t m_maxSceneBytes;
  size_t m_maxTransferBytes;
  size_t m_blockBytes;

  nvvk::Buffer                       m_transferHostBuffer;
  nvvk::BufferSubAllocator::InitInfo m_dataInfo;
  nvvk::BufferSubAllocator           m_dataAllocator;
  std::vector<uint32_t>              m_dataQueueFamilies;

  std::vector<CopyInfo>     m_copyInfos;
  std::vector<VkBufferCopy> m_copyRegions;

  TaskInfo m_taskOperations[STREAMING_MAX_ACTIVE_TASKS];
};

//////////////////////////////////////////////////////////////////////////
//
// StreamingTaskQueue
//
// This is the central data structure to manage the lifetime of a task
// represented by a simple "taskIndex".
// We can test if a task has completed on the device and is available,
// furthermore we can pop such available tasks or push new ones.
//
// Each task is represented using a simple "taskIndex" (we recycle these)
// and when it is pushed an appropriate timeline semaphore with timeline value
// must be provided.
//
// A task can optionally store another dependent taskIndex.
// This is useful for decoupled transfers where we don't know when the transfer
// is completed and therefore have to use a dependent update task that is triggered
// later.
//
// For a given task queue there can be only STREAMING_MAX_ACTIVE_TASKS many
// tasks in-flight at any given time.
// We use busy waits till at least one slot is available top pop (see `ensureAcquisition`) to enforce this.
// That slot can the be re-cycled after it's release.
//
// ``` cpp
// // produce
// newTaskIndex = queue.acquireTaskIndex();
// ... do stuff associating task's actual data with the index
// queue.push(newTaskIndex...);
//
// // consume
// if (queue.canPop(... ensureAcquisition))
// {
//   completedTaskIndex = pop();
//   ... do stuff getting task's actual data using the index
//   queue.releaseTaskIndex(completedTaskIndex);
// }
//
// ```


class StreamingTaskQueue
{
public:
  static_assert(STREAMING_MAX_ACTIVE_TASKS < 32);

  StreamingTaskQueue() { m_availableTaskBits = (1 << STREAMING_MAX_ACTIVE_TASKS) - 1; }

  uint32_t acquireTaskIndex()
  {
    // find available bit
    for(uint32_t i = 0; i < STREAMING_MAX_ACTIVE_TASKS; i++)
    {
      if(m_availableTaskBits & (1 << i))
      {
        m_availableTaskBits &= ~(1 << i);
        return i;
      }
    }

    return INVALID_TASK_INDEX;
  }

  void releaseTaskIndex(uint32_t index)
  {
    assert((m_availableTaskBits & (1 << index)) == 0);
    m_availableTaskBits |= (1 << index);
  }

  bool canPop(VkDevice device, bool ensureAcquisition)
  {
    if(ensureAcquisition && !m_availableTaskBits && !m_taskQueue.empty())
    {
      // if there is no task bits available we must enforce a wait,
      // cause we must guarantee to have at least one available index
      // every frame
      if(m_taskQueue.front().semaphoreState.wait(device, ~0ULL) == VK_TIMEOUT)
      {
        LOGE("Failure to wait for semaphore");
        {
          exit(-1);
        }
      }
    }

    return !m_taskQueue.empty() && m_taskQueue.front().semaphoreState.testSignaled(device);
  }

  void push(uint32_t taskIndex, nvvk::SemaphoreState semaphoreState, uint32_t dependentIndex = INVALID_TASK_INDEX)
  {
    Task task = {
        .semaphoreState = semaphoreState,
        .taskIndex      = taskIndex,
        .dependentIndex = dependentIndex,
    };
    m_taskQueue.push(task);
  }

  uint32_t pop()
  {
    uint32_t taskIndex = m_taskQueue.front().taskIndex;
    assert(taskIndex != INVALID_TASK_INDEX);
    m_taskQueue.pop();
    return taskIndex;
  }

  uint32_t popWithDependent(uint32_t& dependentIndex)
  {
    uint32_t taskIndex = m_taskQueue.front().taskIndex;
    assert(taskIndex != INVALID_TASK_INDEX);
    dependentIndex = m_taskQueue.front().dependentIndex;
    m_taskQueue.pop();
    return taskIndex;
  }

private:
  struct Task
  {
    nvvk::SemaphoreState semaphoreState;
    uint32_t             taskIndex      = INVALID_TASK_INDEX;
    uint32_t             dependentIndex = INVALID_TASK_INDEX;
  };

  std::queue<Task> m_taskQueue;
  uint32_t         m_availableTaskBits;
};  // namespace lodclusters
}  // namespace lodclusters
