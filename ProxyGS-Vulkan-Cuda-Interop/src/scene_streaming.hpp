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

#include "scene_streaming_utils.hpp"

namespace lodclusters {
/*
  The scene is not loaded fully but streamed in.
  The lowest detail clusters (along with their CLAS) are persistently loaded,
  anything else is handled dynamically.

  Streaming requests run through four stages:

  1. On the device we fill the request task details.
     During traversal missing geometry groups are appended to the request
     load array.

     see `USE_STREAMING` within `traversal_run.comp.glsl`

     After traversal any groups that haven't been accessed in a while are 
     appended to the request unload array.
     
     see `stream_agefilter_groups.comp.glsl`

     At the end of the frame we download the request to host.

  2. The request is handled on the host after checking its availability.
     It triggers the storage upload of newly loaded geometry groups.
     It also prepares an update task, which encodes the patching of the scene.

     see `SceneStreaming::handleCompletedRequest`

  3. Once the storage upload is completed, the appropriate update task is run.
     This update task actually patches the device side buffers so the loads
     and unloads become effective.
     When ray tracing is active, we will also build the CLAS of the newly loaded
     groups and handle their allocation management along with the patching.

     see `stream_update_scene.comp.glsl`.

     CLAS allocation management is done either through a persistent
     allocator system (`stream_allocator...` shader files) or through a simple
     compaction system (`stream_compaction...` shader files).

  4. After the update task is completed on the device the host can
     safely release the memory of unloaded groups. This memory is then
     recycled when we load new geometry groups in (2).

   The streaming system has quite some configurable options, mostly balancing how 
   much operations should be done within a single frame.
   There is also the ability to use an asynchronous transfer queue for the data uploads,
   otherwise we just upload on the main queue prior the patch operations.

   Lastly another major option is how the CLAS are allocated within the
   fixed size clas buffer. Since the actual size of a CLAS is only
   known on the device after it was built and the estimates from the host
   can be a lot higher. We used solutions that can be implemented
   on the device, not relying on further host readbacks but still
   trying to make efficient use of the memory based on actual sizes.

   Two options are provided, and they both first build new CLAS into
   scratch space before moving them to their resident location.

   - Compaction: 
     This simple scheme is based on a basic compaction algorithm that 
     packs all resident cluster CLAS tightly before appending newly built ones.
     This can cause bursts of high amount of memory movement and a lot of bandwidth 
     and scratch space consumption. This is despite the fact that the new cluster
     API does provide functionality for moving objects to overlapping memory destinations.
     
     We do not recommend this, but it is the easiest to get going.

     see `stream_compaction...` shader files
   
   - Persistent Allocator:
     In this option we implement a persistent memory manager on the device
     so that clas are moved only once after initial building. The system
     builds arrays of free gaps up to the maximum clas size and then
     leverages them during the allocation process.
     This avoids the big peaks in clas memory movement of the simple
     compaction.
     
     A bit array is used to track which memory arrays are in use.
     Each bit represents a power of two multiple of the alignment size
     of a clas (at time of writing on NVIDIA hardware 128 bytes).

     During the loading phase of the geometry groups appropriate gaps for
     clas storage are found for all groups and the bits they cover are set.
     Respectively, the unloading of groups will clear the bits.
     A kernel is run after unloading to build the arrays of free gaps that
     the allocation can use.

     see `stream_allocator...` shader files

*/
class SceneStreaming
{
public:
  // pointers must stay valid during lifetime
  bool init(Resources* res, const Scene* scene, const StreamingConfig& config);

  // run prior the renderer starts referencing resources
  // if true CLAS for all clusters will be built
  bool updateClasRequired(bool state);

  // tear down, safe to call without init
  void deinit();

  // reset streaming state
  void reset();

  // reload internal shaders
  bool reloadShaders()
  {
    deinitShadersAndPipelines();
    return initShadersAndPipelines();
  }

  // called by render thread
  // render thread must take care of barriers prior/after these operations
  // triggers main setup, ensures data is uploaded, unloaded etc.
  //    implicitly does "handleCompletedUpdate" and "handleCompletedStorage"
  //    explicitly calls `handleCompletedRequest`
  // barriers: none
  void cmdBeginFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, QueueState& asyncQueueState, uint32_t ageThreshold, nvvk::ProfilerGpuTimer& profiler);

  // triggers scene update & clas build if required
  // barriers: requires transfers from `cmdBeginFrame` to be completed
  void cmdPreTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, nvvk::ProfilerGpuTimer& profiler);

  // triggers age filter & clas move
  // barriers: requires compute shader writes from `cmdPreTraversal` to be completed,
  //           for ray tracing also requires acceleration structure builds
  void cmdPostTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, nvvk::ProfilerGpuTimer& profiler);

  // triggers request download
  // barriers: requires compute shader writes from `cmdPostTraversal` to be completed
  void cmdEndFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, nvvk::ProfilerGpuTimer& profiler);

  // statistics on streaming operations
  void getStats(StreamingStats& stats) const;

  // scratch space is provided by renderer so it can alias/reuse the memory for other operations
  // this is the required size
  size_t getRequiredClasScratchSize() const { return m_clasScratchTotalSize; }
  size_t getRequiredClasScratchAlginment() const { return m_clasScratchAlignment; }

  // used by renderer
  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }
  const nvvk::Buffer&                          getShaderStreamingBuffer() const { return m_shaderBuffer; }

  // device side memory usage, reserved or current state
  size_t getClasSize(bool reserved) const;
  size_t getGeometrySize(bool reserved) const;
  size_t getOperationsSize() const { return m_operationsSize + m_clasOperationsSize; }

  // for debugging one can limit the amount of streaming requests to perform
  //  < 0 means all requests are handled (regular usage)
  // == 0 stops handling requests
  //  > 0 does as many requests until decremented to zero
#ifdef _DEBUG
  static const int32_t s_defaultDebugFrameLimit = -1;
#else
  static const int32_t s_defaultDebugFrameLimit = -1;
#endif
  int32_t m_debugFrameLimit = s_defaultDebugFrameLimit;

private:
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  StreamingConfig m_config;
  bool            m_requiresClas;
  size_t          m_persistentGeometrySize;
  size_t          m_operationsSize;
  size_t          m_clasOperationsSize;
  uint32_t        m_lastUpdateIndex;
  uint32_t        m_frameIndex;
  StreamingStats  m_stats;

  // persistent scene data

  struct PersistentGeometry
  {
    nvvk::BufferTyped<shaderio::Node>     nodes;
    nvvk::BufferTyped<shaderio::BBox>     nodeBboxes;
    nvvk::BufferTyped<shaderio::LodLevel> lodLevels;
    nvvk::BufferTyped<uint64_t>           groupAddresses;
    nvvk::Buffer                          lowDetailGroupsData;
    uint32_t                              lodLevelsCount                                = 0;
    uint32_t                              lodLoadedGroupsCount[SHADERIO_MAX_LOD_LEVELS] = {};
    uint32_t                              lodGroupsCount[SHADERIO_MAX_LOD_LEVELS]       = {};
    uint32_t                              lastUpdateFrame                               = 0;
  };

  std::vector<PersistentGeometry>       m_persistentGeometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;
  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;

  void initGeometries(Resources& res, const Scene* scene);
  void resetGeometryGroupAddresses(Resources::BatchedUploader& uploader);

  // streaming system related

  // the main buffer and its content used to provide data to all compute kernels,
  // it is updated within `cmdBeginFrame`
  shaderio::SceneStreaming m_shaderData;
  nvvk::Buffer             m_shaderBuffer;

  // streaming requests run through 3 stages:
  // first a request is tasked (what geometry groups to load or unload)
  StreamingTaskQueue m_requestsTaskQueue;
  // when a request is handled it will trigger an upload of newly requested geometry data
  // it also prepares an update task to the scene
  StreamingTaskQueue m_storageTaskQueue;
  // once the storage transfer has completed an update task is run to configure
  // the scene's state making the changes of loaded and unloaded groups effective.
  StreamingTaskQueue m_updatesTaskQueue;

  // each frame we prepare a request of what geometry groups to load or unload
  StreamingRequests m_requests;
  // manages the table of resident cluster groups
  StreamingResident m_resident;
  // manages the clas memory of resident cluster groups if `usePersistentClasAllocator` was
  // set in the options
  StreamingAllocator m_clasAllocator;
  // manages the upload and storage of the cluster group geometry data
  StreamingStorage m_storage;
  // manages updates to the scene, this is where loads/unloads are becoming effective
  StreamingUpdates m_updates;

  // This is the main function where we react on streaming requests.
  // It processes the request by issuing new storage upload work and prepares the scene patching and resident update task.
  // returns the updateTaskIndex to be handled immediately in this frame if it's != INVALID_TASK_INDEX
  uint32_t handleCompletedRequest(VkCommandBuffer cmd, QueueState& cmdQueueState, QueueState& asyncQueueState, uint32_t popRequestIndex);

  // ray tracing specific
  VkClusterAccelerationStructureTriangleClusterInputNV m_clasTriangleInput;

  nvvk::Buffer m_clasLowDetailBuffer;
  size_t       m_clasLowDetailSize;

  nvvk::Buffer m_clasLowDetailBlasBuffer;

  // max size of a clas can have
  size_t m_clasSingleMaxSize;
  // max storage size of all newly clas built in a frame
  size_t m_clasScratchNewClasSize;
  // scratch space for building the new clas
  size_t m_clasScratchNewBuildSize;
  // scratch space for moving clas
  size_t m_clasScratchMoveSize;
  // total scratch space needed in a frame
  size_t m_clasScratchTotalSize;
  // alignment of scratch operations
  size_t m_clasScratchAlignment;

  bool initClas();
  void deinitClas();

  // shaders & pipelines

  struct Shaders
  {
    shaderc::SpvCompilationResult computeAgeFilterGroups;
    shaderc::SpvCompilationResult computeUpdateSceneRaster;
    shaderc::SpvCompilationResult computeUpdateSceneRay;
    shaderc::SpvCompilationResult computeSetup;

    // if usePersistentClasAllocator
    shaderc::SpvCompilationResult computeAllocatorBuildFreeGaps;
    shaderc::SpvCompilationResult computeAllocatorFreeGapsInsert;
    shaderc::SpvCompilationResult computeAllocatorSetupInsertion;
    shaderc::SpvCompilationResult computeAllocatorUnloadGroups;
    shaderc::SpvCompilationResult computeAllocatorLoadGroups;
    // else
    shaderc::SpvCompilationResult computeCompactionClasOld;
    shaderc::SpvCompilationResult computeCompactionClasNew;
  };

  struct Pipelines
  {
    VkPipeline computeAllocatorBuildFreeGaps  = nullptr;
    VkPipeline computeAllocatorFreeGapsInsert = nullptr;
    VkPipeline computeAllocatorSetupInsertion = nullptr;
    VkPipeline computeAllocatorUnloadGroups   = nullptr;
    VkPipeline computeAllocatorLoadGroups     = nullptr;

    // if usePersistentClasAllocator
    VkPipeline computeAgeFilterGroups   = nullptr;
    VkPipeline computeUpdateSceneRaster = nullptr;
    VkPipeline computeUpdateSceneRay    = nullptr;
    VkPipeline computeSetup             = nullptr;
    // else
    VkPipeline computeCompactionClasOld = nullptr;
    VkPipeline computeCompactionClasNew = nullptr;
  };

  Shaders              m_shaders;
  Pipelines            m_pipelines;
  VkPipelineLayout     m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;

  bool initShadersAndPipelines();
  void deinitShadersAndPipelines();
};
}  // namespace lodclusters
