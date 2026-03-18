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

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

namespace lodclusters {

struct DepthExportInfo {
  int memory_fd = -1;                    // VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  int timeline_semaphore_fd = -1;        // VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
  uint32_t width = 0;
  uint32_t height = 0; 
  uint32_t row_pitch_bytes = 0;
  uint64_t size = 0;
  uint64_t offset = 0;
  VkFormat format = VK_FORMAT_D24_UNORM_S8_UINT;  // Stage 4: Set proper default format
  uint64_t last_signaled_payload = 0;
};

// Complete end-to-end interop structure with three timeline semaphores
struct InteropExportInfo {
  // Memory resources
  int      depth_mem_fd         = -1;         // External memory FD (Opaque FD)
  uint64_t depth_mem_size       = 0;          // bytes
  uint64_t depth_mem_offset     = 0;          // bytes, if no sub-offset = 0

  // Timeline semaphores for coordination
  int      scene_ready_sem_fd   = -1;         // Timeline FD (Vulkan->Python)
  int      camera_ready_sem_fd  = -1;         // Timeline FD (Python->Vulkan)
  int      frame_done_sem_fd    = -1;         // Timeline FD (Vulkan->Python)

  // Image layout/format metadata
  uint32_t width                = 0;
  uint32_t height               = 0;
  uint32_t row_pitch_bytes      = 0;          // Important: row alignment
  VkFormat depth_format         = VK_FORMAT_D24_UNORM_S8_UINT;

  // Timeline values (for debugging/monitoring)
  uint64_t last_signaled_frame_done = 0;
};

struct ExternalMemoryConfig {
  bool enabled = false;
  std::string udsPath = "/tmp/vk2torch.sock";
  bool offscreen = false;
  uint32_t width = 1920;
  uint32_t height = 1080;
  VkFormat format = VK_FORMAT_D24_UNORM_S8_UINT;
  // New in-process mode support:
  VkFormat exportDepthFormat = VK_FORMAT_R32_UINT;  // 24→32bit packed depth
  bool     pack24In32 = true;                       // Python side uses &0x00FFFFFF
};

class ExternalMemoryManager {
public:
  ExternalMemoryManager() = default;
  ~ExternalMemoryManager();

  bool init(VkDevice device, VkPhysicalDevice physicalDevice, const ExternalMemoryConfig& config);
  void deinit();

  // In-process mode (no UDS socket communication)
  bool initInProcess(VkDevice device, VkPhysicalDevice physicalDevice, const ExternalMemoryConfig& config);
  int  exportDepthBufferFdDup() const;        // Returns dup(fd) for safe transfer
  int  exportFrameDoneSemaphoreFdDup() const; // Returns dup(fd) for safe transfer
  uint32_t rowPitchBytes() const;              // Real row pitch for CuPy strides
  VkExtent2D extent() const;                   // Image extent
  VkSemaphore timelineSemaphore() const;       // Timeline semaphore handle
  
  // Stage 4: Export info aggregation with cached member
  const DepthExportInfo& getDepthExportInfo() const { return m_exportInfo; }

  // Complete end-to-end interop interface
  const InteropExportInfo& getInteropInfo() const { return m_interopInfo; }

  // T3 requirements: Timeline semaphore access and payload tracking
  VkSemaphore frameDoneTimeline() const { return m_frameDoneSemaphore; }
  void setLastSignaled(uint64_t payload);

  // Timeline semaphore coordination methods
  bool signalSceneReady(uint64_t value = 1);                          // Vulkan -> Python
  bool waitSceneReady(uint64_t value, uint32_t timeout_ms = 5000);    // CPU wait for scene ready
  bool signalCameraReady(uint64_t value);                             // Python -> Vulkan (CPU signal)
  bool waitCameraReady(uint64_t value, uint64_t timeout_ns = ~0ull);  // Vulkan CPU wait timeline
  bool signalFrameDone(uint64_t value, VkQueue queue);               // Vulkan -> Python (vkQueueSubmit2)
  void setCurrentFrameValue(uint64_t v);                             // Record current frame count
  uint64_t currentFrameValue() const;

  // Create exportable resources
  bool createExportableBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                              VkBuffer* buffer, VkDeviceMemory* memory, int* fd, 
                              VkDeviceSize* actualSize = nullptr, bool* isDedicated = nullptr);
  bool createExportableTimelineSemaphore(VkSemaphore* semaphore, uint64_t initialValue, int* fd);
  
  // Export existing resources
  int exportMemoryFd(VkDeviceMemory memory);
  int exportSemaphoreFd(VkSemaphore semaphore);

  // UDS communication
  bool setupUDS();
  bool acceptClient();
  bool sendHandshakeInfo();
  bool sendFds(const std::vector<int>& fds);
  
  // Socket-based camera control
  bool sendReadyMessage();
  bool receiveCameraMatrices(float* viewMatrix, float* projMatrix);
  
  // Shared memory camera control
  bool ReadCamera32f(float out[32]);
  bool tryOpenSharedMemory();  // Try to open shared memory if not already open

  // Frame synchronization  
  bool waitForCameraReady(uint64_t frameNumber);
  
  // Camera data reading
  bool readCameraData(void* destination, size_t size);
  
  // Window mode frame protocol integration
  void addCameraWaitToSubmit(VkSubmitInfo& submitInfo, uint64_t frameNumber, VkTimelineSemaphoreSubmitInfo& timelineInfo, 
                             VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  void addFrameDoneSignalToSubmit(VkSubmitInfo& submitInfo, uint64_t frameNumber, VkTimelineSemaphoreSubmitInfo& timelineInfo);
  
  void cmdCopyImageToColorBuffer(VkCommandBuffer cmd,
                                                        VkImage srcImage,
                                                        VkImageLayout currentLayout, // 传"真实当前布局"
                                                        uint32_t width, uint32_t height);

  void cmdCopyDepthToBuffer(VkCommandBuffer cmd,
                            VkImage         depthImage,
                            VkImageLayout   depthOldLayout,
                            uint32_t        width,
                            uint32_t        height,
                            VkImageLayout   restoreLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

  // Frame counter management
  uint64_t getNextFrameNumber() { 
    if (m_currentFrameNumber == 0) {
      // Skip frame 0 since Python starts at 1
      m_currentFrameNumber = 1;
    }
    return m_currentFrameNumber++; 
  }
  uint64_t getCurrentFrameNumber() const { return m_currentFrameNumber; }

  // Getters
  VkBuffer getCameraBuffer() const { return m_cameraBuffer; }
  VkBuffer getColorReadbackBuffer() const { return m_colorReadbackBuffer; }
  VkSemaphore getCameraSemaphore() const { return m_cameraSemaphore; }
  VkSemaphore getFrameDoneSemaphore() const { return m_frameDoneSemaphore; }
  
  bool isConnected() const;  // Check if client is still connected
  uint32_t getWidth() const { return m_config.width; }
  uint32_t getHeight() const { return m_config.height; }
  VkFormat getFormat() const { return m_config.format; }
  
  // Semaphore test support
  void startSemaphoreEchoThread();
  void stopSemaphoreEchoThread();

  // Vulkan device extensions required
  static std::vector<const char*> getRequiredDeviceExtensions() {
    return {
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
      VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
    };
  }

private:
  VkDevice m_device = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  ExternalMemoryConfig m_config;

  // Exportable resources
  VkBuffer m_cameraBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_cameraMemory = VK_NULL_HANDLE;
  
  VkBuffer m_colorReadbackBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_colorReadbackMemory = VK_NULL_HANDLE;
  
  // Depth buffer for in-process mode  
  VkBuffer m_depthReadbackBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_depthReadbackMemory = VK_NULL_HANDLE;
  VkDeviceSize m_depthBufferSize = 0;
  bool m_depthBufferDedicated = false;
  
  VkSemaphore m_cameraSemaphore = VK_NULL_HANDLE;
  VkSemaphore m_frameDoneSemaphore = VK_NULL_HANDLE;

  // UDS socket
  int m_serverSocket = -1;
  int m_clientSocket = -1;

  // Actual allocated sizes (may be larger than requested due to alignment)
  VkDeviceSize m_cameraBufferSize = 0;
  VkDeviceSize m_colorBufferSize = 0;
  
  // Track if buffers use dedicated allocation
  bool m_cameraBufferDedicated = false;
  bool m_colorBufferDedicated = false;
  
  // Memory mapped pointers for direct access
  void* m_cameraBufferMapped = nullptr;
  void* m_colorBufferMapped = nullptr;
  
  // Semaphore echo thread
  std::thread* m_echoThread = nullptr;
  std::atomic<bool> m_echoThreadRunning{false};
  
  // Frame counter for window mode protocol
  uint64_t m_currentFrameNumber = 1;
  
  // Frame number channel for PyBridge/LodClusters coordination
  mutable std::mutex m_frameValueMutex;
  uint64_t m_currentFrameValue = 1;  // Shared frame value for timeline signals
  
  // In-process mode file descriptors (stored for dup() export)
  int m_depthBufferFd = -1;
  int m_frameDoneSemaphoreFd = -1;
  uint32_t m_actualRowPitch = 0;  // Actual row pitch in bytes
  bool m_inProcessMode = false;   // Track if initialized in in-process mode
  
  // T3 requirement: Track last signaled timeline payload
  uint64_t m_lastSignaledPayload = 0;
  
  // Stage 4: Cached export info for efficient access
  mutable DepthExportInfo m_exportInfo;
  
  // Complete end-to-end interop info with three timeline semaphores
  mutable InteropExportInfo m_interopInfo;
  
  // Timeline semaphores for complete coordination
  VkSemaphore m_sceneReadyTimeline = VK_NULL_HANDLE;   // Vulkan->Python (scene ready)
  VkSemaphore m_cameraReadyTimeline = VK_NULL_HANDLE;  // Python->Vulkan (camera ready)
  // Note: m_frameDoneSemaphore already exists for frame done signaling
  
  // Shared memory for camera matrices
  int m_shmFd = -1;
  void* m_shmPtr = nullptr;
  static const size_t m_shmSize = 256;  // 256 bytes total
  static constexpr const char* m_shmName = "/py2vk_cam32f";

  // Helper functions
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
  uint32_t findMemoryTypeWithExport(uint32_t typeFilter, VkMemoryPropertyFlags properties, 
                                     VkExternalMemoryHandleTypeFlagBits handleType);
  bool checkExtensionSupport(const char* extensionName);
  void semaphoreEchoWorker();
  
  // Stage 4: Helper to populate export info when resources are ready
  void updateExportInfo();
  void updateInteropInfo();  // Update complete interop info with all three semaphores
};

} // namespace lodclusters