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

#include <span>

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <glm/glm.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvkglsl/glsl.hpp>
#include <vk_radix_sort.h>
#include <filesystem>

#if VK_HEADER_VERSION < 309
#error Update Vulkan SDK >= 1.4.309.0
#endif

#if USE_DLSS
#include "dlss_denoiser.hpp"
#endif

#include "hbao_pass.hpp"
#include "nvhiz_vk.hpp"
#include "../shaders/shaderio.h"

namespace lodclusters {

class ExternalMemoryManager;

struct FrameConfig
{
  VkExtent2D windowSize;

  bool  showInstanceBboxes = false;
  bool  freezeCulling      = false;
  bool  useHizOcclusion    = false;
  bool  hbaoActive         = true;
  float lodPixelError      = 1.0f;
  // increase error by this for instances not having primary visibility in ray tracing
  float culledErrorScale = 2.0f;
  // how many frames until we schedule a group for unloading
  uint32_t streamingAgeThreshold = 16;
  // how much threads to use in the persistent kernels
  uint32_t traversalPersistentThreads = 2048;

  uint32_t sharingMinInstances   = 2;
  uint32_t sharingToleranceLevel = 2;
  uint32_t sharingMinLevel       = 1;
  bool     sharingPushCulled     = true;

  HbaoPass::Settings hbaoSettings;

  uint32_t visualize = VISUALIZE_LOD;

  // External memory manager for Python integration
  ExternalMemoryManager* externalMemoryManager = nullptr;

  // must be kept next to each other
  shaderio::FrameConstants frameConstants;
  shaderio::FrameConstants frameConstantsLast;
};

//////////////////////////////////////////////////////////////////////////

struct PhysicalDeviceInfo
{
  VkPhysicalDeviceProperties         properties10;
  VkPhysicalDeviceVulkan11Properties properties11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceVulkan12Properties properties12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
  VkPhysicalDeviceVulkan13Properties properties13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceVulkan14Properties properties14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES};

  VkPhysicalDeviceFeatures         features10;
  VkPhysicalDeviceVulkan11Features features11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features features12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan14Features features14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};

  VkPhysicalDeviceMemoryProperties memoryProperties = {};

  void init(VkPhysicalDevice physicalDevice, uint32_t apiVersion = VK_API_VERSION_1_4)
  {
    assert(apiVersion >= VK_API_VERSION_1_2);

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext                       = &properties11;
    properties11.pNext                = &properties12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      properties12.pNext = &properties13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      properties13.pNext = &properties14;
    }
    vkGetPhysicalDeviceProperties2(physicalDevice, &props);
    properties10 = props.properties;

    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext                     = &features11;
    features11.pNext                   = &features12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      features12.pNext = &features13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      features13.pNext = &features14;
    }
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
    features10 = features.features;
  }
};

inline void cmdCopyBuffer(VkCommandBuffer cmd, const nvvk::Buffer& src, const nvvk::Buffer& dst)
{
  VkBufferCopy cpy = {0, 0, src.bufferSize};
  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}

//////////////////////////////////////////////////////////////////////////

std::string formatMemorySize(size_t sizeInBytes);

inline size_t logMemoryUsage(size_t size, const char* memtype, const char* what)
{
  LOGI("%s memory: %s - %s\n", memtype, formatMemorySize(size).c_str(), what);
  return size;
}

//////////////////////////////////////////////////////////////////////////

struct BufferRanges
{
  VkDeviceSize tempOffset = 0;

  VkDeviceSize beginOffset = 0;
  VkDeviceSize splitOffset = 0;

  VkDeviceSize append(VkDeviceSize size, VkDeviceSize alignment)
  {
    tempOffset = nvutils::align_up(tempOffset, alignment);

    VkDeviceSize offset = tempOffset;
    tempOffset += size;

    return offset;
  }

  void beginOverlap()
  {
    beginOffset = tempOffset;
    splitOffset = 0;
  }
  void splitOverlap()
  {
    splitOffset = std::max(splitOffset, tempOffset);
    tempOffset  = beginOffset;
  }
  void endOverlap() { tempOffset = std::max(splitOffset, tempOffset); }

  VkDeviceSize getSize(VkDeviceSize alignment = 4) { return nvutils::align_up(tempOffset, alignment); }
};

//////////////////////////////////////////////////////////////////////////

class QueueState
{
public:
  VkDevice    m_device            = nullptr;
  VkQueue     m_queue             = nullptr;
  uint32_t    m_familyIndex       = 0;
  VkSemaphore m_timelineSemaphore = nullptr;
  uint64_t    m_timelineValue     = 1;

  std::vector<VkSemaphoreSubmitInfo> m_pendingWaits;

  void init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue);
  void deinit();

  VkResult getTimelineValue(uint64_t& timelineValue) const
  {
    return vkGetSemaphoreCounterValue(m_device, m_timelineSemaphore, &timelineValue);
  }

  nvvk::SemaphoreState getCurrentState() const
  {
    return nvvk::SemaphoreState::makeFixed(m_timelineSemaphore, m_timelineValue);
  }

  VkSemaphoreSubmitInfo getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0) const;

  // increments timeline
  VkSemaphoreSubmitInfo advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0);
};

struct QueueStateManager
{
  QueueState primary;
  QueueState transfer;
};

//////////////////////////////////////////////////////////////////////////

class Resources
{
public:
  static constexpr VkPipelineStageFlags2 ALL_SHADER_STAGES =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;


  struct FrameBuffer
  {
    VkExtent2D renderSize{};
    VkExtent2D targetSize{};
    VkExtent2D windowSize{};

    int  supersample = 0;
    bool useResolved = false;

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthStencilFormat;

    VkViewport viewport;
    VkRect2D   scissor;

    nvvk::Image imgColor         = {};
    nvvk::Image imgColorResolved = {};
    nvvk::Image imgDepthStencil  = {};

    VkImageView viewDepth = VK_NULL_HANDLE;

    VkFormat    raytracingDepthFormat = VK_FORMAT_R32_SFLOAT;
    nvvk::Image imgRaytracingDepth    = {};

    nvvk::Image imgHizFar = {};

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

#if USE_DLSS
    bool                        hasDenoiser  = false;
    DlssDenoiser                dlssDenoiser = {};
    NVSDK_NGX_PerfQuality_Value dlssQuality  = NVSDK_NGX_PerfQuality_Value(-1);
#endif
  };

  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, const nvvk::QueueInfo& queue, const nvvk::QueueInfo& queueTransfer);
  void deinit();
  void setAssetRoot(const std::filesystem::path& assetRoot);

  bool initFramebuffer(const VkExtent2D& windowSize, int supersample, bool hbaoFullRes);
  void updateFramebufferRenderSizeDependent(VkCommandBuffer cmd);
#if USE_DLSS
  void updateFramebufferDlss(VkCommandBuffer cmd);
  void setFramebufferDlss(bool enabled, NVSDK_NGX_PerfQuality_Value dlssQuality);
#endif
  void deinitFramebufferRenderSizeDependent();
  void deinitFramebuffer();

  glm::vec2 getFramebufferWindow2RenderScale() const;

  void beginFrame(uint32_t cycleIndex);
  void postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void endFrame();

  void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void cmdHBAO(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);

  void getReadbackData(shaderio::Readback& readback);

  //////////////////////////////////////////////////////////////////////////

  shaderc::CompileOptions makeCompilerOptions() { return shaderc::CompileOptions(m_glslCompiler.options()); }

  bool compileShader(shaderc::SpvCompilationResult& compiled,
                     VkShaderStageFlagBits          shader,
                     const std::filesystem::path&   filePath,
                     shaderc::CompileOptions*       options = nullptr);

  // tests if all shaders compiled well, returns false if not
  // also destroys all shaders if not all were successful.
  bool verifyShaders(size_t numShaders, shaderc::SpvCompilationResult* shaders)
  {
    for(size_t i = 0; i < numShaders; i++)
    {
      if(shaders[i].GetCompilationStatus() != shaderc_compilation_status_null_result_object
         && shaders[i].GetCompilationStatus() != shaderc_compilation_status_success)
        return false;
    }

    return true;
  }
  template <typename T>
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(shaderc::SpvCompilationResult), (shaderc::SpvCompilationResult*)&container);
  }

  void destroyPipelines(size_t numPipelines, VkPipeline* pipelines)
  {
    for(size_t i = 0; i < numPipelines; i++)
    {
      vkDestroyPipeline(m_device, pipelines[i], nullptr);
      pipelines[i] = nullptr;
    }
  }
  template <typename T>
  void destroyPipelines(T& container)
  {
    destroyPipelines(sizeof(T) / sizeof(VkPipeline), (VkPipeline*)&container);
  }

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createTempCmdBuffer();
  void            tempSyncSubmit(VkCommandBuffer cmd);

  //////////////////////////////////////////////////////////////////////////

  void cmdBeginRendering(VkCommandBuffer    cmd,
                         bool               hasSecondary = false,
                         VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR);
  void cmdBeginRayTracing(VkCommandBuffer cmd);

  void cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;

  //////////////////////////////////////////////////////////////////////////

  template <typename T>
  VkResult createBufferTyped(nvvk::BufferTyped<T>&     buffer,
                             size_t                    elementCount,
                             VkBufferUsageFlagBits2    bufferUsageFlags,
                             VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                             VmaAllocationCreateFlags  vmaAllocFlags = 0,
                             VkDeviceSize              minAlignment  = 0,
                             std::span<const uint32_t> queueFamilies = {})
  {
    return m_allocator.createBuffer(buffer, elementCount * nvvk::BufferTyped<T>::value_size, bufferUsageFlags,
                                    vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }

  VkResult createBuffer(nvvk::Buffer&             buffer,
                        VkDeviceSize              bufferSize,
                        VkBufferUsageFlagBits2    bufferUsageFlags,
                        VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                        VmaAllocationCreateFlags  vmaAllocFlags = 0,
                        VkDeviceSize              minAlignment  = 0,
                        std::span<const uint32_t> queueFamilies = {})
  {
    return m_allocator.createBuffer(buffer, bufferSize, bufferUsageFlags, vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }

  VkResult createLargeBuffer(nvvk::LargeBuffer& buffer, VkDeviceSize bufferSize, VkBufferUsageFlagBits2 bufferUsageFlags)
  {
    return m_allocator.createLargeBuffer(buffer, bufferSize, bufferUsageFlags, m_queue.queue);
  }

  VkDeviceSize getDeviceLocalHeapSize() const;

  bool isBufferSizeValid(VkDeviceSize size) const;

  //////////////////////////////////////////////////////////////////////////

  void simpleUploadBuffer(const nvvk::Buffer& buffer, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, 0, buffer.bufferSize, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  void simpleUploadBuffer(const nvvk::Buffer& buffer, size_t offset, size_t sz, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, offset, sz, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  enum FlushState
  {
    ALLOW_FLUSH,
    DONT_FLUSH,
  };

  class BatchedUploader
  {
  public:
    BatchedUploader(Resources& resources, VkDeviceSize maxBatchSize = 128 * 1024 * 1024)
        : m_resources(resources)
        , m_maxBatchSize(maxBatchSize)
    {
    }

    VkCommandBuffer getCmd()
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      return m_cmd;
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, size_t offset, size_t sz, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      if(sz)
      {
        if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz) && flushState == FlushState::ALLOW_FLUSH)
        {
          flush();
        }

        if(!m_cmd)
        {
          m_cmd = m_resources.createTempCmdBuffer();
        }
        T* mapping = nullptr;
        NVVK_CHECK(m_resources.m_uploader.appendBufferMapping(dst, offset, sz, mapping));

        if(src)
        {
          memcpy(mapping, src, sz);
        }

        return mapping;
      }
      return nullptr;
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      return uploadBuffer(dst, 0, dst.bufferSize, src, flushState);
    }

    void fillBuffer(const nvvk::Buffer& dst, uint32_t fillValue)
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      vkCmdFillBuffer(m_cmd, dst.buffer, 0, dst.bufferSize, fillValue);
    }

    // must call flush at end of operations
    void flush()
    {
      if(m_cmd)
      {
        m_resources.m_uploader.cmdUploadAppended(m_cmd);
        m_resources.tempSyncSubmit(m_cmd);
        m_resources.m_uploader.releaseStaging();
        m_cmd = nullptr;
      }
    }

    void abort()
    {
      m_resources.m_uploader.cancelAppended();
      m_resources.m_uploader.releaseStaging();
    }

    ~BatchedUploader() { assert(!m_cmd && "must call flush at end"); }

  private:
    Resources&      m_resources;
    VkDeviceSize    m_maxBatchSize = 0;
    VkCommandBuffer m_cmd          = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////

  static constexpr VkPipelineStageFlags2 s_supportedShaderStages =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

  VkDevice         m_device          = {};
  VkPhysicalDevice m_physicalDevice  = {};
  nvvk::QueueInfo  m_queue           = {};
  nvvk::QueueInfo  m_queueTransfer   = {};
  VkCommandPool    m_tempCommandPool = {};

  nvvk::ResourceAllocator m_allocator     = {};
  nvvk::SamplerPool       m_samplerPool   = {};
  VkSampler               m_samplerLinear = {};
  nvvkglsl::GlslCompiler  m_glslCompiler  = {};
  nvvk::StagingUploader   m_uploader      = {};

  FrameBuffer m_frameBuffer;
  struct CommonBuffers
  {
    nvvk::BufferTyped<shaderio::FrameConstants> frameConstants;
    nvvk::BufferTyped<shaderio::Readback>       readBack;
    nvvk::BufferTyped<shaderio::Readback>       readBackHost;
  } m_commonBuffers;

  PhysicalDeviceInfo          m_physicalDeviceInfo = {};
  nvvk::GraphicsPipelineState m_basicGraphicsState = {};
  uint32_t                    m_cycleIndex         = 0;
  size_t                      m_fboChangeID        = ~0;
  glm::vec4                   m_bgColor            = {0.1, 0.13, 0.15, 1.0};

  bool m_supportsClusters = false;
  bool m_dumpSpirv        = false;

  bool            m_hbaoFullRes = false;
  HbaoPass        m_hbaoPass;
  HbaoPass::Frame m_hbaoFrame;
  
  // Asset root for shader loading
  std::filesystem::path m_assetRoot;

  NVHizVK                       m_hiz;
  NVHizVK::Update               m_hizUpdate;
  shaderc::SpvCompilationResult m_hizShaders[NVHizVK::SHADER_COUNT];

  QueueStateManager m_queueStates;
  VrdxSorter        m_vrdxSorter{};

private:
};


}  // namespace lodclusters
