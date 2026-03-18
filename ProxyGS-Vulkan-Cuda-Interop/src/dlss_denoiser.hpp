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

// This class adds support for DLSS denoising
// It initialize the NGX and create the G-Buffers for the denoiser
// It also provides the descriptor set for the denoiser

#include <glm/glm.hpp>


#include "../shaders/dlss_util.h"

#include "nvvk/gbuffers.hpp"
#include "nvvk/resource_allocator.hpp"
#include "nvvk/sampler_pool.hpp"
#include "nvutils/parameter_registry.hpp"

// #DLSS
#include "dlss_wrapper.hpp"
#include "nvsdk_ngx_helpers_vk.h"

class DlssDenoiser
{
public:
  enum DlssBufferType
  {
    eDlssRenderImage     = SHADERIO_eDlssRenderImage,
    eDlssAlbedo          = SHADERIO_eDlssAlbedo,
    eDlssSpecAlbedo      = SHADERIO_eDlssSpecAlbedo,
    eDlssNormalRoughness = SHADERIO_eDlssNormalRoughness,
    eDlssMotion          = SHADERIO_eDlssMotion,
    eDlssCount,
  };

  struct InitInfo
  {
    VkInstance               instance{};
    nvvk::ResourceAllocator* resourceAllocator{};

    // optional for UI/imgui
    nvvk::SamplerPool* samplerPool{};
    VkDescriptorPool   descriptorPool{};
  };

  struct Settings
  {
    bool enable = false;
  };


  DlssDenoiser()  = default;
  ~DlssDenoiser() = default;

  void init(const InitInfo& initInfo);
  void deinit();

  void deinitResources();

  void initDenoiser();
  // Return the descriptor info for the DLSS buffer
  VkDescriptorImageInfo getDescriptorImageInfo(DlssBufferType name) const;

  // Return if DLSS can be used
  bool isAvailable() const;

  // When the size of the rendering changes, we need to update the DLSS buffers
  // also calls setResource for all internal buffers
  VkExtent2D updateSize(VkCommandBuffer cmd, VkExtent2D size, NVSDK_NGX_PerfQuality_Value quality);

  // To be called externally only for `DlssRayReconstruction::ResourceType::eColorOut`
  void setResource(DlssRayReconstruction::ResourceType resourceId, VkImage image, VkImageView imageView, VkFormat format);

  // This is the actual denoising call
  void denoise(VkCommandBuffer cmd, glm::vec2 jitter, const glm::mat4& modelView, const glm::mat4& projection, bool reset = false);

  // This is for the debug UI
  void onUi();

  // Return the render size
  VkExtent2D getRenderSize() const { return m_dlssGBuffers.getSize(); }

  const nvvk::GBuffer& getGBuffers() { return m_dlssGBuffers; }

  bool ensureInitialized();

private:
  InitInfo m_info{};
  Settings m_settings{};
  bool     m_initialized  = false;
  bool     m_hasResources = false;

  // #DLSS - Wrapper for DLSS
  NgxContext            m_ngx{};
  DlssRayReconstruction m_dlss{};

  std::vector<VkFormat> m_bufferInfos = {
      {VK_FORMAT_R8G8B8A8_UNORM},       // #DLSS - Rendered image       : eDlssRenderImage
      {VK_FORMAT_R8G8B8A8_UNORM},       // #DLSS - BaseColor            : eDlssAlbedo
      {VK_FORMAT_R16G16B16A16_SFLOAT},  // #DLSS - SpecAlbedo           : eDlssSpecAlbedo
      {VK_FORMAT_R16G16B16A16_SFLOAT},  // #DLSS - Normal / Roughness   : eDlssNormalRoughness
      {VK_FORMAT_R16G16_SFLOAT},        // #DLSS - Motion vectors       : eDlssMotion
  };

  nvvk::GBuffer m_dlssGBuffers{};  // G-Buffers: for denoising
  bool          m_dlssSupported = false;
  VkExtent2D    m_renderingSize{};
  VkDevice      m_device{};
  VkSampler     m_sampler{};
};
