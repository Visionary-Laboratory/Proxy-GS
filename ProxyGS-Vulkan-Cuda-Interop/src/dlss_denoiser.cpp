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

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/debug_util.hpp>
#include <imgui/imgui.h>

#include "dlss_denoiser.hpp"

bool DlssDenoiser::ensureInitialized()
{
  if(!m_initialized)
  {
    initDenoiser();
    return true;
  }
  return false;
}

void DlssDenoiser::init(const InitInfo& info)
{
  assert(m_info.resourceAllocator == nullptr);

  m_info = info;
  if(info.samplerPool)
  {
    info.samplerPool->acquireSampler(m_sampler);
  }
}

void DlssDenoiser::deinit()
{
  if(m_info.resourceAllocator != nullptr)
  {
    m_dlssGBuffers.deinit();
    m_dlss.deinit();
    m_ngx.deinit();
    m_initialized  = false;
    m_hasResources = false;
    if(m_info.samplerPool)
    {
      //m_info.samplerPool->releaseSampler(m_sampler);
    }

    m_info = {};
  }
}

void DlssDenoiser::deinitResources()
{
  m_dlssGBuffers.deinit();
  m_dlss.deinit();
  m_hasResources = false;
}

void DlssDenoiser::initDenoiser()
{
  if(m_initialized)
    return;
  SCOPED_TIMER("Initializing DLSS Denoiser");

  m_device = m_info.resourceAllocator->getDevice();

  // #DLSS - Create the DLSS
  NgxContext::InitInfo ngxInitInfo{
      .instance       = m_info.instance,
      .physicalDevice = m_info.resourceAllocator->getPhysicalDevice(),
      .device         = m_info.resourceAllocator->getDevice(),
  };
  // ngxInitInfo.loggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;

  NVSDK_NGX_Result ngxResult = m_ngx.init(ngxInitInfo);
  if(ngxResult == NVSDK_NGX_Result_Success)
  {
    m_dlssSupported = (m_ngx.isDlssRRAvailable() == NVSDK_NGX_Result_Success);
  }

  if(!m_dlssSupported)
  {
    LOGW("NGX init failed: %d - DLSS unsupported\n", ngxResult);
  }
  m_initialized = true;
}


VkDescriptorImageInfo DlssDenoiser::getDescriptorImageInfo(DlssBufferType name) const
{
  assert(m_hasResources);
  return m_dlssGBuffers.getDescriptorImageInfo(name);
}

bool DlssDenoiser::isAvailable() const
{
  return m_dlssSupported && m_initialized;
}

VkExtent2D DlssDenoiser::updateSize(VkCommandBuffer cmd, VkExtent2D size, NVSDK_NGX_PerfQuality_Value quality)
{
  if(!isAvailable())
    return size;

  // Choose the size of the DLSS buffers
  DlssRayReconstruction::SupportedSizes supportedSizes{};
  DlssRayReconstruction::querySupportedInputSizes(m_ngx, {size, quality}, &supportedSizes);
  m_renderingSize = supportedSizes.optimalSize;

  DlssRayReconstruction::InitInfo initInfo{
      .inputSize  = m_renderingSize,
      .outputSize = size,
  };
  m_dlss.deinit();
  vkDeviceWaitIdle(m_device);
  m_dlss.cmdInit(cmd, m_ngx, initInfo);

  if(!m_hasResources)
  {
    // G-Buffer
    m_dlssGBuffers.init({.allocator      = m_info.resourceAllocator,
                         .colorFormats   = m_bufferInfos,
                         .imageSampler   = m_sampler,
                         .descriptorPool = m_info.descriptorPool});
  }

  // Recreate the G-Buffers
  m_dlssGBuffers.update(cmd, m_renderingSize);
  m_hasResources = true;

  auto dlssResourceFromGBufTexture = [&](DlssRayReconstruction::ResourceType resource, DlssBufferType gbufIndex) {
    m_dlss.setResource({resource, m_dlssGBuffers.getColorImage(gbufIndex), m_dlssGBuffers.getColorImageView(gbufIndex),
                        m_dlssGBuffers.getColorFormat(gbufIndex)});
  };

  // #DLSS Fill the user pool with our textures
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eColorIn, DlssBufferType::eDlssRenderImage);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eDiffuseAlbedo, DlssBufferType::eDlssAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eSpecularAlbedo, DlssBufferType::eDlssSpecAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eNormalRoughness, DlssBufferType::eDlssNormalRoughness);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eMotionVector, DlssBufferType::eDlssMotion);

  return m_renderingSize;
}

void DlssDenoiser::setResource(DlssRayReconstruction::ResourceType resourceId, VkImage image, VkImageView imageView, VkFormat format)
{
  m_dlss.setResource({resourceId, image, imageView, format});
}

void DlssDenoiser::denoise(VkCommandBuffer cmd, glm::vec2 jitter, const glm::mat4& modelView, const glm::mat4& projection, bool reset /*= false*/)
{
  m_dlss.cmdDenoise(cmd, m_ngx, {jitter, modelView, projection, reset});
}

void DlssDenoiser::onUi()
{
  if(!isAvailable() || !m_hasResources)
  {
    return;
  }

  ImVec2 tumbnailSize = {100 * m_dlssGBuffers.getAspectRatio(), 100};
  int    m_showBuffer = -1;
  auto   showBuffer   = [&](const char* name, DlssBufferType buffer) {
    ImGui::Text("%s", name);
    if(ImGui::ImageButton(name, ImTextureID(m_dlssGBuffers.getDescriptorSet(buffer)), tumbnailSize))
      m_showBuffer = buffer;
  };

  if(ImGui::CollapsingHeader("DLSS Guide Images"))
  {
    if(ImGui::BeginTable("thumbnails", 2))
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      showBuffer("Color", DlssBufferType::eDlssAlbedo);
      ImGui::TableNextColumn();
      showBuffer("Normal", DlssBufferType::eDlssNormalRoughness);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      showBuffer("Motion", DlssBufferType::eDlssMotion);
      ImGui::TableNextColumn();
      showBuffer("Specular Albedo", DlssBufferType::eDlssSpecAlbedo);
      ImGui::EndTable();
    }
  }
}