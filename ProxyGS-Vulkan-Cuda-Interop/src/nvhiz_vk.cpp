/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>

#include <volk.h>
#include <fmt/format.h>

#include "nvhiz_vk.hpp"

static const VkFormat NVHIZ_FORMAT = VK_FORMAT_R32_SFLOAT;

void NVHizVK::TextureInfo::getShaderFactors(float factors[4]) const
{
  factors[0] = float(usedWidth) / float(width);
  factors[1] = float(usedHeight) / float(height);
  factors[2] = float(usedWidth - 2) / float(width);
  factors[3] = float(usedHeight - 2) / float(height);
}

float NVHizVK::TextureInfo::getSizeMax() const
{
  return float(std::max(width, height));
}

void NVHizVK::deinit()
{
  if(!m_device)
    return;

  deinitPipelines();

  vkDestroySampler(m_device, m_readDepthSampler, nullptr);
  vkDestroySampler(m_device, m_readFarSampler, nullptr);
  vkDestroySampler(m_device, m_readNearSampler, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descrLayout, nullptr);

  if(m_descrSetsCount)
  {
    vkDestroyDescriptorPool(m_device, m_descrPool, nullptr);
    delete[] m_descrSets;
  }

  *this = {};
}


void NVHizVK::init(VkDevice device, const Config& config, uint32_t descrSetsCount)
{
  deinit();

  VkResult result;

  m_device = device;

  m_descrSetsCount = descrSetsCount;

  (Config&)m_config = config;

  if(m_config.supportsSubGroupShuffle)
  {
    m_config.hizLevels = 3;
  }

  {
    VkSamplerReductionModeCreateInfo infoReduc = {};
    infoReduc.sType                            = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
    infoReduc.reductionMode = m_config.reversedZ ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX;

    VkSamplerCreateInfo info     = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    info.anisotropyEnable        = VK_FALSE;
    info.maxAnisotropy           = 0;
    info.flags                   = 0;
    info.compareEnable           = VK_FALSE;
    info.compareOp               = VK_COMPARE_OP_ALWAYS;
    info.unnormalizedCoordinates = VK_FALSE;
    info.mipLodBias              = 0;
    info.minLod                  = 0;
    info.maxLod                  = 16.0f;
    info.minFilter               = m_config.msaaSamples ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    info.magFilter               = m_config.msaaSamples ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    info.pNext = !m_config.msaaSamples && m_config.supportsMinmaxFilter ? &infoReduc : nullptr;

    result = vkCreateSampler(m_device, &info, nullptr, &m_readDepthSampler);
    assert(result == VK_SUCCESS);

    info.pNext     = m_config.supportsMinmaxFilter ? &infoReduc : nullptr;
    info.minFilter = VK_FILTER_LINEAR;
    info.magFilter = VK_FILTER_LINEAR;

    result = vkCreateSampler(m_device, &info, nullptr, &m_readFarSampler);
    assert(result == VK_SUCCESS);

    info.pNext      = nullptr;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.minFilter  = VK_FILTER_NEAREST;
    info.magFilter  = VK_FILTER_NEAREST;
    result          = vkCreateSampler(m_device, &info, nullptr, &m_readNearSampler);
    assert(result == VK_SUCCESS);
  }

  {
    VkDescriptorSetLayoutBinding bindings[BINDING_COUNT];
    bindings[BINDING_READ_DEPTH].binding            = BINDING_READ_DEPTH;
    bindings[BINDING_READ_DEPTH].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[BINDING_READ_DEPTH].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[BINDING_READ_DEPTH].descriptorCount    = 1;
    bindings[BINDING_READ_DEPTH].pImmutableSamplers = nullptr;

    bindings[BINDING_READ_FAR].binding            = BINDING_READ_FAR;
    bindings[BINDING_READ_FAR].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[BINDING_READ_FAR].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[BINDING_READ_FAR].descriptorCount    = 1;
    bindings[BINDING_READ_FAR].pImmutableSamplers = nullptr;

    bindings[BINDING_WRITE_NEAR].binding            = BINDING_WRITE_NEAR;
    bindings[BINDING_WRITE_NEAR].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[BINDING_WRITE_NEAR].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[BINDING_WRITE_NEAR].descriptorCount    = 1;
    bindings[BINDING_WRITE_NEAR].pImmutableSamplers = nullptr;

    bindings[BINDING_WRITE_FAR].binding            = BINDING_WRITE_FAR;
    bindings[BINDING_WRITE_FAR].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[BINDING_WRITE_FAR].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[BINDING_WRITE_FAR].descriptorCount    = MAX_MIP_LEVELS;
    bindings[BINDING_WRITE_FAR].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount                    = BINDING_COUNT;
    info.pBindings                       = bindings;

    result = vkCreateDescriptorSetLayout(m_device, &info, nullptr, &m_descrLayout);
    assert(result == VK_SUCCESS);

    m_poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    m_poolSizes[0].descriptorCount = 2;
    m_poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    m_poolSizes[1].descriptorCount = 1 + MAX_MIP_LEVELS;
  }

  if(m_descrSetsCount)
  {
    m_descrSets = new VkDescriptorSet[m_descrSetsCount];

    VkDescriptorPoolCreateInfo info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.poolSizeCount              = 2;
    info.pPoolSizes                 = m_poolSizes;
    info.maxSets                    = m_descrSetsCount;
    result                          = vkCreateDescriptorPool(m_device, &info, nullptr, &m_descrPool);
    assert(result == VK_SUCCESS);

    VkDescriptorSetLayout* setLayouts = new VkDescriptorSetLayout[m_descrSetsCount];
    for(uint32_t i = 0; i < m_descrSetsCount; i++)
    {
      setLayouts[i] = m_descrLayout;
    }

    VkDescriptorSetAllocateInfo allocateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool              = m_descrPool;
    allocateInfo.descriptorSetCount          = m_descrSetsCount;
    allocateInfo.pSetLayouts                 = setLayouts;
    result                                   = vkAllocateDescriptorSets(m_device, &allocateInfo, m_descrSets);
    assert(result == VK_SUCCESS);

    delete[] setLayouts;
  }

  {
    VkPushConstantRange range;
    range.offset     = 0;
    range.size       = sizeof(PushConstants);
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.pSetLayouts                = &m_descrLayout;
    info.setLayoutCount             = 1;
    info.pPushConstantRanges        = &range;
    info.pushConstantRangeCount     = 1;

    result = vkCreatePipelineLayout(m_device, &info, nullptr, &m_pipelineLayout);
    assert(result == VK_SUCCESS);
  }
}

VkSampler NVHizVK::getReadFarSampler() const
{
  return m_readFarSampler;
}

const VkDescriptorPoolSize* NVHizVK::getDescriptorPoolSizes(uint32_t& count) const
{
  count = POOLSIZE_COUNT;
  return m_poolSizes;
}

VkDescriptorSetLayout NVHizVK::getDescriptorSetLayout() const
{
  return m_descrLayout;
}

void NVHizVK::appendShaderDefines(uint32_t shader, shaderc::CompileOptions& options) const
{
  ProgHizMode  hiz;
  ProgViewMode view;
  getShaderIndexConfig(shader, hiz, view);

  options.AddMacroDefinition("NV_HIZ_LEVELS", std::to_string(m_config.hizLevels));
  options.AddMacroDefinition("NV_HIZ_MSAA_SAMPLES", std::to_string(m_config.msaaSamples));
  options.AddMacroDefinition("NV_HIZ_REVERSED_Z", std::to_string(m_config.reversedZ ? 1u : 0u));
  options.AddMacroDefinition("NV_HIZ_NEAR_LEVEL", std::to_string(m_config.hizNearLevel));
  options.AddMacroDefinition("NV_HIZ_FAR_LEVEL", std::to_string(m_config.hizFarLevel));
  options.AddMacroDefinition("NV_HIZ_IS_FIRST", std::to_string(hiz != PROG_HIZ_FAR_REST ? 1u : 0u));
  options.AddMacroDefinition("NV_HIZ_OUTPUT_NEAR", std::to_string(hiz == PROG_HIZ_FAR_AND_NEAR ? 1u : 0u));
  options.AddMacroDefinition("NV_HIZ_USE_STEREO", std::to_string(view == PROG_VIEW_STEREO ? 1u : 0u));
}

void NVHizVK::deinitPipelines()
{
  if(!m_device)
    return;

  if(!m_pipelines[0])
    return;

  for(uint32_t i = 0; i < SHADER_COUNT; i++)
  {
    vkDestroyPipeline(m_device, m_pipelines[i], nullptr);
  }

  memset(&m_pipelines, 0, sizeof(m_pipelines));
}

void NVHizVK::initPipelines(const shaderc::SpvCompilationResult spvResults[SHADER_COUNT])
{
  deinitPipelines();

  for(uint32_t i = 0; i < SHADER_COUNT; i++)
  {
    VkComputePipelineCreateInfo info       = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(spvResults[i]);

    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.pNext = &shaderInfo;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.pName = "main";
    info.layout      = m_pipelineLayout;
    VkResult result;
    result = vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines[i]);
    assert(result == VK_SUCCESS);

    std::ignore = result;
  }
}

void NVHizVK::setupUpdateInfos(Update& update, uint32_t width, uint32_t height, VkFormat sourceFormat, VkImageAspectFlags sourceAspect) const
{
  {
    update.sourceInfo.width      = width;
    update.sourceInfo.height     = height;
    update.sourceInfo.usedWidth  = width;
    update.sourceInfo.usedHeight = height;
    update.sourceInfo.mipLevels  = 1;
    update.sourceInfo.format     = sourceFormat;
    update.sourceInfo.aspect     = sourceAspect;
  }
  {
    uint32_t divisor = 2 << m_config.hizFarLevel;
    uint32_t dim     = width > height ? width : height;
    dim /= divisor;

    uint32_t hiz  = 1;
    uint32_t mips = 1;
    while(hiz < dim)
    {
      hiz *= 2;
      mips++;
    }

    update.farInfo.format     = NVHIZ_FORMAT;
    update.farInfo.aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
    update.farInfo.width      = hiz;
    update.farInfo.height     = hiz;
    update.farInfo.mipLevels  = mips;
    update.farInfo.usedWidth  = width / divisor;
    update.farInfo.usedHeight = height / divisor;
  }
  {
    uint32_t divisor           = 2 << m_config.hizNearLevel;
    update.nearInfo.format     = NVHIZ_FORMAT;
    update.nearInfo.aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
    update.nearInfo.width      = width / divisor;
    update.nearInfo.height     = height / divisor;
    update.nearInfo.mipLevels  = 1;
    update.nearInfo.usedWidth  = width / divisor;
    update.nearInfo.usedHeight = height / divisor;
  }

  update.farImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  update.farImageInfo.sampler     = m_readFarSampler;

  update.nearImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  update.nearImageInfo.sampler     = m_readNearSampler;
}

void NVHizVK::setupDescriptorUpdate(DescriptorUpdate& write, const Update& update, VkDescriptorSet set) const
{
  for(uint32_t i = 0; i < BINDING_COUNT; i++)
  {
    write.writeSets[i].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.writeSets[i].pNext            = 0;
    write.writeSets[i].pBufferInfo      = nullptr;
    write.writeSets[i].pTexelBufferView = nullptr;
    write.writeSets[i].dstSet           = set;
    write.writeSets[i].dstBinding       = i;
    write.writeSets[i].dstArrayElement  = 0;
    write.writeSets[i].descriptorCount  = 1;

    write.writeSets[i].pImageInfo = &write.imageInfos[i];

    if(i == BINDING_READ_DEPTH)
    {
      write.writeSets[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

      write.imageInfos[i].imageView   = update.sourceImageView;
      write.imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      write.imageInfos[i].sampler     = m_readDepthSampler;
    }
    else if(i == BINDING_READ_FAR)
    {
      write.writeSets[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

      write.imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      write.imageInfos[i].imageView   = update.farImageView;
      write.imageInfos[i].sampler     = m_readFarSampler;
    }
    else if(i == BINDING_WRITE_NEAR)
    {
      write.writeSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

      write.imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      write.imageInfos[i].imageView   = update.nearImageView ? update.nearImageView : update.farImageViews[0];
      write.imageInfos[i].sampler     = VK_NULL_HANDLE;
    }
    else if(i == BINDING_WRITE_FAR)
    {
      write.writeSets[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      write.writeSets[i].descriptorCount = MAX_MIP_LEVELS;

      for(uint32_t m = 0; m < MAX_MIP_LEVELS; m++)
      {
        write.imageInfos[i + m].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        write.imageInfos[i + m].imageView   = update.farImageViews[m];
        write.imageInfos[i + m].sampler     = VK_NULL_HANDLE;
      }
    }
  }
}

void NVHizVK::updateDescriptorSet(const Update& update, uint32_t setIdx) const
{
  DescriptorUpdate write;
  setupDescriptorUpdate(write, update, m_descrSets[setIdx]);
  vkUpdateDescriptorSets(m_device, BINDING_COUNT, write.writeSets, 0, nullptr);
}

void NVHizVK::initUpdateViews(Update& update) const
{
  deinitUpdateViews(update);

  VkResult              result;
  VkImageViewCreateInfo info           = {};
  info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.components.r                    = VK_COMPONENT_SWIZZLE_R;
  info.components.g                    = VK_COMPONENT_SWIZZLE_G;
  info.components.b                    = VK_COMPONENT_SWIZZLE_B;
  info.components.a                    = VK_COMPONENT_SWIZZLE_A;
  info.viewType                        = update.stereo ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
  info.subresourceRange.layerCount     = update.stereo ? 2 : 1;
  info.subresourceRange.baseArrayLayer = 0;

  // source
  info.image                         = update.sourceImage;
  info.format                        = update.sourceInfo.format;
  info.subresourceRange.aspectMask   = VK_IMAGE_ASPECT_DEPTH_BIT;
  info.subresourceRange.baseMipLevel = 0;
  info.subresourceRange.levelCount   = 1;

  result = vkCreateImageView(m_device, &info, nullptr, &update.sourceImageView);

  // all subsequent are color formats
  info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

  // far read
  info.image                       = update.farImage;
  info.format                      = update.farInfo.format;
  info.subresourceRange.levelCount = update.farInfo.mipLevels;

  result = vkCreateImageView(m_device, &info, nullptr, &update.farImageView);

  // far writes
  for(uint32_t i = 0; i < MAX_MIP_LEVELS; i++)
  {
    info.image                         = update.farImage;
    info.format                        = update.farInfo.format;
    info.subresourceRange.baseMipLevel = i < update.farInfo.mipLevels ? i : std::max(update.farInfo.mipLevels, 1U) - 1;
    info.subresourceRange.levelCount   = 1;

    result = vkCreateImageView(m_device, &info, nullptr, &update.farImageViews[i]);
  }

  // near write
  if(update.nearImage)
  {
    info.image                         = update.nearImage;
    info.format                        = update.nearInfo.format;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount   = 1;

    result = vkCreateImageView(m_device, &info, nullptr, &update.nearImageView);
  }

  update.farImageInfo.imageView  = update.farImageView;
  update.nearImageInfo.imageView = update.nearImageView;

  std::ignore = result;
}

void NVHizVK::deinitUpdateViews(Update& update) const
{
  if(!m_device)
    return;

  if(update.sourceImageView)
  {
    vkDestroyImageView(m_device, update.sourceImageView, nullptr);
    update.sourceImageView = nullptr;
  }

  if(update.nearImageView)
  {
    vkDestroyImageView(m_device, update.nearImageView, nullptr);
    update.nearImageView = nullptr;
  }

  if(update.farImageView)
  {
    vkDestroyImageView(m_device, update.farImageView, nullptr);
    update.farImageView = nullptr;
  }

  for(uint32_t i = 0; i < MAX_MIP_LEVELS; i++)
  {
    if(update.farImageViews[i])
    {
      vkDestroyImageView(m_device, update.farImageViews[i], nullptr);
      update.farImageViews[i] = nullptr;
    }
  }
}

void NVHizVK::cmdUpdateHiz(VkCommandBuffer cmd, const Update& update, VkDescriptorSet set) const
{
  uint32_t inputW = update.sourceInfo.usedWidth;
  uint32_t inputH = update.sourceInfo.usedHeight;

  uint32_t halfW = inputW / 2;
  uint32_t halfH = inputH / 2;

  uint32_t subW = (inputW + 1) / 2;
  uint32_t subH = (inputH + 1) / 2;

  uint32_t farMultiplier  = 1 << m_config.hizFarLevel;
  uint32_t nearMultiplier = 1 << m_config.hizNearLevel;

  if(halfW != update.farInfo.usedWidth * farMultiplier || halfH != update.farInfo.usedHeight * farMultiplier
     || (update.nearImageView
         && (halfW != update.nearInfo.usedWidth * nearMultiplier || halfH != update.nearInfo.usedHeight * nearMultiplier)))
  {
    assert(0);
  }

  uint32_t align = 8;

  ProgHizMode  hizMode  = update.nearImage ? PROG_HIZ_FAR_AND_NEAR : PROG_HIZ_FAR;
  ProgViewMode viewMode = update.stereo ? PROG_VIEW_STEREO : PROG_VIEW_MONO;

  uint32_t viewCount = viewMode == PROG_VIEW_STEREO ? 2 : 1;
  uint32_t mips      = update.farInfo.mipLevels;

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &set, 0, nullptr);

  VkImageMemoryBarrier imageBarriers[2]        = {{}, {}};
  imageBarriers[0].sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageBarriers[0].image                       = update.farImage;
  imageBarriers[0].dstAccessMask               = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  imageBarriers[0].srcAccessMask               = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  imageBarriers[0].newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
  imageBarriers[0].oldLayout                   = VK_IMAGE_LAYOUT_GENERAL;
  imageBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageBarriers[0].subresourceRange.layerCount = viewCount;
  imageBarriers[0].subresourceRange.levelCount = update.farInfo.mipLevels;

  imageBarriers[1].sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageBarriers[1].image                       = update.nearImage;
  imageBarriers[1].dstAccessMask               = VK_ACCESS_SHADER_WRITE_BIT;
  imageBarriers[1].srcAccessMask               = VK_ACCESS_SHADER_WRITE_BIT;
  imageBarriers[1].newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
  imageBarriers[1].oldLayout                   = VK_IMAGE_LAYOUT_GENERAL;
  imageBarriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageBarriers[1].subresourceRange.layerCount = viewCount;
  imageBarriers[1].subresourceRange.levelCount = 1;

  PushConstants push = {};


  for(uint32_t i = 0; i < mips; i += m_config.hizLevels)
  {
    uint32_t inputLod = (i == 0) ? 0 : i - 1;

    if(i == 0)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[getShaderIndex(hizMode, viewMode)]);
    }
    else if(i == m_config.hizLevels)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[getShaderIndex(PROG_HIZ_FAR_REST, viewMode)]);
    }

    if(i != 0)
    {
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, imageBarriers);
    }

    for(uint32_t level = 0; level < m_config.hizLevels; level++)
    {
      int32_t active          = level + i < mips;
      push.levelActive[level] = active;
    }

    subW = ((subW + align - 1) / align) * align;
    subH = ((subH + align - 1) / align) * align;

    push.srcSize[0] = inputW;
    push.srcSize[1] = inputH;
    push.srcSize[2] = inputW - 2;
    push.srcSize[3] = inputH - 2;

    push.startLod = inputLod;
    push.writeLod = i;

    for(uint32_t v = 0; v < viewCount; v++)
    {
      push.layer = v;

      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
      vkCmdDispatch(cmd, (subW + 7) / 8, (subH + 7) / 8, 1);
    }

    for(uint32_t level = 0; level < m_config.hizLevels; level++)
    {
      subW = (subW + 1) / 2;
      subH = (subH + 1) / 2;
    }

    subW = subW ? subW : 1;
    subH = subH ? subH : 1;

    inputW = subW * 2;
    inputH = subH * 2;
  }

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, update.nearImageView ? 2 : 1, imageBarriers);
}
