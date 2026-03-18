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

#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <span>

#include <vulkan/vulkan_core.h>

#include <nvvkglsl/glsl.hpp>

class NVHizVK
{
private:
  enum ProgViewMode : uint32_t
  {
    PROG_VIEW_MONO,
    PROG_VIEW_STEREO,
    PROG_VIEW_COUNT,
  };

  enum ProgHizMode : uint32_t
  {
    PROG_HIZ_FAR,
    PROG_HIZ_FAR_AND_NEAR,
    PROG_HIZ_FAR_REST,
    PROG_HIZ_COUNT,
  };

public:
  static const uint32_t MAX_MIP_LEVELS = 16;
  static const uint32_t SHADER_COUNT   = (uint32_t(PROG_HIZ_COUNT) * uint32_t(PROG_VIEW_COUNT));
  static const uint32_t POOLSIZE_COUNT = 2;

  enum BindingSlots
  {
    // keep in sync with glsl
    BINDING_READ_DEPTH,
    BINDING_READ_FAR,
    BINDING_WRITE_NEAR,
    BINDING_WRITE_FAR,
    BINDING_COUNT,
  };

  struct TextureInfo
  {
    // allocation
    uint32_t           width;
    uint32_t           height;
    uint32_t           mipLevels;
    VkFormat           format;
    VkImageAspectFlags aspect;

    // the system may use only a sub-rectangle of the allocated width/height
    // you should clamp access to this, when sampling the texture
    uint32_t usedWidth;
    uint32_t usedHeight;

    // xy scale and zw clamp
    // use min(uv*factor.xy,factor.zw) for lookups
    void  getShaderFactors(float factors[4]) const;
    float getSizeMax() const;
  };

  struct Update
  {
    // provide texture/views that are not layered
    VkImageView sourceImageView;                // 2DMS if createInfo.msaaLevel set, otherwise 2D
    VkImageView nearImageView;                  // 2D optional
    VkImageView farImageView;                   // 2D all mips
    VkImageView farImageViews[MAX_MIP_LEVELS];  // 2D single mip

    VkDescriptorImageInfo farImageInfo;
    VkDescriptorImageInfo nearImageInfo;

    VkImage sourceImage;
    VkImage nearImage;  // optional
    VkImage farImage;

    TextureInfo sourceInfo;
    TextureInfo farInfo;
    TextureInfo nearInfo;
    bool        stereo;  // textures are layered, and updates layer 0,1

    Update() { memset(this, 0, sizeof(Update)); }
  };

  struct DescriptorUpdate
  {
    VkWriteDescriptorSet  writeSets[BINDING_COUNT];
    VkDescriptorImageInfo imageInfos[BINDING_COUNT + MAX_MIP_LEVELS - 1];
  };

  struct Config
  {
    int  msaaSamples             = 0;
    bool reversedZ               = false;
    bool supportsSubGroupShuffle = false;
    bool supportsMinmaxFilter    = false;
  };


  void init(VkDevice device, const Config& config, uint32_t descrSetsCount);

  VkSampler                   getReadFarSampler() const;
  const VkDescriptorPoolSize* getDescriptorPoolSizes(uint32_t& count) const;
  VkDescriptorSetLayout       getDescriptorSetLayout() const;
  void                        appendShaderDefines(uint32_t shader, shaderc::CompileOptions& options) const;
  void                        initPipelines(const shaderc::SpvCompilationResult spvResults[SHADER_COUNT]);

  void deinit();

  void setupUpdateInfos(Update& update, uint32_t width, uint32_t height, VkFormat sourceFormat, VkImageAspectFlags sourceAspect) const;
  void setupDescriptorUpdate(DescriptorUpdate& updateWrite, const Update& update, VkDescriptorSet set) const;

  void cmdUpdateHiz(VkCommandBuffer cmd, const Update& update, VkDescriptorSet set) const;

  // optional utility functions
  void initUpdateViews(Update& update) const;
  void deinitUpdateViews(Update& update) const;

  // if descrSetsCount was non zero
  void updateDescriptorSet(const Update& update, uint32_t setIdx) const;
  // if descrSetsCount was non zero
  void cmdUpdateHiz(VkCommandBuffer cmd, const Update& update, uint32_t setIdx) const
  {
    cmdUpdateHiz(cmd, update, m_descrSets[setIdx]);
  }

private:
  struct InternalConfig : public Config
  {
    uint32_t hizLevels    = 1;
    uint32_t hizNearLevel = 0;
    uint32_t hizFarLevel  = 0;
  };

  static void getShaderIndexConfig(uint32_t index, ProgHizMode& hiz, ProgViewMode& view)
  {
    hiz  = ProgHizMode(index % uint32_t(PROG_HIZ_COUNT));
    view = ProgViewMode(index / uint32_t(PROG_HIZ_COUNT));
  }

  static uint32_t getShaderIndex(ProgHizMode hiz, ProgViewMode view) { return view * uint32_t(PROG_HIZ_COUNT) + hiz; }

  struct PushConstants
  {
    // keep in sync with glsl
    int srcSize[4];
    int writeLod;
    int startLod;
    int layer;
    int _pad0;
    int levelActive[4];
  };

  void deinitPipelines();

  InternalConfig        m_config                  = {};
  VkDevice              m_device                  = {};
  VkSampler             m_readDepthSampler        = {};
  VkSampler             m_readFarSampler          = {};
  VkSampler             m_readNearSampler         = {};
  VkPipeline            m_pipelines[SHADER_COUNT] = {0};
  VkPipelineLayout      m_pipelineLayout          = {};
  VkDescriptorSetLayout m_descrLayout             = {};
  VkDescriptorPoolSize  m_poolSizes[POOLSIZE_COUNT];
  uint32_t              m_descrSetsCount = 0;
  VkDescriptorPool      m_descrPool      = {};
  VkDescriptorSet*      m_descrSets      = {};
};
