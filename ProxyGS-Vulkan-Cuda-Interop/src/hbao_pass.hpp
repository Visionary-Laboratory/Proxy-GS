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

#include <cassert>

#include <vulkan/vulkan_core.h>
#include <nvvkglsl/glsl.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <glm/glm.hpp>
#include <filesystem>

//////////////////////////////////////////////////////////////////////////

/// HbaoSystem implements a screen-space
/// ambient occlusion effect using
/// horizon-based ambient occlusion.
/// See https://github.com/nvpro-samples/gl_ssao
/// for more details

class HbaoPass
{
public:
  static const int RANDOM_SIZE     = 4;
  static const int RANDOM_ELEMENTS = RANDOM_SIZE * RANDOM_SIZE;

  struct Config
  {
    VkFormat targetFormat;
    uint32_t maxFrames;
  };

  bool init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool, nvvkglsl::GlslCompiler* glslCompiler, const Config& config);
  void setAssetRoot(const std::filesystem::path& assetRoot);
  bool reloadShaders();
  void deinit();

  struct FrameConfig
  {
    bool blend;

    uint32_t sourceWidthScale;
    uint32_t sourceHeightScale;

    uint32_t targetWidth;
    uint32_t targetHeight;

    VkDescriptorImageInfo sourceDepth;
    VkDescriptorImageInfo targetColor;
  };

  struct FrameIMGs
  {
    nvvk::Image depthlinear, viewnormal, result, blur, resultarray, deptharray;
  };

  struct Frame
  {
    uint32_t slot = ~0u;

    FrameIMGs images;
    int       width;
    int       height;

    FrameConfig config;
  };

  bool initFrame(Frame& frame, const FrameConfig& config, VkCommandBuffer cmd);
  void deinitFrame(Frame& frame);


  struct View
  {
    bool      isOrtho;
    float     nearPlane;
    float     farPlane;
    float     halfFovyTan;
    glm::mat4 projectionMatrix;
  };

  struct Settings
  {
    View view;

    float unit2viewspace = 1.0f;
    float intensity      = 1.0f;
    float radius         = 1.0f;
    float bias           = 0.1f;
    float blurSharpness  = 40.0f;
  };

  // before: must do appropriate barriers for color write access and depth read access
  // after:  from compute write to whatever output image needs
  void cmdCompute(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;

private:
  struct Shaders
  {
    shaderc::SpvCompilationResult depth_linearize{};
    shaderc::SpvCompilationResult viewnormal{};
    shaderc::SpvCompilationResult blur{};
    shaderc::SpvCompilationResult blur_apply{};
    shaderc::SpvCompilationResult deinterleave{};
    shaderc::SpvCompilationResult calc{};
    shaderc::SpvCompilationResult reinterleave{};
  };

  struct Pipelines
  {
    VkPipeline depth_linearize{};
    VkPipeline viewnormal{};
    VkPipeline blur{};
    VkPipeline blur_apply{};
    VkPipeline deinterleave{};
    VkPipeline calc{};
    VkPipeline reinterleave{};
  };

  VkDevice                 m_device{};
  nvvk::ResourceAllocator* m_allocator{};
  nvvk::SamplerPool*       m_samplerPool{};
  nvvkglsl::GlslCompiler*  m_glslCompiler{};
  std::filesystem::path    m_assetRoot;

  uint64_t m_slotsUsed = {};
  Config   m_config;

  nvvk::DescriptorPack m_dsetPack;
  VkPipelineLayout     m_pipelineLayout{};

  nvvk::Buffer           m_ubo;
  VkDescriptorBufferInfo m_uboInfo;

  VkSampler m_linearSampler{};
  VkSampler m_nearestSampler{};

  Shaders   m_shaders;
  Pipelines m_pipelines;

  glm::vec4 m_hbaoRandom[RANDOM_ELEMENTS];

  void updatePipelines();
  void updateUbo(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;
};
