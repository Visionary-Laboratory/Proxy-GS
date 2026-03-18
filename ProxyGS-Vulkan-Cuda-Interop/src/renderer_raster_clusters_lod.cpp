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

#include <nvutils/alignment.hpp>
#include <fmt/format.h>

#include "renderer.hpp"
#include "external_memory.hpp"
#include "../shaders/shaderio.h"

namespace lodclusters {

class RendererRasterClustersLod : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) override;
  virtual void deinit(Resources& res) override;

private:
  bool initShaders(Resources& res, RenderScene& scene, const RendererConfig& config);

  struct Shaders
  {
    shaderc::SpvCompilationResult graphicsMesh;
    shaderc::SpvCompilationResult graphicsFragment;

    shaderc::SpvCompilationResult computeTraversalPresort;
    shaderc::SpvCompilationResult computeTraversalInit;
    shaderc::SpvCompilationResult computeTraversalRun;
    shaderc::SpvCompilationResult computeTraversalGroups;
    shaderc::SpvCompilationResult computeBuildSetup;
  };

  struct Pipelines
  {
    VkPipeline graphicsMesh            = nullptr;
    VkPipeline computeTraversalPresort = nullptr;
    VkPipeline computeTraversalInit    = nullptr;
    VkPipeline computeTraversalRun     = nullptr;
    VkPipeline computeTraversalGroups  = nullptr;
    VkPipeline computeBuildSetup       = nullptr;
  };

  RendererConfig     m_config;
  Shaders            m_shaders;
  Pipelines          m_pipelines;
  VkShaderStageFlags m_stageFlags{};
  VkPipelineLayout   m_pipelineLayout{};

  nvvk::DescriptorPack m_dsetPack;

  nvvk::Buffer m_sceneBuildBuffer;
  nvvk::Buffer m_sceneTraversalBuffer;
  nvvk::Buffer m_sceneDataBuffer;

  shaderio::SceneBuilding m_sceneBuildShaderio;
};

bool RendererRasterClustersLod::initShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  shaderc::CompileOptions options = res.makeCompilerOptions();

  uint32_t meshletTriangles = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterTriangles);
  uint32_t meshletVertices  = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterVertices);
  LOGI("mesh shader config: %d triangles %d vertices\n", meshletTriangles, meshletVertices);

  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", meshletVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", meshletTriangles));
  options.AddMacroDefinition("TARGETS_RASTERIZATION", "1");
  options.AddMacroDefinition("USE_STREAMING", rscene.useStreaming ? "1" : "0");
  options.AddMacroDefinition("USE_SORTING", config.useSorting ? "1" : "0");
  options.AddMacroDefinition("USE_CULLING", config.useCulling ? "1" : "0");
  options.AddMacroDefinition("USE_RENDER_STATS", config.useRenderStats ? "1" : "0");
  options.AddMacroDefinition("USE_SEPARATE_GROUPS", config.useSeparateGroups ? "1" : "0");
  options.AddMacroDefinition("USE_DLSS", "0");
  options.AddMacroDefinition("DEBUG_VISUALIZATION", config.useDebugVisualization ? "1" : "0");
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", "32");

  res.compileShader(m_shaders.graphicsMesh, VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters.mesh.glsl", &options);
  res.compileShader(m_shaders.graphicsFragment, VK_SHADER_STAGE_FRAGMENT_BIT, "render_raster.frag.glsl", &options);
  res.compileShader(m_shaders.computeTraversalPresort, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_presort.comp.glsl", &options);
  res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", &options);
  res.compileShader(m_shaders.computeTraversalRun, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", &options);
  res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options);

  if(config.useSeparateGroups)
  {
    res.compileShader(m_shaders.computeTraversalGroups, VK_SHADER_STAGE_COMPUTE_BIT,
                      "traversal_run_separate_groups.comp.glsl", &options);
  }

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, rscene, config);
}

bool RendererRasterClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_config = config;

  if(!initShaders(res, rscene, config))
  {
    return false;
  }

  if(!rscene.updateClasRequired(false))
  {
    return false;
  }

#if USE_DLSS
  // not supported in raster for now
  res.setFramebufferDlss(false, config.dlssQuality);
#endif

  initBasics(res, rscene, config);

  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.operationsMemBytes = logMemoryUsage(rscene.getOperationsSize(), "operations", "rscene total");

  {
    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneBuildBuffer.bufferSize, "operations", "build shaderio");

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.maxRenderClusters  = uint32_t(1u << config.numRenderClusterBits);
    m_sceneBuildShaderio.maxTraversalInfos  = uint32_t(1u << config.numTraversalTaskBits);

    m_sceneBuildShaderio.indirectDispatchGroups.gridY = 1;
    m_sceneBuildShaderio.indirectDispatchGroups.gridZ = 1;

    BufferRanges mem = {};
    m_sceneBuildShaderio.renderClusterInfos =
        mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);

    if(config.useSorting)
    {
      m_sceneBuildShaderio.instanceSortKeys   = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.instanceSortValues = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    }

    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, 8);
    }

    res.createBuffer(m_sceneDataBuffer, mem.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneDataBuffer.bufferSize, "operations", "build data");

    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortKeys += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortValues += m_sceneDataBuffer.address;
    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos += m_sceneDataBuffer.address;
    }

    res.createBuffer(m_sceneTraversalBuffer, sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneTraversalBuffer.bufferSize, "operations", "build traversal");

    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;
  }

  {
    m_stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    m_dsetPack.bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    if(rscene.useStreaming)
    {
      m_dsetPack.bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
      m_dsetPack.bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    }
    m_dsetPack.initFromBindings(res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.layout}, {{m_stageFlags, 0, sizeof(uint32_t)}});

    nvvk::WriteSetContainer writeSets;

    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_READBACK_SSBO), &res.m_commonBuffers.readBack);
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_SCENEBUILDING_SSBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_SCENEBUILDING_UBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_HIZ_TEX), &res.m_hizUpdate.farImageInfo);
    if(rscene.useStreaming)
    {
      writeSets.append(m_dsetPack.getWriteSet(BINDINGS_STREAMING_SSBO), rscene.sceneStreaming.getShaderStreamingBuffer());
      writeSets.append(m_dsetPack.getWriteSet(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
    }
    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  {
    nvvk::GraphicsPipelineCreator graphicsGen;
    nvvk::GraphicsPipelineState   state              = res.m_basicGraphicsState;
    graphicsGen.pipelineInfo.layout                  = m_pipelineLayout;
    graphicsGen.renderingState.depthAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
    graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
    graphicsGen.colorFormats = {res.m_frameBuffer.colorFormat};

    state.rasterizationState.frontFace = config.flipWinding ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    if(config.twoSided)
    {
      state.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    }
    graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsMesh));
    graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsFragment));
    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_pipelines.graphicsMesh);
  }

  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;

    if(config.useSorting)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalPresort);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalPresort);
    }

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalInit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalRun);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);

    if(config.useSeparateGroups)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalGroups);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalGroups);
    }
  }

  return true;
}

static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void RendererRasterClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  glm::vec2 renderScale = res.getFramebufferWindow2RenderScale();
  float     pixelScale  = std::min(renderScale.x, renderScale.y);

  m_sceneBuildShaderio.traversalViewMatrix =
      frame.freezeCulling ? frame.frameConstantsLast.viewMatrix : frame.frameConstants.viewMatrix;
  m_sceneBuildShaderio.errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(frame.lodPixelError * pixelScale, frame.frameConstants.fov,
                                    frame.frameConstants.viewportf.y);
  m_sceneBuildShaderio.useHizOcclusion = frame.useHizOcclusion ? 1u : 0u;

  const bool useSky = true;  // When using Sky, the sky is rendered first and the rest of the scene is rendered on top of it.

  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants) * 2,
                    (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0);

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer,
                                        frame.streamingAgeThreshold, profiler);
  }

  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, 0, profiler);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Preparation");
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.sets.data(), 0, nullptr);

    if(m_config.useSorting)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalPresort);
      vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_PRESORT_WORKGROUP), 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vrdxCmdSortKeyValue(cmd, res.m_vrdxSorter, m_sceneBuildShaderio.numRenderInstances, m_sceneDataBuffer.buffer,
                          m_sceneBuildShaderio.instanceSortKeys - m_sceneDataBuffer.address, m_sceneDataBuffer.buffer,
                          m_sceneBuildShaderio.instanceSortValues - m_sceneDataBuffer.address,
                          m_sortingAuxBuffer.buffer, 0, nullptr, 0);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.sets.data(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
    vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP), 1, 1);

    // this barrier covers init & streaming pre traversal
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Run");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
    vkCmdDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP), 1, 1);

    if(m_config.useSeparateGroups)
    {
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
    }

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    uint32_t buildSetupID = BUILD_SETUP_DRAW;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPostTraversal(cmd, 0, profiler);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Draw");

    VkAttachmentLoadOp op = useSky ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR;

    res.cmdBeginRendering(cmd, false, op, op);

    if(useSky)
    {
      writeBackgroundSky(cmd);
    }

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dsetPack.sets.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMesh);
    vkCmdDrawMeshTasksIndirectNV(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClusters), 1, 0);

    if(frame.showInstanceBboxes)
    {
      renderInstanceBboxes(cmd);
    }

    vkCmdEndRendering(cmd);
  }

  // Copy to external memory buffer for Python integration
  if (frame.externalMemoryManager && frame.externalMemoryManager->isConnected()) {
    auto timerSection = profiler.cmdFrameSection(cmd, "Copy to External Buffer");
    
    VkBuffer colorReadbackBuffer = frame.externalMemoryManager->getColorReadbackBuffer();
    uint32_t width = frame.externalMemoryManager->getWidth();
    uint32_t height = frame.externalMemoryManager->getHeight();
    
    // Image layout transition: COLOR_ATTACHMENT -> TRANSFER_SRC
    VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = res.m_frameBuffer.imgColor.image;
    imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    
    // Copy image to buffer
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = width;
    region.bufferImageHeight = height;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyImageToBuffer(cmd, res.m_frameBuffer.imgColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           colorReadbackBuffer, 1, &region);
    
    // Buffer memory barrier: TRANSFER_WRITE -> MEMORY_READ (for external access)
    VkBufferMemoryBarrier bufferBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = colorReadbackBuffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;
    
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);
    
    // Restore image layout: TRANSFER_SRC -> COLOR_ATTACHMENT
    imageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }

  if(!frame.freezeCulling)
  {
    res.cmdBuildHiz(cmd, frame, profiler);
  }

  // reservation for geometry may change
  m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);
  m_resourceActualUsage                    = m_resourceReservedUsage;
  m_resourceActualUsage.geometryMemBytes   = rscene.getGeometrySize(false);
}

void RendererRasterClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  vkDeviceWaitIdle(res.m_device);
  std::array<VkWriteDescriptorSet, 1> writeSets;
  writeSets[0]            = m_dsetPack.getWriteSet(BINDINGS_HIZ_TEX);
  writeSets[0].pImageInfo = &res.m_hizUpdate.farImageInfo;
  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res, rscene);
}

void RendererRasterClustersLod::deinit(Resources& res)
{
  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);

  m_dsetPack.deinit();

  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);

  deinitBasics(res);
}

std::unique_ptr<Renderer> makeRendererRasterClustersLod()
{
  return std::make_unique<RendererRasterClustersLod>();
}
}  // namespace lodclusters
