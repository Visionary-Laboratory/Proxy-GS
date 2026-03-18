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

#include <nvvk/sbt_generator.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/alignment.hpp>
#include <fmt/format.h>

#include "renderer.hpp"
#include "external_memory.hpp"

//////////////////////////////////////////////////////////////////////////

namespace lodclusters {

class RendererRayTraceClustersLod : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene);

private:
  bool initShaders(Resources& res, RenderScene& scene, const RendererConfig& config);

  void initRayTracingPipeline(Resources& res);
  bool initRayTracingBlas(Resources& res, RenderScene& scene, const RendererConfig& config, VkDeviceSize& scratchSize);
  void initRayTracingTlas(Resources& res, const RendererConfig& config, VkDeviceSize& scratchSize);

  void updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, bool update = false);


  struct Shaders
  {
    shaderc::SpvCompilationResult rayGen;
    shaderc::SpvCompilationResult rayClosestHit;
    shaderc::SpvCompilationResult rayMiss;
    shaderc::SpvCompilationResult rayMissAO;

    shaderc::SpvCompilationResult computeTraversalPresort;
    shaderc::SpvCompilationResult computeTraversalInit;
    shaderc::SpvCompilationResult computeTraversalRun;
    shaderc::SpvCompilationResult computeTraversalGroups;
    shaderc::SpvCompilationResult computeBuildSetup;

    shaderc::SpvCompilationResult computeBlasInsertClusters;
    shaderc::SpvCompilationResult computeBlasSetupInsertion;

    shaderc::SpvCompilationResult computeInstanceAssignBlas;
    shaderc::SpvCompilationResult computeInstanceClassifyLod;
    shaderc::SpvCompilationResult computeGeometryBlasSharing;
  };

  struct Pipelines
  {
    VkPipeline rayTracing = nullptr;

    VkPipeline computeTraversalPresort = nullptr;
    VkPipeline computeTraversalInit    = nullptr;
    VkPipeline computeTraversalRun     = nullptr;
    VkPipeline computeTraversalGroups  = nullptr;
    VkPipeline computeBuildSetup       = nullptr;

    VkPipeline computeBlasInsertClusters  = nullptr;
    VkPipeline computeBlasSetupInsertion  = nullptr;
    VkPipeline computeInstanceAssignBlas  = nullptr;
    VkPipeline computeInstanceClassifyLod = nullptr;
    VkPipeline computeGeometryBlasSharing = nullptr;
  };

  RendererConfig m_config;
  uint32_t       m_maxRenderClusters = 0;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV m_rtClasProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};

  nvvk::SBTGenerator::Regions m_sbtRegions;  // Shading binding table wrapper
  nvvk::Buffer                m_sbtBuffer;

  Shaders            m_shaders;
  Pipelines          m_pipelines;
  VkShaderStageFlags m_stageFlags{};
  VkPipelineLayout   m_pipelineLayout{};

  nvvk::DescriptorPack m_dsetPack;

  nvvk::Buffer            m_sceneBuildBuffer;
  nvvk::Buffer            m_sceneDataBuffer;
  nvvk::LargeBuffer       m_sceneBlasDataBuffer;
  nvvk::Buffer            m_sceneTraversalBuffer;
  nvvk::Buffer            m_sceneGeometryHistogramBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio;

  VkClusterAccelerationStructureClustersBottomLevelInputNV m_blasInput{};
  VkDeviceSize                                             m_blasDataSize = 0;

  bool                                        m_tlasDoBuild = true;
  nvvk::Buffer                                m_tlasInstancesBuffer;
  VkAccelerationStructureGeometryKHR          m_tlasGeometry{};
  VkAccelerationStructureBuildGeometryInfoKHR m_tlasBuildInfo{};
  nvvk::AccelerationStructure                 m_tlas;

  nvvk::Buffer m_scratchBuffer;
};

bool RendererRayTraceClustersLod::initShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  shaderc::CompileOptions options = res.makeCompilerOptions();

  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", rscene.scene->m_maxClusterVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", rscene.scene->m_maxClusterTriangles));
  options.AddMacroDefinition("TARGETS_RASTERIZATION", "0");
  options.AddMacroDefinition("USE_STREAMING", rscene.useStreaming ? "1" : "0");
  options.AddMacroDefinition("USE_SORTING", config.useSorting ? "1" : "0");
  options.AddMacroDefinition("USE_CULLING", config.useCulling ? "1" : "0");
  options.AddMacroDefinition("USE_BLAS_SHARING", config.useBlasSharing ? "1" : "0");
  options.AddMacroDefinition("USE_RENDER_STATS", config.useRenderStats ? "1" : "0");
  options.AddMacroDefinition("USE_SEPARATE_GROUPS", config.useSeparateGroups ? "1" : "0");
#if USE_DLSS
  options.AddMacroDefinition("USE_DLSS", config.useDlss ? "1" : "0");
#else
  options.AddMacroDefinition("USE_DLSS", "0");
#endif
  options.AddMacroDefinition("DEBUG_VISUALIZATION", config.useDebugVisualization ? "1" : "0");

  shaderc::CompileOptions optionsAO = options;
  options.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "0");
  optionsAO.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "1");

  res.compileShader(m_shaders.rayGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR, "render_raytrace.rgen.glsl", &options);
  res.compileShader(m_shaders.rayClosestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "render_raytrace_clusters.rchit.glsl", &options);
  res.compileShader(m_shaders.rayMiss, VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl", &options);
  res.compileShader(m_shaders.rayMissAO, VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl", &optionsAO);

  if(m_config.useSorting)
  {
    res.compileShader(m_shaders.computeTraversalPresort, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_presort.comp.glsl", &options);
  }

  if(m_config.useBlasSharing)
  {
    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init_blas_sharing.comp.glsl", &options);
  }
  else
  {
    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", &options);
  }

  if(m_config.useSeparateGroups)
  {
    res.compileShader(m_shaders.computeTraversalGroups, VK_SHADER_STAGE_COMPUTE_BIT,
                      "traversal_run_separate_groups.comp.glsl", &options);
  }

  res.compileShader(m_shaders.computeTraversalRun, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", &options);
  res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasInsertClusters, VK_SHADER_STAGE_COMPUTE_BIT, "blas_clusters_insert.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasSetupInsertion, VK_SHADER_STAGE_COMPUTE_BIT, "blas_setup_insertion.comp.glsl", &options);
  res.compileShader(m_shaders.computeInstanceAssignBlas, VK_SHADER_STAGE_COMPUTE_BIT, "instance_assign_blas.comp.glsl", &options);

  if(m_config.useBlasSharing)
  {
    res.compileShader(m_shaders.computeInstanceClassifyLod, VK_SHADER_STAGE_COMPUTE_BIT, "instance_classify_lod.comp.glsl", &options);
    res.compileShader(m_shaders.computeGeometryBlasSharing, VK_SHADER_STAGE_COMPUTE_BIT, "geometry_blas_sharing.comp.glsl", &options);
  }

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, rscene, config);
}

bool RendererRayTraceClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_config                = config;
  m_maxRenderClusters     = 1u << config.numRenderClusterBits;
  m_resourceReservedUsage = {};

  if(!initShaders(res, rscene, config))
  {
    LOGE("RendererRayTraceClustersLod shaders failed\n");
    return false;
  }

  if(!rscene.updateClasRequired(true))
  {
    LOGE("RendererRayTraceClustersLod rscene.updateClasRequired failed\n");
    return false;
  }

#if USE_DLSS
  res.setFramebufferDlss(config.useDlss, config.dlssQuality);
#endif

  initBasics(res, rscene, config);

  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.rtClasMemBytes     = rscene.getClasSize(true);
  m_resourceReservedUsage.operationsMemBytes = logMemoryUsage(rscene.getOperationsSize(), "operations", "rscene total");

  {
    // get ray tracing properties

    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_rtProperties};
    m_rtProperties.pNext = &m_rtClasProperties;
    vkGetPhysicalDeviceProperties2(res.m_physicalDevice, &prop2);

    VkDeviceSize scratchSize = 0;
    if(rscene.useStreaming)
    {
      scratchSize = rscene.sceneStreaming.getRequiredClasScratchSize();
    }

    if(!initRayTracingBlas(res, rscene, config, scratchSize))
    {
      LOGE("Resources exceeding max buffer allocation size\n");
      deinit(res);
      return false;
    }

    // TLAS creation
    initRayTracingTlas(res, config, scratchSize);

    // streaming also stores newly built clas in scratch
    res.createBuffer(m_scratchBuffer, scratchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_scratchBuffer.bufferSize, "operations", "rt scratch");

    // Update tlas build information
    m_tlasBuildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
    m_tlasBuildInfo.dstAccelerationStructure  = m_tlas.accel;
    m_tlasBuildInfo.scratchData.deviceAddress = m_scratchBuffer.address;
  }

  // scene building data

  {
    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneBuildBuffer.bufferSize, "operations", "build shaderio");

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.maxRenderClusters  = m_maxRenderClusters;
    m_sceneBuildShaderio.maxTraversalInfos  = uint32_t(1u << config.numTraversalTaskBits);
    m_sceneBuildShaderio.tlasInstances      = m_tlasInstancesBuffer.address;
    m_sceneBuildShaderio.numGeometries      = uint32_t(rscene.scene->getActiveGeometryCount());

    m_sceneBuildShaderio.indirectDispatchGroups.gridY = 1;
    m_sceneBuildShaderio.indirectDispatchGroups.gridZ = 1;

    BufferRanges mem = {};
    m_sceneBuildShaderio.renderClusterInfos =
        mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);

    m_sceneBuildShaderio.instanceVisibility = mem.append(sizeof(uint8_t) * m_renderInstances.size(), 4);
    m_sceneBuildShaderio.blasBuildInfos = mem.append(sizeof(shaderio::BlasBuildInfo) * m_renderInstances.size(), 16);
    m_sceneBuildShaderio.instanceBuildInfos = mem.append(sizeof(shaderio::InstanceBuildInfo) * m_renderInstances.size(), 16);

    if(config.useBlasSharing)
    {
      m_sceneBuildShaderio.geometryBuildInfos =
          mem.append(sizeof(shaderio::GeometryBuildInfo) * m_sceneBuildShaderio.numGeometries, 16);
    }

    if(config.useSorting)
    {
      // can alias some data required for sorting, with other data used at traversal/blas time.
      mem.beginOverlap();
      m_sceneBuildShaderio.instanceSortKeys   = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.instanceSortValues = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      mem.splitOverlap();
    }

    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, 8);
    }

    m_sceneBuildShaderio.blasBuildSizes     = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    m_sceneBuildShaderio.blasBuildAddresses = mem.append(sizeof(uint64_t) * m_renderInstances.size(), 8);

    m_sceneBuildShaderio.blasClusterAddresses = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxRenderClusters, 8);

    if(config.useSorting)
    {
      mem.endOverlap();
    }

    res.createBuffer(m_sceneDataBuffer, mem.getSize(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneDataBuffer.bufferSize, "operations", "build data");

    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceVisibility += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildSizes += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasClusterAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortKeys += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortValues += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.geometryBuildInfos += m_sceneDataBuffer.address;
    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos += m_sceneDataBuffer.address;
    }

    res.createBuffer(m_sceneTraversalBuffer, sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneTraversalBuffer.bufferSize, "operations", "build traversal");

    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;

    res.createLargeBuffer(m_sceneBlasDataBuffer, m_blasDataSize,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceReservedUsage.rtBlasMemBytes += m_blasDataSize;

    if(m_config.useBlasSharing)
    {
      res.createBuffer(m_sceneGeometryHistogramBuffer, sizeof(shaderio::GeometryBuildHistogram) * m_sceneBuildShaderio.numGeometries,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      m_resourceReservedUsage.operationsMemBytes +=
          logMemoryUsage(m_sceneGeometryHistogramBuffer.bufferSize, "operations", "build geo");

      m_sceneBuildShaderio.geometryHistograms = m_sceneGeometryHistogramBuffer.address;
    }
  }

  // use a single common descriptor set for all operations

  {
    m_stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                   | VK_SHADER_STAGE_COMPUTE_BIT;

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
    m_dsetPack.bindings.addBinding(BINDINGS_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
    m_dsetPack.bindings.addBinding(BINDINGS_RENDER_TARGET, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
#if USE_DLSS
    if(config.useDlss)
    {
      // skip first
      for(uint32_t i = 1; i < DlssDenoiser::eDlssCount; i++)
      {
        m_dsetPack.bindings.addBinding(BINDINGS_RENDER_TARGET + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
      }
    }
#endif

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
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_TLAS), m_tlas);

    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth);
#if USE_DLSS
    if(config.useDlss)
    {
      // apply all
      for(uint32_t i = 0; i < DlssDenoiser::eDlssCount; i++)
      {
        VkDescriptorImageInfo renderTargetInfo =
            res.m_frameBuffer.dlssDenoiser.getDescriptorImageInfo(DlssDenoiser::DlssBufferType(i));
        writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RENDER_TARGET + i), &renderTargetInfo);
      }
    }
    else
#endif
    {
      VkDescriptorImageInfo renderTargetInfo;
      renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      renderTargetInfo.imageView   = res.m_frameBuffer.imgColor.descriptor.imageView;
      writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RENDER_TARGET), &renderTargetInfo);
    }

    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  // initialize ray tracing pipeline

  initRayTracingPipeline(res);

  // initialize traversal pipeline

  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    if(config.useSorting)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalPresort);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalPresort);
    }

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalInit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalRun);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBlasInsertClusters);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasInsertClusters);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBlasSetupInsertion);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasSetupInsertion);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeInstanceAssignBlas);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeInstanceAssignBlas);

    if(config.useBlasSharing)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeInstanceClassifyLod);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeInstanceClassifyLod);

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeGeometryBlasSharing);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeGeometryBlasSharing);
    }

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

void RendererRayTraceClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  m_sceneBuildShaderio.traversalViewMatrix =
      frame.freezeCulling ? frame.frameConstantsLast.viewMatrix : frame.frameConstants.viewMatrix;

  glm::vec2 renderScale = res.getFramebufferWindow2RenderScale();
  float     pixelScale  = std::min(renderScale.x, renderScale.y);

  m_sceneBuildShaderio.errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(frame.lodPixelError * pixelScale, frame.frameConstants.fov,
                                    frame.frameConstants.viewportf.y);

  m_sceneBuildShaderio.culledErrorScale      = std::max(1.0f, frame.culledErrorScale);
  m_sceneBuildShaderio.useHizOcclusion       = frame.useHizOcclusion ? 1u : 0u;
  m_sceneBuildShaderio.sharingMinInstances   = frame.sharingMinInstances;
  m_sceneBuildShaderio.sharingPushCulled     = frame.sharingPushCulled;
  m_sceneBuildShaderio.sharingToleranceLevel = std::max(1u, frame.sharingToleranceLevel);
  m_sceneBuildShaderio.sharingMinLevel       = frame.sharingMinLevel;

  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants) * 2,
                    (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0);

  if(m_config.useBlasSharing)
  {
    vkCmdFillBuffer(cmd, m_sceneGeometryHistogramBuffer.buffer, 0, m_sceneGeometryHistogramBuffer.bufferSize, 0);
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer,
                                        frame.streamingAgeThreshold, profiler);
  }


  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);
  res.cmdImageTransition(cmd, res.m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);


  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, m_scratchBuffer.address, profiler);
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
    }

    if(m_config.useBlasSharing)
    {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.sets.data(), 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstanceClassifyLod);
      vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, INSTANCES_CLASSIFY_LOD_WORKGROUP), 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeGeometryBlasSharing);
      vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numGeometries, GEOMETRY_BLAS_SHARING_WORKGROUP), 1, 1);
    }

    if(m_config.useBlasSharing || m_config.useSorting)
    {
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    // we prepare traversal by filling in instance root nodes into the traversal queue
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.sets.data(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
    vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP), 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in traversal queue

    uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Run");

    // this does the main traversal
    // it returns a list of render clusters

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
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in render list

    uint32_t buildSetupID = BUILD_SETUP_BLAS_INSERTION;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);
  }

  if(rscene.useStreaming)
  {
    // this operation gives us new CLAS addresses
    rscene.sceneStreaming.cmdPostTraversal(cmd, m_scratchBuffer.address, profiler);

    // no barrier needed here, given the critical barrier prior using these addresses
    // is directly prior running `m_pipelines.computeBlasInsertCluster`
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Blas Build Preparation");

    // this kernel prepares the per-blas clas reference list starting position.
    // it also resets the per-blas clas counters.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.sets.data(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasSetupInsertion);
    vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, BLAS_SETUP_INSERTION_WORKGROUP), 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);

    // let's fill in the clusters from the unsorted render list, into the per-blas clas reference lists.

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasInsertClusters);
    vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchBlasInsertion));

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  if(rscene.useStreaming)
  {
    // initialize download work here so it can overlap with next
    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }

  // what is this? nah we never had any bugs in building and allocating the cluster data, totally not needed
#if !STREAMING_DEBUG_WITHOUT_RT
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Blas Build");

    // after we prepared the build information for the blas we can run it.

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    // setup blas inputs
    inputs.maxAccelerationStructureCount = uint32_t(m_renderInstances.size());
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel  = &m_blasInput;
    inputs.flags                         = m_config.clusterBlasFlags;

    // we feed the generated blas addresses directly into the ray instances
    cmdInfo.dstAddressesArray.deviceAddress = m_sceneBuildShaderio.blasBuildAddresses;
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_renderInstances.size();
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    // for statistics we keep track of blas sizes
    cmdInfo.dstSizesArray.deviceAddress = m_sceneBuildShaderio.blasBuildSizes;
    cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * m_renderInstances.size();
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.blasBuildInfos;
    cmdInfo.srcInfosArray.size =
        sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * m_renderInstances.size();
    cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

    // in implicit mode we provide one big chunk from which outputs are sub-allocated
    cmdInfo.dstImplicitData = m_sceneBlasDataBuffer.address;

    // we may actually build less BLAS than instances, due pre-build low detail blas or recycling
    cmdInfo.srcInfosCount = m_sceneBuildBuffer.address + offsetof(shaderio::SceneBuilding, blasBuildCounter);

    cmdInfo.scratchData = m_scratchBuffer.address;
    cmdInfo.input       = inputs;

    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Tlas Preparation");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstanceAssignBlas);
    vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, INSTANCES_ASSIGN_BLAS_WORKGROUP), 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Tlas Build");

    updateRayTracingTlas(cmd, res, !m_tlasDoBuild);
    m_tlasDoBuild = false;

    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }
#endif

  // Ray trace
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Render");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelines.rayTracing);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1, m_dsetPack.sets.data(), 0, nullptr);
#if !STREAMING_DEBUG_WITHOUT_RT
    vkCmdTraceRaysKHR(cmd, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                      frame.frameConstants.viewport.x, frame.frameConstants.viewport.y, 1);
#endif
    res.cmdBeginRendering(cmd, false, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    writeRayTracingDepthBuffer(cmd);
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

  if(!frame.freezeCulling)
  {
    res.cmdBuildHiz(cmd, frame, profiler);
  }

#if USE_DLSS
  if(m_config.useDlss)
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "DLSS");

    res.m_frameBuffer.dlssDenoiser.denoise(cmd, frame.frameConstants.jitter, frame.frameConstants.viewMatrix,
                                           frame.frameConstants.projMatrix);
  }
#endif

  {
    // reservation for geometry may change
    m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);

    m_resourceActualUsage                  = m_resourceReservedUsage;
    m_resourceActualUsage.geometryMemBytes = rscene.getGeometrySize(false);
    m_resourceActualUsage.rtClasMemBytes   = rscene.getClasSize(false);

    shaderio::Readback readback;
    res.getReadbackData(readback);
    m_resourceActualUsage.rtBlasMemBytes = readback.blasActualSizes;
  }
}

void RendererRayTraceClustersLod::deinit(Resources& res)
{
  deinitBasics(res);

  res.m_allocator.destroyBuffer(m_tlasInstancesBuffer);
  res.m_allocator.destroyBuffer(m_scratchBuffer);
  res.m_allocator.destroyAcceleration(m_tlas);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);
  res.m_allocator.destroyLargeBuffer(m_sceneBlasDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneGeometryHistogramBuffer);

  res.m_allocator.destroyBuffer(m_sbtBuffer);

  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);

  m_dsetPack.deinit();
  m_resourceReservedUsage = {};
}


bool RendererRayTraceClustersLod::initRayTracingBlas(Resources& res, RenderScene& rscene, const RendererConfig& config, VkDeviceSize& scratchSize)
{
  // BLAS space requirement (implicit)
  // the size of the generated blas is dynamic, need to query prebuild info.

  uint32_t numInstances = (uint32_t)m_renderInstances.size();

  m_blasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
  // Just using m_hiPerGeometryClusters here is problematic, as the intermediate state
  // of a continuous lod can yield higher numbers (especially when streaming may temporarily cause overlapping of different levels).
  // Therefore, we use the highest sum of all clusters across all lod levels.
  m_blasInput.maxClusterCountPerAccelerationStructure = std::min(rscene.scene->m_maxPerGeometryClusters, m_maxRenderClusters);
  m_blasInput.maxTotalClusterCount = m_maxRenderClusters;

  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = numInstances;
  inputs.opMode                                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputs.opInput.pClustersBottomLevel = &m_blasInput;
  inputs.flags                        = config.clusterBlasFlags;

  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  scratchSize = std::max(scratchSize, sizesInfo.buildScratchSize);

  m_blasDataSize = sizesInfo.accelerationStructureSize;

  return true;
}

void RendererRayTraceClustersLod::initRayTracingPipeline(Resources& res)
{
  VkDevice device = res.m_device;

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissAO,
    eClosestHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  std::array<VkShaderModuleCreateInfo, eShaderGroupCount>        stageShaders{};
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stageShaders[s].sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  }
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stages[s].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[s].pNext = &stageShaders[s];
    stages[s].pName = "main";
  }

  stages[eRaygen].stage              = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stageShaders[eRaygen].codeSize     = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayGen);
  stageShaders[eRaygen].pCode        = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayGen);
  stages[eMiss].stage                = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMiss].codeSize       = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMiss);
  stageShaders[eMiss].pCode          = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMiss);
  stages[eMissAO].stage              = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMissAO].codeSize     = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMissAO);
  stageShaders[eMissAO].pCode        = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMissAO);
  stages[eClosestHit].stage          = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stageShaders[eClosestHit].codeSize = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayClosestHit);
  stageShaders[eClosestHit].pCode    = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayClosestHit);

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             .generalShader      = VK_SHADER_UNUSED_KHR,
                                             .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                             .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                             .intersectionShader = VK_SHADER_UNUSED_KHR};

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // Miss Ao
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissAO;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  shaderGroups.push_back(group);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount                   = uint32_t(eShaderGroupCount),
      .pStages                      = stages.data(),
      .groupCount                   = static_cast<uint32_t>(shaderGroups.size()),
      .pGroups                      = shaderGroups.data(),
      .maxPipelineRayRecursionDepth = 2,
      .layout                       = m_pipelineLayout,
  };

  // NEW for clusters! we need to enable their usage explicitly for a ray tracing pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
  pipeClusters.allowClusterAccelerationStructure = true;

  rayPipelineInfo.pNext = &pipeClusters;

  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_pipelines.rayTracing));
  NVVK_DBG_NAME(m_pipelines.rayTracing);

  // Creating the SBT
  {
    // Shader Binding Table (SBT) setup
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(res.m_device, m_rtProperties);

    // Prepare SBT data from ray pipeline
    size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_pipelines.rayTracing, rayPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(res.m_allocator.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
                                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    nvvk::StagingUploader uploader;
    uploader.init(&res.m_allocator);

    void* mapping = nullptr;
    NVVK_CHECK(uploader.appendBufferMapping(m_sbtBuffer, 0, bufferSize, mapping));
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_sbtBuffer.address, bufferSize, mapping));

    VkCommandBuffer cmd = res.createTempCmdBuffer();
    uploader.cmdUploadAppended(cmd);
    res.tempSyncSubmit(cmd);
    uploader.deinit();

    // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
    m_sbtRegions = sbtGenerator.getSBTRegions();

    sbtGenerator.deinit();
  }
}

void RendererRayTraceClustersLod::initRayTracingTlas(Resources& res, const RendererConfig& config, VkDeviceSize& scratchSize)
{
  std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(m_renderInstances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform                              = nvvk::toTransformMatrixKHR(m_renderInstances[i].worldMatrix);
    instance.instanceCustomIndex                    = static_cast<uint32_t>(i);  // gl_InstanceCustomIndexEX
    instance.mask                                   = 0xFF;                      // All objects
    instance.instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
        instance.flags                              = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    if(config.flipWinding)
    {
      instance.flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
    }
    // patched in at render time
    instance.accelerationStructureReference = 0;
    tlasInstances[i]                        = instance;
  }

  // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
  res.createBuffer(m_tlasInstancesBuffer, tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_tlasInstancesBuffer.bufferSize, "operations", "rt instances");
  res.simpleUploadBuffer(m_tlasInstancesBuffer, tlasInstances.data());

  VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_tlasInstancesBuffer.buffer};
  VkDeviceAddress instBufferAddr = vkGetBufferDeviceAddress(res.m_device, &bufferInfo);

  // Wraps a device pointer to the above uploaded instances.
  VkAccelerationStructureGeometryInstancesDataKHR instancesVk{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
  instancesVk.data.deviceAddress = instBufferAddr;

  // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
  m_tlasGeometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  m_tlasGeometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  m_tlasGeometry.geometry.instances = instancesVk;

  // Find sizes
  m_tlasBuildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  m_tlasBuildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  m_tlasBuildInfo.geometryCount = 1;
  m_tlasBuildInfo.pGeometries   = &m_tlasGeometry;
  // FIXME
  m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  m_tlasBuildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

  uint32_t                                 instanceCount = uint32_t(m_renderInstances.size());
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(res.m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &m_tlasBuildInfo, &instanceCount, &sizeInfo);

  // Create TLAS
  VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  createInfo.size = sizeInfo.accelerationStructureSize;

  res.m_allocator.createAcceleration(m_tlas, createInfo);
  m_resourceReservedUsage.rtTlasMemBytes += createInfo.size;

  scratchSize = std::max(scratchSize, sizeInfo.buildScratchSize);
}

void RendererRayTraceClustersLod::updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, bool update)
{
  if(update)
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = m_tlas.accel;
  }
  else
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  }

  // Build Offsets info: n instances
  VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{uint32_t(m_renderInstances.size()), 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

  // Build the TLAS
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &m_tlasBuildInfo, &pBuildOffsetInfo);
}

std::unique_ptr<Renderer> makeRendererRayTraceClustersLod()
{
  return std::make_unique<RendererRayTraceClustersLod>();
}

void RendererRayTraceClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  vkDeviceWaitIdle(res.m_device);

  nvvk::WriteSetContainer writeSets;

  writeSets.append(m_dsetPack.getWriteSet(BINDINGS_HIZ_TEX), &res.m_hizUpdate.farImageInfo);
  writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth);

#if USE_DLSS
  if(m_config.useDlss)
  {
    // apply all
    for(uint32_t i = 0; i < DlssDenoiser::eDlssCount; i++)
    {
      VkDescriptorImageInfo renderTargetInfo =
          res.m_frameBuffer.dlssDenoiser.getDescriptorImageInfo(DlssDenoiser::DlssBufferType(i));
      writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RENDER_TARGET + i), &renderTargetInfo);
    }
  }
  else
#endif
  {
    VkDescriptorImageInfo renderTargetInfo;
    renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    renderTargetInfo.imageView   = res.m_frameBuffer.imgColor.descriptor.imageView;
    writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RENDER_TARGET), &renderTargetInfo);
  }

  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res, rscene);
}

}  // namespace lodclusters
