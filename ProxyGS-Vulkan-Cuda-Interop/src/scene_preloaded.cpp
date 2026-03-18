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

#include "scene_preloaded.hpp"

namespace lodclusters {

bool ScenePreloaded::init(Resources* res, const Scene* scene, const Config& config)
{
  assert(m_resources == nullptr && "init called without prior deinit");

  m_resources = res;
  m_scene     = scene;
  m_config    = config;

  Resources::BatchedUploader uploader(*res);

  m_shaderGeometries.resize(scene->getActiveGeometryCount());
  m_geometries.resize(scene->getActiveGeometryCount());

  VkDeviceSize sizeLimit           = (res->getDeviceLocalHeapSize() * 1000) / 800;
  VkDeviceSize clusterGeometrySize = 0;

  uint32_t instancesOffset = 0;
  for(size_t geometryIndex = 0; geometryIndex < scene->getActiveGeometryCount(); geometryIndex++)
  {
    shaderio::Geometry&        shaderGeometry  = m_shaderGeometries[geometryIndex];
    ScenePreloaded::Geometry&  preloadGeometry = m_geometries[geometryIndex];
    const Scene::GeometryView& sceneGeometry   = scene->getActiveGeometry(geometryIndex);

    // normally we would recommend using less buffers, and just aggregate this information in a single buffer per geometry.

    res->createBufferTyped(preloadGeometry.localTriangles, sceneGeometry.localTriangles.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    res->createBufferTyped(preloadGeometry.vertices, sceneGeometry.vertices.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    size_t numClusters = sceneGeometry.lodMesh.clusterTriangleRanges.size();
    res->createBufferTyped(preloadGeometry.clusters, numClusters, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    res->createBufferTyped(preloadGeometry.clusterBboxes, numClusters, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    res->createBufferTyped(preloadGeometry.clusterGeneratingGroups, numClusters, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    size_t numGroups = sceneGeometry.lodMesh.groupClusterRanges.size();
    res->createBufferTyped(preloadGeometry.groups, numGroups, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    size_t numNodes = sceneGeometry.lodHierarchy.nodes.size();
    res->createBufferTyped(preloadGeometry.nodes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    res->createBufferTyped(preloadGeometry.nodeBboxes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    uint32_t numLodLevels = sceneGeometry.lodLevelsCount;
    res->createBufferTyped(preloadGeometry.lodLevels, numLodLevels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);


    m_geometrySize += preloadGeometry.localTriangles.bufferSize;
    m_geometrySize += preloadGeometry.vertices.bufferSize;
    m_geometrySize += preloadGeometry.clusters.bufferSize;
    m_geometrySize += preloadGeometry.clusterBboxes.bufferSize;
    m_geometrySize += preloadGeometry.clusterGeneratingGroups.bufferSize;
    m_geometrySize += preloadGeometry.groups.bufferSize;
    m_geometrySize += preloadGeometry.nodes.bufferSize;
    m_geometrySize += preloadGeometry.nodeBboxes.bufferSize;

    clusterGeometrySize += preloadGeometry.localTriangles.bufferSize;
    clusterGeometrySize += preloadGeometry.vertices.bufferSize;
    clusterGeometrySize += preloadGeometry.clusters.bufferSize;

    // simple estimate extra clas size as raw copy
    if(m_geometrySize + clusterGeometrySize > sizeLimit)
    {
      LOGW("Likely exceeding device memory limit for preloaded scene\n");
      uploader.abort();
      deinit();
      return false;
    }

    // setup shaderio
    shaderGeometry                   = {};
    shaderGeometry.bbox              = sceneGeometry.bbox;
    shaderGeometry.nodes             = preloadGeometry.nodes.address;
    shaderGeometry.nodeBboxes        = preloadGeometry.nodeBboxes.address;
    shaderGeometry.preloadedGroups   = preloadGeometry.groups.address;
    shaderGeometry.preloadedClusters = preloadGeometry.clusters.address;
    shaderGeometry.lodLevelsCount    = uint32_t(numLodLevels);
    shaderGeometry.lodLevels         = preloadGeometry.lodLevels.address;
    shaderGeometry.lodsCompletedMask = (1 << shaderGeometry.lodLevelsCount) - 1;
    shaderGeometry.instancesCount    = sceneGeometry.instanceReferenceCount * scene->getGeometryInstanceFactor();
    shaderGeometry.instancesOffset   = instancesOffset;

    instancesOffset += shaderGeometry.instancesCount;

    // lowest detail group must have just a single cluster
    nvcluster_Range lastGroupRange = sceneGeometry.lodMesh.lodLevelGroupRanges.back();
    assert(lastGroupRange.count == 1);
    assert(sceneGeometry.lodMesh.groupClusterRanges[lastGroupRange.offset].count == 1);

    shaderGeometry.lowDetailClusterID = sceneGeometry.lodMesh.groupClusterRanges[lastGroupRange.offset].offset;
    shaderGeometry.lowDetailTriangles = sceneGeometry.lodMesh.clusterTriangleRanges[shaderGeometry.lowDetailClusterID].count;

    // basic uploads

    uploader.uploadBuffer(preloadGeometry.nodes, sceneGeometry.lodHierarchy.nodes.data());
    uploader.uploadBuffer(preloadGeometry.nodeBboxes, sceneGeometry.nodeBboxes.data());

    uploader.uploadBuffer(preloadGeometry.localTriangles, sceneGeometry.localTriangles.data());
    uploader.uploadBuffer(preloadGeometry.vertices, sceneGeometry.vertices.data());

    uploader.uploadBuffer(preloadGeometry.clusterBboxes, sceneGeometry.clusterBboxes.data());
    uploader.uploadBuffer(preloadGeometry.clusterGeneratingGroups, sceneGeometry.lodMesh.clusterGeneratingGroups.data());

    uploader.uploadBuffer(preloadGeometry.lodLevels, sceneGeometry.lodLevels.data());

    // clusters and groups need to be filled manually

    shaderio::Cluster* clusters = uploader.uploadBuffer(preloadGeometry.clusters, (shaderio::Cluster*)nullptr);

    for(size_t c = 0; c < numClusters; c++)
    {
      nvcluster_Range vertexRange   = sceneGeometry.clusterVertexRanges[c];
      nvcluster_Range triangleRange = sceneGeometry.lodMesh.clusterTriangleRanges[c];

      shaderio::Cluster& cluster    = clusters[c];
      cluster                       = {};
      cluster.triangleCountMinusOne = uint8_t(triangleRange.count - 1);
      cluster.vertexCountMinusOne   = uint8_t(vertexRange.count - 1);

      // setup pointers to where relevant data is stored
      cluster.vertices = preloadGeometry.vertices.addressAt(vertexRange.offset, vertexRange.count);
      cluster.localTriangles = preloadGeometry.localTriangles.addressAt(triangleRange.offset * 3, triangleRange.count * 3);
    }

    shaderio::Group* groups = uploader.uploadBuffer(preloadGeometry.groups, (shaderio::Group*)nullptr, Resources::DONT_FLUSH);

    for(size_t g = 0; g < sceneGeometry.lodMesh.groupClusterRanges.size(); g++)
    {
      nvcluster_Range clusterRange = sceneGeometry.lodMesh.groupClusterRanges[g];
      uint8_t         lodLevel     = sceneGeometry.groupLodLevels[g];

      shaderio::Group& group                = groups[g];
      group                                 = {};
      group.geometryID                      = uint32_t(geometryIndex);
      group.groupID                         = uint32_t(g);
      group.lodLevel                        = lodLevel;
      group.clusterCount                    = clusterRange.count;
      group.traversalMetric.boundingSphereX = sceneGeometry.lodHierarchy.groupCumulativeBoundingSpheres[g].center.x;
      group.traversalMetric.boundingSphereY = sceneGeometry.lodHierarchy.groupCumulativeBoundingSpheres[g].center.y;
      group.traversalMetric.boundingSphereZ = sceneGeometry.lodHierarchy.groupCumulativeBoundingSpheres[g].center.z;
      group.traversalMetric.boundingSphereRadius = sceneGeometry.lodHierarchy.groupCumulativeBoundingSpheres[g].radius;
      group.traversalMetric.maxQuadricError      = sceneGeometry.lodHierarchy.groupCumulativeQuadricError[g];

      // setup pointers to where relevant data is stored
      group.clusterGeneratingGroups =
          preloadGeometry.clusterGeneratingGroups.addressAt(clusterRange.offset, clusterRange.count);
      group.clusterBboxes = preloadGeometry.clusterBboxes.addressAt(clusterRange.offset, clusterRange.count);

      // for preloaded data we match the per-geometry global id
      // since all arrays are fully accessible
      group.residentID        = uint32_t(g);
      group.clusterResidentID = clusterRange.offset;

      // fill in properties of clusters
      for(uint32_t c = 0; c < clusterRange.count; c++)
      {
        uint32_t clusterIndex                  = clusterRange.offset + c;
        clusters[clusterIndex].lodLevel        = uint8_t(lodLevel);
        clusters[clusterIndex].groupChildIndex = uint8_t(c);
        clusters[clusterIndex].groupID         = uint32_t(g);
      }
    }
  }

  res->createBufferTyped(m_shaderGeometriesBuffer, scene->getActiveGeometryCount(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_operationsSize += logMemoryUsage(m_shaderGeometriesBuffer.bufferSize, "operations", "preloaded geo buffer");

  uploader.uploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());
  uploader.flush();

  return true;
}

bool ScenePreloaded::updateClasRequired(bool state)
{
  if(state != m_hasClas)
  {
    if(state)
    {
      return initClas();
    }
    else
    {
      deinitClas();
    }
  }

  return true;
}

void ScenePreloaded::deinit()
{
  if(!m_resources)
    return;

  for(auto& it : m_geometries)
  {
    m_resources->m_allocator.destroyBuffer(it.localTriangles);
    m_resources->m_allocator.destroyBuffer(it.vertices);
    m_resources->m_allocator.destroyBuffer(it.clusters);
    m_resources->m_allocator.destroyBuffer(it.clusterGeneratingGroups);
    m_resources->m_allocator.destroyBuffer(it.clusterBboxes);
    m_resources->m_allocator.destroyBuffer(it.clusterClasSizes);
    m_resources->m_allocator.destroyBuffer(it.clusterClasAddresses);
    m_resources->m_allocator.destroyBuffer(it.groups);
    m_resources->m_allocator.destroyBuffer(it.nodes);
    m_resources->m_allocator.destroyBuffer(it.nodeBboxes);
    m_resources->m_allocator.destroyBuffer(it.lodLevels);
    m_resources->m_allocator.destroyBuffer(it.clasData);
  }

  m_resources->m_allocator.destroyBuffer(m_clasLowDetailBlasBuffer);

  m_resources->m_allocator.destroyBuffer(m_shaderGeometriesBuffer);
  m_resources = nullptr;
  m_scene     = nullptr;
}

bool ScenePreloaded::initClas()
{
  Resources* res = m_resources;
  m_hasClas      = true;

  m_clasOperationsSize = 0;
  m_clasSize           = 0;

  VkClusterAccelerationStructureTriangleClusterInputNV clusterTriangleInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  clusterTriangleInput.maxClusterTriangleCount       = m_scene->m_maxClusterTriangles;
  clusterTriangleInput.maxClusterVertexCount         = m_scene->m_maxClusterVertices;
  clusterTriangleInput.maxClusterUniqueGeometryCount = 1;
  clusterTriangleInput.maxGeometryIndexValue         = 0;
  clusterTriangleInput.minPositionTruncateBitCount   = m_config.clasPositionTruncateBits;
  clusterTriangleInput.maxTotalTriangleCount         = m_scene->m_maxPerGeometryTriangles;
  clusterTriangleInput.maxTotalVertexCount           = m_scene->m_maxPerGeometryVertices;
  clusterTriangleInput.vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT;

  VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
  // low detail blas has only one cluster per blas
  blasInput.maxClusterCountPerAccelerationStructure = 1;
  blasInput.maxTotalClusterCount                    = uint32_t(m_scene->getActiveGeometryCount());

  VkDeviceSize scratchSize = 0;
  VkDeviceSize blasSize    = 0;

  VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  VkClusterAccelerationStructureInputInfoNV clusterInputInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  clusterInputInfo.flags                         = m_config.clasBuildFlags;
  clusterInputInfo.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  clusterInputInfo.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
  clusterInputInfo.opInput.pTriangleClusters     = &clusterTriangleInput;
  clusterInputInfo.maxAccelerationStructureCount = m_scene->m_maxPerGeometryClusters;
  vkGetClusterAccelerationStructureBuildSizesNV(res->m_device, &clusterInputInfo, &buildSizesInfo);
  scratchSize = std::max(scratchSize, buildSizesInfo.buildScratchSize);

  VkClusterAccelerationStructureInputInfoNV blasInputInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  blasInputInfo.flags                        = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  blasInputInfo.opMode                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  blasInputInfo.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  blasInputInfo.opInput.pClustersBottomLevel = &blasInput;
  blasInputInfo.maxAccelerationStructureCount = uint32_t(m_scene->getActiveGeometryCount());
  vkGetClusterAccelerationStructureBuildSizesNV(res->m_device, &blasInputInfo, &buildSizesInfo);
  scratchSize = std::max(scratchSize, buildSizesInfo.buildScratchSize);
  blasSize    = buildSizesInfo.accelerationStructureSize;

  nvvk::Buffer scratchTemp;
  res->createBuffer(scratchTemp, scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  nvvk::BufferTyped<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> clasBuildInfosHost;
  res->createBufferTyped(clasBuildInfosHost, m_scene->m_maxPerGeometryClusters,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::BufferTyped<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> blasBuildInfosHost;
  res->createBufferTyped(blasBuildInfosHost, m_scene->getActiveGeometryCount(),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::BufferTyped<uint32_t> buildSizesHost;
  res->createBufferTyped(buildSizesHost,
                         std::max(m_scene->m_maxPerGeometryClusters, uint32_t(m_scene->getActiveGeometryCount())),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  nvvk::BufferTyped<uint64_t> buildAddressesHost;
  res->createBufferTyped(buildAddressesHost,
                         std::max(m_scene->m_maxPerGeometryClusters, uint32_t(m_scene->getActiveGeometryCount())),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  VkCommandBuffer cmd;
  VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};

  // for every geometry build clas

  for(size_t g = 0; g < m_scene->getActiveGeometryCount(); g++)
  {
    ScenePreloaded::Geometry&  preloadGeometry = m_geometries[g];
    shaderio::Geometry&        shaderGeometry  = m_shaderGeometries[g];
    const Scene::GeometryView& sceneGeometry   = m_scene->getActiveGeometry(g);

    VkClusterAccelerationStructureBuildTriangleClusterInfoNV* buildInfos = clasBuildInfosHost.data();
    for(uint32_t c = 0; c < sceneGeometry.totalClustersCount; c++)
    {
      nvcluster_Range triangleRange = sceneGeometry.lodMesh.clusterTriangleRanges[c];
      nvcluster_Range vertexRange   = sceneGeometry.clusterVertexRanges[c];

      VkClusterAccelerationStructureBuildTriangleClusterInfoNV& buildInfo = buildInfos[c];
      buildInfo                                                           = {};

      buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
      buildInfo.clusterID         = c;
      buildInfo.triangleCount     = triangleRange.count;
      buildInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
      buildInfo.indexBufferStride = 1;
      buildInfo.indexBuffer = preloadGeometry.localTriangles.addressAt(triangleRange.offset * 3, triangleRange.count * 3);
      buildInfo.vertexCount              = vertexRange.count;
      buildInfo.vertexBufferStride       = uint16_t(sizeof(glm::vec4));
      buildInfo.vertexBuffer             = preloadGeometry.vertices.addressAt(vertexRange.offset, vertexRange.count);
      buildInfo.positionTruncateBitCount = m_config.clasPositionTruncateBits;
    }

    size_t numClusters = sceneGeometry.totalClustersCount;
    res->createBufferTyped(preloadGeometry.clusterClasAddresses, numClusters,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    res->createBufferTyped(preloadGeometry.clusterClasSizes, numClusters,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    m_clasOperationsSize += preloadGeometry.clusterClasAddresses.bufferSize;
    m_clasOperationsSize += preloadGeometry.clusterClasSizes.bufferSize;

    // update shader visible pointers
    shaderGeometry.preloadedClusterClasAddresses = preloadGeometry.clusterClasAddresses.address;
    shaderGeometry.preloadedClusterClasSizes     = preloadGeometry.clusterClasSizes.address;

    clusterTriangleInput.maxTotalTriangleCount = sceneGeometry.totalTriangleCount;
    clusterTriangleInput.maxTotalVertexCount   = sceneGeometry.totalVerticesCount;

    clusterInputInfo.maxAccelerationStructureCount = sceneGeometry.totalClustersCount;
    clusterInputInfo.opInput.pTriangleClusters     = &clusterTriangleInput;

    cmdInfo.srcInfosArray.deviceAddress = clasBuildInfosHost.address;
    cmdInfo.srcInfosArray.size          = clasBuildInfosHost.bufferSize;
    cmdInfo.srcInfosArray.stride        = sizeof(shaderio::ClasBuildInfo);

    cmdInfo.dstSizesArray.deviceAddress = preloadGeometry.clusterClasSizes.address;
    cmdInfo.dstSizesArray.size          = preloadGeometry.clusterClasSizes.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.scratchData = scratchTemp.address;

    cmd = res->createTempCmdBuffer();
    {
      // build size mode only

      cmdInfo.input        = clusterInputInfo;
      cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;

      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

      // download sizes

      VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      memBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);

      VkBufferCopy region = {};
      region.size         = sizeof(uint32_t) * sceneGeometry.totalClustersCount;
      vkCmdCopyBuffer(cmd, preloadGeometry.clusterClasSizes.buffer, buildSizesHost.buffer, 1, &region);
    }
    res->tempSyncSubmit(cmd);

    // allocate clas data and setup per-clas offsets
    {
      uint64_t        sumClasSizes = 0;
      const uint32_t* clasSizes    = buildSizesHost.data();
      for(uint32_t c = 0; c < sceneGeometry.totalClustersCount; c++)
      {
        assert(clasSizes[c] && "clas with invalid size");
        sumClasSizes += clasSizes[c];
      }

      res->createBuffer(preloadGeometry.clasData, sumClasSizes, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

      uint64_t  clasOffset    = 0;
      uint64_t* clasAddresses = buildAddressesHost.data();
      for(uint32_t c = 0; c < sceneGeometry.totalClustersCount; c++)
      {
        clasAddresses[c] = preloadGeometry.clasData.address + clasOffset;
        clasOffset += clasSizes[c];
      }
    }

    cmd = res->createTempCmdBuffer();
    {
      // upload addresses
      VkBufferCopy region = {};
      region.size         = sizeof(uint64_t) * sceneGeometry.totalClustersCount;
      vkCmdCopyBuffer(cmd, buildAddressesHost.buffer, preloadGeometry.clusterClasAddresses.buffer, 1, &region);

      VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
      memBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);

      // build for real

      cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

      cmdInfo.dstAddressesArray.deviceAddress = preloadGeometry.clusterClasAddresses.address;
      cmdInfo.dstAddressesArray.size          = preloadGeometry.clusterClasAddresses.bufferSize;
      cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
    }
    res->tempSyncSubmit(cmd);


    {
      // setup blas build info
      VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV* blasInfos = blasBuildInfosHost.data();
      // just a single cluster
      blasInfos[g].clusterReferencesCount  = 1;
      blasInfos[g].clusterReferencesStride = uint32_t(sizeof(uint64_t));
      blasInfos[g].clusterReferences = preloadGeometry.clusterClasAddresses.addressAt(shaderGeometry.lowDetailClusterID);
    }

    m_clasSize += preloadGeometry.clasData.bufferSize;
  }

  logMemoryUsage(m_clasOperationsSize, "operations", "preloaded clas");

  // create low detail blas
  {
    res->createBuffer(m_clasLowDetailBlasBuffer, blasSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_clasOperationsSize += logMemoryUsage(blasSize, "operations", "preloaded clas lowdetail blas");

    cmd = res->createTempCmdBuffer();

    cmdInfo.input        = blasInputInfo;
    cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;

    cmdInfo.dstImplicitData = m_clasLowDetailBlasBuffer.address;

    cmdInfo.srcInfosArray.deviceAddress = blasBuildInfosHost.address;
    cmdInfo.srcInfosArray.size          = blasBuildInfosHost.bufferSize;
    cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

    cmdInfo.dstSizesArray.deviceAddress = buildSizesHost.address;
    cmdInfo.dstSizesArray.size          = buildSizesHost.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.dstAddressesArray.deviceAddress = buildAddressesHost.address;
    cmdInfo.dstAddressesArray.size          = buildAddressesHost.bufferSize;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res->tempSyncSubmit(cmd);

    const uint64_t* blasAddresses = buildAddressesHost.data();

    for(size_t g = 0; g < m_scene->getActiveGeometryCount(); g++)
    {
      ScenePreloaded::Geometry& preloadGeometry = m_geometries[g];
      shaderio::Geometry&       shaderGeometry  = m_shaderGeometries[g];

      shaderGeometry.lowDetailBlasAddress = blasAddresses[g];
    }
  }

  m_resources->simpleUploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  res->m_allocator.destroyBuffer(scratchTemp);
  res->m_allocator.destroyBuffer(buildSizesHost);
  res->m_allocator.destroyBuffer(buildAddressesHost);
  res->m_allocator.destroyBuffer(clasBuildInfosHost);
  res->m_allocator.destroyBuffer(blasBuildInfosHost);

  return true;
}

void ScenePreloaded::deinitClas()
{
  for(size_t g = 0; g < m_geometries.size(); g++)
  {
    ScenePreloaded::Geometry& preloadGeometry = m_geometries[g];
    shaderio::Geometry&       shaderGeometry  = m_shaderGeometries[g];
    m_resources->m_allocator.destroyBuffer(preloadGeometry.clusterClasAddresses);
    m_resources->m_allocator.destroyBuffer(preloadGeometry.clusterClasSizes);
    m_resources->m_allocator.destroyBuffer(preloadGeometry.clasData);
    shaderGeometry.preloadedClusterClasAddresses = 0;
    shaderGeometry.preloadedClusterClasSizes     = 0;
    shaderGeometry.lowDetailBlasAddress          = 0;
  }

  m_resources->m_allocator.destroyBuffer(m_clasLowDetailBlasBuffer);

  m_resources->simpleUploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  m_clasSize           = 0;
  m_clasOperationsSize = 0;
  m_hasClas            = false;
}
}  // namespace lodclusters
