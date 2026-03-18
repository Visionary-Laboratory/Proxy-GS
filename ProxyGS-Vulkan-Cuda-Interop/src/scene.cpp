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

#include <random>
#include <string.h>

#include <meshoptimizer.h>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/file_operations.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "nvclusterlod/nvclusterlod_common.h"
#include "nvclusterlod/nvclusterlod_hierarchy.h"
#include "scene.hpp"


namespace lodclusters {
static_assert(sizeof(shaderio::Node) == sizeof(nvclusterlod_HierarchyNode));
static_assert(offsetof(shaderio::Node, nodeRange) == offsetof(nvclusterlod_HierarchyNode, children));
static_assert(offsetof(shaderio::Node, traversalMetric) == offsetof(nvclusterlod_HierarchyNode, boundingSphere));
static_assert(offsetof(shaderio::Node, traversalMetric) + offsetof(shaderio::TraversalMetric, maxQuadricError)
              == offsetof(nvclusterlod_HierarchyNode, maxClusterQuadricError));
static_assert(sizeof(glm::vec4) == sizeof(nvclusterlod_Sphere));
static_assert(NVCLUSTERLOD_ORIGINAL_MESH_GROUP == SHADERIO_ORIGINAL_MESH_GROUP);


void Scene::ProcessingInfo::init(float processingThreadsPct)
{
  numPoolThreadsOriginal = nvutils::get_thread_pool().get_thread_count();

  numPoolThreads = numPoolThreadsOriginal;
  if(processingThreadsPct > 0.0f && processingThreadsPct < 1.0f)
  {
    numPoolThreads = std::min(numPoolThreads, std::max(1u, uint32_t(ceilf(float(numPoolThreads) * processingThreadsPct))));

    if(numPoolThreads != numPoolThreadsOriginal)
      nvutils::get_thread_pool().reset(numPoolThreads);
  }
}

void Scene::ProcessingInfo::setupParallelism(size_t geometryCount_)
{
  geometryCount = geometryCount_;

  bool preferInnerParallelism = geometryCount < numPoolThreads;

  numOuterThreads = preferInnerParallelism ? 1 : numPoolThreads;
  numInnerThreads = preferInnerParallelism ? numPoolThreads : 1;

  nvcluster_ContextCreateInfo clusterContextInfo;
  clusterContextInfo.parallelize = preferInnerParallelism ? 1 : 0;
  nvclusterCreateContext(&clusterContextInfo, &clusterContext);

  nvclusterlod_ContextCreateInfo lodContextInfo;
  lodContextInfo.parallelize    = preferInnerParallelism ? 1 : 0;
  lodContextInfo.clusterContext = clusterContext;
  nvclusterlodCreateContext(&lodContextInfo, &lodContext);
}

void Scene::ProcessingInfo::logBegin()
{
  LOGI("... geometry load & processing: geometries %lu, threads outer %d inner %d\n", static_cast<unsigned long>(geometryCount), numOuterThreads, numInnerThreads);

  startTime = clock.getMicroseconds();
}

void Scene::ProcessingInfo::logCompletedGeometry()
{
  std::lock_guard lock(progressMutex);

  progressGeometriesCompleted++;

  // statistics
  const uint32_t precentageGranularity = 5;
  uint32_t       percentage            = uint32_t(size_t(progressGeometriesCompleted * 100) / geometryCount);
  percentage                           = (percentage / precentageGranularity) * precentageGranularity;

  if(percentage > progressLastPercentage)
  {
    progressLastPercentage = percentage;
    LOGI("... geometry load & processing: %3d%%\n", percentage);
  }
}

void Scene::ProcessingInfo::logEnd()
{
  double endTime = clock.getMicroseconds();

  LOGI("... geometry load & processing: %f milliseconds\n", (endTime - startTime) / 1000.0f);
}

void Scene::ProcessingInfo::deinit()
{
  if(lodContext)
    nvclusterlodDestroyContext(lodContext);
  if(clusterContext)
    nvclusterDestroyContext(clusterContext);

  if(numPoolThreads != numPoolThreadsOriginal)
    nvutils::get_thread_pool().reset(numPoolThreadsOriginal);
}

bool Scene::init(const std::filesystem::path& filePath, const SceneConfig& config, bool skipCache)
{
  *this = {};

  m_filePath        = filePath;
  m_config          = config;
  m_loadedFromCache = false;

  m_cacheFilePath = filePath;

  std::string oldExtension = filePath.extension().string();
  m_cacheFilePath.replace_extension(oldExtension + ".nvsngeo");

  std::string cacheFileName = nvutils::utf8FromPath(m_cacheFilePath);

  if(!skipCache && m_config.autoLoadCache && m_cacheFileMapping.open(m_cacheFilePath))
  {
    m_cacheFileView.init(m_cacheFileMapping.size(), m_cacheFileMapping.data());
    if(m_cacheFileView.isValid())
    {
      // when loading results from the cache, we cannot change the cluster or lod settings of a scene,
      // it's considered read only.
      m_loadedFromCache = true;
      LOGI("Scene::init using cache file %s\n", cacheFileName.c_str());
    }
    else
    {
      m_cacheFileView.deinit();
      m_cacheFileMapping.close();
    }
  }

  ProcessingInfo processingInfo;
  processingInfo.init(m_config.processingThreadsPct);

  bool loadSuccess = loadGLTF(processingInfo, filePath);

  processingInfo.deinit();

  if(!loadSuccess)
  {
    if(m_cacheFileView.isValid())
    {
      m_cacheFileView.deinit();
      m_cacheFileMapping.close();
    }

    return false;
  }

  m_originalInstanceCount = m_instances.size();
  m_originalGeometryCount = m_geometryViews.size();
  m_activeGeometryCount   = m_originalGeometryCount;

  computeClusterStats();
  computeInstanceBBoxes();
  m_gridBbox = m_bbox;

  glm::vec3 modelExtent = m_bbox.hi - m_bbox.lo;
  m_isBig = modelExtent.y < 0.15f * std::max(modelExtent.x, modelExtent.z) && m_originalInstanceCount > 1024;

  for(auto& geometry : m_geometryViews)
  {
    m_hiPerGeometryTriangles  = std::max(m_hiPerGeometryTriangles, geometry.hiTriangleCount);
    m_hiPerGeometryVertices   = std::max(m_hiPerGeometryVertices, geometry.hiVerticesCount);
    m_hiPerGeometryClusters   = std::max(m_hiPerGeometryClusters, geometry.hiClustersCount);
    m_maxPerGeometryTriangles = std::max(m_maxPerGeometryTriangles, geometry.totalTriangleCount);
    m_maxPerGeometryVertices  = std::max(m_maxPerGeometryVertices, geometry.totalVerticesCount);
    m_maxPerGeometryClusters  = std::max(m_maxPerGeometryClusters, geometry.totalClustersCount);
    m_maxClusterVertices      = std::max(m_maxClusterVertices, geometry.clusterMaxVerticesCount);
    m_maxClusterTriangles     = std::max(m_maxClusterTriangles, geometry.clusterMaxTrianglesCount);
    m_hiTrianglesCount += geometry.hiTriangleCount;
    m_hiClustersCount += geometry.hiClustersCount;
    m_totalClustersCount += geometry.totalClustersCount;
  }
  for(size_t i = 0; i < m_instances.size(); i++)
  {
    const GeometryView& geometry = m_geometryViews[m_instances[i].geometryID];
    m_hiTrianglesCountInstanced += geometry.hiTriangleCount;
    m_hiClustersCountInstanced += geometry.hiClustersCount;
  }

  if(m_config.clusterStripify && (processingInfo.numTotalStrips.load() > 0))
  {
    LOGI("Average triangles per strip %.2f\n",
         double(processingInfo.numTotalTriangles.load()) / double(processingInfo.numTotalStrips.load()));
  }

  if(!m_loadedFromCache && m_config.autoSaveCache)
  {
    saveCache();
  }

  if(m_loadedFromCache && !m_config.memoryMappedCache)
  {
    // everything was loaded into system memory,
    // close file mappings
    m_cacheFileView.deinit();
    m_cacheFileMapping.close();
  }

  return true;
}

void Scene::deinit()
{
  *this = {};
}

void Scene::updateSceneGrid(const SceneGridConfig& gridConfig)
{
  m_gridConfig = gridConfig;

  size_t copiesCount = std::max(1u, gridConfig.numCopies);

  size_t numOldCopies = m_instances.size() / m_originalInstanceCount;

  m_instances.resize(m_originalInstanceCount * copiesCount);
  m_activeGeometryCount = gridConfig.uniqueGeometriesForCopies ? m_originalGeometryCount * copiesCount : m_originalGeometryCount;

  std::default_random_engine            rng(2342);
  std::uniform_real_distribution<float> randomUnorm(0.0f, 1.0f);

  uint32_t axis    = gridConfig.gridBits;
  size_t   sq      = 1;
  int      numAxis = 0;
  if(!axis)
    axis = 3;

  for(int i = 0; i < 3; i++)
  {
    numAxis += (axis & (1 << i)) ? 1 : 0;
  }

  switch(numAxis)
  {
    case 1:
      sq = copiesCount;
      break;
    case 2:
      while(sq * sq < copiesCount)
      {
        sq++;
      }
      break;
    case 3:
      while(sq * sq * sq < copiesCount)
      {
        sq++;
      }
      break;
  }


  size_t lastCopyIndex = 0;

  glm::vec3 modelExtent = (m_bbox.hi - m_bbox.lo);
  glm::vec3 modelCenter = (m_bbox.hi + m_bbox.lo) * 0.5f;
  float     modelSize   = glm::length(modelExtent);
  glm::vec3 gridShift;
  glm::mat4 gridRotMatrix;

  for(size_t copyIndex = 1; copyIndex < copiesCount; copyIndex++)
  {
    gridShift = gridConfig.refShift * modelExtent;
    size_t c  = copyIndex;

    float u = 0;
    float v = 0;
    float w = 0;

    switch(numAxis)
    {
      case 1:
        u = float(c);
        break;
      case 2:
        u = float(c % sq);
        v = float(c / sq);
        break;
      case 3:
        u = float(c % sq);
        v = float((c / sq) % sq);
        w = float(c / (sq * sq));
        break;
    }

    float use = u;

    if(axis & (1 << 0))
    {
      gridShift.x *= -use;
      if(numAxis > 1)
        use = v;
    }
    else
    {
      gridShift.x = 0;
    }

    if(axis & (1 << 1))
    {
      gridShift.y *= use;
      if(numAxis > 2)
        use = w;
      else if(numAxis > 1)
        use = v;
    }
    else
    {
      gridShift.y = 0;
    }

    if(axis & (1 << 2))
    {
      gridShift.z *= -use;
    }
    else
    {
      gridShift.z = 0;
    }

    glm::mat4 scaleMatrix = glm::mat4(1.0f);

    if(gridConfig.minScale != 1.0f || gridConfig.maxScale != 1.0f)
    {
      float scale = glm::mix(gridConfig.minScale, gridConfig.maxScale, randomUnorm(rng));
      scaleMatrix = glm::scale(scaleMatrix, glm::vec3(scale));

      if(scale < 1.0f)
      {
        gridShift.y += modelSize * (1.0f - scale);
      }
      else
      {
        gridShift.y -= modelSize * (scale - 1.0f);
      }
    }

    if(axis & (8 | 16 | 32))
    {
      glm::vec3 mask    = {axis & 8 ? 1.0f : 0.0f, axis & 16 ? 1.0f : 0.0f, axis & 32 ? 1.0f : 0.0f};
      glm::vec3 gridDir = glm::vec3(randomUnorm(rng), randomUnorm(rng), randomUnorm(rng));
      gridDir           = glm::max(gridDir * mask, mask * 0.00001f);
      float gridAngle   = randomUnorm(rng) * 360.0f;
      gridDir           = glm::normalize(gridDir);

      // snap angle
      if(gridConfig.snapAngle > 0.0)
      {
        float remainder = std::fmod(gridAngle, gridConfig.snapAngle);
        gridAngle       = gridAngle - remainder;
      }

      // to radians
      gridAngle = gridAngle * glm::pi<float>() / 180.0f;

      gridRotMatrix = glm::rotate(glm::mat4(1), gridAngle, gridDir);
    }

    for(size_t i = 0; i < m_originalInstanceCount; i++)
    {
      Instance& instance = m_instances[i + copyIndex * m_originalInstanceCount];
      // copy state from reference
      instance = m_instances[i];

      if(gridConfig.uniqueGeometriesForCopies)
      {
        // apply unique set if geometryIDs to this set of instances
        instance.geometryID += uint32_t(c * m_originalGeometryCount);
      }

      // modify matrix for the grid
      glm::mat4 worldMatrix = m_instances[i].matrix;
      glm::vec3 translation = worldMatrix[3];
      worldMatrix[3]        = glm::vec4(translation - modelCenter, 1.f);

      worldMatrix = scaleMatrix * worldMatrix;

      if(axis & (8 | 16 | 32))
      {
        worldMatrix = gridRotMatrix * worldMatrix;
      }
      translation    = worldMatrix[3];
      worldMatrix[3] = glm::vec4(translation + modelCenter + gridShift, 1.f);

      instance.matrix = worldMatrix;
    }
  }

  m_gridBbox = m_bbox;
  computeInstanceBBoxes();
  std::swap(m_gridBbox, m_bbox);
}

void Scene::computeInstanceBBoxes()
{
  m_bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  for(auto& instance : m_instances)
  {
    const GeometryView& geometry = getActiveGeometry(instance.geometryID);

    instance.bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

    for(uint32_t v = 0; v < 8; v++)
    {
      bool x = (v & 1) != 0;
      bool y = (v & 2) != 0;
      bool z = (v & 4) != 0;

      glm::bvec3 weight(x, y, z);
      glm::vec3  corner = glm::mix(geometry.bbox.lo, geometry.bbox.hi, weight);
      corner            = instance.matrix * glm::vec4(corner, 1.0f);
      instance.bbox.lo  = glm::min(instance.bbox.lo, corner);
      instance.bbox.hi  = glm::max(instance.bbox.hi, corner);
    }

    m_bbox.lo = glm::min(m_bbox.lo, instance.bbox.lo);
    m_bbox.hi = glm::max(m_bbox.hi, instance.bbox.hi);
  }
}

void Scene::computeClusterStats()
{
  for(size_t i = 0; i < m_geometryViews.size(); i++)
  {
    m_maxClusterVertices = std::max(m_maxClusterVertices, m_geometryViews[i].clusterMaxVerticesCount);
  }

  // reset settings in case we had a valid cache file
  if(m_cacheFileView.isValid())
  {
    // override settings
    m_config.clusterTriangles         = 0;
    m_config.clusterVertices          = 0;
    m_config.clusterGroupSize         = 0;
    m_config.lodLevelDecimationFactor = 0;

    for(size_t g = 0; g < m_geometryViews.size(); g++)
    {
      GeometryView& geometry = m_geometryViews[g];

      m_config.clusterTriangles = std::max(m_config.clusterTriangles, geometry.lodInfo.clusterConfig.maxClusterSize);

      // special case for vertices
      if(geometry.lodInfo.clusterConfig.maxClusterVertices == ~0 || geometry.lodInfo.clusterConfig.maxClusterVertices == 0)
      {
        m_config.clusterVertices = std::max(m_config.clusterVertices, geometry.lodInfo.clusterConfig.maxClusterSize * 3);
      }
      else
      {
        m_config.clusterVertices = std::max(m_config.clusterVertices, geometry.lodInfo.clusterConfig.maxClusterVertices);
      }

      m_config.clusterGroupSize = std::max(m_config.clusterGroupSize, geometry.lodInfo.groupConfig.maxClusterSize);
      m_config.lodLevelDecimationFactor = std::max(m_config.lodLevelDecimationFactor, geometry.lodInfo.decimationFactor);
    }
  }

  computeHistograms();
}

void Scene::processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached)
{
  GeometryStorage& geometryStorage = m_geometryStorages[geometryIndex];
  GeometryView&    geometryView    = m_geometryViews[geometryIndex];
  
  // Show progress for current geometry being processed
  LOGI("  Processing geometry %zu/%zu %s\n", geometryIndex + 1, processingInfo.geometryCount, 
       isCached ? "(cached)" : "(new)");

  bool viewFromStorage = true;

  if(isCached)
  {
    if(m_config.memoryMappedCache)
    {
      m_cacheFileView.getGeometryView(geometryView, geometryIndex);

      viewFromStorage = false;
    }
    else
    {
      loadCachedGeometry(geometryStorage, geometryIndex);
    }
  }
  else
  {
    if(geometryStorage.globalTriangles.empty())
    {
      geometryStorage = {};
    }
    else
    {
      buildGeometryClusters(processingInfo, geometryStorage);

      // no longer need original triangles
      geometryStorage.globalTriangles = {};

      if(geometryStorage.lodMesh.clusterTriangleRanges.empty())
        return;

      if(m_config.clusterStripify)
      {
        LOGI("    Building cluster strips...\n");
        buildGeometryClusterStrips(processingInfo, geometryStorage);
      }

      LOGI("    Building cluster vertices...\n");
      buildGeometryClusterVertices(processingInfo, geometryStorage);

      // no longer need vertex indirection
      geometryStorage.localVertices = {};

      LOGI("    Building bounding boxes...\n");
      buildGeometryBboxes(processingInfo, geometryStorage);
    }
  }

  if(viewFromStorage)
  {
    (GeometryBase&)geometryView = geometryStorage;

    geometryView.vertices            = geometryStorage.vertices;
    geometryView.localTriangles      = geometryStorage.localTriangles;
    geometryView.clusterVertexRanges = geometryStorage.clusterVertexRanges;
    geometryView.clusterBboxes       = geometryStorage.clusterBboxes;
    geometryView.groupLodLevels      = geometryStorage.groupLodLevels;
    geometryView.lodLevels           = geometryStorage.lodLevels;

    nvclusterlod::toView(geometryStorage.lodHierarchy, geometryView.lodHierarchy);
    nvclusterlod::toView(geometryStorage.lodMesh, geometryView.lodMesh);
    geometryView.nodeBboxes = geometryStorage.nodeBboxes;
  }

  if(m_processingOnlyFile)
  {
    saveProcessingOnly(processingInfo, geometryIndex);
  }
}

void Scene::buildGeometryClusters(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  nvclusterlod_Result result;

  nvclusterlod_MeshInput lodMeshInput;
  lodMeshInput.decimationFactor = m_config.lodLevelDecimationFactor;
  lodMeshInput.triangleCount    = uint32_t(geometry.globalTriangles.size());
  lodMeshInput.triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(geometry.globalTriangles.data());
  lodMeshInput.vertexCount      = uint32_t(geometry.vertices.size());
  lodMeshInput.vertexStride     = sizeof(glm::vec4);
  lodMeshInput.vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(geometry.vertices.data());

  lodMeshInput.clusterConfig = getClusterConfig();
  lodMeshInput.groupConfig   = getGroupConfig();

  geometry.lodInfo.clusterConfig            = lodMeshInput.clusterConfig;
  geometry.lodInfo.groupConfig              = lodMeshInput.groupConfig;
  geometry.lodInfo.decimationFactor         = lodMeshInput.decimationFactor;
  geometry.lodInfo.inputTriangleCount       = geometry.globalTriangles.size();
  geometry.lodInfo.inputVertexCount         = geometry.vertices.size();
  geometry.lodInfo.inputTriangleIndicesHash = 0;
  geometry.lodInfo.inputVerticesHash        = 0;

  // Add progress reporting for mesh clustering
  LOGI("    Processing mesh: %d triangles, %d vertices -> meshoptimizer clustering...\n", 
       lodMeshInput.triangleCount, lodMeshInput.vertexCount);
  
  nvutils::PerformanceTimer meshTimer;
  double meshStartTime = meshTimer.getMicroseconds();
  
  result = nvclusterlod::generateLodMesh(processingInfo.lodContext, lodMeshInput, geometry.lodMesh);
  
  double meshEndTime = meshTimer.getMicroseconds();
  LOGI("    Mesh clustering completed: %.2f seconds\n", (meshEndTime - meshStartTime) / 1000000.0);
  
  if(result != NVCLUSTERLOD_SUCCESS)
  {
    assert(0);
    LOGE("nvclusterlod::generateLodMesh failed: %d\n", result);
    std::exit(-1);
  }
  
  // Report LOD generation results
  LOGI("    Generated %zu clusters, %zu LOD levels\n", 
       geometry.lodMesh.clusterTriangleRanges.size(),
       geometry.lodMesh.lodLevelGroupRanges.size());

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.lodMesh.triangleVertices.shrink_to_fit();
  geometry.lodMesh.clusterTriangleRanges.shrink_to_fit();
  geometry.lodMesh.clusterGeneratingGroups.shrink_to_fit();
  geometry.lodMesh.clusterBoundingSpheres.shrink_to_fit();
  geometry.lodMesh.groupQuadricErrors.shrink_to_fit();
  geometry.lodMesh.groupClusterRanges.shrink_to_fit();
  geometry.lodMesh.lodLevelGroupRanges.shrink_to_fit();


  nvclusterlod_HierarchyInput hierarchyInput = {};
  hierarchyInput.clusterCount                = uint32_t(geometry.lodMesh.clusterBoundingSpheres.size());
  hierarchyInput.clusterBoundingSpheres      = geometry.lodMesh.clusterBoundingSpheres.data();
  hierarchyInput.clusterGeneratingGroups     = geometry.lodMesh.clusterGeneratingGroups.data();
  hierarchyInput.groupClusterRanges          = geometry.lodMesh.groupClusterRanges.data();
  hierarchyInput.groupQuadricErrors          = geometry.lodMesh.groupQuadricErrors.data();
  hierarchyInput.lodLevelGroupRanges         = geometry.lodMesh.lodLevelGroupRanges.data();
  hierarchyInput.lodLevelCount               = uint32_t(geometry.lodMesh.lodLevelGroupRanges.size());
  hierarchyInput.groupCount                  = int32_t(geometry.lodMesh.groupClusterRanges.size());

  // required to later traverse the lod hierarchy in parallel
  // Note we build hierarchies over each lod level's groups
  // and then join them together to a single hierarchy.
  // This is key to the parallel traversal algorithm that traverse multiple lod levels
  // at once.

  result = nvclusterlod::generateLodHierarchy(processingInfo.lodContext, hierarchyInput, geometry.lodHierarchy);
  if(result != NVCLUSTERLOD_SUCCESS)
  {
    assert(0);
    LOGE("nvclusterlod::generateLodHierarchy failed: %d\n", result);
    std::exit(-1);
  }

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.lodHierarchy.nodes.shrink_to_fit();
  geometry.lodHierarchy.groupCumulativeBoundingSpheres.shrink_to_fit();
  geometry.lodHierarchy.groupCumulativeQuadricError.shrink_to_fit();

  // build localized index buffers

  geometry.localTriangles.resize(geometry.lodMesh.triangleVertices.size() * 3);
  geometry.localVertices.resize(geometry.lodMesh.triangleVertices.size() * 3);
  geometry.clusterVertexRanges.resize(geometry.lodMesh.clusterTriangleRanges.size());

  std::vector<uint32_t> threadClusterMaxVertices(processingInfo.numInnerThreads, 0);
  std::vector<uint32_t> threadClusterMaxTriangles(processingInfo.numInnerThreads, 0);
  std::vector<uint32_t> threadCacheEarly(processingInfo.numInnerThreads * 256 * 2);

  nvutils::parallel_batches_pooled(
      geometry.lodMesh.clusterTriangleRanges.size(),
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        nvcluster_Range& vertexRange             = geometry.clusterVertexRanges[idx];
        nvcluster_Range  triangleRange           = geometry.lodMesh.clusterTriangleRanges[idx];
        const uint32_t* __restrict inputTriangle = &geometry.lodMesh.triangleVertices[triangleRange.offset].x;
        uint8_t* __restrict outputTriangle       = &geometry.localTriangles[triangleRange.offset * 3];
        uint32_t* __restrict outVertices         = &geometry.localVertices[triangleRange.offset * 3];
        uint32_t* vertexCacheEarlyValue          = &threadCacheEarly[threadInnerIdx * 256 * 2];
        uint32_t* vertexCacheEarlyPos            = vertexCacheEarlyValue + 256;
        memset(vertexCacheEarlyValue, ~0, sizeof(uint32_t) * 256);

        uint32_t count = 0;

        for(uint32_t i = 0; i < triangleRange.count * 3; i++)
        {
          uint32_t vertexIndex = inputTriangle[i];
          uint32_t cacheIndex  = ~0;

          // quick early out, have we seen the index
          uint32_t cacheEarlyValue = vertexCacheEarlyValue[vertexIndex & 0xFF];
          if(cacheEarlyValue == vertexIndex)
          {
            cacheIndex = vertexCacheEarlyPos[vertexIndex & 0xFF];
          }
          else
          {
            // look for it serially
            for(uint32_t v = 0; v < count; v++)
            {
              if(outVertices[v] == vertexIndex)
              {
                cacheIndex = v;
              }
            }
          }

          if(cacheIndex == ~0)
          {
            cacheIndex                                = count++;
            outVertices[cacheIndex]                   = vertexIndex;
            vertexCacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;
            vertexCacheEarlyPos[vertexIndex & 0xFF]   = cacheIndex;
          }
          outputTriangle[i] = uint8_t(cacheIndex);
        }

        vertexRange.count = count;
        threadClusterMaxVertices[threadInnerIdx] = std::max(threadClusterMaxVertices[threadInnerIdx], vertexRange.count);
        threadClusterMaxTriangles[threadInnerIdx] = std::max(threadClusterMaxTriangles[threadInnerIdx], triangleRange.count);

        assert(triangleRange.count <= m_config.clusterTriangles);
        assert(vertexRange.count <= m_config.clusterVertices);
      },
      processingInfo.numInnerThreads);

  // no longer needed
  geometry.lodMesh.triangleVertices = {};

  geometry.clusterMaxVerticesCount  = 0;
  geometry.clusterMaxTrianglesCount = 0;
  for(uint32_t t = 0; t < processingInfo.numInnerThreads; t++)
  {
    geometry.clusterMaxVerticesCount  = std::max(geometry.clusterMaxVerticesCount, threadClusterMaxVertices[t]);
    geometry.clusterMaxTrianglesCount = std::max(geometry.clusterMaxTrianglesCount, threadClusterMaxTriangles[t]);
  }

  // compaction pass
  uint32_t* localVertices = geometry.localVertices.data();

  uint32_t offset = 0;
  for(size_t c = 0; c < geometry.lodMesh.clusterTriangleRanges.size(); c++)
  {
    nvcluster_Range& vertexRange   = geometry.clusterVertexRanges[c];
    nvcluster_Range  triangleRange = geometry.lodMesh.clusterTriangleRanges[c];

    vertexRange.offset = offset;

    memmove(&localVertices[offset], &localVertices[triangleRange.offset * 3], sizeof(uint32_t) * vertexRange.count);

    offset += vertexRange.count;
  }

  geometry.localVertices.resize(offset);
  geometry.localVertices.shrink_to_fit();

  shaderio::LodLevel initLevel{};
  initLevel.minBoundingSphereRadius = FLT_MAX;
  initLevel.minMaxQuadricError      = FLT_MAX;

  geometry.lodLevelsCount = uint32_t(geometry.lodMesh.lodLevelGroupRanges.size());
  geometry.lodLevels.resize(geometry.lodMesh.lodLevelGroupRanges.size(), initLevel);

  // for later, easier access
  geometry.groupLodLevels.resize(geometry.lodMesh.groupClusterRanges.size());

  geometry.hiClustersCount = 0;
  geometry.hiTriangleCount = 0;
  geometry.hiVerticesCount = 0;

  for(size_t level = 0; level < geometry.lodMesh.lodLevelGroupRanges.size(); level++)
  {
    nvcluster_Range groupRange = geometry.lodMesh.lodLevelGroupRanges[level];

    geometry.lodLevels[level].groupCount  = groupRange.count;
    geometry.lodLevels[level].groupOffset = groupRange.offset;

    for(size_t g = groupRange.offset; g < groupRange.offset + groupRange.count; g++)
    {
      nvcluster_Range clusterRange = geometry.lodMesh.groupClusterRanges[g];

      geometry.groupLodLevels[g] = uint8_t(level);

      // USE_BLAS_SHARING
      //
      // For the BLAS sharing technique we need to figure out the conservative
      // lod range that an instance may cover. We store for each lod level
      // the smallest possible group bounding sphere as well as the smallest
      // maximum error found in any group.

      // The technique will use these values to artificially place a lod sphere
      // at the far end of an instance and evaluate its lod metric. The minima
      // ensure that there can't be any group in the instance's sphere that would
      // behave such a way that it requires lower detail.
      //
      // See `instance_classify_lod.comp.glsl` shader.

      geometry.lodLevels[level].minBoundingSphereRadius =
          std::min(geometry.lodLevels[level].minBoundingSphereRadius,
                   geometry.lodHierarchy.groupCumulativeBoundingSpheres[g].radius);
      geometry.lodLevels[level].minMaxQuadricError =
          std::min(geometry.lodLevels[level].minMaxQuadricError, geometry.lodHierarchy.groupCumulativeQuadricError[g]);

      if(level == 0)
      {
        uint32_t lastCluster  = clusterRange.offset + clusterRange.count - 1;
        uint32_t firstCluster = clusterRange.offset;

        geometry.hiClustersCount += clusterRange.count;
        geometry.hiTriangleCount += geometry.lodMesh.clusterTriangleRanges[lastCluster].count
                                    + geometry.lodMesh.clusterTriangleRanges[lastCluster].offset
                                    - geometry.lodMesh.clusterTriangleRanges[firstCluster].offset;
        geometry.hiVerticesCount += geometry.clusterVertexRanges[lastCluster].count
                                    + geometry.clusterVertexRanges[lastCluster].offset
                                    - geometry.clusterVertexRanges[firstCluster].offset;
      }
    }
  }

  geometry.totalTriangleCount = uint32_t(geometry.localTriangles.size() / 3);
  geometry.totalVerticesCount = uint32_t(geometry.localVertices.size());
  geometry.totalClustersCount = uint32_t(geometry.lodMesh.clusterTriangleRanges.size());
}

void Scene::computeLodBboxes_recursive(GeometryStorage& geometry, size_t i)
{
  const nvclusterlod_HierarchyNode& node = geometry.lodHierarchy.nodes[i];
  shaderio::BBox&                   bbox = geometry.nodeBboxes[i];

  bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f};

  if(node.children.isClusterGroup)
  {
    nvcluster_Range clusterRange = geometry.lodMesh.groupClusterRanges[node.clusterGroup.group];
    for(size_t c = clusterRange.offset; c < clusterRange.offset + clusterRange.count; c++)
    {
      bbox.lo = glm::min(bbox.lo, geometry.clusterBboxes[c].lo);
      bbox.hi = glm::max(bbox.hi, geometry.clusterBboxes[c].hi);
    }
  }
  else
  {
    for(uint32_t n = 0; n <= node.children.childCountMinusOne; n++)
    {
      computeLodBboxes_recursive(geometry, node.children.childOffset + n);
    }

    for(uint32_t n = 0; n <= node.children.childCountMinusOne; n++)
    {
      bbox.lo = glm::min(bbox.lo, geometry.nodeBboxes[node.children.childOffset + n].lo);
      bbox.hi = glm::max(bbox.hi, geometry.nodeBboxes[node.children.childOffset + n].hi);
    }
  }
}

void Scene::buildGeometryBboxes(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  geometry.clusterBboxes.resize(geometry.lodMesh.clusterTriangleRanges.size());

  const glm::vec4* positions      = geometry.vertices.data();
  const uint8_t*   localTriangles = geometry.localTriangles.data();

  nvutils::parallel_batches_pooled(
      geometry.lodMesh.clusterTriangleRanges.size(),
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        nvcluster_Range& vertexRange   = geometry.clusterVertexRanges[idx];
        nvcluster_Range& triangleRange = geometry.lodMesh.clusterTriangleRanges[idx];

        shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, FLT_MAX, -FLT_MAX};
        for(uint32_t v = 0; v < vertexRange.count; v++)
        {
          uint32_t  vertexIndex = vertexRange.offset + v;
          glm::vec3 pos         = positions[vertexIndex];

          bbox.lo = glm::min(bbox.lo, pos);
          bbox.hi = glm::max(bbox.hi, pos);
        }

        // find longest edge
        for(uint32_t t = 0; t < triangleRange.count; t++)
        {
          glm::vec3 trianglePositions[3];

          for(uint32_t v = 0; v < 3; v++)
          {
            trianglePositions[v] = positions[uint32_t(localTriangles[(triangleRange.offset + t) * 3 + v]) + vertexRange.offset];
          }

          for(uint32_t e = 0; e < 3; e++)
          {
            float distance    = glm::distance(trianglePositions[e], trianglePositions[(e + 1) % 3]);
            bbox.shortestEdge = std::min(bbox.shortestEdge, distance);
            bbox.longestEdge  = std::max(bbox.longestEdge, distance);
          }
        }

        geometry.clusterBboxes[idx] = bbox;
      },
      processingInfo.numInnerThreads);


  // now build lod node bounding boxes
  geometry.nodeBboxes.resize(geometry.lodHierarchy.nodes.size());
  computeLodBboxes_recursive(geometry, 0);
}


void Scene::computeHistograms()
{
  m_clusterTriangleHistogram.resize(m_config.clusterTriangles + 1, 0);
  m_clusterVertexHistogram.resize(m_config.clusterVertices + 1, 0);
  m_groupClusterHistogram.resize(m_config.clusterGroupSize + 1, 0);
  m_nodeChildrenHistogram.resize(32 + 1, 0);
  m_lodLevelsHistogram.resize(128, 0);

  m_maxLodLevelsCount = 0;

  for(GeometryView& geometry : m_geometryViews)
  {
    assert(geometry.lodLevelsCount < 128);
    m_maxLodLevelsCount = std::max(m_maxLodLevelsCount, geometry.lodLevelsCount);

    m_lodLevelsHistogram[geometry.lodLevelsCount]++;

    for(size_t c = 0; c < geometry.lodMesh.clusterTriangleRanges.size(); c++)
    {
      const nvcluster_Range& vertexRange   = geometry.clusterVertexRanges[c];
      const nvcluster_Range& triangleRange = geometry.lodMesh.clusterTriangleRanges[c];

      m_clusterTriangleHistogram[triangleRange.count]++;
      m_clusterVertexHistogram[vertexRange.count]++;
    }

    for(size_t g = 0; g < geometry.lodMesh.groupClusterRanges.size(); g++)
    {
      const nvcluster_Range& groupRange = geometry.lodMesh.groupClusterRanges[g];
      if(groupRange.count + 1 > m_groupClusterHistogram.size())
      {
        m_groupClusterHistogram.resize(groupRange.count + 1, 0);
      }
      m_groupClusterHistogram[groupRange.count]++;
    }

    for(size_t n = 0; n < geometry.lodHierarchy.nodes.size(); n++)
    {
      const nvclusterlod_HierarchyNode& node = geometry.lodHierarchy.nodes[n];
      if(node.children.isClusterGroup)
      {
        continue;
      }

      if(node.children.childCountMinusOne + 1 + 1 > m_nodeChildrenHistogram.size())
      {
        m_nodeChildrenHistogram.resize(node.children.childCountMinusOne + 1 + 1, 0);
      }
      m_nodeChildrenHistogram[node.children.childCountMinusOne + 1]++;
    }
  }

  m_lodLevelsHistogram.resize(m_maxLodLevelsCount + 1);

  m_clusterTriangleHistogramMax = 0u;
  m_clusterVertexHistogramMax   = 0u;
  m_groupClusterHistogramMax    = 0u;
  m_nodeChildrenHistogramMax    = 0u;
  m_lodLevelsHistogramMax       = 0u;

  for(size_t i = 0; i < m_clusterTriangleHistogram.size(); i++)
  {
    m_clusterTriangleHistogramMax = std::max(m_clusterTriangleHistogramMax, m_clusterTriangleHistogram[i]);
  }
  for(size_t i = 0; i < m_clusterVertexHistogram.size(); i++)
  {
    m_clusterVertexHistogramMax = std::max(m_clusterVertexHistogramMax, m_clusterVertexHistogram[i]);
  }
  for(size_t i = 0; i < m_groupClusterHistogram.size(); i++)
  {
    m_groupClusterHistogramMax = std::max(m_groupClusterHistogramMax, m_groupClusterHistogram[i]);
  }
  for(size_t i = 0; i < m_nodeChildrenHistogram.size(); i++)
  {
    m_nodeChildrenHistogramMax = std::max(m_nodeChildrenHistogramMax, m_nodeChildrenHistogram[i]);
  }
  for(size_t i = 0; i < m_lodLevelsHistogram.size(); i++)
  {
    m_lodLevelsHistogramMax = std::max(m_lodLevelsHistogramMax, m_lodLevelsHistogram[i]);
  }
}

void Scene::buildGeometryClusterStrips(ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  uint32_t numMaxTriangles = m_config.clusterTriangles;
  uint32_t numThreadIndices = numMaxTriangles * 3 + numMaxTriangles * 3 + uint32_t(meshopt_stripifyBound(numMaxTriangles * 3));
  std::vector<uint32_t> threadIndices(processingInfo.numInnerThreads * numThreadIndices);

  std::atomic_uint32_t numStrips = 0;

  uint8_t* localTriangles = geometry.localTriangles.data();

  nvutils::parallel_batches_pooled(
      geometry.lodMesh.clusterTriangleRanges.size(),
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        nvcluster_Range triangleRange = geometry.lodMesh.clusterTriangleRanges[idx];
        nvcluster_Range vertexRange   = geometry.clusterVertexRanges[idx];

        uint32_t* meshletIndices      = &threadIndices[threadInnerIdx * numThreadIndices];
        uint32_t* meshletOptim        = meshletIndices + triangleRange.count * 3;
        uint32_t* meshletStripIndices = meshletOptim + triangleRange.count * 3;

        // convert u8 to u32
        for(uint32_t i = 0; i < triangleRange.count * 3; i++)
        {
          meshletIndices[i] = localTriangles[triangleRange.offset * 3 + i];
        }

        meshopt_optimizeVertexCache(meshletOptim, meshletIndices, triangleRange.count * 3, vertexRange.count);
        size_t stripIndexCount =
            meshopt_stripify(meshletStripIndices, meshletOptim, triangleRange.count * 3, vertexRange.count, ~0);
        size_t newIndexCount = meshopt_unstripify(meshletIndices, meshletStripIndices, stripIndexCount, ~0);

        triangleRange.count = uint32_t(newIndexCount / 3);

        for(uint32_t i = 0; i < uint32_t(newIndexCount); i++)
        {
          localTriangles[triangleRange.offset * 3 + i] = uint8_t(meshletIndices[i]);
        }

        // just for stats
        numStrips++;
        for(uint32_t t = 1; t < uint32_t(triangleRange.count); t++)
        {
          const uint32_t* current = meshletIndices + t * 3;
          const uint32_t* prev    = meshletIndices + (t - 1) * 3;

          if(!((current[0] == prev[0] || current[0] == prev[2]) && (current[1] == prev[1] || current[1] == prev[2])))
            numStrips++;
        }
      },
      processingInfo.numInnerThreads);

  processingInfo.numTotalTriangles += geometry.localTriangles.size();
  processingInfo.numTotalStrips += numStrips;
}

void Scene::buildGeometryClusterVertices(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  // build per-cluster vertices
  std::vector<glm::vec4> oldVerticesData = std::move(geometry.vertices);

  geometry.vertices.resize(geometry.localVertices.size());
  geometry.vertices.shrink_to_fit();

  const glm::vec4* oldVertices          = oldVerticesData.data();
  glm::vec4*       newVertices          = geometry.vertices.data();
  uint32_t*        clusterLocalVertices = geometry.localVertices.data();

  nvutils::parallel_batches_pooled(
      geometry.clusterVertexRanges.size(),
      [&](uint64_t c, uint32_t threadInnerIdx) {
        nvcluster_Range& vertexRange = geometry.clusterVertexRanges[c];

        for(uint32_t v = 0; v < vertexRange.count; v++)
        {
          uint32_t oldIdx                              = clusterLocalVertices[v + vertexRange.offset];
          clusterLocalVertices[v + vertexRange.offset] = v + vertexRange.offset;
          newVertices[v + vertexRange.offset]          = oldVertices[oldIdx];
        }
      },
      processingInfo.numInnerThreads);
}


}  // namespace lodclusters
