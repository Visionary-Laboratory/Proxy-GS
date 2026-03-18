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

#include <vector>
#include <string>
#include <atomic>
#include <mutex>

#include <glm/glm.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/timers.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy_storage.hpp>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <nvclusterlod/nvclusterlod_cache.hpp>

#include "../shaders/shaderio_scene.h"

namespace lodclusters {

struct SceneConfig
{
  // cluster and cluster group settings
  uint32_t clusterVertices  = 64;
  uint32_t clusterTriangles = 64;
  uint32_t clusterGroupSize = 32;

  // at each lod step reduce cluster group triangles by this factor
  float lodLevelDecimationFactor = 0.5f;

  // build triangle strips within clusters
  bool clusterStripify = true;

  // Influence the number of geometries that can be processed in parallel.
  // Percentage of threads of maximum hardware concurrency
  float processingThreadsPct = 0.5;
  // We only process the data and save a cache file, then
  // terminate the app. This allows to greatly reduce peak memory
  // consumption during processing.
  bool processingOnly = false;

  // save cache file after load automatically
  bool autoSaveCache = true;
  // try load from cache file if file was found
  bool autoLoadCache = true;
  // when loading from cache file, memory map it,
  // rather than loading it into system RAM.
  bool memoryMappedCache = false;
};

struct SceneGridConfig
{
  // when set to true each new set of instance on the grid gets
  // its own unique set of geometries. This stresses the streaming system
  // and memory consumption a lot more.
  bool      uniqueGeometriesForCopies = false;
  uint32_t  numCopies                 = 1;
  uint32_t  gridBits                  = 13;
  glm::vec3 refShift                  = {1.0f, 1.0f, 1.0f};
  float     snapAngle                 = 0;
  float     minScale                  = 1.0f;
  float     maxScale                  = 1.0f;
};

class Scene
{
public:
  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
    uint32_t       materialID = ~0U;
    glm::vec4      color{0.8, 0.8, 0.8, 1.0f};
  };

  struct GeometryBase
  {
    uint32_t clusterMaxVerticesCount{};
    uint32_t clusterMaxTrianglesCount{};

    uint32_t lodLevelsCount{};

    // based on highest detail lod
    uint32_t hiTriangleCount{};
    uint32_t hiVerticesCount{};
    uint32_t hiClustersCount{};

    // total sum
    uint32_t totalTriangleCount{};
    uint32_t totalVerticesCount{};
    uint32_t totalClustersCount{};

    shaderio::BBox bbox{};

    nvclusterlod::LodGeometryInfo lodInfo;

    uint32_t instanceReferenceCount{};
  };

  // not fully leveraged yet, but preparation to have a readonly view on
  // data accessed from persistently memory mapped files
  struct GeometryView : GeometryBase
  {
    std::span<const glm::vec4> vertices;

    std::span<const uint8_t> localTriangles;

    std::span<const nvcluster_Range> clusterVertexRanges;
    std::span<const shaderio::BBox>  clusterBboxes;
    std::span<const uint8_t>         groupLodLevels;

    std::span<const shaderio::BBox>     nodeBboxes;
    std::span<const shaderio::LodLevel> lodLevels;

    nvclusterlod::LodMeshView      lodMesh;
    nvclusterlod::LodHierarchyView lodHierarchy;

    inline uint64_t getCachedSize() const
    {
      uint64_t cachedSize = 0;

      cachedSize += (sizeof(GeometryBase) + nvclusterlod::detail::ALIGN_MASK) & ~nvclusterlod::detail::ALIGN_MASK;
      cachedSize += nvclusterlod::detail::getCachedSize(vertices);
      cachedSize += nvclusterlod::detail::getCachedSize(localTriangles);
      cachedSize += nvclusterlod::detail::getCachedSize(clusterVertexRanges);
      cachedSize += nvclusterlod::detail::getCachedSize(clusterBboxes);
      cachedSize += nvclusterlod::detail::getCachedSize(groupLodLevels);
      cachedSize += nvclusterlod::detail::getCachedSize(nodeBboxes);
      cachedSize += nvclusterlod::detail::getCachedSize(lodLevels);
      cachedSize += nvclusterlod::getCachedSize(lodMesh);
      cachedSize += nvclusterlod::getCachedSize(lodHierarchy);

      return cachedSize;
    }
  };

  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };

  bool init(const std::filesystem::path& filePath, const SceneConfig& config, bool skipCache);
  bool saveCache() const;
  void deinit();

  void updateSceneGrid(const SceneGridConfig& gridConfig);

  bool isMemoryMappedCache() const { return m_loadedFromCache && m_cacheFileMapping.valid(); }

  const std::filesystem::path& getFilePath() const { return m_filePath; }
  const std::filesystem::path& getCacheFilePath() const { return m_cacheFilePath; }

  SceneConfig     m_config;
  SceneGridConfig m_gridConfig;

  shaderio::BBox m_bbox;
  shaderio::BBox m_gridBbox;

  std::vector<Instance> m_instances;
  std::vector<Camera>   m_cameras;

  // we virtually instance geometries to avoid higher cpu memory consumption
  // happens when the grid config is larger
  const GeometryView& getActiveGeometry(size_t idx) const { return m_geometryViews[idx % m_originalGeometryCount]; }
  size_t              getActiveGeometryCount() const { return m_activeGeometryCount; }

  uint32_t getGeometryInstanceFactor() const
  {
    return m_gridConfig.uniqueGeometriesForCopies ? 1u : uint32_t(m_instances.size() / m_originalInstanceCount);
  }

  bool m_isBig = false;

  uint32_t m_maxClusterTriangles       = 0;
  uint32_t m_maxClusterVertices        = 0;
  uint32_t m_maxPerGeometryClusters    = 0;
  uint32_t m_maxPerGeometryTriangles   = 0;
  uint32_t m_maxPerGeometryVertices    = 0;
  uint32_t m_maxLodLevelsCount         = 0;
  uint32_t m_hiPerGeometryClusters     = 0;
  uint32_t m_hiPerGeometryTriangles    = 0;
  uint32_t m_hiPerGeometryVertices     = 0;
  uint64_t m_hiClustersCount           = 0;
  uint64_t m_hiTrianglesCount          = 0;
  uint64_t m_hiClustersCountInstanced  = 0;
  uint64_t m_hiTrianglesCountInstanced = 0;
  uint64_t m_totalClustersCount        = 0;

  std::vector<uint32_t> m_clusterTriangleHistogram;
  std::vector<uint32_t> m_clusterVertexHistogram;
  std::vector<uint32_t> m_groupClusterHistogram;
  std::vector<uint32_t> m_nodeChildrenHistogram;
  std::vector<uint32_t> m_lodLevelsHistogram;

  uint32_t m_clusterTriangleHistogramMax;
  uint32_t m_clusterVertexHistogramMax;
  uint32_t m_groupClusterHistogramMax;
  uint32_t m_nodeChildrenHistogramMax;
  uint32_t m_lodLevelsHistogramMax;

  bool m_loadedFromCache = false;

  size_t m_originalInstanceCount = 0;
  size_t m_originalGeometryCount = 0;

private:
  static bool     loadCached(GeometryView& view, uint64_t dataSize, const void* data);
  static bool     storeCached(const GeometryView& view, uint64_t dataSize, void* data);
  static uint64_t storeCached(const GeometryView& view, FILE* outFile);

  // GeometryStorage allows building and modifying the data in system RAM
  struct GeometryStorage : GeometryBase
  {
    std::vector<glm::vec4> vertices;

    // temporary, removed after processing
    std::vector<glm::uvec3> globalTriangles;

    // local to a cluster: indices of triangle vertices and global vertices
    std::vector<uint8_t> localTriangles;

    // temporary, removed after processing
    std::vector<uint32_t> localVertices;

    std::vector<nvcluster_Range> clusterVertexRanges;
    std::vector<shaderio::BBox>  clusterBboxes;
    std::vector<uint8_t>         groupLodLevels;

    std::vector<shaderio::LodLevel> lodLevels;
    nvclusterlod::LodMesh           lodMesh;
    nvclusterlod::LodHierarchy      lodHierarchy;
    std::vector<shaderio::BBox>     nodeBboxes;
  };

  class CacheHeader
  {
  public:
    CacheHeader()
    {
      header = {};
      if(sizeof(CacheHeader) - sizeof(Header))
      {
        memset(&data[sizeof(Header)], 0, sizeof(CacheHeader) - sizeof(Header));
      }
    }

    bool isValid() const
    {
      Header reference = {};

      return header.magic == reference.magic && header.geoVersion == reference.geoVersion
             && header.structSize == reference.structSize && header.lodVersion == reference.lodVersion
             && header.clusterVersion == reference.clusterVersion && header.alignment == reference.alignment;
    }

  private:
    struct Header
    {
      uint64_t magic          = 0x006f65676e73766eULL;  // nvsngeo
      uint32_t geoVersion     = 5;
      uint32_t structSize     = uint32_t(sizeof(GeometryView));
      uint32_t lodVersion     = NVCLUSTERLOD_VERSION;
      uint32_t clusterVersion = NVCLUSTER_VERSION;
      uint64_t alignment      = nvclusterlod::detail::ALIGNMENT;

      // geoVersion history:
      // 1 initial
      // 2 bugfix wrong storage of `lodInfo`
      // 3 octant vertices
      // 4 table is 2 x 64-bit per geometry (offse + size) to allow out of order storage
    };

    union
    {
      Header  header;
      uint8_t data[(sizeof(Header) + nvclusterlod::detail::ALIGN_MASK) & ~(nvclusterlod::detail::ALIGN_MASK)];
    };
  };

  class CacheFileView
  {
    // Optionally if you want to have a simple cache file for this
    // data, we provide a canonical layout, and this simple class
    // to open it.
    //
    // The cache data must be stored in three sections:
    //
#if 0
    struct CacheFile
    {
      // first: library version specific header
      CacheHeader header;
      // second: for each geometry serialized data of the `LodGeometryView`
      uint8_t geometryViewData[];
      // third: offset table
      // offsets where each `LodGeometry` data is stored + size
      // ordered with ascending offsets
      // `geometryDataSize = geometryOffsets[geometryIndex * 2 + 1];`
      uint64_t geometryOffsets[geometryCount * 2];
      uint64_t geometryCount;
    };
#endif

  public:
    bool isValid() const { return m_dataSize != 0; }

    bool init(uint64_t dataSize, const void* data);

    void deinit() { *(this) = {}; }

    uint64_t getGeometryCount() const { return m_geometryCount; }

    bool getGeometryView(GeometryView& view, uint64_t geometryIndex) const;

  private:
    template <class T>
    const T* getPointer(uint64_t offset, uint64_t count = 1) const
    {
      assert(offset + sizeof(T) * count <= m_dataSize);
      return reinterpret_cast<const T*>(m_dataBytes + offset);
    }

    uint64_t       m_dataSize      = 0;
    uint64_t       m_tableStart    = 0;
    const uint8_t* m_dataBytes     = nullptr;
    uint64_t       m_geometryCount = 0;
  };


  size_t m_activeGeometryCount = 0;

  std::vector<GeometryStorage> m_geometryStorages;
  std::vector<GeometryView>    m_geometryViews;

  std::filesystem::path m_filePath;
  std::filesystem::path m_cacheFilePath;

  // When loading a scene from a cache file, we can actually
  // directly load all data from the memory mapped file, rather than
  // copying it into system memory.
  // This view and mapping are kept alive after init when
  // `SceneConfig::memoryMappedCache` is true, otherwise they are closed
  // within `Scene::init`.

  nvutils::FileReadMapping m_cacheFileMapping;
  CacheFileView            m_cacheFileView;

  // only used in `processingOnly` mode
  FILE*                 m_processingOnlyFile       = nullptr;
  uint64_t              m_processingOnlyFileOffset = 0;
  std::vector<uint64_t> m_processingOnlyGeometryOffsets;

  struct ProcessingInfo
  {
    // how we perform multi-threading:
    // - either over geometries (outer loop)
    // - or within a geometry (inner loops)

    nvcluster_Context    clusterContext{};
    nvclusterlod_Context lodContext{};

    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;
    uint32_t numInnerThreads = 1;

    size_t geometryCount = 0;

    std::mutex processOnlySaveMutex;

    // some stats

    std::atomic_uint64_t numTotalTriangles = 0;
    std::atomic_uint64_t numTotalStrips    = 0;

    // logging progress

    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    std::mutex progressMutex;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;

    void init(float pct);
    void setupParallelism(size_t geometryCount_);
    void deinit();

    void logBegin();
    void logCompletedGeometry();
    void logEnd();
  };

  bool loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath);

  nvcluster_Config getClusterConfig() const
  {
    nvcluster_Config clusterConfig = {};
    clusterConfig.minClusterSize   = (m_config.clusterTriangles * 3) / 4;
    clusterConfig.maxClusterSize   = m_config.clusterTriangles;

    if(m_config.clusterVertices < m_config.clusterTriangles * 3)
    {
      clusterConfig.maxClusterVertices = m_config.clusterVertices;
      clusterConfig.itemVertexCount    = 3;
    }
    return clusterConfig;
  }
  nvcluster_Config getGroupConfig() const
  {
    nvcluster_Config groupConfig = {};
    groupConfig.minClusterSize   = (m_config.clusterGroupSize * 3) / 4;
    groupConfig.maxClusterSize   = m_config.clusterGroupSize;
    return groupConfig;
  }

  bool checkCache(const nvclusterlod::LodGeometryInfo& info, size_t geometryIndex);

  void processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached);
  void loadCachedGeometry(GeometryStorage& geometry, size_t geometryIndex);

  void buildGeometryClusters(const ProcessingInfo& processingInfo, GeometryStorage& geometry);
  void computeLodBboxes_recursive(GeometryStorage& geometry, size_t nodeIdx);
  void buildGeometryBboxes(const ProcessingInfo& processingInfo, GeometryStorage& geometry);
  void buildGeometryClusterStrips(ProcessingInfo& processingInfo, GeometryStorage& geometry);
  void buildGeometryClusterVertices(const ProcessingInfo& processingInfo, GeometryStorage& geometry);

  void beginProcessingOnly(size_t geometryCount);
  void saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex);
  void endProcessingOnly(bool hadError);

  void computeClusterStats();
  void computeHistograms();
  void computeInstanceBBoxes();
};

}  // namespace lodclusters
