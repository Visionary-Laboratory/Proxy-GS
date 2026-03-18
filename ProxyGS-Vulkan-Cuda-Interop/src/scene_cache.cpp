
/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/logger.hpp>

#include "scene.hpp"

namespace lodclusters {


bool Scene::storeCached(const GeometryView& view, uint64_t dataSize, void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = (dataAddress % nvclusterlod::detail::ALIGNMENT) == 0 && (dataAddress + sizeof(GeometryBase)) <= dataEnd;

  if(isValid)
  {
    memcpy(reinterpret_cast<void*>(dataAddress), (const GeometryBase*)&view, sizeof(GeometryBase));
    dataAddress += (sizeof(GeometryBase) + nvclusterlod::detail::ALIGN_MASK) & ~nvclusterlod::detail::ALIGN_MASK;
  }

  if(isValid)
  {
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.vertices);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.localTriangles);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.clusterVertexRanges);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.clusterBboxes);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupLodLevels);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.nodeBboxes);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.lodLevels);
  }


  isValid = isValid && nvclusterlod::storeCached(view.lodMesh, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodMesh);
  isValid = isValid && nvclusterlod::storeCached(view.lodHierarchy, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodHierarchy);

  return isValid;
}

static bool fileWriteAligned(uint64_t& outAccumulatedSize, FILE* outFile, size_t dataSize, const void* data)
{
  assert(outAccumulatedSize % nvclusterlod::detail::ALIGNMENT == 0);

  static const uint8_t padBytes[nvclusterlod::detail::ALIGNMENT] = {};

  if(fwrite(data, dataSize, 1, outFile) != 1)
    return false;

  uint64_t newDataSize = (dataSize + nvclusterlod::detail::ALIGN_MASK) & ~nvclusterlod::detail::ALIGN_MASK;

  uint64_t padSize = newDataSize - dataSize;
  if(padSize)
  {
    if(fwrite(padBytes, padSize, 1, outFile) != 1)
    {
      return false;
    }
  }

  outAccumulatedSize += newDataSize;
  return true;
}

template <typename T>
inline void fileWriteAligned(bool& isValid, uint64_t& outAccumulatedSize, FILE* outFile, const std::span<const T>& view)
{
  assert(outAccumulatedSize % nvclusterlod::detail::ALIGNMENT == 0);

  if(isValid)
  {
    union
    {
      uint64_t count;
      uint8_t  countData[nvclusterlod::detail::ALIGNMENT];
    };
    memset(countData, 0, sizeof(countData));

    count = view.size();

    if(fwrite(countData, nvclusterlod::detail::ALIGNMENT, 1, outFile) != 1)
    {
      isValid = false;
      return;
    }

    outAccumulatedSize += nvclusterlod::detail::ALIGNMENT;

    if(view.size() && !fileWriteAligned(outAccumulatedSize, outFile, view.size_bytes(), view.data()))
    {
      isValid = false;
    }
  }
}

uint64_t Scene::storeCached(const GeometryView& view, FILE* outFile)
{
  uint64_t dataSize = 0;

  bool isValid = fileWriteAligned(dataSize, outFile, sizeof(GeometryBase), (const GeometryBase*)&view);

  if(isValid)
  {
    fileWriteAligned(isValid, dataSize, outFile, view.vertices);
    fileWriteAligned(isValid, dataSize, outFile, view.localTriangles);
    fileWriteAligned(isValid, dataSize, outFile, view.clusterVertexRanges);
    fileWriteAligned(isValid, dataSize, outFile, view.clusterBboxes);
    fileWriteAligned(isValid, dataSize, outFile, view.groupLodLevels);
    fileWriteAligned(isValid, dataSize, outFile, view.nodeBboxes);
    fileWriteAligned(isValid, dataSize, outFile, view.lodLevels);


    fileWriteAligned(isValid, dataSize, outFile, view.lodMesh.triangleVertices);
    fileWriteAligned(isValid, dataSize, outFile, view.lodMesh.clusterTriangleRanges);
    fileWriteAligned(isValid, dataSize, outFile, view.lodMesh.clusterGeneratingGroups);
    fileWriteAligned(isValid, dataSize, outFile, view.lodMesh.clusterBoundingSpheres);
    fileWriteAligned(isValid, dataSize, outFile, view.lodMesh.groupQuadricErrors);
    fileWriteAligned(isValid, dataSize, outFile, view.lodMesh.groupClusterRanges);
    fileWriteAligned(isValid, dataSize, outFile, view.lodMesh.lodLevelGroupRanges);

    fileWriteAligned(isValid, dataSize, outFile, view.lodHierarchy.nodes);
    fileWriteAligned(isValid, dataSize, outFile, view.lodHierarchy.groupCumulativeBoundingSpheres);
    fileWriteAligned(isValid, dataSize, outFile, view.lodHierarchy.groupCumulativeQuadricError);
  }

  return dataSize;
}

bool Scene::loadCached(GeometryView& view, uint64_t dataSize, const void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  if(dataAddress % nvclusterlod::detail::ALIGNMENT == 0 && dataAddress + sizeof(GeometryBase) <= dataEnd)
  {
    memcpy((GeometryBase*)&view, data, sizeof(GeometryBase));
    dataAddress += (sizeof(GeometryBase) + nvclusterlod::detail::ALIGN_MASK) & ~nvclusterlod::detail::ALIGN_MASK;
  }
  else
  {
    view = {};
    return false;
  }

  if(isValid)
  {
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.vertices);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.localTriangles);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.clusterVertexRanges);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.clusterBboxes);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupLodLevels);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.nodeBboxes);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.lodLevels);
  }

  isValid = isValid && nvclusterlod::loadCached(view.lodMesh, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodMesh);
  isValid = isValid && nvclusterlod::loadCached(view.lodHierarchy, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodHierarchy);

  return isValid;
}

bool Scene::CacheFileView::init(uint64_t dataSize, const void* data)
{
  m_dataSize  = dataSize;
  m_dataBytes = reinterpret_cast<const uint8_t*>(data);

  if(dataSize <= sizeof(CacheHeader) + sizeof(uint64_t))
  {
    m_dataSize = 0;
    return false;
  }

  const CacheHeader* fileHeader = (const CacheHeader*)data;

  if(!fileHeader->isValid())
  {
    m_dataSize = 0;
    return false;
  }

  m_geometryCount = *getPointer<uint64_t>(m_dataSize - sizeof(uint64_t));

  if(!m_geometryCount || (dataSize <= (sizeof(CacheHeader) + sizeof(uint64_t) * (m_geometryCount * 2 + 1))))
  {
    m_dataSize = 0;
    return false;
  }

  m_tableStart = m_dataSize - sizeof(uint64_t) * (m_geometryCount * 2 + 1);

  return true;
}

bool Scene::CacheFileView::getGeometryView(GeometryView& view, uint64_t geometryIndex) const
{
  constexpr uint64_t ALIGN_MASK = nvclusterlod::detail::ALIGNMENT - 1;

  if(geometryIndex >= m_geometryCount)
  {
    assert(0);
    return false;
  }

  const uint64_t* geometryOffsets = getPointer<uint64_t>(m_tableStart, m_geometryCount * 2);
  uint64_t        base            = geometryOffsets[geometryIndex * 2 + 0];

  if(base + sizeof(GeometryBase) > m_tableStart)
  {
    // this must not happen on a valid file
    assert(0);
    return false;
  }

  uint64_t geometryTotalSize = geometryOffsets[geometryIndex * 2 + 1];
  uint64_t baseEnd           = base + geometryTotalSize;

  const uint8_t* geoData = getPointer<uint8_t>(base, geometryTotalSize);

  return Scene::loadCached(view, geometryTotalSize, geoData);
}

bool Scene::checkCache(const nvclusterlod::LodGeometryInfo& info, size_t geometryIndex)
{
  if(m_cacheFileView.isValid() && geometryIndex < m_cacheFileView.getGeometryCount())
  {
    GeometryView cacheView = {};
    if(!m_cacheFileView.getGeometryView(cacheView, geometryIndex))
    {
      return false;
    }

    // ignore these during compare
    cacheView.lodInfo.decimationFactor = info.decimationFactor;
    cacheView.lodInfo.groupConfig      = info.groupConfig;
    cacheView.lodInfo.clusterConfig    = info.clusterConfig;

    return memcmp(&info, &cacheView.lodInfo, sizeof(nvclusterlod::LodGeometryInfo)) == 0;
  }
  return false;
}

template <typename T>
static inline void fillVector(std::vector<T>& storageVec, const std::span<const T>& viewSpan)
{
  storageVec.resize(viewSpan.size());
  memcpy(storageVec.data(), viewSpan.data(), viewSpan.size_bytes());
}

void Scene::loadCachedGeometry(GeometryStorage& storage, size_t geometryIndex)
{
  GeometryView view = {};
  m_cacheFileView.getGeometryView(view, geometryIndex);
  (GeometryBase&)storage = view;

  fillVector(storage.vertices, view.vertices);
  fillVector(storage.localTriangles, view.localTriangles);
  fillVector(storage.clusterVertexRanges, view.clusterVertexRanges);
  fillVector(storage.clusterBboxes, view.clusterBboxes);
  fillVector(storage.groupLodLevels, view.groupLodLevels);
  fillVector(storage.nodeBboxes, view.nodeBboxes);
  fillVector(storage.lodLevels, view.lodLevels);

  nvclusterlod::toStorage(view.lodMesh, storage.lodMesh);
  nvclusterlod::toStorage(view.lodHierarchy, storage.lodHierarchy);
}

bool Scene::saveCache() const
{
  uint64_t dataOffset = sizeof(Scene::CacheHeader);

  std::vector<uint64_t> geometryOffsets;
  geometryOffsets.reserve(m_geometryViews.size() * 2 + 1);

  for(const GeometryView& geom : m_geometryViews)
  {
    uint64_t geomDataSize = geom.getCachedSize();
    geometryOffsets.push_back(dataOffset);
    geometryOffsets.push_back(geomDataSize);

    dataOffset += geomDataSize;
  }
  geometryOffsets.push_back(m_geometryViews.size());

  uint64_t tableOffset = dataOffset;

  dataOffset += geometryOffsets.size() * sizeof(uint64_t);

  nvutils::FileReadOverWriteMapping outMapping;

  std::string outFilename = nvutils::utf8FromPath(m_filePath) + ".nvsngeo";

  if(!outMapping.open(outFilename.c_str(), dataOffset))
  {
    LOGE("Scene::saveCache failed to save file %s\n", outFilename.c_str());
    return false;
  }

  uint8_t* mappingData = static_cast<uint8_t*>(outMapping.data());

  // write header
  Scene::CacheHeader cacheHeader;
  memcpy(mappingData, &cacheHeader, sizeof(cacheHeader));
  // write offset table at end
  memcpy(mappingData + tableOffset, geometryOffsets.data(), sizeof(uint64_t) * geometryOffsets.size());

  bool hadError = false;
  nvutils::parallel_batches(m_geometryViews.size(), [&](uint64_t idx) {
    const GeometryView& view = m_geometryViews[idx];

    uint64_t dataOffset = geometryOffsets[idx * 2 + 0];
    uint64_t dataSize   = geometryOffsets[idx * 2 + 1];

    if(!Scene::storeCached(view, dataSize, mappingData + dataOffset))
    {
      hadError = true;
    }
  });

  if(hadError)
  {
    LOGE("Scene::saveCache had errors while saving %s\n", outFilename.c_str());
  }
  else
  {
    LOGI("Scene::saveCache saved %s\n", outFilename.c_str());
  }

  return !hadError;
}


void Scene::beginProcessingOnly(size_t geometryCount)
{
  // don't trigger this code path if not valid
  if(!m_config.processingOnly || m_cacheFileView.isValid())
  {
    return;
  }

  std::string outFilename = nvutils::utf8FromPath(m_filePath) + ".nvsngeo";

  m_processingOnlyFile = nullptr;
  int result           = 0;
#ifdef WIN32
  result = fopen_s(&m_processingOnlyFile, outFilename.c_str(), "wb") == 0;
#else
  m_processingOnlyFile = fopen(outFilename.c_str(), "wb");
  result               = (m_processingOnlyFile) != nullptr;
#endif

  if(!result)
  {
    LOGE("Scene::beginProcessOnlySave failed to save file %s\n", outFilename.c_str());
    return;
  }

  Scene::CacheHeader header;

  fwrite(&header, sizeof(header), 1, m_processingOnlyFile);

  m_processingOnlyFileOffset = sizeof(Scene::CacheHeader);

  m_processingOnlyGeometryOffsets.resize(geometryCount * 2 + 1);
  m_processingOnlyGeometryOffsets[geometryCount * 2] = geometryCount;

  LOGI("Scene::beginProcessOnlySave started save file %s\n", outFilename.c_str());
}


void Scene::saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex)
{
  // unfortunately single-threaded writing for now
  {
    std::lock_guard lock(processingInfo.processOnlySaveMutex);

    uint64_t dataSize = storeCached(m_geometryViews[geometryIndex], m_processingOnlyFile);
    m_processingOnlyGeometryOffsets[geometryIndex * 2 + 0] = m_processingOnlyFileOffset;
    m_processingOnlyGeometryOffsets[geometryIndex * 2 + 1] = dataSize;

    m_processingOnlyFileOffset += dataSize;
  }

  // in process only mode we deallocate all data after storage
  m_geometryViews[geometryIndex]    = {};
  m_geometryStorages[geometryIndex] = {};
}

void Scene::endProcessingOnly(bool hadError)
{
  if(!m_processingOnlyFile)
    return;

  if(!hadError)
  {
    fwrite(m_processingOnlyGeometryOffsets.data(), m_processingOnlyGeometryOffsets.size() * sizeof(uint64_t), 1, m_processingOnlyFile);
  }

  fclose(m_processingOnlyFile);

  std::string outFilename = nvutils::utf8FromPath(m_cacheFilePath);

  if(hadError)
  {
    std::filesystem::remove(m_cacheFilePath);

    LOGI("Scene::endProcessOnlySave had error, deleted %s\n", outFilename.c_str());
  }
  else
  {
    LOGI("Scene::endProcessOnlySave finished to save file %s\n", outFilename.c_str());
  }

  exit(0);
}

}  // namespace lodclusters
