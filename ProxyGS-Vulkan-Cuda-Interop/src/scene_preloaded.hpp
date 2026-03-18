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

#include "scene.hpp"
#include "resources.hpp"

namespace lodclusters {

// With this class we pre-load all lod levels of the rendered scene.
// It is much more memory intensive.
class ScenePreloaded
{
public:
  struct Config
  {
    VkBuildAccelerationStructureFlagsKHR clasBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    uint32_t                             clasPositionTruncateBits = 0;
  };

  // pointers must stay valid during lifetime
  bool init(Resources* res, const Scene* scene, const Config& config);

  // run prior the renderer starts referencing resources
  // if true CLAS for all clusters will be built
  bool updateClasRequired(bool state);

  // tear down, safe to call without init
  void deinit();

  // renderers need to access this buffer
  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }

  // device memory usage
  size_t getClasSize() const { return m_clasSize; }
  size_t getGeometrySize() const { return m_geometrySize; }
  size_t getOperationsSize() const { return m_operationsSize + m_clasOperationsSize; }

private:
  struct Geometry
  {
    nvvk::BufferTyped<shaderio::Node> nodes;
    nvvk::BufferTyped<shaderio::BBox> nodeBboxes;

    nvvk::BufferTyped<shaderio::Group> groups;

    nvvk::BufferTyped<uint8_t>   localTriangles;
    nvvk::BufferTyped<glm::vec4> vertices;

    nvvk::BufferTyped<shaderio::Cluster> clusters;
    nvvk::BufferTyped<uint32_t>          clusterGeneratingGroups;
    nvvk::BufferTyped<shaderio::BBox>    clusterBboxes;

    nvvk::BufferTyped<shaderio::LodLevel> lodLevels;

    // for ray tracing
    nvvk::BufferTyped<uint64_t> clusterClasAddresses;
    nvvk::BufferTyped<uint32_t> clusterClasSizes;
    nvvk::Buffer                clasData;
  };

  Config       m_config;
  bool         m_hasClas   = false;
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  size_t m_clasSize           = 0;
  size_t m_clasOperationsSize = 0;
  size_t m_geometrySize       = 0;
  size_t m_operationsSize     = 0;

  std::vector<ScenePreloaded::Geometry> m_geometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;

  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;

  nvvk::Buffer m_clasLowDetailBlasBuffer;

  bool initClas();
  void deinitClas();
};
}  // namespace lodclusters
