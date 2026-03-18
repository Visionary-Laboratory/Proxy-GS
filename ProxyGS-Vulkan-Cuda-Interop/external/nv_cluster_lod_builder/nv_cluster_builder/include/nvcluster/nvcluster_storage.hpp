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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vector>

#include <nvcluster/nvcluster.h>

namespace nvcluster {

// Utility storage for clustering output
// Construct with generateClusters()
struct ClusterStorage
{
  std::vector<nvcluster_Range> clusterItemRanges;
  std::vector<uint32_t>        items;

  void shrink_to_fit()
  {
    // clusterItemRanges is conservatively sized for clustering output. If this
    // object is kept around, memory can be saved by reallocating.
    clusterItemRanges.shrink_to_fit();
  }
};

// Utility storage for segmented clustering output
// Construct with generateSegmentedClusters()
struct SegmentedClusterStorage
{
  std::vector<nvcluster_Range> segmentClusterRanges;
  std::vector<nvcluster_Range> clusterItemRanges;
  std::vector<uint32_t>        items;

  void shrink_to_fit()
  {
    // clusterItemRanges is conservatively sized for clustering output. If this
    // object is kept around, memory can be saved by reallocating.
    clusterItemRanges.shrink_to_fit();
  }
};

// ClusterStorage delayed init constructor
inline nvcluster_Result generateClusters(nvcluster_Context       context,
                                         const nvcluster_Config& config,
                                         const nvcluster_Input&  input,
                                         ClusterStorage&         clusterStorage)
{
  // Query output upper limit
  nvcluster_Counts requiredCounts;
  nvcluster_Result result = nvclusterGetRequirements(context, &config, input.itemCount, &requiredCounts);
  if(result != nvcluster_Result::NVCLUSTER_SUCCESS)
  {
    return result;
  }

  // Resize to the upper limit
  clusterStorage.clusterItemRanges.resize(requiredCounts.clusterCount);
  clusterStorage.items.resize(input.itemCount);

  // Build clusters
  nvcluster_OutputClusters outputClusters{
      .clusterItemRanges = clusterStorage.clusterItemRanges.data(),
      .items             = clusterStorage.items.data(),
      .clusterCount      = uint32_t(clusterStorage.clusterItemRanges.size()),
      .itemCount         = uint32_t(clusterStorage.items.size()),
  };
  result = nvclusterBuild(context, &config, &input, &outputClusters);
  if(result != nvcluster_Result::NVCLUSTER_SUCCESS)
  {
    return result;
  }

  // Resize down to what was written. Let the user call shrink_to_fit() if the
  // object is not temporary.
  clusterStorage.clusterItemRanges.resize(outputClusters.clusterCount);
  return result;
}

inline nvcluster_Result generateSegmentedClusters(nvcluster_Context         context,
                                                  const nvcluster_Config&   config,
                                                  const nvcluster_Input&    input,
                                                  const nvcluster_Segments& segments,
                                                  SegmentedClusterStorage&  segmentedClusterStorage)
{
  // Query output upper limit
  nvcluster_Counts requiredCounts;
  nvcluster_Result result = nvclusterGetRequirementsSegmented(context, &config, input.itemCount, &segments, &requiredCounts);
  if(result != nvcluster_Result::NVCLUSTER_SUCCESS)
  {
    return result;
  }

  // Resize to the upper limit
  segmentedClusterStorage.segmentClusterRanges.resize(segments.segmentCount);
  segmentedClusterStorage.clusterItemRanges.resize(requiredCounts.clusterCount);
  segmentedClusterStorage.items.resize(input.itemCount);

  // Build clusters
  nvcluster_OutputClusters outputClusters{
      .clusterItemRanges = segmentedClusterStorage.clusterItemRanges.data(),
      .items             = segmentedClusterStorage.items.data(),
      .clusterCount      = uint32_t(segmentedClusterStorage.clusterItemRanges.size()),
      .itemCount         = uint32_t(segmentedClusterStorage.items.size()),
  };
  result = nvclusterBuildSegmented(context, &config, &input, &segments, &outputClusters,
                                   segmentedClusterStorage.segmentClusterRanges.data());
  if(result != nvcluster_Result::NVCLUSTER_SUCCESS)
  {
    return result;
  }

  // Resize down to what was written. Let the user call shrink_to_fit() if the
  // object is not temporary.
  segmentedClusterStorage.clusterItemRanges.resize(outputClusters.clusterCount);
  return result;
}

}  // namespace nvcluster
