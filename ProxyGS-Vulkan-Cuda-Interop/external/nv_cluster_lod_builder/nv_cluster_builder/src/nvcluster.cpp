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

#include <clusterizer.hpp>
#include <nvcluster/nvcluster.h>
#include <stddef.h>

#if !defined(NVCLUSTER_BUILDER_COMPILING)
#error NVCLUSTER_BUILDER_COMPILING must be defined when building the library
#endif

struct nvcluster_Context_t
{
  uint32_t       version     = 0u;
  nvcluster_Bool parallelize = NVCLUSTER_FALSE;
};

const char* nvclusterResultString(nvcluster_Result result)
{
  // clang-format off
  switch(result){
  case NVCLUSTER_SUCCESS: return "SUCCESS";
  case NVCLUSTER_ERROR_CONTEXT_VERSION_MISMATCH: return "NVCLUSTER_ERROR_CONTEXT_VERSION_MISMATCH";
  case NVCLUSTER_ERROR_INVALID_CONFIG_CLUSTER_SIZES: return "NVCLUSTER_ERROR_INVALID_CONFIG_CLUSTER_SIZES";
  case NVCLUSTER_ERROR_MISSING_SPATIAL_BOUNDING_BOXES: return "NVCLUSTER_ERROR_MISSING_SPATIAL_BOUNDING_BOXES";
  case NVCLUSTER_ERROR_MISSING_SPATIAL_CENTROIDS: return "NVCLUSTER_ERROR_MISSING_SPATIAL_CENTROIDS";
  case NVCLUSTER_ERROR_INVALID_OUTPUT_ITEM_INDICES_SIZE: return "NVCLUSTER_ERROR_INVALID_OUTPUT_ITEM_INDICES_SIZE";
  case NVCLUSTER_ERROR_SPATIAL_AND_CONNECTIONS_ITEM_COUNT_MISMATCH: return "NVCLUSTER_ERROR_SPATIAL_AND_CONNECTIONS_ITEM_COUNT_MISMATCH";
  case NVCLUSTER_ERROR_SEGMENT_AND_ITEM_COUNT_CONTRADICTION: return "NVCLUSTER_ERROR_SEGMENT_AND_ITEM_COUNT_CONTRADICTION";
  case NVCLUSTER_ERROR_SEGMENT_COUNT_MISMATCH: return "NVCLUSTER_ERROR_SEGMENT_COUNT_MISMATCH";
  case NVCLUSTER_ERROR_MAX_CLUSTER_VERTICES_WITHOUT_CONNECTION_BITS: return "NVCLUSTER_ERROR_MAX_CLUSTER_VERTICES_WITHOUT_CONNECTION_BITS";
  case NVCLUSTER_ERROR_MAX_VERTICES_LESS_THAN_ITEM_VERTICES: return "NVCLUSTER_ERROR_MAX_VERTICES_LESS_THAN_ITEM_VERTICES";
  case NVCLUSTER_ERROR_NO_CONNECTION_ATTRIBUTES: return "NVCLUSTER_ERROR_NO_CONNECTION_ATTRIBUTES";
  case NVCLUSTER_ERROR_ITEM_VERTEX_COUNT_OVERFLOW: return "NVCLUSTER_ERROR_ITEM_VERTEX_COUNT_OVERFLOW";
  case NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTICES_PROVIDED: return "NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTICES_PROVIDED";
  case NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTEX_COUNT_PROVIDED: return "NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTEX_COUNT_PROVIDED";
  case NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_PER_ITEM_VERTEX_COUNT: return "NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_PER_ITEM_VERTEX_COUNT";
  case NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_VERTEX_COUNT: return "NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_VERTEX_COUNT";
  case NVCLUSTER_ERROR_NULL_INPUT: return "NVCLUSTER_ERROR_NULL_INPUT";
  case NVCLUSTER_ERROR_NULL_CONTEXT: return "NVCLUSTER_ERROR_NULL_CONTEXT";
  case NVCLUSTER_ERROR_NULL_OUTPUT: return "NVCLUSTER_ERROR_NULL_OUTPUT";
  case NVCLUSTER_ERROR_INTERNAL_MULTIPLE_UNDERFLOW: return "NVCLUSTER_ERROR_INTERNAL_MULTIPLE_UNDERFLOW";
  default: return "<Invalid nvcluster_Result>";
  }
  // clang-format on
}

uint32_t nvclusterVersion(void)
{
  return NVCLUSTER_VERSION;
}

nvcluster_Result nvclusterCreateContext(const nvcluster_ContextCreateInfo* createInfo, nvcluster_Context* context)
{
  if(createInfo == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_INPUT;
  }
  if(context == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_CONTEXT;
  }
  if(createInfo->version != NVCLUSTER_VERSION)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_CONTEXT_VERSION_MISMATCH;
  }

  *context = new nvcluster_Context_t{
      .version     = createInfo->version,
      .parallelize = createInfo->parallelize,
  };

  return nvcluster_Result::NVCLUSTER_SUCCESS;
}

nvcluster_Result nvclusterDestroyContext(nvcluster_Context context)
{
  if(context == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_CONTEXT;
  }

  delete context;

  return nvcluster_Result::NVCLUSTER_SUCCESS;
}

nvcluster_Result nvclusterGetRequirements(nvcluster_Context context, const nvcluster_Config* config, uint32_t itemCount, nvcluster_Counts* outputRequiredCounts)
{
  if(context == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_CONTEXT;
  }
  if(config == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_INPUT;
  }
  if(outputRequiredCounts == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_OUTPUT;
  }
  if(config->minClusterSize == 0 || config->maxClusterSize == 0 || config->minClusterSize > config->maxClusterSize)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_INVALID_CONFIG_CLUSTER_SIZES;
  }

  const size_t n  = itemCount;
  const size_t Ca = config->minClusterSize;

  // Pre-splitting ignores alignment and each can introduce an extra
  // under-filled cluster
  const size_t P                = config->preSplitThreshold;
  const size_t preSplitClusters = P == 0 ? 0 : (n + P - 1) / P;

  size_t maxClusters = std::min(n, (n + Ca - 1u) / Ca + preSplitClusters);

  // Check if maxClusterVertices has been set/is not the default
  if(config->maxClusterVertices != 0u && config->maxClusterVertices != ~0u)
  {
    if(config->maxClusterVertices < config->itemVertexCount)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_MAX_VERTICES_LESS_THAN_ITEM_VERTICES;
    }

    // Worst case, every item is disconnected, forming clusters of size
    // (maxClusterVertices / itemVertexCount). That number of clusters would be
    // doubled if we only split overflowing clusters exactly in half, but SAH
    // does not guarantee this and in fact may form single item clusters. While
    // uncommon, it is safer to just return 'n'.
    maxClusters = n;
  }

  *outputRequiredCounts = nvcluster_Counts{
      .clusterCount = uint32_t(maxClusters),
  };

  return nvcluster_Result::NVCLUSTER_SUCCESS;
}

inline nvcluster_Result buildMaybeWithConnections(nvcluster_Context         context,
                                                  const nvcluster_Config*   config,
                                                  const nvcluster_Input*    input,
                                                  nvcluster_OutputClusters* outputClusters,
                                                  const nvcluster_Segments* segments             = nullptr,
                                                  nvcluster_Range*          segmentClusterRanges = nullptr)
{
  if(input->itemCount && !input->itemBoundingBoxes)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_MISSING_SPATIAL_BOUNDING_BOXES;
  }
  if(input->itemCount && !input->itemCentroids)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_MISSING_SPATIAL_CENTROIDS;
  }

  // API permutation consistency checks
  if(input->itemVertices)
  {
    // The user is passing in vertices, which implies connections are to be
    // generated. If there are any manually provided connections, it's probably
    // a mistake.
    if(input->itemConnectionRanges || input->connectionTargetItems || input->connectionWeights || input->connectionVertexBits)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTICES_PROVIDED;
    }

    if(config->itemVertexCount == 0)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_PER_ITEM_VERTEX_COUNT;
    }

    if(input->vertexCount == 0)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_ITEM_VERTICES_WITHOUT_VERTEX_COUNT;
    }
  }
  else
  {
    // If the user provided connections, itemVertices and vertexCount should not
    // be set.
    if(input->itemConnectionRanges || input->connectionTargetItems || input->connectionWeights || input->connectionVertexBits)
    {
      if(input->vertexCount)
      {
        return nvcluster_Result::NVCLUSTER_ERROR_BOTH_CONNECTIONS_AND_VERTEX_COUNT_PROVIDED;
      }
    }

    // If the user wants a vertex limit, they must provide either
    // connectionVertexBits or itemVertices (to generate connectionVertexBits).
    if(config->maxClusterVertices != 0u && config->maxClusterVertices != ~0u && input->connectionVertexBits != nullptr)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_MAX_CLUSTER_VERTICES_WITHOUT_CONNECTION_BITS;
    }

    // Should have weights and/or vertex bits if connections are provided
    if(input->connectionCount > 0u && input->connectionWeights == nullptr && input->connectionVertexBits == nullptr)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_NO_CONNECTION_ATTRIBUTES;
    }
  }

  nvcluster_Range    singleSegmentRange{0, input->itemCount};
  nvcluster_Segments singleSegment{
      .segmentItemRanges = &singleSegmentRange,
      .segmentCount      = 1,
  };
  nvcluster_Range outputSegmentIgnored;
  if(segments == nullptr && segmentClusterRanges == nullptr)
  {
    segments             = &singleSegment;
    segmentClusterRanges = &outputSegmentIgnored;
  }

  // Skip computing connections if the item limit makes the vertex limit
  // redundant.
  bool skipVertexLimit = input->itemVertices && config->maxClusterSize * config->itemVertexCount <= config->maxClusterVertices;
  if(input->itemVertices && !skipVertexLimit)
  {
    nvcluster::MeshConnections meshConnections = nvcluster::makeMeshConnections(context->parallelize, *config, *input);
    return nvcluster::clusterize(context->parallelize, nvcluster::Input(*config, *input, *segments, meshConnections),
                                 nvcluster::OutputClusters(*outputClusters, segmentClusterRanges, segments->segmentCount));
  }
  else
  {
    // Translate 0u to imply no limit and disable the vertex limit if
    // skipVertexLimit is set.
    nvcluster_Config configCopy = *config;
    if(config->maxClusterVertices == 0u || skipVertexLimit)
    {
      configCopy.maxClusterVertices = ~0u;
      config                        = &configCopy;
    }
    return nvcluster::clusterize(context->parallelize, nvcluster::Input(*config, *input, *segments),
                                 nvcluster::OutputClusters(*outputClusters, segmentClusterRanges, segments->segmentCount));
  }
}

nvcluster_Result nvclusterBuild(nvcluster_Context context, const nvcluster_Config* config, const nvcluster_Input* input, nvcluster_OutputClusters* outputClusters)
{
  if(context == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_CONTEXT;
  }
  if(input == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_INPUT;
  }
  if(outputClusters == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_OUTPUT;
  }

  return buildMaybeWithConnections(context, config, input, outputClusters);
}

nvcluster_Result nvclusterGetRequirementsSegmented(nvcluster_Context         context,
                                                   const nvcluster_Config*   config,
                                                   uint32_t                  itemCount,
                                                   const nvcluster_Segments* segments,
                                                   nvcluster_Counts*         outputRequiredCounts)
{
  if(context == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_CONTEXT;
  }
  if(config == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_INPUT;
  }
  if(outputRequiredCounts == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_OUTPUT;
  }

  uint32_t outputClusterCount = 0u;
  for(uint32_t itemSegmentIndex = 0; itemSegmentIndex < segments->segmentCount; itemSegmentIndex++)
  {
    const nvcluster_Range& segmentItemRange = segments->segmentItemRanges[itemSegmentIndex];
    if(itemCount < segmentItemRange.offset + segmentItemRange.count)
    {
      return nvcluster_Result::NVCLUSTER_ERROR_SEGMENT_AND_ITEM_COUNT_CONTRADICTION;
    }
    nvcluster_Counts segmentResult{};
    nvcluster_Result res = nvclusterGetRequirements(context, config, segmentItemRange.count, &segmentResult);
    if(res != nvcluster_Result::NVCLUSTER_SUCCESS)
    {
      return res;
    }
    outputClusterCount += segmentResult.clusterCount;
  }
  *outputRequiredCounts = nvcluster_Counts{
      .clusterCount = outputClusterCount,
  };

  return nvcluster_Result::NVCLUSTER_SUCCESS;
}

// TODO: this is a naive implementation with no parallelism across segments. The
// internal implementation is already parallel and should eventually be able to
// handle clustering multiple ranges at the same time.
nvcluster_Result nvclusterBuildSegmented(nvcluster_Context         context,
                                         const nvcluster_Config*   config,
                                         const nvcluster_Input*    input,
                                         const nvcluster_Segments* segments,
                                         nvcluster_OutputClusters* outputClusters,
                                         nvcluster_Range*          segmentClusterRanges)
{
  if(context == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_CONTEXT;
  }
  if(config == nullptr || input == nullptr || segments == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_INPUT;
  }
  if(outputClusters == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_OUTPUT;
  }
  if(segments->segmentCount && segmentClusterRanges == nullptr)
  {
    return nvcluster_Result::NVCLUSTER_ERROR_NULL_OUTPUT;
  }
  return buildMaybeWithConnections(context, config, input, outputClusters, segments, segmentClusterRanges);
}
