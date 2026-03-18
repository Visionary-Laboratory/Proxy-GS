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

/*
  
  Shader Description
  ==================
  
  Only used for TARGETS_RAY_TRACING && USE_BLAS_SHARING
  
  This compute shader evaluates the geometry lod histogram to find
  a suitable sharable instance, whose blas can be used by many
  other instances of lesser lod range.
  
  A single thread represents one geometry
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};
#endif

////////////////////////////////////////////

#include "traversal.glsl"

////////////////////////////////////////////

layout(local_size_x=GEOMETRY_BLAS_SHARING_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint geometryID = gl_GlobalInvocationID.x;
  
  if (geometryID < build.numGeometries)
  {
    Geometry geometry           = geometries[geometryID];
    
    uint shareLevelMin   = 0xFFFF;
    uint shareLevelMax   = 0xFFFF;
    uint shareInstanceID = 0xFFFFFFFF;
    
    if (testForBlasSharing(geometry))
    {
      // every instance built per frame that is not sharing
      uint numBuildInstances  = 0;
      // every instance that is sharing
      uint numSharedInstances = 0;
  
      // used in the loop for temporarily storing previous iteration's result
      uint numPrevInstances    = 0;
      for (uint lodLevel = 0; lodLevel < geometry.lodLevelsCount; lodLevel++)
      {
        uint numInstances = build.geometryHistograms.d[geometryID].lodLevelMinHistogram[lodLevel];
        
        // check if we still need to find the sharing level, and 
        //       if any instance is ending at this lod level
        if (lodLevel >= build.sharingMinLevel &&
            shareLevelMax == 0xFFFF &&
            numBuildInstances + numInstances >= build.sharingMinInstances && 
            build.geometryHistograms.d[geometryID].lodLevelMaxHistogram[lodLevel] > 0)
        {
          uint packedLodInstance = build.geometryHistograms.d[geometryID].lodLevelMaxPackedInstance[lodLevel];
          shareLevelMax   = lodLevel;
          shareLevelMin   = packedLodInstance >> 27;
          shareInstanceID = packedLodInstance & ((1<<27)-1);
          
          // from some lod level onwards add tolerance of one level
          if (lodLevel >= build.sharingToleranceLevel && shareLevelMin < lodLevel)
          {
            // pretend we "end" one lod level earlier
            shareLevelMax  = lodLevel - 1;
            // subtract or add previous lod level instances accordingly
            numSharedInstances += numPrevInstances;
            numBuildInstances  -= numPrevInstances;
          }
        }
        if (shareLevelMax != 0xFFFF)
        {
          // all instances whose lodLevelMin is >= shareLevelMax, can use the shared blas
          numSharedInstances += numInstances;
        }
        else
        {
          numBuildInstances += numInstances;
        }
        
        numPrevInstances = numInstances;
      }
      
    #if USE_RENDER_STATS
      build.geometryBuildInfos.d[geometryID].shareUseCount = numSharedInstances;
    #endif
    }
  
    build.geometryBuildInfos.d[geometryID].shareLevelMin   = uint16_t(shareLevelMin);
    build.geometryBuildInfos.d[geometryID].shareLevelMax   = uint16_t(shareLevelMax);
    build.geometryBuildInfos.d[geometryID].shareInstanceID = shareInstanceID;
  }
}
