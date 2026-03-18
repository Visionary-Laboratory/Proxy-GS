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

/*
  
  Shader Description
  ==================
  
  Only used for TARGETS_RAY_TRACING && USE_BLAS_SHARING
  
  This compute shader classifies the lod range
  of each instance and updates the geometry's lod 
  histogram information accordingly.
  
  It also ensure that each tlas instance is initialized
  to use the pre-built low detail blas. This blas assignment
  may later be overridden in `instance_assign_blas.comp.glsl`

  A thread represents one instance.
  
  The follow up procedure to this is
  `geometry_blas_sharing.comp.glsl`
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

#extension GL_NV_shader_subgroup_partitioned : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
  FrameConstants viewLast;
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


////////////////////////////////////////////

layout(local_size_x=INSTANCES_CLASSIFY_LOD_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////

void main()
{
  uint instanceID   = gl_GlobalInvocationID.x;
  uint instanceLoad = min(build.numRenderInstances-1, instanceID);
  bool isValid      = instanceID == instanceLoad;

  // TODO optimization:
  // For better loading behavior when streaming, the instances should be sorted
  // relative to camera position.
  
  RenderInstance instance = instances[instanceLoad];
  uint geometryID = instance.geometryID;
  Geometry geometry = geometries[geometryID];
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
  bool inFrustum = intersectFrustum(geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax) && intersectHiz(clipMin, clipMax)));
  
  uint visibilityState = isVisible ? INSTANCE_VISIBLE_BIT : 0;
  
  uint rootNodePacked = geometry.nodes.d[0].packed;

  if (isValid)
  {
    // setup evaluation of lod metric
    mat4  worldMatrix  = instances[instanceID].worldMatrix;
    float uniformScale = computeUniformScale(worldMatrix);
    float errorScale   = 1.0;
  #if USE_CULLING
    // instance is not primary visible, apply different error scale
    if (visibilityState == 0) errorScale = build.culledErrorScale;
  #endif
  
    mat4 transform = build.traversalViewMatrix * worldMatrix;
    vec3 oViewPos  = (inverse(worldMatrix) * vec4(view.viewPos.xyz,1)).xyz;
    
    // The geometry's root node contains one child node for each lod level.
    // We will iterate over them.
    
    uint childOffset        = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);
    uint childCountMinusOne = PACKED_GET(rootNodePacked, Node_packed_nodeChildCountMinusOne);
  
    // Iterate over each lod level and determine whether it is potentially used
    // buy this instance.
    
    bool geometryUsesBlasSharing = testForBlasSharing(geometry);
    uint geometryLodLevelMax     = geometry.lodLevelsCount - 1;
    
    bool findMin     = true;
    bool findMax     = true;
    uint lodLevelMin = 0;
    uint lodLevelMax = geometryLodLevelMax;
    
    // lodLevelMin means a low lod/mip level and represents higher detail
    // lodLevelMax represents lower detail
    //
    // An instance may span multiple lod levels, meaning it has cluster
    // groups from different lod levels. 
    // This result depends on distance and orientation of the instance towards
    // the camera.
    //
    //   camera ->             [instance lodLevelMin  .... lodLevelMax]
    //

    for (uint lodLevel = 0; lodLevel < geometry.lodLevelsCount; lodLevel++)
    {
      Node childNode                  = geometry.nodes.d[childOffset + lodLevel];
      TraversalMetric traversalMetric = childNode.traversalMetric;

      // During lod traversal, we use the offline accumulated maximum sphere of the cluster groups stored into lod nodes
      // to test whether there is potentially something to be rendered. We want to optimize for the highest
      // error we get away with. So we test in detail if something is "coarse enough" (error > threshold).
      // `childNode.traversalMetric` provides the data for the maximum sphere.
      //
      // An actual cluster is rendered if 
      //  1) cluster's group            ` error over distance > threshold` (group's lod level is coarse enough)
      //  2) clusters' generating group `!error over distance > threshold` (generating group is in lod level - 1)
      
      // Example:
      //
      //  lod level                   | 0 | 1 | 2 | 3 | 4
      //  testForTraversal(maxSphere) | - | x | x | x | x
      //
      // In the example it's guaranteed that at least 1 cluster could be rendered at lod level 1.
      //   1) is ensured due to one group being represented within the accumulated maximum sphere
      //   2) is ensured because no such parent group can exist, otherwise `testForTraversal(maxSphere)` 
      //      would have evaluated to true for lod level 0.
      //
      // The first transition of the metric determines the lod level of a certain surface region.
      // This serves as the "fine enough".
      // Clusters from higher, less detailed, lod levels (e.g 2,3,4) of the same
      // region will not trigger, because their generating group's will not pass 2)
      //
      // We will use this reasoning to find the highest possible lod level as well.

      if (findMin && testForTraversal(mat4x3(transform), uniformScale, traversalMetric, errorScale))
      {
        findMin     = false;
        lodLevelMin = lodLevel;
      }
      
      // When using blas sharing for this geometry, we also need to find
      // the lodLevelMax value. This way we know from which lod level onwards
      // an instance's blas can be used rotational invariant.
      if (geometryUsesBlasSharing && !findMin)
      {
        // This time we use the smallest possible sphere for each lod level.
        // 
        // The smallest possible sphere was pre-computed for the geometry for each lod level.
        // We took the smallest radius, and the smallest `maxQuadraticError` found in any group,
        // and the sphere is put at the furthest possible distance from the camera,
        // while still within the maximum sphere.
        // These conditions ensure that nothing with a smaller `error over distance`
        // behavior can exist.
        
        vec3 oSpherePos = TraversalMetric_getSphere(traversalMetric);
        vec3 oViewDir   = normalize(oSpherePos - oViewPos);
        
        oSpherePos.xyz += oViewDir * (traversalMetric.boundingSphereRadius - geometry.lodLevels.d[lodLevel].minBoundingSphereRadius);
        
        traversalMetric.boundingSphereX = oSpherePos.x;
        traversalMetric.boundingSphereY = oSpherePos.y;
        traversalMetric.boundingSphereZ = oSpherePos.z;
        traversalMetric.boundingSphereRadius = geometry.lodLevels.d[lodLevel].minBoundingSphereRadius;
        traversalMetric.maxQuadricError      = geometry.lodLevels.d[lodLevel].minMaxQuadricError;
      
        // Example: 
        //
        //  lod level                   | 0 | 1 | 2 | 3 | 4
        //  testForTraversal(minSphere) | - | - | - | x | x
        //
        // If even the smallest possible group in lod level 3 is coarse enough, it
        // means there cannot be a group that would first transition in lod level 4
        //
        // Therefore lod level 3 is guaranteed to be the last active lod level.
        
        if (testForTraversal(mat4x3(transform), uniformScale, traversalMetric, errorScale))
        {
          lodLevelMax = lodLevel;
          break;
        }
      }
    }
    
    if (visibilityState == 0 && build.sharingPushCulled != 0)
    {
      // For invisible instances we might want to artificially push out
      // the minimum lod level that this instance will think it requires.
      // That way we increase the instance's likelihood to share another blas.
      
      lodLevelMin = min(lodLevelMin + 1, lodLevelMax);
    }
    
    // If the minimum lod level used is actually the maximum lod level available
    // for this geometry, it means the instance only uses the lowest detail
    // cluster group / pre-built blas.
    bool lowestDetailOnly = lodLevelMin == geometryLodLevelMax;
    
    if (lowestDetailOnly)
    {
    #if USE_RENDER_STATS
      atomicAdd(readback.numRenderedTriangles, geometry.lowDetailTriangles); 
      atomicAdd(readback.numRenderedClusters, 1);
    #endif
    }
    else if (geometryUsesBlasSharing)
    {
      // fill geometry histogram
      atomicAdd(build.geometryHistograms.d[geometryID].lodLevelMinHistogram[lodLevelMin], 1);
      atomicAdd(build.geometryHistograms.d[geometryID].lodLevelMaxHistogram[lodLevelMax], 1);
      
      // we want to find the instance with the highest lod min level (meaning it likely has less detail)
      // for each lod max level
      
      // pack lod min in upper 5 most significant bits, and instanceID in lower 27
      // this should give us a "stable" result on a static camera
      uint packedLodInstance = (lodLevelMin << 27) | instanceID & 0x7FFFFFF;
      atomicMax(build.geometryHistograms.d[geometryID].lodLevelMaxPackedInstance[lodLevelMax], packedLodInstance);
      
      // The `build.geometryBuildInfos` is read in the
      // `geometry_blas_sharing.comp.glsl` kernel
    }
    
    // used during `traversal_run.comp.glsl` to allow lower detail for "invisible" instances
    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);
    
    // used in `traversal_init_blas_sharing.comp.glsl` to drive the actual decision
    // which blas an instance should use
    build.instanceBuildInfos.d[instanceID].lodLevelMin            = uint8_t(lodLevelMin);
    build.instanceBuildInfos.d[instanceID].lodLevelMax            = uint8_t(lodLevelMax);
    build.instanceBuildInfos.d[instanceID].geometryLodLevelMax    = uint16_t(geometryLodLevelMax);
    build.instanceBuildInfos.d[instanceID].instanceUseCount       = 1;
    
    // ensure that the tlas instances are always initialized to the pre-built low detail blas and
    // renderable in some way.
    // May be overwritten at a later time in `instance_assign_blas.comp.glsl`
    build.tlasInstances.d[instanceID].blasReference               = geometry.lowDetailBlasAddress;
  }
}