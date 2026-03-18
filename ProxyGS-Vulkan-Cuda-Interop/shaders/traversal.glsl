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

float computeUniformScale(mat4 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}

vec3 TraversalMetric_getSphere(TraversalMetric metric)
{
  return vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
}
void TraversalMetric_setSphere(inout TraversalMetric metric, vec3 sphere)
{
  metric.boundingSphereX = sphere.x;
  metric.boundingSphereY = sphere.y;
  metric.boundingSphereZ = sphere.z;
}

// key function for the lod metric evaluation
// returns true if error is over threshold ("coarse enough")
bool testForTraversal(mat4x3 instanceToEye, float uniformScale, TraversalMetric metric, float errorScale)
{
  return true;
  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(vec3(instanceToEye * vec4(boundingSpherePos, 1.0f)));
  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;
  
  // error is over threshold, we are coarse enough
  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}

// variant of the above, assumes world space for view position and metric sphere position
bool testForTraversal(vec3 wViewPos, float uniformScale, TraversalMetric metric, float errorScale)
{
  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(wViewPos - boundingSpherePos);
  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;
  
  // error is over threshold, we are coarse enough
  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}

bool testForBlasSharing(Geometry geometry)
{
  return geometry.instancesCount >= max(build.sharingMinInstances,2);
}