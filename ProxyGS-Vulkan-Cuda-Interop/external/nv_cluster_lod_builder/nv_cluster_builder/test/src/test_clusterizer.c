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

#ifdef __cplusplus
#error This file verifies the API is C compatible
#endif

#include <nvcluster/nvcluster.h>
#include <stdio.h>

int runCTest(void)
{
  nvcluster_ContextCreateInfo createInfo   = nvcluster_defaultContextCreateInfo();
  nvcluster_Context           context      = 0;
  nvcluster_Result            createResult = nvclusterCreateContext(&createInfo, &context);
  if(createResult != NVCLUSTER_SUCCESS)
  {
    printf("Create Context Result: %s\n", nvclusterResultString(createResult));
    return 0;
  }

  nvcluster_Result destroyResult = nvclusterDestroyContext(context);
  if(destroyResult != NVCLUSTER_SUCCESS)
  {
    printf("Destroy Context Result: %s\n", nvclusterResultString(destroyResult));
    return 0;
  }
  return 1;
}
