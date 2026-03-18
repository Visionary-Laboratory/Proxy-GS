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

#include <nvvk/context.hpp>

namespace core {

struct BootstrapConfig {
  bool needExternalInterop = false; // 需要导出外部内存/信号量 FD 时置 true（无论 UDS 还是 pybind）
  int  forcedGpuIndex = -1;
  bool enableValidation = false;
  // 可按需增加 mesh/rt/barycentric 等开关，先给默认值
};

struct BootstrapResult {
  nvvk::Context ctx; // 按 main 里原有流程创建完成
};

BootstrapResult createVulkanContext(const BootstrapConfig& cfg);

/**
 * Append external interop extensions to ContextInitInfo if needed
 * @param info - context initialization info to modify
 * @param needInterop - true if external memory/semaphore FD export is needed (UDS || pybind)
 */
void appendExternalInteropExtensionsIfNeeded(nvvk::ContextInitInfo& info, bool needInterop);

} // namespace core