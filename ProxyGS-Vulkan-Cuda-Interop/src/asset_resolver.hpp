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
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <nvutils/logger.hpp>

namespace assets {

// 构造常用搜索路径（用于模型/贴图等）
// 顺序很重要：先 asset_root，后各子目录
std::vector<std::filesystem::path> build_search_paths(const std::filesystem::path& asset_root);

// 通用解析：给定相对名，返回第一个存在的绝对路径；找不到则返回空 path
std::filesystem::path resolve_in(const std::filesystem::path& asset_root,
                                 const std::string& rel,
                                 const std::vector<std::string>& subdirs);

// Shader解析：始终返回绝对路径
inline std::filesystem::path resolve_shader(const std::filesystem::path& root,
                                            std::string_view rel) {
  namespace fs = std::filesystem;
  const fs::path r{rel};

  const fs::path candidates[] = {
    r,                      // as-is (绝不优先，但保留兼容)
    root / r,
    root / "shaders" / r,
    root / "resources" / r
  };

  for (const auto& c : candidates) {
    auto abs = fs::absolute(c.lexically_normal());
    if (fs::exists(abs)) {
      LOGI("[assets] resolved '%s' -> %s\n", std::string(rel).c_str(), abs.string().c_str());
      return abs; // **返回绝对路径**
    }
  }

  // 兜底：也返回一个绝对路径，便于日志清晰
  auto guess = fs::absolute((root / "shaders" / r).lexically_normal());
  LOGW("[assets] resolve miss '%s', guessed %s\n", std::string(rel).c_str(), guess.string().c_str());
  return guess;
}

inline std::filesystem::path resolve_model(const std::filesystem::path& root, const std::string& name) {
  return resolve_in(root, name, {"", "resources", "assets", "assets/models", "models", "scenes", "meshes"});
}

} // namespace assets