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

#include "asset_resolver.hpp"
#include <filesystem>
#include <nvutils/logger.hpp>

namespace fs = std::filesystem;

namespace assets {

std::vector<fs::path> build_search_paths(const fs::path& root) {
  return {
    root,
    root / "resources",
    root / "assets",
    root / "assets/models",
    root / "models",
    root / "scenes",
    root / "shaders",
    root / "shaders/hbao",
    root / "glsl",
    root / "post"
  };
}

fs::path resolve_in(const fs::path& root, const std::string& rel, const std::vector<std::string>& subdirs) {
  // 绝对路径直接返回
  fs::path relp(rel);
  if (relp.is_absolute() && fs::exists(relp)) {
    LOGI("[assets] absolute path '%s' -> %s\n", rel.c_str(), relp.string().c_str());
    return relp;
  }

  // 尝试 asset_root 本身和常见子目录
  for (const auto& sub : subdirs) {
    fs::path cand = sub.empty() ? (root / relp) : (root / sub / relp);
    if (fs::exists(cand)) {
     // LOGI("[assets] resolved '%s' -> %s\n", rel.c_str(), cand.string().c_str());
      return cand;
    }
  }
  
  // 找不到时打印详细的搜索路径
  std::string searchedPaths = "";
  for (const auto& sub : subdirs) {
    if (!searchedPaths.empty()) searchedPaths += ", ";
    searchedPaths += sub.empty() ? root.string() : (root / sub).string();
  }
  
  LOGW("[assets] MISSING '%s' (searched under assetRoot=%s, subdirs=[%s])\n", 
       rel.c_str(), root.string().c_str(), searchedPaths.c_str());
  
  return {};
}

} // namespace assets