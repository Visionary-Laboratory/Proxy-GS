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

#include <filesystem>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvgui/camera.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/window.hpp>
#include <nvgui/file_dialog.hpp>

#include "lodclusters.hpp"

namespace lodclusters {

std::string formatMemorySize(size_t sizeInBytes)
{
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

std::string formatMetric(size_t size)
{
  static const std::string units[]     = {"", "K", "M", "G"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(size < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float fsize = float(size) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", fsize, units[currentUnit]);
}

template <typename T>
void uiPlot(const std::string& plotName, const std::string& tooltipFormat, const std::vector<T>& data, const T& maxValue, int offset = 0)
{
  ImVec2 plotSize = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2);

  // Ensure minimum height to avoid overly squished graphics
  plotSize.y = std::max(plotSize.y, ImGui::GetTextLineHeight() * 20);

  const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
  const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;
  const ImColor         plotColor = ImColor(0.07f, 0.9f, 0.06f, 1.0f);

  if(ImPlot::BeginPlot(plotName.c_str(), plotSize, plotFlags))
  {
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
    ImPlot::SetupAxes(nullptr, "Count", axesFlags, axesFlags);
    ImPlot::SetupAxesLimits(0, double(data.size()), 0, static_cast<double>(maxValue), ImPlotCond_Always);

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
    ImPlot::SetNextFillStyle(plotColor);
    ImPlot::PlotShaded("", data.data(), (int)data.size(), -INFINITY, 1.0, 0.0, 0, offset);
    ImPlot::PopStyleVar();

    if(ImPlot::IsPlotHovered())
    {
      ImPlotPoint mouse       = ImPlot::GetPlotMousePos();
      int         mouseOffset = (int(mouse.x)) % (int)data.size();
      ImGui::BeginTooltip();
      ImGui::Text(tooltipFormat.c_str(), mouseOffset, data[mouseOffset]);
      ImGui::EndTooltip();
    }

    ImPlot::EndPlot();
  }
}
static uint32_t getUsagePct(uint64_t requested, uint64_t reserved)
{
  bool     exceeds = requested > reserved;
  uint32_t pct     = uint32_t(double(requested) * 100.0 / double(reserved));
  // artificially raise pct over 100 to trigger warning
  if(exceeds && pct < 101)
    pct = 101;
  return pct;
}

struct UsagePercentages
{
  uint32_t pctRender    = 0;
  uint32_t pctTraversal = 0;
  uint32_t pctResident  = 0;
  uint32_t pctClasLeft  = 100;
  uint32_t pctGeoMemory = 0;

  void setupPercentages(shaderio::Readback& readback, const RendererConfig& rendererConfig)
  {
    pctRender    = getUsagePct(readback.numRenderClusters, uint64_t(1) << rendererConfig.numRenderClusterBits);
    pctTraversal = getUsagePct(readback.numTraversalInfos, uint64_t(1) << rendererConfig.numTraversalTaskBits);
  }

  void setupPercentages(StreamingStats& stats, const StreamingConfig& streamingConfig)
  {
    pctResident = uint32_t(double(stats.residentGroups) * 100.0 / double(stats.maxGroups));
    pctClasLeft = stats.reservedClasBytes ? uint32_t(double(stats.maxSizedLeft) * 100.0 / double(stats.maxSizedReserved)) : 100;
    pctGeoMemory = uint32_t(double(stats.usedDataBytes) * 100.0 / double(stats.maxDataBytes));
  }

  const char* getWarning()
  {
    if(pctRender > 100)
      return "WARNING: Scene: Render clusters limit exceeded";
    if(pctTraversal > 100)
      return "WARNING: Scene: Traversal task limit exceeded";
    if(pctResident == 100)
      return "WARNING: Streaming: No resident groups left";
    if(pctClasLeft <= 1)
      return "WARNING: Streaming: Few CLAS groups left";
    if(pctGeoMemory >= 99)
      return "WARNING: Streaming: Little geometry memory left";
    return nullptr;
  }
};

void LodClusters::viewportUI(ImVec2 corner)
{
  ImVec2     mouseAbsPos = ImGui::GetMousePos();
  glm::uvec2 mousePos    = {mouseAbsPos.x - corner.x, mouseAbsPos.y - corner.y};

  m_frameConfig.frameConstants.mousePosition = glm::uvec2(glm::vec2(mousePos) * m_resources.getFramebufferWindow2RenderScale());

  if(m_renderer)
  {
    shaderio::Readback readback;
    m_resources.getReadbackData(readback);

    UsagePercentages pct;
    pct.setupPercentages(readback, m_rendererConfig);

    if(m_renderScene->useStreaming)
    {
      StreamingStats streamingStats;
      m_renderScene->sceneStreaming.getStats(streamingStats);
      pct.setupPercentages(streamingStats, m_streamingConfig);
    }


    const char* warning = pct.getWarning();

    if(warning)
    {
      ImVec4 warn_color = {0.75f, 0.2f, 0.2f, 1};
      ImVec4 hi_color   = {0.85f, 0.3f, 0.3f, 1};
      ImVec4 lo_color   = {0, 0, 0, 1};

      ImGui::SetWindowFontScale(2.0);

      // poor man's outline
      ImGui::SetCursorPos({7, 7});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 9});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 7});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({7, 9});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({8, 8});
      ImGui::TextColored(hi_color, "%s", warning);

      ImGui::SetWindowFontScale(1.0);
    }
  }
}

void LodClusters::onUIRender()
{
  ImGuiWindow* viewport = ImGui::FindWindowByName("Viewport");

  bool requestCameraRecenter = false;
  bool requestMirrorBox      = false;

  if(viewport)
  {
    if(nvgui::isWindowHovered(viewport))
    {
      if(ImGui::IsKeyDown(ImGuiKey_R))
      {
        m_reloadShaders = true;
      }
      if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
      {
        requestCameraRecenter = true;
      }
      if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right) || ImGui::IsKeyPressed(ImGuiKey_M))
      {
        requestMirrorBox = true;
      }
    }
  }

  bool earlyOut = !m_scene;

  if(earlyOut)
  {
    return;
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  bool pickingValid = isPickingValid(readback);
  // camera control, recenter
  if((requestCameraRecenter || requestMirrorBox) && pickingValid)
  {
    glm::uvec2 mousePos = {m_frameConfig.frameConstants.mousePosition.x, m_frameConfig.frameConstants.mousePosition.y};

    const glm::mat4 view = m_info.cameraManipulator->getViewMatrix();
    const glm::mat4 proj = m_frameConfig.frameConstants.projMatrix;

    float d = decodePickingDepth(readback);

    if(d < 1.0F)  // Ignore infinite
    {
      glm::vec4 win_norm     = {0, 0, m_frameConfig.frameConstants.viewport.x, m_frameConfig.frameConstants.viewport.y};
      const glm::vec3 hitPos = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);

      glm::vec3 eye, center, up;
      m_info.cameraManipulator->getLookat(eye, center, up);

      if(requestCameraRecenter)
      {
        // Set the interest position
        m_info.cameraManipulator->setLookat(eye, hitPos, up, false);
        m_info.cameraManipulator->setSpeed(glm::length(eye - hitPos) * m_tweak.clickSpeedScale);
      }

      if(requestMirrorBox)
      {
        m_frameConfig.frameConstants.useMirrorBox = 1;
        m_frameConfig.frameConstants.wMirrorBox = glm::vec4(hitPos, glm::distance(eye, hitPos) * m_tweak.mirrorBoxScale);
      }
    }
    else
    {
      if(requestMirrorBox)
      {
        m_frameConfig.frameConstants.useMirrorBox = 0;
      }
    }
  }
  else if(requestMirrorBox && !pickingValid)
  {
    m_frameConfig.frameConstants.useMirrorBox = 0;
  }

  ImVec4 text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  ImVec4 warn_color = text_color;
  warn_color.y *= 0.5f;
  warn_color.z *= 0.5f;

  // for emphasized parameter we want to recommend to the user
  const ImVec4 recommendedColor = ImVec4(0.0, 1.0, 0.0, 1.0);

  UsagePercentages pct = {};
  if(m_renderer)
  {
    pct.setupPercentages(readback, m_rendererConfig);
  }

  ImGui::Begin("Settings");
  ImGui::PushItemWidth(170 * ImGui::GetWindowDpiScale());

  namespace PE = nvgui::PropertyEditor;

  if(ImGui::CollapsingHeader("Scene Modifiers"))  //, nullptr, ImGuiTreeNodeFlags_DefaultOpen ))
  {
    PE::begin("##Scene Complexity");
    PE::Checkbox("Flip faces winding", &m_rendererConfig.flipWinding);
    PE::Checkbox("Disable back-face culling", &m_rendererConfig.twoSided);
    PE::InputInt("Render grid copies", (int*)&m_sceneGridConfig.numCopies, 1, 16, ImGuiInputTextFlags_EnterReturnsTrue,
                 "Instances the entire scene on a grid");
    PE::entry(
        "Render grid bits",
        [&] {
          for(uint32_t i = 0; i < 6; i++)
          {
            ImGui::PushID(i);
            bool used = (m_sceneGridConfig.gridBits & (1 << i)) != 0;

            ImGui::Checkbox("##hidden", &used);
            if(i < 5)
              ImGui::SameLine();
            if(used)
              m_sceneGridConfig.gridBits |= (1 << i);
            else
              m_sceneGridConfig.gridBits &= ~(1 << i);
            ImGui::PopID();
          }
          return false;
        },
        "0..2 bit enabled axis, 3..5 bit enabled rotation");

    //PE::InputInt("Render grid bits", (int*)&m_sceneGridConfig.gridBits, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
    //             "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
    PE::Checkbox("Render grid unique geometries", &m_sceneGridConfig.uniqueGeometriesForCopies,
                 "New Instances of the grid also get their own set of geometries, stresses streaming & memory consumption");
    PE::InputFloat("Render grid X", &m_sceneGridConfig.refShift.x, 0.1f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                   "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
    PE::InputFloat("Render grid Y", &m_sceneGridConfig.refShift.y, 0.1f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                   "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
    PE::InputFloat("Render grid Z", &m_sceneGridConfig.refShift.z, 0.1f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                   "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
    PE::InputFloat("Render grid snap angle", &m_sceneGridConfig.snapAngle, 5.0f, 10.f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "If rotation is active snaps angle");
    PE::InputFloat("Render grid min scale", &m_sceneGridConfig.minScale, 0.1f, 1.f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "Scale object");
    PE::InputFloat("Render grid max scale", &m_sceneGridConfig.maxScale, 0.1f, 1.f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "Scale object");
    PE::end();
  }

  if(ImGui::CollapsingHeader("Rendering", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {

    PE::begin("##Rendering");
    PE::entry("Renderer", [&]() { return m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer); });
    PE::entry("Super Resolution", [&]() { return m_ui.enumCombobox(GUI_SUPERSAMPLE, "sampling", &m_tweak.supersample); });
#if USE_DLSS
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD && m_resources.m_frameBuffer.dlssDenoiser.isAvailable())
    {
      PE::entry("DLSS - Denoiser", [&]() {
        ImGui::Checkbox("Enabled", &m_rendererConfig.useDlss);
        ImGui::SameLine();
        bool dlaa = m_rendererConfig.dlssQuality == NVSDK_NGX_PerfQuality_Value_DLAA;
        ImGui::Checkbox("DLAA", &dlaa);
        m_rendererConfig.dlssQuality = dlaa ? NVSDK_NGX_PerfQuality_Value_DLAA : NVSDK_NGX_PerfQuality_Value_MaxQuality;
        return false;
      });
    }
#endif
    PE::Text("Render Resolution:", "%d x %d", m_resources.m_frameBuffer.renderSize.width,
             m_resources.m_frameBuffer.renderSize.height);

    PE::Checkbox("Facet shading", &m_tweak.facetShading);
    PE::Checkbox("Wireframe", (bool*)&m_frameConfig.frameConstants.doWireframe);
    PE::Checkbox("Instance BBoxes", &m_frameConfig.showInstanceBboxes);

    // conditional UI, declutters the UI, prevents presenting many sections in disabled state
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
    {
      PE::Checkbox("Cast shadow rays", (bool*)&m_frameConfig.frameConstants.doShadow);
      PE::Checkbox("Reflective box", (bool*)&m_frameConfig.frameConstants.useMirrorBox);
      PE::SliderFloat("Ambient occlusion radius", &m_frameConfig.frameConstants.ambientOcclusionRadius, 0.001f, 1.f);
      PE::SliderInt("Ambient occlusion rays", &m_frameConfig.frameConstants.ambientOcclusionSamples, 0, 64);
    }
    if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)
    {
      PE::Checkbox("Ambient occlusion (HBAO)", &m_tweak.hbaoActive);
      if(PE::treeNode("HBAO settings"))
      {
        PE::Checkbox("Full resolution", &m_tweak.hbaoFullRes);
        PE::InputFloat("Radius", &m_tweak.hbaoRadius, 0.01f, 0, "%.6f");
        PE::InputFloat("Blur sharpness", &m_frameConfig.hbaoSettings.blurSharpness, 1.0f);
        PE::InputFloat("Intensity", &m_frameConfig.hbaoSettings.intensity, 0.1f);
        PE::InputFloat("Bias", &m_frameConfig.hbaoSettings.bias, 0.01f);
        PE::treePop();
      }
    }
    ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
    PE::entry("Visualize", [&]() {
      ImGui::PopStyleColor();  // pop text color here so it only applies to the label
      return m_ui.enumCombobox(GUI_VISUALIZE, "visualize", &m_frameConfig.visualize);
    });
    PE::end();
  }
  if(ImGui::CollapsingHeader("Traversal"))
  {
    PE::begin("##TraversalSpecifics");
    PE::InputIntClamped("Max traversal tasks (bits)", (int*)&m_rendererConfig.numTraversalTaskBits, 8, 25, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputIntClamped("Max render clusters (bits)", (int*)&m_rendererConfig.numRenderClusterBits, 8, 25, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputFloat("LoD pixel error", &m_frameConfig.lodPixelError, 0.25f, 0.25f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
    PE::Checkbox("Separate Groups Kernel", &m_rendererConfig.useSeparateGroups,
                 "optimization that splits traversal into two separate kernels");
    PE::Checkbox("Instance Sorting", &m_rendererConfig.useSorting);
    PE::Checkbox("Rendered Statistics", &m_rendererConfig.useRenderStats,
                 "Adds additional atomic counters for statistics, impacts performance");
    PE::Checkbox("Culling (Occlusion & Frustum)", &m_rendererConfig.useCulling);
    ImGui::BeginDisabled(!m_rendererConfig.useCulling);
    PE::Checkbox("Culling / LoD Freeze", &m_frameConfig.freezeCulling);
    ImGui::EndDisabled();

    m_frameConfig.lodPixelError = std::max(0.001f, m_frameConfig.lodPixelError);

    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
    {
      ImGui::BeginDisabled(!m_rendererConfig.useCulling);
      PE::InputFloat("Culled error scale", &m_frameConfig.culledErrorScale, 1.f, 1.f, "%.3f",
                     ImGuiInputTextFlags_EnterReturnsTrue, "scale the pixel error for occluded objects in ray tracing");
      ImGui::EndDisabled();

      m_frameConfig.culledErrorScale = std::max(1.0f, m_frameConfig.culledErrorScale);

      PE::Checkbox("Blas Sharing", &m_rendererConfig.useBlasSharing);
      ImGui::BeginDisabled(!m_rendererConfig.useBlasSharing);
      PE::Checkbox("Blas Sharing push culled", &m_frameConfig.sharingPushCulled, "culled instances artificially pushed by a lod level");
      PE::InputIntClamped("Blas Sharing min instances", (int*)&m_frameConfig.sharingMinInstances, 1, 0x7FFFFFFF, 1, 1,
                          ImGuiInputTextFlags_EnterReturnsTrue, "how many instances may be built before sharing");
      PE::InputIntClamped("Blas Sharing min level", (int*)&m_frameConfig.sharingMinLevel, 0, 0x7FFFFFFF, 1, 1,
                          ImGuiInputTextFlags_EnterReturnsTrue, "minimum lod level from which sharing may be used");
      PE::InputIntClamped("Blas Sharing tolerance level", (int*)&m_frameConfig.sharingToleranceLevel, 1, 32, 1, 1,
                          ImGuiInputTextFlags_EnterReturnsTrue,
                          "from which lod level we allow sharing a blas that may have too little detail for an instance");
      ImGui::EndDisabled();
    }
    PE::end();

    if(ImGui::BeginTable("##Render stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      //ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Traversal tasks");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctTraversal > 100 ? warn_color : text_color, "%d (%d%%)", readback.numTraversalInfos, pct.pctTraversal);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctTraversal > 100 ? warn_color : text_color, "%s",
                         formatMetric(readback.numTraversalInfos).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Render list clusters");
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctRender > 100 ? warn_color : text_color, "%d (%d%%)", readback.numRenderClusters, pct.pctRender);
      ImGui::TableNextColumn();
      ImGui::TextColored(pct.pctRender > 100 ? warn_color : text_color, "%s", formatMetric(readback.numRenderClusters).c_str());
      if(m_rendererConfig.useRenderStats)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Rendered clusters");
        ImGui::TableNextColumn();
        ImGui::Text("%d", readback.numRenderedClusters);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numRenderedClusters).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Rendered triangles");
        ImGui::TableNextColumn();
        ImGui::Text("%lu", static_cast<unsigned long>(readback.numRenderedTriangles));
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numRenderedTriangles).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Rendered Tri/Cluster");
        ImGui::TableNextColumn();
        if(readback.numRenderedClusters > 0)
        {
          ImGui::Text("%.1f", float(readback.numRenderedTriangles) / float(readback.numRenderedClusters));
        }
        else
        {
          ImGui::Text("N/A");
        }
        ImGui::TableNextColumn();
        ImGui::Text("%s", "");
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Built BLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%d", readback.numBlasBuilds);
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(readback.numBlasBuilds).c_str());
      ImGui::EndTable();
    }
  }
  if(ImGui::CollapsingHeader("Clusters & LoDs generation", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Clusters");
    if(false && m_scene && m_scene->m_loadedFromCache)
    {
      PE::Text("Cluster size from cache", "%dT_%dV", m_scene->m_config.clusterTriangles, m_scene->m_config.clusterVertices);
      PE::Text("LoD group size", "%d", m_scene->m_config.clusterGroupSize);
    }
    else
    {
      PE::entry("Cluster/meshlet size",
                [&]() { return m_ui.enumCombobox(GUI_MESHLET, "##cluster", &m_tweak.clusterConfig); });
      PE::InputIntClamped("LoD group size", (int*)&m_sceneConfig.clusterGroupSize, 8, 256, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                          "number of clusters that make lod group. Their triangles are decimated together and they share a common error property");
    }

    PE::InputIntClamped("CLAS Mantissa drop bits", (int*)&m_streamingConfig.clasPositionTruncateBits, 0, 22, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue,
                        "number of mantissa bits to drop (zeroed) to reduce memory consumption");
    PE::entry("CLAS build mode",
              [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##clasbuild", &m_streamingConfig.clasBuildFlags); });
    PE::end();
  }

  StreamingStats stats = {};
  if(m_renderScene->useStreaming)
  {
    m_renderScene->sceneStreaming.getStats(stats);
  }

  if(m_scene && ImGui::CollapsingHeader("Streaming", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Streaming");
    ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
    PE::Checkbox("Enable", &m_tweak.useStreaming);
    ImGui::PopStyleColor();

    ImGui::BeginDisabled(m_renderScene == nullptr);
    if(PE::entry("Streaming state", [&] { return ImGui::Button("Reset"); }, "resets the streaming state"))
    {
      m_renderScene->streamingReset();
    }
    ImGui::EndDisabled();

    PE::Checkbox("Async transfer", &m_streamingConfig.useAsyncTransfer, "Use asynchronous transfer queue for uploads");
    ImGui::BeginDisabled(!m_streamingConfig.useAsyncTransfer);
    PE::Checkbox("Decoupled transfer", &m_streamingConfig.useDecoupledAsyncTransfer,
                 "Allow asynchronous transfers to take multiple frames");
    ImGui::EndDisabled();

    PE::InputIntClamped("Unused frames until unloaded", (int*)&m_frameConfig.streamingAgeThreshold, 2, 1024, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);

    PE::InputIntClamped("Max Resident Groups", (int*)&m_streamingConfig.maxGroups, uint32_t(m_scene->getActiveGeometryCount()),
                        16 * 1024 * 1024, 128, 128, ImGuiInputTextFlags_EnterReturnsTrue);

    PE::InputIntClamped("Max Frame Group Loads", (int*)&m_streamingConfig.maxPerFrameLoadRequests, 1, 16 * 1024 * 1024,
                        128, 128, ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputIntClamped("Max Frame Group Unloads", (int*)&m_streamingConfig.maxPerFrameUnloadRequests, 1,
                        16 * 1024 * 1024, 128, 128, ImGuiInputTextFlags_EnterReturnsTrue);

    PE::InputIntClamped("Max Geometry MB", (int*)&m_streamingConfig.maxGeometryMegaBytes, 128, 1024 * 48, 128, 128,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputIntClamped("Max Transfer MB", (int*)&m_streamingConfig.maxTransferMegaBytes, 16, 1024, 16, 16,
                        ImGuiInputTextFlags_EnterReturnsTrue);
    PE::InputIntClamped("Max CLAS MB", (int*)&m_streamingConfig.maxClasMegaBytes, 128, 1024 * 48, 16, 16,
                        ImGuiInputTextFlags_EnterReturnsTrue);

    PE::Checkbox("Persistent Clas Allocator", &m_streamingConfig.usePersistentClasAllocator,
                 "Use persistent allocation on the device for clas memory, otherwise move based compaction");
    ImGui::BeginDisabled(!m_streamingConfig.usePersistentClasAllocator);
    PE::InputIntClamped("Allocator granularity shift bits", (int*)&m_streamingConfig.clasAllocatorGranularityShift, 0,
                        8, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                        "Clas Allocation byte granularity: (clas alignment value) < shift");
    PE::InputIntClamped("Allocator sector shift bits", (int*)&m_streamingConfig.clasAllocatorSectorSizeShift, 5, 16, 1,
                        1, ImGuiInputTextFlags_EnterReturnsTrue,
                        "Clas Allocation is scanning for free gaps using unused bits is done per sector of (1 << shift) of 32-bit values");
    PE::Text("Allocator sector size", "%d", 1 << m_streamingConfig.clasAllocatorSectorSizeShift);
    ImGui::EndDisabled();
    PE::end();


    if(ImGui::BeginTable("Streaming stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 170.0f);
      ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthStretch);
      //ImGui::TableHeadersRow(); // we do not show the header, it is not visually usefull
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Geometry");
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.couldNotStore ? warn_color : text_color, "%s", formatMemorySize(stats.usedDataBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.couldNotStore ? warn_color : text_color, "%d%%", getUsagePct(stats.usedDataBytes, stats.maxDataBytes));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        ImGui::Text("CLAS memory");
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%s",
                           formatMemorySize(stats.usedClasBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%d%%",
                           getUsagePct(stats.usedClasBytes, stats.reservedClasBytes));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGui::Text("CLAS waste");
        ImGui::TableNextColumn();
        ImGui::TextColored(text_color, "%s", formatMemorySize(stats.wastedClasBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(text_color, "%d%%", getUsagePct(stats.wastedClasBytes, stats.usedClasBytes));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGui::Text("CLAS groups left");
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%s", formatMetric(stats.maxSizedLeft).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%d%%",
                           getUsagePct(stats.maxSizedLeft, stats.maxSizedReserved));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
      }
      ImGui::Text("Resident groups");
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.couldNotAllocateGroup ? warn_color : text_color, "%s", formatMetric(stats.residentGroups).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.couldNotAllocateGroup ? warn_color : text_color, "%d%%",
                         getUsagePct(stats.residentGroups, stats.maxGroups));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      ImGui::Text("Resident clusters");
      uint32_t pctClusters = getUsagePct(stats.residentClusters, stats.maxClusters);
      ImGui::TableNextColumn();
      ImGui::TextColored(pctClusters > 99 ? warn_color : text_color, "%s", formatMetric(stats.residentClusters).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(pctClusters > 99 ? warn_color : text_color, "%d%%", pctClusters);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      ImGui::Text("Last Completed Transfer");
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.couldNotTransfer ? warn_color : text_color, "%s", formatMemorySize(stats.transferBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.couldNotTransfer ? warn_color : text_color, "%d%%",
                         getUsagePct(stats.transferBytes, stats.maxTransferBytes));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
#if 0
      ImGui::Text("Last Completed Transfers");
      ImGui::TableNextColumn();
      ImGui::TextColored(text_color, "%s", formatMetric(stats.transferCount).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(text_color, "-");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
#endif

      uint32_t pctLoad =
          stats.loadCount == m_streamingConfig.maxPerFrameLoadRequests ?
              100 :
              std::min(99u, uint32_t(float(stats.loadCount) * 100.0f / float(m_streamingConfig.maxPerFrameLoadRequests)));

      ImGui::Text("Last Completed Loads");
      ImGui::TableNextColumn();
      ImGui::TextColored(pctLoad == 100 ? warn_color : text_color, "%s", formatMetric(stats.loadCount).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(pctLoad == 100 ? warn_color : text_color, "%d%%", pctLoad);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      uint32_t pctUnLoad =
          stats.unloadCount == m_streamingConfig.maxPerFrameUnloadRequests ?
              100 :
              std::min(99u, uint32_t(float(stats.unloadCount) * 100.0f / float(m_streamingConfig.maxPerFrameUnloadRequests)));

      ImGui::Text("Last Completed Unloads");
      ImGui::TableNextColumn();
      ImGui::TextColored(pctUnLoad == 100 ? warn_color : text_color, "%s", formatMetric(stats.unloadCount).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(pctUnLoad == 100 ? warn_color : text_color, "%d%%", pctUnLoad);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      uint32_t pctUncompleted =
          stats.uncompletedLoadCount == m_streamingConfig.maxPerFrameLoadRequests ?
              100 :
              std::min(99u, uint32_t(float(stats.unloadCount) * 100.0f / float(m_streamingConfig.maxPerFrameUnloadRequests)));

      ImGui::Text("Last Uncompleted Loads");
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.uncompletedLoadCount ? warn_color : text_color, "%s",
                         formatMetric(stats.uncompletedLoadCount).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(stats.uncompletedLoadCount ? warn_color : text_color, "%d%%", pctUncompleted);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      ImGui::EndTable();
    }
  }

  Renderer::ResourceUsageInfo resourceActual = m_renderer ? m_renderer->getResourceUsage(false) : Renderer::ResourceUsageInfo();
  Renderer::ResourceUsageInfo resourceReserved = m_renderer ? m_renderer->getResourceUsage(true) : Renderer::ResourceUsageInfo();

  ImGui::Begin("Streaming memory");
  const uint32_t maxSlots = 512;
  if(m_streamGeometryHistogram.empty() == m_tweak.useStreaming)
  {
    m_streamGeometryHistogramMax = 0;
    m_streamHistogramOffset      = 0;
    m_streamGeometryHistogram.resize(m_tweak.useStreaming ? maxSlots : 0, 0);
    m_streamClasHistogram.resize(m_tweak.useStreaming ? maxSlots : 0, 0);
  }

  if(!m_streamGeometryHistogram.empty())
  {
    m_renderScene->sceneStreaming.getStats(stats);

    uint32_t mbGeometry          = uint32_t((stats.usedDataBytes + 999999) / 1000000);
    uint32_t mbClas              = uint32_t((stats.usedClasBytes + 999999) / 1000000);
    m_streamGeometryHistogramMax = std::max(m_streamGeometryHistogramMax, mbGeometry);
    m_streamClasHistogramMax     = std::max(m_streamClasHistogramMax, mbClas);
    {
      m_streamHistogramOffset = (m_streamHistogramOffset + 1) % maxSlots;
      m_streamGeometryHistogram[(m_streamHistogramOffset + maxSlots - 1) % maxSlots] = mbGeometry;
      m_streamClasHistogram[(m_streamHistogramOffset + maxSlots - 1) % maxSlots]     = mbClas;
    }

    uiPlot(std::string("Streaming Geometry Memory (MB)"), std::string("past %d MB %d"), m_streamGeometryHistogram,
           m_streamGeometryHistogramMax, m_streamHistogramOffset);
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
    {
      uiPlot(std::string("Streaming CLAS Memory (MB)"), std::string("past %d MB %d"), m_streamClasHistogram,
             m_streamClasHistogramMax, m_streamHistogramOffset);
    }
  }
  ImGui::End();

  ImGui::Begin("Statistics");
  if(ImGui::CollapsingHeader("Scene", nullptr, ImGuiTreeNodeFlags_DefaultOpen) && m_renderer)
  {
    if(ImGui::BeginTable("Scene stats", 3, ImGuiTableFlags_None))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Scene", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Triangles");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_hiTrianglesCountInstanced * m_sceneGridConfig.numCopies).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_hiTrianglesCount).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Clusters");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_hiClustersCountInstanced * m_sceneGridConfig.numCopies).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_hiClustersCount).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Instances");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_instances.size()).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_originalInstanceCount).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Geometries");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->getActiveGeometryCount()).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_originalGeometryCount).c_str());
      ImGui::EndTable();
    }
  }
  if(ImGui::CollapsingHeader("Memory", nullptr, ImGuiTreeNodeFlags_DefaultOpen) && m_renderer)
  {
    if(ImGui::BeginTable("Memory stats", 3, ImGuiTableFlags_RowBg))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Actual", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Geometry");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.geometryMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("==");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("TLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.rtTlasMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("-");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("BLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.rtBlasMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceReserved.rtBlasMemBytes).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("CLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.rtClasMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceReserved.rtClasMemBytes).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Operations");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.operationsMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("-");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Total");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceActual.getTotalSum()).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceReserved.getTotalSum()).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::EndTable();
    }
  }

  if(m_scene && ImGui::CollapsingHeader("Model Cluster Stats"))  //, nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::Text("Cluster max triangles: %d", m_scene->m_maxClusterTriangles);
    ImGui::Text("Cluster max vertices: %d", m_scene->m_maxClusterVertices);
    ImGui::Text("Cluster count: %lu", static_cast<unsigned long>(m_scene->m_totalClustersCount));
    ImGui::Text("Clusters with config (%u) triangles: %u (%.1f%%)", m_scene->m_config.clusterTriangles,
                m_scene->m_clusterTriangleHistogram.back(),
                float(m_scene->m_clusterTriangleHistogram.back()) * 100.f / float(m_scene->m_totalClustersCount));
    ImGui::Text("Geometry max lod levels: %d", m_scene->m_maxLodLevelsCount);

    uiPlot(std::string("Cluster Triangles Histogram"), std::string("Cluster count with %d triangles: %u"),
           m_scene->m_clusterTriangleHistogram, m_scene->m_clusterTriangleHistogramMax);
    uiPlot(std::string("Cluster Vertices Histogram"), std::string("Cluster count with %d vertices: %u"),
           m_scene->m_clusterVertexHistogram, m_scene->m_clusterVertexHistogramMax);
    uiPlot(std::string("Lod-Group Clusters Histogram"), std::string("Group count with %d clusters: %u"),
           m_scene->m_groupClusterHistogram, m_scene->m_groupClusterHistogramMax);
    uiPlot(std::string("Lod-Node Children Histogram"), std::string("Node count with %d children: %u"),
           m_scene->m_nodeChildrenHistogram, m_scene->m_nodeChildrenHistogramMax);
    uiPlot(std::string("Geometry Lod Levels Histogram"), std::string("Geometry count with %d lod levels: %u"),
           m_scene->m_lodLevelsHistogram, m_scene->m_lodLevelsHistogramMax);
  }
  ImGui::End();
  ImGui::End();

  ImGui::Begin("Misc Settings");

  if(ImGui::CollapsingHeader("Camera", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    nvgui::CameraWidget(m_info.cameraManipulator);
    namespace PE = nvgui::PropertyEditor;
    PE::begin();
    PE::InputFloat("Speed distance factor", &m_tweak.clickSpeedScale, 0, 0, "%.2f", 0,
                   "double click causes speed to be based on this percentage of the distance to hit point");
    PE::end();
  }

  if(ImGui::CollapsingHeader("Lighting", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    namespace PE = nvgui::PropertyEditor;
    PE::begin();
    PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                    "Mix between flashlight and sun light");
    PE::end();
    ImGui::TextDisabled("Sun & Sky");
    nvgui::skySimpleParametersUI(m_frameConfig.frameConstants.skyParams);
  }

  if(ImGui::CollapsingHeader("Mirror Box", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    namespace PE = nvgui::PropertyEditor;
    PE::begin();
    PE::Checkbox("Use", (bool*)&m_frameConfig.frameConstants.useMirrorBox);
    PE::InputFloat3("Position", &m_frameConfig.frameConstants.wMirrorBox.x);
    PE::InputFloat("Size", &m_frameConfig.frameConstants.wMirrorBox.w);
    PE::InputFloat("Size distance factor", &m_tweak.mirrorBoxScale, 0, 0, "%.3f", 0,
                   "When event is used, determine mirror box size based on `distance * factor`");
    PE::end();
  }

  if(ImGui::CollapsingHeader("Rendering Advanced", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Renderer Advanced");
    PE::InputIntClamped("Persistent traversal threads", (int*)&m_frameConfig.traversalPersistentThreads, 128,
                        256 * 1024, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
    PE::end();
  }

  ImGui::End();

  if(m_showDebugUI)
  {
    ImGui::Begin("Debug");
    if(ImGui::CollapsingHeader("Misc settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("##HiddenID");
      PE::InputInt("Colorize xor", (int*)&m_frameConfig.frameConstants.colorXor);
      PE::Checkbox("Auto reset timer", &m_tweak.autoResetTimers);
      PE::end();
    }
    if(ImGui::CollapsingHeader("Debug Shader Values", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("##HiddenID");
      PE::InputInt("dbgInt", (int*)&m_frameConfig.frameConstants.dbgUint, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue);
      PE::InputFloat("dbgFloat", &m_frameConfig.frameConstants.dbgFloat, 0.1f, 1.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
      PE::end();

      ImGui::Text(" debugI :  %10d", readback.debugI);
      ImGui::Text(" debugUI:  %10u", readback.debugUI);
      ImGui::Text(" debugU64:  %lX", static_cast<unsigned long>(readback.debugU64));
      static bool debugFloat = false;
      static bool debugHex   = false;
      static bool debugAll   = false;
      ImGui::Checkbox(" as float", &debugFloat);
      ImGui::SameLine();
      ImGui::Checkbox("hex", &debugHex);
      ImGui::SameLine();
      ImGui::Checkbox("all", &debugAll);
      //ImGui::SameLine();
      //bool     doPrint = ImGui::Button("print");
      uint32_t count = debugAll ? 64 : 32;

      if(ImGui::BeginTable("##Debug", 4, ImGuiTableFlags_BordersOuter))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 32);
        ImGui::TableSetupColumn("A", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("B", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("C", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        for(uint32_t i = 0; i < count; i++)
        {
          ImGui::Text("%2d", i);
          if(debugFloat)
          {
            ImGui::TableNextColumn();
            ImGui::Text("%f", *(float*)&readback.debugA[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%f", *(float*)&readback.debugB[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%f", *(float*)&readback.debugC[i]);
          }
          else if(debugHex)
          {
            ImGui::TableNextColumn();
            ImGui::Text("%X", readback.debugA[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%X", readback.debugB[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%X", readback.debugC[i]);
          }
          else
          {
            ImGui::TableNextColumn();
            ImGui::Text("%d", readback.debugA[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%d", readback.debugB[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%d", readback.debugC[i]);
          }

          ImGui::TableNextRow();
          ImGui::TableNextColumn();
        }

        ImGui::EndTable();
      }
    }
    ImGui::End();
  }

  handleChanges();

  // Rendered image displayed fully in 'Viewport' window
  ImGui::Begin("Viewport");
  ImVec2 corner = ImGui::GetCursorScreenPos();  // Corner of the viewport
  ImGui::Image((ImTextureID)m_imguiTexture, ImGui::GetContentRegionAvail());
  viewportUI(corner);
  ImGui::End();
}

void LodClusters::onUIMenu()
{
  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem("Open", "Ctrl+O"))
    {
      std::filesystem::path filename =
          nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF", "glTF(.gltf, .glb)|*.gltf;*.glb");
      if(!filename.empty())
      {
        onFileDrop(filename);
      }
    }
    if(m_scene)
    {
      if(ImGui::MenuItem("Reload", "Ctrl+R"))
      {
        std::filesystem::path filePath = m_scene->getFilePath();
        onFileDrop(filePath);
      }

      if(!m_scene->m_loadedFromCache)
      {
        if(ImGui::MenuItem("Save Cache", "Ctrl+S"))
        {
          saveCacheFile();
        }
      }

      if(!m_scene->isMemoryMappedCache() && std::filesystem::exists(m_scene->getCacheFilePath()))
      {
        if(ImGui::MenuItem("Delete Cache", "Ctrl+D"))
        {
          try
          {
            if(std::filesystem::remove(m_scene->getCacheFilePath()))
            {
              LOGI("Cache file deleted successfully.\n");
            }
          }
          catch(const std::filesystem::filesystem_error& e)
          {
            LOGW("Problem deleting cache file: %s\n", e.what());
          }
        }
      }
    }


    ImGui::EndMenu();
  }
}

}  // namespace lodclusters
