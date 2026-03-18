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

#ifdef _DEBUG
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, (format), __VA_ARGS__);                                                                            \
    fprintf(stderr, "\n");                                                                                             \
  } while(false)
#endif


#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <imgui/imgui.h>

#include <nvvk/validation_settings.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/parameter_parser.hpp>

#include "lodclusters.hpp"
#include "core/context_bootstrap.hpp"
#include "scene.hpp"
#include "external_memory.hpp"
#include <filesystem>
#include <thread>
#include <chrono>

using namespace lodclusters;

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name    = TARGET_NAME;
  appInfo.useMenu = true;
  appInfo.vSync = false;  // Disable VSync by default for better performance

  appInfo.windowSize = {1000,1000};
  appInfo.headless = false;  // Will be set later based on offscreen/uds flags
  appInfo.headlessFrameCount = 100000;  // Increased for testing socket communication

  // Basic Vulkan setup info (full setup moved to bootstrap)
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
      .queues             = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
  };

  nvutils::ProfilerManager                    profilerManager;
  std::shared_ptr<nvutils::CameraManipulator> cameraManipulator = std::make_shared<nvutils::CameraManipulator>();

  nvutils::ParameterRegistry parameterRegistry;
  nvutils::ParameterParser   parameterParser;

  // Add local variables that we can access directly
  bool processingOnly = false;
  std::filesystem::path sceneFilePath;
  
  // External memory / Python integration parameters
  std::string udsPath = "";  // Empty by default, set via command line to enable
  bool enableUDS = false;
  bool offscreen = false;

  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);
  parameterRegistry.add({"processingonly", "directly terminate app once cache file was saved. default false"}, &processingOnly);
  parameterRegistry.add({"scene"}, {".gltf", ".glb"}, &sceneFilePath);
  parameterRegistry.add({"uds", "enable Python integration via Unix Domain Socket at specified path"}, &udsPath);
  parameterRegistry.add({"offscreen", "enable offscreen rendering for Python integration"}, &offscreen);

  LodClusters::Info sampleInfo;
  sampleInfo.cameraManipulator               = cameraManipulator;
  sampleInfo.profilerManager                 = &profilerManager;
  sampleInfo.parameterRegistry               = &parameterRegistry;
  sampleInfo.externalMemoryManager           = nullptr; // Will be set after initialization
  std::shared_ptr<LodClusters> sampleElement = std::make_shared<LodClusters>(sampleInfo);

  parameterParser.add(parameterRegistry);
  parameterParser.parse(argc, argv);

  // Check if UDS integration is enabled (path specified or offscreen requested)
  if (!udsPath.empty() || offscreen) {
    enableUDS = true;
    // Set default path if not specified but offscreen is enabled
    if (udsPath.empty()) {
      udsPath = "/tmp/vk2torch.sock";
    }
  }
  
  // Set headless mode based on offscreen flag or UDS path
  appInfo.headless = offscreen || !udsPath.empty();

  // Check if we're in processing-only mode to skip Vulkan initialization
  
  if (processingOnly) {
    LOGI("Running in processing-only mode, skipping Vulkan initialization...\n");
    
    // Create Scene configuration from parameters
    lodclusters::SceneConfig sceneConfig;
    sceneConfig.processingOnly = true;
    sceneConfig.clusterVertices = 64;
    sceneConfig.clusterTriangles = 64; 
    sceneConfig.clusterGroupSize = 32;
    sceneConfig.lodLevelDecimationFactor = 0.5f;
    sceneConfig.clusterStripify = true;
    sceneConfig.processingThreadsPct = 0.5f;
    sceneConfig.autoSaveCache = true;
    sceneConfig.autoLoadCache = true;
    sceneConfig.memoryMappedCache = false;
    
    // Create a Scene object and process the specified file
    lodclusters::Scene scene;
    std::filesystem::path inputPath;
    
    // Check if user specified a scene file
    if (!sceneFilePath.empty()) {
        // User specified a file path
        inputPath = sceneFilePath;
        if (!std::filesystem::exists(inputPath)) {
            LOGE("Specified scene file does not exist: %s\n", inputPath.string().c_str());
            return -1;
        }
    } else {
        // No file specified, try to find the default bunny file
        std::filesystem::path bunnyPath = "resources/bunny_v2/bunny.gltf";
        
        const std::vector<std::filesystem::path> searchPaths = {
            std::filesystem::absolute(std::filesystem::current_path() / "_downloaded_resources"),
            std::filesystem::absolute(std::filesystem::current_path() / "resources"),
            std::filesystem::absolute(std::filesystem::current_path() / "../resources"),
            std::filesystem::absolute(std::filesystem::current_path() / "downloads")
        };
        
        bool foundBunny = false;
        for (const auto& searchPath : searchPaths) {
            auto fullPath = searchPath / "bunny_v2" / "bunny.gltf";
            if (std::filesystem::exists(fullPath)) {
                inputPath = fullPath;
                foundBunny = true;
                break;
            }
        }
        
        if (!foundBunny) {
            LOGE("No scene file specified and could not find default bunny.gltf in standard locations\n");
            LOGE("Usage: %s --processingonly 1 --scene /path/to/your/model.gltf\n", argv[0]);
            return -1;
        }
    }
    
    LOGI("Processing: %s\n", inputPath.string().c_str());
    
    // Initialize the scene - this will do all the CPU mesh processing
    bool success = scene.init(inputPath, sceneConfig, false);
    
    if (success) {
        LOGI("Mesh processing completed successfully\n");
    } else {
        LOGE("Mesh processing failed\n");
    }
    
    // Clean up
    scene.deinit();
    
    return success ? 0 : -1;
  }

  // Create Vulkan context using bootstrap
  core::BootstrapConfig bootstrapConfig;
  // Unified condition: needInterop = offscreen || !udsPath.empty() || inProcessInterop
  bool needInterop = offscreen || !udsPath.empty();  // main() criteria for external interop
  bootstrapConfig.needExternalInterop = needInterop;
  bootstrapConfig.forcedGpuIndex = vkSetup.forceGPU;
  bootstrapConfig.enableValidation = vkSetup.enableValidationLayers;
  
  core::BootstrapResult bootstrapResult = core::createVulkanContext(bootstrapConfig);
  nvvk::Context& vkContext = bootstrapResult.ctx;

  sampleElement->setSupportsClusters(vkContext.hasExtensionEnabled(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME));

  // Initialize external memory manager if UDS is enabled
  std::unique_ptr<ExternalMemoryManager> externalMemoryManager;
  if (enableUDS) {
    LOGI("Initializing external memory manager for Python integration\n");
    externalMemoryManager = std::make_unique<ExternalMemoryManager>();
    
    ExternalMemoryConfig config;
    config.enabled = true;
    config.udsPath = udsPath;
    config.offscreen = offscreen;
    config.width = 1000;  // TODO: make configurable
    config.height = 1000; // TODO: make configurable
    config.format = VK_FORMAT_R8G8B8A8_UNORM;
    
    if (!externalMemoryManager->init(vkContext.getDevice(), vkContext.getPhysicalDevice(), config)) {
      LOGE("Failed to initialize external memory manager\n");
      return -1;
    }
    
    // In offscreen mode, handle client connection differently
    if (offscreen) {
      LOGI("Running in offscreen mode - waiting for Python client...\n");
      
      // Block and wait for client connection
      if (!externalMemoryManager->acceptClient()) {
        LOGE("Failed to accept Python client in offscreen mode\n");
        return -1;
      }
      
      LOGI("Python client connected in offscreen mode\n");
      
      // Now wait for the client to disconnect
      while (externalMemoryManager->isConnected()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      
      LOGI("Python client disconnected. Exiting offscreen mode.\n");
      
      // Clean up Vulkan context
      vkContext.deinit();
      return 0;
    }
    
    // Pass the external memory manager to sample element immediately
    // The connection blocking will happen AFTER scene initialization in onRender()
    if (enableUDS) {
      LOGI("UDS enabled - external memory manager will wait for connection after scene init\n");
      sampleElement->setExternalMemoryManager(externalMemoryManager.get());
    }


  }

  // Normal GUI mode - set up application
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  bool hasDebugUI = sampleElement->getShowDebugUI();

  // Setting up the layout of the application
  appInfo.dockSetup = [&hasDebugUI](ImGuiID viewportID) {
    if(hasDebugUI)
    {
      // left side panel container
      ImGuiID debugID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.15F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Debug", debugID);
    }

    // right side panel container
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGui::DockBuilderDockWindow("Misc Settings", settingID);

    // bottom panel container
    ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", loggerID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.75F, nullptr, &loggerID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID streamingID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Right, 0.66F, nullptr, &profilerID);
    ImGui::DockBuilderDockWindow("Streaming memory", streamingID);
    ImGuiID statisticsID = ImGui::DockBuilderSplitNode(streamingID, ImGuiDir_Right, 0.5F, nullptr, &streamingID);
    ImGui::DockBuilderDockWindow("Statistics", statisticsID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  auto                  logger      = std::make_shared<nvapp::ElementLogger>();
  nvapp::ElementLogger* loggerDeref = logger.get();
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    loggerDeref->addLog(logLevel, "%s", text.c_str());
  });

  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());
  app.addElement(logger);
  app.addElement(sampleElement);
  app.addElement(std::make_shared<nvapp::ElementCamera>(cameraManipulator));
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager));
  
  // Run the GUI application
  app.run();

  nvutils::Logger::getInstance().setLogCallback(nullptr);

  // Cleanup in reverse order
  app.deinit();
  vkContext.deinit();

  return 0;
}
