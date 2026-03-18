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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <array>
#include <string>
#include <utility>
#include <cstdint>
#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <chrono>

// Vulkan and nvpro_core2 includes
#include <vulkan/vulkan.h>
#include <nvvk/context.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvapp/application.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <filesystem>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>     // ::dup

// Project includes
#include "core/context_bootstrap.hpp"
#include "external_memory.hpp"
#include "lodclusters.hpp"
#include "scene.hpp"

// Asset path resolution utility functions
static std::filesystem::path module_dir() {
  Dl_info info{};
  dladdr((void*)&module_dir, &info);
  std::filesystem::path p(info.dli_fname ? info.dli_fname : "");
  return p.empty() ? std::filesystem::current_path() : p.parent_path();
}

static std::filesystem::path default_asset_root() {
#ifndef VK2TORCH_SOURCE_DIR
#define VK2TORCH_SOURCE_DIR ""
#endif
  // 优先环境变量，其次编译期内置源目录/resources，再其次.so同级的resources
  const char* env = std::getenv("VK2TORCH_DATA_ROOT");
  if(env && *env) return std::filesystem::path(env);
  if constexpr (sizeof(VK2TORCH_SOURCE_DIR) > 1) {
    auto p = std::filesystem::path(VK2TORCH_SOURCE_DIR) / "resources";
    if (std::filesystem::exists(p)) return p;
  }
  auto p = module_dir() / "resources";
  if (std::filesystem::exists(p)) return p;
  // 再退一步：.so 上级两层找 resources（适配 build 目录结构）
  auto p2 = module_dir().parent_path().parent_path() / "resources";
  return p2;
}

// 通用资源解析：给定相对文件，自动在若干常见子目录查找
static std::filesystem::path resolve_asset(const std::filesystem::path& assetRoot,
                                           const std::string& rel) {
  using std::filesystem::path;
  const path base = assetRoot;
  const path relp = rel;
  const path candidates[] = {
    base / relp,
    base / "shaders" / relp,
    base / "shaders/hbao" / relp,
    base / "post" / relp,
    base / "glsl" / relp
  };
  for (auto& c : candidates) {
    if (std::filesystem::exists(c)) return c;
  }
  return relp; // 兜底：返回原样，供旧逻辑报错时打印
}

/**
 * Vk2TorchApp - Main interface class for in-process VK2Torch integration
 * 
 * This class creates and manages an nvapp::Application running in headless mode,
 * with LodClusters component for zero-copy CUDA integration.
 */
class Vk2TorchApp {
public:
    /**
     * Constructor - Creates and starts headless Application with LOD rendering
     * @param width Render target width
     * @param height Render target height
     * @param raster Whether to use rasterization (true) or ray tracing (false)
     * @param scene_path Path to GLTF scene file (default: "matrix_city.glb")
     * @param asset_root Absolute path to asset root directory (default: auto-detect)
     */
    Vk2TorchApp(int width, int height, bool raster = true, const std::string& scene_path = "matrix_city.glb", const std::string& asset_root = "")
        : m_width(width), m_height(height), m_raster(raster), m_scene_path(scene_path)
    {
        pybind11::gil_scoped_release release;  // Release GIL during initialization
        
        try {
            // Initialize asset root directory
            if(!asset_root.empty()) {
                m_assetRoot = std::filesystem::path(asset_root);
            } else {
                m_assetRoot = default_asset_root(); // 现有函数可继续用
            }
            m_assetRootStr = m_assetRoot.string();
            printf("[vk2torch] assetRoot = %s\n", m_assetRootStr.c_str());
            
            initializeVulkan();
            createApplication();
            // startRenderThread();  // Scene will be loaded inside render thread
            
            printf("Vk2TorchApp: Successfully initialized with real 3D rendering\n");
        }
        catch (const std::exception& e) {
            cleanup();
            throw;
        }
    }

    /**
     * Destructor - stops Application and cleans up resources
     */
    ~Vk2TorchApp() {
        stop();
    }

    /**
     * Get render target dimensions
     * @return {height, width} pair
     */
    std::pair<int, int> size() const {
        return {m_height, m_width};
    }

    /**
     * Get complete interop export information with all FDs and metadata
     * @return Dictionary with memory/semaphore FDs and metadata for CUDA interop
     */
    pybind11::dict get_interop_info() {
        if (!m_externalMemory) {
            throw std::runtime_error("ExternalMemoryManager not initialized");
        }

        // Get complete export info from ExternalMemoryManager
        auto info = m_externalMemory->getInteropInfo();
        
        pybind11::dict d;
        d["depth_mem_fd"]             = info.depth_mem_fd;
        d["scene_ready_sem_fd"]       = info.scene_ready_sem_fd;
        d["camera_ready_sem_fd"]      = info.camera_ready_sem_fd;
        d["frame_done_sem_fd"]        = info.frame_done_sem_fd;
        d["width"]                    = info.width;
        d["height"]                   = info.height;
        d["row_pitch_bytes"]          = info.row_pitch_bytes;
        d["size"]                     = pybind11::int_(info.depth_mem_size);
        d["offset"]                   = pybind11::int_(info.depth_mem_offset);
        d["depth_format"]             = static_cast<uint32_t>(info.depth_format);
        d["last_signaled_frame_done"] = pybind11::int_(info.last_signaled_frame_done);
        
        // Add format name for convenience
        d["format_name"] = "VK_FORMAT_D24_UNORM_S8_UINT";
        d["handle_types"] = "OPAQUE_FD";
        
        return d;
    }

    /**
     * Set camera view and projection matrices from numpy arrays
     * @param proj 4x4 projection matrix (row-major numpy array)
     * @param view 4x4 view matrix (row-major numpy array)
     */
    void set_camera_matrices(pybind11::array proj_arr, pybind11::array view_arr) {
        // Validate input arrays
        // if (proj_arr.ndim() != 2 || proj_arr.shape(0) != 4 || proj_arr.shape(1) != 4) {
        //     throw std::runtime_error("proj must be 4x4 matrix");
        // }
        // if (view_arr.ndim() != 2 || view_arr.shape(0) != 4 || view_arr.shape(1) != 4) {
        //     throw std::runtime_error("view must be 4x4 matrix");
        // }

        // Convert to float arrays (handle both float32 and float64)
        std::array<float, 16> proj_data{}, view_data{};
        
        pybind11::buffer_info proj_buf = proj_arr.request();
        pybind11::buffer_info view_buf = view_arr.request();
        
        // Convert projection matrix
        if (proj_buf.itemsize == 8) {
            const double* src = static_cast<const double*>(proj_buf.ptr);
            for (int i = 0; i < 16; ++i) proj_data[i] = float(src[i]);
        } else {
            const float* src = static_cast<const float*>(proj_buf.ptr);
            for (int i = 0; i < 16; ++i) proj_data[i] = src[i];
        }
        
        // Convert view matrix
        if (view_buf.itemsize == 8) {
            const double* src = static_cast<const double*>(view_buf.ptr);
            for (int i = 0; i < 16; ++i) view_data[i] = float(src[i]);
        } else {
            const float* src = static_cast<const float*>(view_buf.ptr);
            for (int i = 0; i < 16; ++i) view_data[i] = src[i];
        }
        
        // Convert row-major to column-major (GLM format)
        glm::mat4 proj(1.0f), view(1.0f);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                proj[r][c] = proj_data[r * 4 + c];  // row-major -> column-major
                view[r][c] = view_data[r * 4 + c];
            }
        }
        
        // Apply camera override to LodClusters and signal camera ready
        if (m_lodclusters) {
            m_lodclusters->enableOverrideCamera(proj, view);
            
            // // Signal camera ready if external memory manager is available
            // if (m_externalMemory) {
            //     uint64_t frameValue = m_frameCounter.load();
            //     m_externalMemory->signalCameraReady(frameValue);
            // }
        } else {
            throw std::runtime_error("LodClusters not initialized");
        }

    }

    /**
     * Wait for scene ready signal from CPU (optional CPU-based waiting)
     * @param timeout_ms Timeout in milliseconds (default: 5000ms)
     * @return True if scene ready signal received, false on timeout
     */
    bool wait_scene_ready_cpu(uint32_t timeout_ms = 5000) {
        if (!m_externalMemory) {
            throw std::runtime_error("ExternalMemoryManager not initialized");
        }
        
        return m_externalMemory->waitSceneReady(1, timeout_ms);
    }

    /**
     * Render one frame with CPU control (for frame-by-frame Python control)
     * @return Frame number that was rendered
     */
    uint64_t render_one_frame() {
        if (!m_lodclusters) {
            throw std::runtime_error("LodClusters not initialized");
        }
        
        uint64_t frame_value = m_frameCounter.fetch_add(1) + 1;
        
        // Trigger frame render with timeline signaling
        m_lodclusters->renderOneFrame(frame_value);
        
        return frame_value;
    }

    /**
     * Get row pitch in bytes for the exported buffer
     * @return Row pitch in bytes from ExternalMemoryManager
     */
    int row_pitch_bytes() const {
        if (m_externalMemory) {
            auto info = m_externalMemory->getDepthExportInfo();
            return info.row_pitch_bytes;
        }
        return m_width * 4;  // Fallback
    }

    /**
     * Export depth buffer file descriptor for CUDA import
     * @return Duplicated file descriptor (caller must close)
     */
    int export_depth_buffer_fd() const {
        if (m_externalMemory) {
            auto info = m_externalMemory->getDepthExportInfo();
            return (info.memory_fd >= 0) ? ::dup(info.memory_fd) : -1;
        }
        printf("Vk2TorchApp: export_depth_buffer_fd() - not ready\n");
        return -1;  // Not ready
    }

    /**
     * Export frame done timeline semaphore file descriptor
     * @return Duplicated file descriptor (caller must close)
     */
    int export_frame_done_semaphore_fd() const {
        if (m_externalMemory) {
            auto info = m_externalMemory->getDepthExportInfo();
            return (info.timeline_semaphore_fd >= 0) ? ::dup(info.timeline_semaphore_fd) : -1;
        }
        printf("Vk2TorchApp: export_frame_done_semaphore_fd() - not ready\n");
        return -1;  // Not ready
    }

    /**
     * Set camera matrices and frame number
     * @param frame Frame number for synchronization
     * @param view 4x4 view matrix in column-major order
     * @param proj 4x4 projection matrix in column-major order
     */
    void set_camera(uint64_t frame, const std::array<float, 16>& view, const std::array<float, 16>& proj) {
        // Convert arrays to GLM matrices
        glm::mat4 viewMat(1.0f), projMat(1.0f);
        std::memcpy(&viewMat[0][0], view.data(), 16 * sizeof(float));
        std::memcpy(&projMat[0][0], proj.data(), 16 * sizeof(float));
        
        if (m_lodclusters) {
            m_lodclusters->enableOverrideCamera(projMat, viewMat);
            if (m_externalMemory) {
                m_externalMemory->signalCameraReady(frame);
            }
        } else {
            printf("Vk2TorchApp: set_camera(frame=%lu) - not ready\n", frame);
        }
    }

    /**
     * Get the last signaled frame number
     * @return Timeline semaphore value for last completed frame
     */
    uint64_t last_signaled_frame() const {
        if (m_externalMemory) {
            auto info = m_externalMemory->getInteropInfo();
            return info.last_signaled_frame_done;
        }
        return 0;
    }

    /**
     * Export depth buffer information with file descriptors for CUDA interop
     * @param dup_fds Whether to duplicate file descriptors (default: true)
     * @return Dictionary with memory/semaphore FDs and buffer information
     */
    pybind11::dict get_depth_export_info(bool dup_fds = true) {
        if (!m_externalMemory) {
            throw std::runtime_error("ExternalMemoryManager not initialized");
        }

        // Get export info from ExternalMemoryManager
        auto info = m_externalMemory->getDepthExportInfo();
        
        int mem_fd = info.memory_fd;
        int sem_fd = info.timeline_semaphore_fd;
        
        if (dup_fds) {
            if (mem_fd >= 0) mem_fd = ::dup(mem_fd);
            if (sem_fd >= 0) sem_fd = ::dup(sem_fd);
        }
        
        pybind11::dict d;
        d["mem_fd"]            = mem_fd;
        d["sem_fd"]            = sem_fd;
        d["row_pitch_bytes"]   = info.row_pitch_bytes;
        d["width"]             = info.width;
        d["height"]            = info.height;
        d["size"]              = pybind11::int_(info.size);
        d["offset"]            = pybind11::int_(info.offset);
        d["format_name"]       = "VK_FORMAT_D24_UNORM_S8_UINT"; // TODO: convert enum to string
        d["handle_types"]      = "OPAQUE_FD";
        d["semaphore_payload"] = pybind11::int_(info.last_signaled_payload);
        return d;
    }


    /**
     * Trigger one frame render and signal completion
     * @param sync Whether to wait for frame completion (default: false)
     * @return Timeline semaphore value for this frame
     */
    uint64_t render_and_signal(bool sync = false) {
        if (!m_lodclusters) {
            throw std::runtime_error("LodClusters not initialized");
        }
        
        uint64_t frame_value = m_frameCounter.fetch_add(1) + 1;
        
        // Trigger frame render with timeline signaling
        m_lodclusters->renderOneFrame(frame_value);
        
        if (sync) {
            waitFrameDone(frame_value);
        }
        
        return frame_value;
    }

    /**
     * Initialize headless rendering mode (three-part system)
     * @return True if successful
     */
    bool headless_init() {
        if (!m_app) {
            throw std::runtime_error("Application not created");
        }
        return m_app->headlessInit();
    }

    /**
     * Render one frame in headless mode (three-part system)
     * @return True if more frames available, false when done
     */
    bool headless_step() {
        if (!m_app) {
            throw std::runtime_error("Application not created");
        }
        return m_app->headlessStep();
    }

    /**
     * Shutdown headless rendering mode (three-part system)
     */
    void headless_shutdown() {
        if (!m_app) {
            throw std::runtime_error("Application not created");
        }
        m_app->headlessShutdown();
    }

    /**
     * Stop the Application and cleanup resources
     */
    void stop() {
        printf("Vk2TorchApp: Stopping...\n");
        
        {
            std::lock_guard<std::mutex> lock(m_stopMutex);
            if (m_running) {
                m_running = false;
                if (m_app) {
                    m_app->close();
                }
            }
        }
        
        if (m_renderThread.joinable()) {
            m_renderThread.join();
        }
        
        cleanup();
        printf("Vk2TorchApp: Stopped\n");
    }

private:
    // Configuration
    int m_width;
    int m_height;
    bool m_raster;
    std::string m_scene_path;
    
    // Real application components
    nvvk::Context m_vkContext;
    std::unique_ptr<nvapp::Application> m_app;
    std::unique_ptr<lodclusters::ExternalMemoryManager> m_externalMemory;
    std::shared_ptr<lodclusters::LodClusters> m_lodclusters;
    std::unique_ptr<nvutils::ProfilerManager>   m_profilerManager;
    std::unique_ptr<nvutils::ParameterRegistry> m_parameterRegistry;
    
    // Asset path management
    std::filesystem::path m_assetRoot;
    std::string m_assetRootStr;  // For stable pointer access


    // Threading and synchronization
    std::thread m_renderThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_ready{false};
    std::mutex m_stopMutex;
    std::condition_variable m_stopCondition;
    
    // Frame synchronization for render_and_signal and set_camera_matrices
    std::atomic<uint64_t> m_frameCounter{0};
    
    // Shared components for elements
    std::shared_ptr<nvutils::CameraManipulator> m_cameraManipulator;
    
    // Frame synchronization helper
    void waitFrameDone(uint64_t value) {
        if (m_app) {
            // Simple synchronization - wait for all operations to complete
            // TODO: Implement proper timeline semaphore waiting
            vkDeviceWaitIdle(m_app->getDevice());
        }
    }
    
    // Real Vulkan initialization
    void initializeVulkan() {
        printf("Vk2TorchApp: Initializing Vulkan context...\n");
        
        // 1) Initialize Volk
        NVVK_CHECK(volkInitialize());
        
        // Feature structures (same as context_bootstrap.cpp)
        VkPhysicalDeviceMeshShaderFeaturesNV meshNV = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rayPosKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clustersNV = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};
        VkPhysicalDeviceShaderClockFeaturesKHR clockKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
        VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};

        // Setup context info - headless mode, no surface extensions needed
        nvvk::ContextInitInfo vkSetup{
            .instanceExtensions = {},  // Start empty for headless
            .deviceExtensions   = {},  // No swapchain for headless
            .queues             = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
        };
        
        vkSetup.enableValidationLayers = false;  // Disable validation for performance
        vkSetup.forceGPU = -1;  // Auto-select GPU
        
        // Add standard device extensions
        vkSetup.deviceExtensions.push_back({VK_NV_MESH_SHADER_EXTENSION_NAME, &meshNV});
        vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, nullptr, false});
        vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accKHR, false});
        vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rayKHR, false});
        vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &rayPosKHR, false});
        vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayQueryKHR, false});
        vkSetup.deviceExtensions.push_back({VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, &clustersNV, false, 2});
        vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockKHR, false});
        vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures, false});
        vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &shadingRateFeatures, false});
        vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &barycentricFeatures, false});
        vkSetup.deviceExtensions.push_back({VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, nullptr, false});

        // Add external memory extensions (needed for CUDA interop)
        printf("Vk2TorchApp: Adding external memory extensions for interop\n");
        for (const auto& ext : lodclusters::ExternalMemoryManager::getRequiredDeviceExtensions()) {
            vkSetup.deviceExtensions.push_back({ext, nullptr, false});
        }

        // 3) Optional: Filter available instance extensions (insurance against any residual extensions)
        auto& inst = vkSetup.instanceExtensions;
        uint32_t cnt = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &cnt, nullptr);
        std::vector<VkExtensionProperties> props(cnt);
        if (cnt) vkEnumerateInstanceExtensionProperties(nullptr, &cnt, props.data());
        {
            std::vector<const char*> keep;
            for (const char* e : inst) {
                bool found = false;
                for (auto &p: props) if (!strcmp(p.extensionName, e)) { found = true; break; }
                if (found) keep.push_back(e);
                else fprintf(stderr, "[vk2torch] drop unavailable instance ext: %s\n", e);
            }
            inst.swap(keep);
        }
        fprintf(stderr, "[vk2torch] final instanceExtensions count = %zu\n", inst.size());

        m_vkContext.contextInfo = vkSetup;
        
        // 4) Create Instance (with fallback to empty extensions if needed)
        printf("Vk2TorchApp: Creating Vulkan Instance\n");
        VkResult r = m_vkContext.createInstance();
        if (r == VK_ERROR_EXTENSION_NOT_PRESENT) {
            fprintf(stderr, "[vk2torch] EXT_NOT_PRESENT: retry with EMPTY instance extensions\n");
            // Clear extensions and create new Context
            vkSetup.instanceExtensions.clear();
            nvvk::Context newContext;
            newContext.contextInfo = vkSetup;
            NVVK_CHECK(newContext.createInstance());
            m_vkContext = std::move(newContext);
        } else {
            NVVK_CHECK(r);
        }
        
        NVVK_CHECK(m_vkContext.selectPhysicalDevice());
        NVVK_CHECK(m_vkContext.createDevice());
        
        nvvk::DebugUtil::getInstance().init(m_vkContext.getDevice());
        
        printf("Vk2TorchApp: Vulkan context created successfully\n");
    }
    
    void createApplication() {
        printf("Vk2TorchApp: Creating Application with real 3D scene rendering...\n");
        
        // Create shared components
        m_cameraManipulator = std::make_shared<nvutils::CameraManipulator>();
        
        // Create external memory manager for in-process mode
        m_externalMemory = std::make_unique<lodclusters::ExternalMemoryManager>();
        lodclusters::ExternalMemoryConfig extConfig;
        extConfig.enabled = true;
        extConfig.width = m_width;
        extConfig.height = m_height;
        extConfig.format = VK_FORMAT_R32_UINT;  // Packed 24-bit depth in 32-bit
        extConfig.exportDepthFormat = VK_FORMAT_R32_UINT;
        extConfig.pack24In32 = true;
        
        if (!m_externalMemory->initInProcess(m_vkContext.getDevice(), m_vkContext.getPhysicalDevice(), extConfig)) {
            throw std::runtime_error("Failed to initialize external memory manager");
        }
        
        // Create nvapp::Application
        nvapp::ApplicationCreateInfo appInfo;
        appInfo.name = "VK2Torch In-Process - Real 3D Rendering";
        appInfo.headless = true;
        appInfo.headlessFrameCount = 1000000;  // Run indefinitely
        appInfo.windowSize = {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height)};
        appInfo.vSync = false;
        appInfo.useMenu = false;
        appInfo.instance = m_vkContext.getInstance();
        appInfo.device = m_vkContext.getDevice();
        appInfo.physicalDevice = m_vkContext.getPhysicalDevice();
        appInfo.queues = m_vkContext.getQueueInfos();
        
        m_app = std::make_unique<nvapp::Application>();
        m_app->init(appInfo);
        
        // Create LodClusters element for real 3D scene rendering
        printf("Vk2TorchApp: Creating LodClusters for real scene rendering...\n");
        lodclusters::LodClusters::Info lodInfo;

        m_profilerManager   = std::make_unique<nvutils::ProfilerManager>();
        m_parameterRegistry = std::make_unique<nvutils::ParameterRegistry>();
        m_cameraManipulator = std::make_shared<nvutils::CameraManipulator>();

        lodInfo.cameraManipulator = m_cameraManipulator;
        lodInfo.profilerManager = m_profilerManager.get();  // No profiler needed for headless
        lodInfo.parameterRegistry = m_parameterRegistry.get();  // Use defaults
        lodInfo.externalMemoryManager = m_externalMemory.get();  // Connect to external memory
        lodInfo.assetRoot = m_assetRootStr.c_str();  // Pass asset root for shader loading
        
        m_lodclusters = std::make_shared<lodclusters::LodClusters>(lodInfo);
        m_lodclusters->setSupportsClusters(m_vkContext.hasExtensionEnabled(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME));
        
        // Set the scene file path if provided by user
        if (!m_scene_path.empty()) {
            std::filesystem::path scenePath = m_scene_path;
            if (!scenePath.is_absolute()) {
                // Make it relative to asset root if it's a relative path
                scenePath = m_assetRoot / scenePath;
            }
            m_lodclusters->setSceneFilePath(scenePath);
            printf("Vk2TorchApp: Set scene file path: %s\n", scenePath.string().c_str());
        }
        
        // Add LodClusters element for rendering
        m_app->addElement(m_lodclusters);
        
        printf("Vk2TorchApp: Real 3D rendering pipeline created successfully\n");
        printf("             - LodClusters will render actual 3D geometry\n");
    }
    
    void loadDefaultScene() {
        printf("Vk2TorchApp: Loading default 3D scene...\n");
        
        // Find bunny.gltf in standard locations (same logic as main.cpp)
        std::filesystem::path scenePath;
        
        if (!m_scene_path.empty()) {
            // User specified a scene path
            scenePath = m_scene_path;
            printf("Vk2TorchApp: Using user-specified scene: %s\n", scenePath.string().c_str());
        } else {
            // Search for default bunny scene
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
                    scenePath = fullPath;
                    foundBunny = true;
                    printf("Vk2TorchApp: Found default bunny scene at: %s\n", scenePath.string().c_str());
                    break;
                }
            }
            
            if (!foundBunny) {
                printf("Vk2TorchApp: Warning - bunny.gltf not found, will use empty scene\n");
                printf("             Search paths tried:\n");
                for (const auto& searchPath : searchPaths) {
                    printf("               %s/bunny_v2/bunny.gltf\n", searchPath.string().c_str());
                }
                return;  // Continue without scene - will render empty/test content
            }
        }
        
        // Initialize the scene in LodClusters using public onFileDrop method
        if (m_lodclusters && std::filesystem::exists(scenePath)) {
            // Use onFileDrop which is public and designed for loading scene files
            m_lodclusters->onFileDrop(scenePath);
            printf("Vk2TorchApp: ✅ 3D scene loading initiated: %s\n", scenePath.string().c_str());
        } else {
            printf("Vk2TorchApp: ⚠️ Scene file not found: %s\n", scenePath.string().c_str());
        }
    }
    
    void startRenderThread() {
        printf("Vk2TorchApp: Starting render thread...\n");
        
        m_running = true;
        
        m_renderThread = std::thread([this]() {
            try {
                printf("Vk2TorchApp: Render thread started, running application loop\n");
                
                // Application is ready immediately without PyBridge
                m_ready = true;
                printf("Vk2TorchApp: Application initialized\n");
                
                // Load the default scene
                loadDefaultScene();
                
                // Run the application loop
                m_app->run();
                
                printf("Vk2TorchApp: Application loop exited\n");
            } catch (const std::exception& e) {
                printf("Vk2TorchApp: Render thread exception: %s\n", e.what());
            } catch (...) {
                printf("Vk2TorchApp: Render thread unknown exception caught\n");
            }
            
            m_running = false;
        });
        
        // Brief wait to ensure thread starts
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        printf("Vk2TorchApp: Render thread started successfully\n");
    }
    
    void cleanup() {
        printf("Vk2TorchApp: Cleaning up resources...\n");
        
        // Clean up in reverse order of creation
        if (m_app) {
            m_app->deinit();
            m_app.reset();
        }
        
        // Clean up elements
        m_lodclusters.reset();
        
        if (m_externalMemory) {
            m_externalMemory->deinit();
            m_externalMemory.reset();
        }
        
        m_cameraManipulator.reset();
        
        m_vkContext.deinit();
        
        printf("Vk2TorchApp: Cleanup complete\n");
    }
};

namespace py = pybind11;

PYBIND11_MODULE(vk2torch_ext, m) {
    m.doc() = "VK2Torch Extension - In-process Vulkan to PyTorch integration";

    py::class_<Vk2TorchApp>(m, "Vk2TorchApp")
        .def(py::init<int, int, bool, const std::string&, const std::string&>(),
             "Create Vk2TorchApp instance with headless Application",
             py::arg("width"), py::arg("height"), 
             py::arg("raster") = true, py::arg("scene_path") = "", py::arg("asset_root") = "")
        
        // Dimensions and buffer info
        .def("size", &Vk2TorchApp::size,
             "Get render target dimensions as (height, width) tuple")
        
        .def("row_pitch_bytes", &Vk2TorchApp::row_pitch_bytes,
             "Get row pitch in bytes for exported buffers")
        
        // External memory/semaphore export (FD automatically duplicated)
        .def("export_depth_buffer_fd", &Vk2TorchApp::export_depth_buffer_fd,
             "Export depth buffer file descriptor (caller must close)")
        
        .def("export_frame_done_semaphore_fd", &Vk2TorchApp::export_frame_done_semaphore_fd,
             "Export frame done semaphore file descriptor (caller must close)")
        
        // Camera and frame control
        .def("set_camera", &Vk2TorchApp::set_camera,
             "Set camera view and projection matrices with frame number",
             py::arg("frame"), py::arg("view"), py::arg("proj"))
        
        // New zero-copy rendering API methods
        .def("get_depth_export_info", &Vk2TorchApp::get_depth_export_info,
             "Export depth buffer information with file descriptors for CUDA interop",
             py::arg("dup_fds") = true)
        
        .def("set_camera_matrices", &Vk2TorchApp::set_camera_matrices,
             "Set camera view and projection matrices (numpy 4x4 arrays)",
             py::arg("proj"), py::arg("view"))
        
        .def("render_and_signal", &Vk2TorchApp::render_and_signal,
             "Trigger one frame render and signal completion",
             py::arg("sync") = false)
        
        .def("last_signaled_frame", &Vk2TorchApp::last_signaled_frame,
             "Get last signaled frame number from timeline semaphore")
        
        // New interop methods for complete three-way timeline semaphore coordination
        .def("get_interop_info", &Vk2TorchApp::get_interop_info,
             "Get complete interop export information with all FDs and metadata")
        
        .def("wait_scene_ready_cpu", &Vk2TorchApp::wait_scene_ready_cpu,
             "Wait for scene ready signal from CPU (optional CPU-based waiting)",
             py::arg("timeout_ms") = 5000)
        
        .def("render_one_frame", &Vk2TorchApp::render_one_frame,
             "Render one frame with CPU control (for frame-by-frame Python control)")
        
        // Three-part headless control system
        .def("headless_init", &Vk2TorchApp::headless_init,
             "Initialize headless rendering mode (call once before headless_step)")
        
        .def("headless_step", &Vk2TorchApp::headless_step,
             "Render one frame in headless mode, returns false when done")
        
        .def("headless_shutdown", &Vk2TorchApp::headless_shutdown,
             "Shutdown headless rendering mode (call after headless_step returns false)")
        
        // Lifecycle management
        .def("stop", &Vk2TorchApp::stop,
             "Stop the Application and cleanup resources");

    // Module metadata
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "NVIDIA Corporation";
    m.attr("__doc__") = "VK2Torch Extension - In-process Vulkan LOD rendering with zero-copy PyTorch integration";
}