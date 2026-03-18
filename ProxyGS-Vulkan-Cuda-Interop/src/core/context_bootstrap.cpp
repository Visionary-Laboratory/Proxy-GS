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

#include "context_bootstrap.hpp"
#include "../external_memory.hpp"
#include <nvvk/context.hpp>
#include <nvvk/check_error.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/validation_settings.hpp>

#ifdef USE_DLSS
#include "dlss_wrapper.hpp"
#endif

namespace core {

BootstrapResult createVulkanContext(const BootstrapConfig& cfg) {
    BootstrapResult result;
    
    // Feature structures (same as main.cpp)
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

    // Setup context info (same as main.cpp)
    nvvk::ContextInitInfo vkSetup{
        .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
        .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
        .queues             = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
    };
    
    // Apply config
    vkSetup.enableValidationLayers = cfg.enableValidation;
    vkSetup.forceGPU = cfg.forcedGpuIndex;
    
    // Add standard device extensions (same as main.cpp)
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

    // Add external memory extensions if needed
    appendExternalInteropExtensionsIfNeeded(vkSetup, cfg.needExternalInterop);

    // Setup validation
    nvvk::ValidationSettings validationSettings;
    if (vkSetup.enableValidationLayers) {
        validationSettings.message_id_filter = {"VUID-RuntimeSpirv-storageInputOutput16-06334", "VUID-VkShaderModuleCreateInfo-pCode-08740"};
        vkSetup.instanceCreateInfoExt = validationSettings.buildPNextChain();
    }

    // Add surface extensions
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    
    // Initialize Vulkan loader
    NVVK_CHECK(volkInitialize());

    {
        nvutils::ScopedTimer st("Creating Vulkan Context");
        
#if USE_DLSS
        // Adding the DLSS extensions to the instance
        static std::vector<VkExtensionProperties> extraInstanceExtensions;
        DlssRayReconstruction::getRequiredInstanceExtensions({}, extraInstanceExtensions);
        for(auto& ext : extraInstanceExtensions) {
            vkSetup.instanceExtensions.emplace_back(ext.extensionName);
        }
#endif

        VkResult vkResult{};
        
        result.ctx.contextInfo = vkSetup;
        
        vkResult = result.ctx.createInstance();
        NVVK_CHECK(vkResult);
        
        vkResult = result.ctx.selectPhysicalDevice();
        NVVK_CHECK(vkResult);
        
#if USE_DLSS
        // Adding the extra device extensions required by DLSS
        static std::vector<VkExtensionProperties> extraDeviceExtensions;
        DlssRayReconstruction::getRequiredDeviceExtensions({}, result.ctx.getInstance(), result.ctx.getPhysicalDevice(), extraDeviceExtensions);
        for(auto& ext : extraDeviceExtensions) {
            result.ctx.contextInfo.deviceExtensions.push_back({.extensionName = ext.extensionName, .specVersion = ext.specVersion});
        }
#endif

        vkResult = result.ctx.createDevice();
        NVVK_CHECK(vkResult);
        
        nvvk::DebugUtil::getInstance().init(result.ctx.getDevice());

        if (result.ctx.contextInfo.verbose) {
            NVVK_CHECK(nvvk::Context::printVulkanVersion());
            NVVK_CHECK(nvvk::Context::printInstanceLayers());
            NVVK_CHECK(nvvk::Context::printInstanceExtensions(result.ctx.contextInfo.instanceExtensions));
            NVVK_CHECK(nvvk::Context::printDeviceExtensions(result.ctx.getPhysicalDevice(), result.ctx.contextInfo.deviceExtensions));
            NVVK_CHECK(nvvk::Context::printGpus(result.ctx.getInstance(), result.ctx.getPhysicalDevice()));
            LOGI("_________________________________________________\n");
        }
    }

    return result;
}

void appendExternalInteropExtensionsIfNeeded(nvvk::ContextInitInfo& info, bool needInterop) {
    if (needInterop) {
        LOGI("Adding external memory extensions for interop\n");
        for (const auto& ext : lodclusters::ExternalMemoryManager::getRequiredDeviceExtensions()) {
            info.deviceExtensions.push_back({ext, nullptr, false});
        }
    }
}

} // namespace core