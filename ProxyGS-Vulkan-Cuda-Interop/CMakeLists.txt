cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Global PIC setting for Python extension compatibility - MUST BE SET BEFORE nvpro_core2
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# Force PIC for ALL targets (including third-party dependencies)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)

option(NVPRO2_ENABLE_nvgl OFF)
option(NVPRO2_ENABLE_nvgpu_monitor OFF)
option(NVPRO2_ENABLE_nvslang OFF)
option(NVPRO2_ENABLE_nvvkgltf OFF)


# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

# Apply automatic patches to nvpro_core2
include(nvpro_patch)

find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

# C2) 全局 PIC & 编译器配置 (Global PIC and compiler configuration)
# PIC already set globally at top of file - removed duplicate

# RPATH settings to avoid conda library conflicts
set(CMAKE_SKIP_RPATH OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu")
set(CMAKE_BUILD_RPATH "/usr/lib/x86_64-linux-gnu")

#####################################################################################
# Download the default scene
download_files(FILENAMES bunny_v2.zip EXTRACT)

#####################################################################################
#####################################################################################
# Build options
#
option(BUILD_APP "Build vk_lod_clusters application" ON)
option(VK2TORCH_STUB "Build stub implementation for vk2torch_ext" OFF)

#####################################################################################
# extra libraries

if(NOT TARGET nv_cluster_lod_builder)
  add_subdirectory(external/nv_cluster_lod_builder)
endif()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.*)
list(FILTER SOURCE_FILES EXCLUDE REGEX "dlss*")

# Add VMA implementation file
list(APPEND SOURCE_FILES src/thirdparty/vma_impl.cpp)

# Add core files
file(GLOB CORE_FILES src/core/*.*)

# Add pybridge files
file(GLOB PYBRIDGE_FILES src/pybridge/*.*)

file(GLOB SHADER_FILES shaders/*.glsl shaders/*.h)
file(GLOB VK_RADIX_SORT_FILES thirdparty/vulkan_radix_sort/src/vk_radix_sort.cc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vulkan_radix_sort/include)

#####################################################################################
# A1) 核心对象库（统一真实渲染实现，避免两套分支不一致）
#
add_library(vklod_core_obj OBJECT
  src/lodclusters.cpp
  src/lodclusters_ui.cpp
  src/scene.cpp
  src/scene_cache.cpp
  src/scene_gltf.cpp
  src/scene_preloaded.cpp
  src/scene_streaming.cpp
  src/scene_streaming_utils.cpp
  src/resources.cpp
  src/renderer.cpp
  src/renderer_raster_clusters_lod.cpp
  src/renderer_raytrace_clusters_lod.cpp
  src/cgltf.cpp
  src/nvhiz_vk.cpp
  src/hbao_pass.cpp
  src/asset_resolver.cpp
  thirdparty/vulkan_radix_sort/src/vk_radix_sort.cc
)

# 必须 PIC 以便被 Python 扩展使用
set_property(TARGET vklod_core_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(vklod_core_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders  # 让 #include "hbao.h" 生效
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vulkan_radix_sort/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/nv_cluster_lod_builder/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/nv_cluster_lod_builder/nv_cluster_builder/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/nv_cluster_lod_builder/meshoptimizer/src
  ${Vulkan_INCLUDE_DIR}
)

# 链接接口库以获得头文件路径和编译定义
target_link_libraries(vklod_core_obj PUBLIC
  nvpro2::nvapp
  nvpro2::nvvk
  nvpro2::nvutils
  nvpro2::nvvkglsl
  cgltf
)

# OBJECT 库不能链接实际库，只能链接接口库
# 依赖项将在使用此对象库的目标中处理

# 让 nvpro_core2 的项目宏注入（修复 TARGET_EXE_TO_DOWNLOAD_DIRECTORY 等）
add_project_definitions(vklod_core_obj)

#####################################################################################
# Executable
#
if(BUILD_APP)
  add_executable(${PROJECT_NAME} 
    ${SOURCE_FILES} 
    ${CORE_FILES} 
    ${PYBRIDGE_FILES} 
    ${SHADER_FILES}
    $<TARGET_OBJECTS:vklod_core_obj>  # 使用统一的核心对象库
  )
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

  target_link_libraries(${PROJECT_NAME} PRIVATE
    nvpro2::nvapp
    nvpro2::nvgui
    nvpro2::nvutils
    nvpro2::nvvk
    nvpro2::nvvkglsl
    nv_cluster_lod_builder
    meshoptimizer
    cgltf
    Threads::Threads
  )

  # Force static linking of standard libraries to reduce conda conflicts
  target_link_options(${PROJECT_NAME} PRIVATE -static-libstdc++ -static-libgcc)
endif()

#####################################################################################
# Adding DLSS if selected
option(USE_DLSS "Use DLSS-RR for denoising" ON)
if(USE_DLSS AND BUILD_APP)
  message(STATUS "Using DLSS-RR for denoising")
    download_package(
      NAME DLSSRR
      URLS https://github.com/NVIDIA/DLSS/archive/refs/tags/v310.3.0.zip
      VERSION 310.3.0
      LOCATION DLSSRR_SOURCE_DIR
    )

    # Set NGX SDK root path
    set(NGX_SDK_ROOT "${DLSSRR_SOURCE_DIR}/DLSS-310.3.0/")
    message(STATUS " - DLSS path: " ${NGX_SDK_ROOT})
    include(cmake/ngx.cmake)
    
    target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DLSS)
    list(APPEND SOURCE_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_denoiser.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_denoiser.hpp
        )
    target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
    source_group("Source Files" FILES ${SOURCE_FILES})

    if (WIN32)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            file(GLOB DLSS_DLLS "${NGX_SDK_ROOT}/lib/Windows_x86_64/dev/nvngx_*.dll")
        else()
            file(GLOB DLSS_DLLS "${NGX_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_*.dll")
        endif()
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            file(GLOB DLSS_DLLS "${NGX_SDK_ROOT}/lib/Linux_x86_64/dev/libnvidia-ngx*.so.*")
        else()
            file(GLOB DLSS_DLLS "${NGX_SDK_ROOT}/lib/Linux_x86_64/rel/libnvidia-ngx*.so.*")
        endif()    
    endif()

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLSS_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}> VERBATIM
        )

    install(FILES ${DLSS_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX})

endif()

#####################################################################################
# source groups and project finalization (only for main app)
if(BUILD_APP)
  source_group("Shader Files" FILES ${SHADER_FILES})
  source_group("Source Files" FILES ${SOURCE_FILES})

  add_project_definitions(${PROJECT_NAME})
  copy_to_runtime_and_install( ${PROJECT_NAME}
      NVSHADERS_FILES "${NVSHADERS_DIR}/nvshaders/slang_types.h"
                      "${NVSHADERS_DIR}/nvshaders/sky_io.h.slang"
                      "${NVSHADERS_DIR}/nvshaders/sky_functions.h.slang"
                      "${NVSHADERS_DIR}/nvshaders/constants.h.slang"
      LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
      AUTO
  )
endif()

#####################################################################################
# Python Extension (vk2torch_ext) - optional
#
option(BUILD_PYTHON_EXT "Build Python extension vk2torch_ext" ON)

if(BUILD_PYTHON_EXT)
  # Find Python and pybind11
  find_package(Python3 COMPONENTS Interpreter Development)
  
  if(Python3_FOUND)
    # Try to find pybind11
    find_package(pybind11 QUIET CONFIG)
    
    if(NOT pybind11_FOUND)
      # Try to find pybind11 via Python
      execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RESULT
      )
      
      if(PYBIND11_RESULT EQUAL 0)
        message(STATUS "Found pybind11 via Python at: ${PYBIND11_INCLUDE_DIR}")
        
        # Manually setup pybind11
        include_directories(${PYBIND11_INCLUDE_DIR})
        
        # B2) 手动分支 - 使用统一的核心对象库
        add_library(vk2torch_ext MODULE 
          src/pybind/vk2torch_ext.cpp
          src/core/context_bootstrap.cpp
          src/external_memory.cpp
          src/thirdparty/vma_impl.cpp
          $<TARGET_OBJECTS:vklod_core_obj>  # 统一使用核心对象库
        )
        
        message(STATUS "vk2torch_ext Python extension will be built (manual setup)")
        
      else()
        message(WARNING "pybind11 not found - skipping Python extension")
      endif()
    else()
      message(STATUS "Found pybind11 CONFIG")
      
      # B1) pybind11 CONFIG 分支 - 使用统一的核心对象库
      pybind11_add_module(vk2torch_ext 
        src/pybind/vk2torch_ext.cpp
        src/core/context_bootstrap.cpp
        src/external_memory.cpp
        src/thirdparty/vma_impl.cpp
        $<TARGET_OBJECTS:vklod_core_obj>  # 统一使用核心对象库
      )
      
      message(STATUS "Found pybind11 CONFIG - will use unified configuration")
    endif()
    
    # B3) 统一的公共设置（放在两个分支之后）
    if(TARGET vk2torch_ext)
      # 输出目录 & RPATH & 可见性 & PIC
      set_target_properties(vk2torch_ext PROPERTIES
        PREFIX ""
        OUTPUT_NAME "vk2torch_ext"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/_bin/$<CONFIG>"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
        BUILD_RPATH_USE_ORIGIN ON
        INSTALL_RPATH "$ORIGIN;$ORIGIN/vulkan"
      )

      target_include_directories(vk2torch_ext PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders  # 让 #include "hbao.h" 生效
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vulkan_radix_sort/include
        ${Vulkan_INCLUDE_DIR}
      )

      target_link_libraries(vk2torch_ext PRIVATE
        nvpro2::nvapp
        nvpro2::nvvk
        nvpro2::nvutils
        nvpro2::nvvkglsl
        nv_cluster_lod_builder
        meshoptimizer
        cgltf
        Vulkan::Vulkan
        Threads::Threads
      )

      # nvpro_core2 的宏（修复 TARGET_EXE_TO_DOWNLOAD_DIRECTORY 等）
      add_project_definitions(vk2torch_ext)
      
      # 编译定义
      target_compile_definitions(vk2torch_ext PRIVATE VERSION_INFO="dev")
      target_compile_definitions(vk2torch_ext PRIVATE PYBIND11_MODULE)
      if(VK2TORCH_STUB)
        target_compile_definitions(vk2torch_ext PRIVATE VK2TORCH_STUB)
      endif()
      
      # D) 运行库打包：VulkanSDK .so 就地拷贝 (Runtime library packaging: VulkanSDK .so local copying)
      if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_LIB_DIR "$ENV{VULKAN_SDK}/lib")
        if(EXISTS "${VULKAN_LIB_DIR}")
          file(GLOB VULKAN_RUNTIME_LIBS 
               "${VULKAN_LIB_DIR}/libvulkan.so.*" 
               "${VULKAN_LIB_DIR}/libshaderc_shared.so.*"
               "${VULKAN_LIB_DIR}/libSPIRV-Tools-shared.so.*")
          
          if(VULKAN_RUNTIME_LIBS)
            add_custom_command(TARGET vk2torch_ext POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vk2torch_ext>/vulkan"
              COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VULKAN_RUNTIME_LIBS} "$<TARGET_FILE_DIR:vk2torch_ext>/vulkan/"
              COMMENT "Copying VulkanSDK runtime libraries to Python module directory"
              VERBATIM)
            message(STATUS "VulkanSDK runtime libraries will be packaged with Python module")
          endif()
        endif()
      endif()
      
      # E) Python模块命名与旧产物清理 (Python module naming and cleanup)
      # Clean up any old artifacts in the root directory
      add_custom_command(TARGET vk2torch_ext PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove -f "${CMAKE_SOURCE_DIR}/vk2torch_ext.so"
        COMMAND ${CMAKE_COMMAND} -E remove -f "${CMAKE_SOURCE_DIR}/vk2torch_ext.so.old"
        COMMENT "Cleaning up old Python module artifacts from root directory"
        VERBATIM)
      
      # Ensure proper module naming (handled by target properties above)
      message(STATUS "vk2torch_ext will be named: vk2torch_ext${PYTHON_MODULE_EXTENSION}")
      message(STATUS "vk2torch_ext output directory: ${CMAKE_BINARY_DIR}/_bin/$<CONFIG>")
      
      message(STATUS "vk2torch_ext configured with unified core object library")
    endif()
  else()
    message(WARNING "Python3 not found - skipping Python extension")
  endif()
else()
  message(STATUS "Python extension build disabled")
endif()

