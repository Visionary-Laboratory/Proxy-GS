# CMake toolchain file: Use system compilers, ignore conda sysroot, allow conda Python
# This prevents GLIBC conflicts by avoiding conda's cross-compilation toolchain

# ---- use system compilers (not conda wrappers)
set(CMAKE_C_COMPILER /usr/bin/gcc-10 CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10 CACHE FILEPATH "" FORCE)

# ---- search only system prefixes for ALL non-Python deps
set(CMAKE_SYSTEM_PREFIX_PATH "/usr;/usr/local" CACHE STRING "" FORCE)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)

# ---- ignore conda sysroot & gcc paths so they won't be mistakenly selected
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_IGNORE_PREFIX_PATH
       "$ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu"
       "$ENV{CONDA_PREFIX}/lib/gcc"
       "$ENV{CONDA_PREFIX}/lib64"
       )
endif()

# ---- Force system library search paths to override conda contamination
set(CMAKE_SYSTEM_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu;/usr/lib;/lib/x86_64-linux-gnu;/lib" CACHE STRING "" FORCE)
set(CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "/usr/lib/x86_64-linux-gnu;/usr/lib;/lib/x86_64-linux-gnu;/lib")

# ---- Completely override problematic conda linker flags
set(CMAKE_EXE_LINKER_FLAGS_INIT "")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "")

# ---- RPATH: prioritize system libraries and VulkanSDK at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
# Runtime will also append VulkanSDK path (see command line -DVULKAN_SDK_RPATH)