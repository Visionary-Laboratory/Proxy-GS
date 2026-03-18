# CMake toolchain file: Use conda-compatible system GCC with proper GLIBC targeting
# This ensures the extension is compatible with conda environment GLIBC versions

# Use older system GCC that targets compatible GLIBC
# GCC-9 typically links against GLIBC_2.29/GLIBCXX_3.4.28 which is more conda-compatible  
set(CMAKE_C_COMPILER /usr/bin/gcc-9 CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9 CACHE FILEPATH "" FORCE)

# If GCC-9 not available, try GCC-8 (even more conservative)
if(NOT EXISTS "/usr/bin/gcc-9")
    set(CMAKE_C_COMPILER /usr/bin/gcc-8 CACHE FILEPATH "" FORCE) 
    set(CMAKE_CXX_COMPILER /usr/bin/g++-8 CACHE FILEPATH "" FORCE)
endif()

# Force conservative C++17 standard (not C++20) for better compatibility
set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "" FORCE)

# Target older GLIBC for conda compatibility
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 -fPIC" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "-fPIC" CACHE STRING "" FORCE)

# Use system prefix paths but prefer conda-compatible versions
set(CMAKE_SYSTEM_PREFIX_PATH "/usr;/usr/local" CACHE STRING "" FORCE)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER) 
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)

# Static link standard libraries to avoid version conflicts
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-static-libgcc -static-libstdc++" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS "-static-libgcc -static-libstdc++" CACHE STRING "" FORCE)

# RPATH configuration for runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)