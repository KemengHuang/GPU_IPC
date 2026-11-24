# =============================================================================
# Basic Configuration
# =============================================================================

# Minimum required CMake version
cmake_minimum_required(VERSION 3.24)

# Set policy to avoid compiler ID conflicts
cmake_policy(SET CMP0069 NEW)

# Project settings
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
        "CUDA architectures (for example: native, 86, 89, or 120)")
endif()

# HIP/ROCm support option
option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(USE_HIP)
    # Pass -DCMAKE_HIP_ARCHITECTURES=gfx90a (etc.) to override the default, which CMake
    # takes from the GPUs present on the build host.
    project(gpu_ipc LANGUAGES CXX HIP)
    if(WIN32)
        # CMake 3.29+ passes -fuse-ld=lld-link, which the -fgpu-rdc HIP link rejects
        set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld")
    endif()
else()
    # Initialize project with C++ and CUDA languages
    project(gpu_ipc LANGUAGES CXX CUDA)
endif()

# =============================================================================
# Language Standards Configuration
# =============================================================================

# Set C++17 standard with strict compliance
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_HIP)
    # Set HIP C++17 standard
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
else()
    # Set CUDA 17 standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# =============================================================================
# Dependency Management
# =============================================================================

if(USE_HIP)
    # hip::host carries the ROCm include directories and __HIP_PLATFORM_AMD__, so the
    # plain C++ sources see the same runtime declarations as the HIP sources without
    # anything naming an install prefix.
    find_package(hip REQUIRED)
else()
    # Find CUDA Toolkit with required components
    find_package(CUDAToolkit REQUIRED)
endif()

# Find third-party dependencies
find_package(Eigen3 REQUIRED)    # Linear algebra library
find_package(GLEW REQUIRED)      # OpenGL Extension Wrangler
find_package(GLUT REQUIRED)      # OpenGL Utility Toolkit
find_package(OpenGL REQUIRED)    # Open Graphics Library

# =============================================================================
# Target Configuration
# =============================================================================

# Create main executable target
add_executable(gipc)

if(USE_HIP)
    # Set HIP-specific target properties
    # Enable relocatable device code for cross-TU device symbol linking
    set_target_properties(gipc PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        HIP_SEPARABLE_COMPILATION ON
    )
else()
    # Set CUDA-specific target properties
    set_target_properties(gipc PROPERTIES
        CUDA_EXTENSIONS OFF                    # Disable CUDA compiler extensions
        CUDA_SEPARABLE_COMPILATION ON          # Enable separate compilation
        CUDA_RESOLVE_DEVICE_SYMBOLS ON         # Resolve device symbols at link time
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} # Target CUDA architecture
    )
endif()

# =============================================================================
# Source File Management
# =============================================================================

# Recursively collect all CUDA and C++ source files from GPU_IPC directory
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS 
    "GPU_IPC/*.cu"     # CUDA source files
    "GPU_IPC/*.cpp"    # C++ source files
)

# Recursively collect all header files from GPU_IPC directory
file(GLOB_RECURSE GIPC_HEADER CONFIGURE_DEPENDS 
    "GPU_IPC/*.h"      # C header files
    "GPU_IPC/*.cuh"    # CUDA header files
    "GPU_IPC/*.hpp"    # C++ header files
    "GPU_IPC/*.inl"    # Inline files
    "GPU_IPC/*.inc"    # Include files
)

# For HIP builds, mark .cu files as HIP language
if(USE_HIP)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "GPU_IPC/*.cu")
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

# Add all collected sources to the target
target_sources(gipc PRIVATE ${GIPC_SOURCE} ${GIPC_HEADER})

# =============================================================================
# Include Directories
# =============================================================================

# Set include directories for the target
target_include_directories(gipc PRIVATE 
    "GPU_IPC/"                 # Main source directory
    ${Eigen3_INCLUDE_DIRS}     # Eigen3 include directories
)

# =============================================================================
# Compile Definitions
# =============================================================================

# Set preprocessor definitions
target_compile_definitions(gipc PRIVATE 
    GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/"    # Assets directory path
    GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/"    # Output directory path
    USE_SNK                   # Enable SNK feature
    ADAPTIVE_KAPPA           # Enable adaptive kappa
    $<$<BOOL:${USE_HIP}>:USE_HIP>  # Define USE_HIP when building for AMD
)

# =============================================================================
# Library Linking
# =============================================================================

if(USE_HIP)
    target_link_libraries(gipc PRIVATE hip::host)
else()
    # Link CUDA libraries
    target_link_libraries(gipc PRIVATE
        CUDA::cudart      # CUDA runtime library
        CUDA::cublas      # CUDA BLAS library
        CUDA::cusolver    # CUDA solver library
        CUDA::cusparse    # CUDA sparse matrix library
        CUDA::cufft       # CUDA FFT library
    )
endif()

# Link graphics and math libraries
target_link_libraries(gipc PRIVATE
    GLUT::GLUT        # GLUT library
    GLEW::GLEW        # GLEW library
    OpenGL::GLU    # OpenGL library
    Eigen3::Eigen     # Eigen3 library
)

# =============================================================================
# Compiler Options
# =============================================================================

if(USE_HIP)
    # clang's -ffast-math is far broader than nvcc's --use_fast_math: it also implies
    # finite-math-only, reassociation and reciprocal math for double precision, which the
    # continuous-collision and intersection predicates here cannot tolerate. -ffp-contract
    # is pinned to expression scope for the same reason: clang defaults it to "fast" when
    # compiling HIP and then contracts across expressions, which nvcc never does.
    target_compile_options(gipc PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:
            -ffp-contract=on         # Contract within one expression, matching nvcc
            -fgpu-rdc                # Enable relocatable device code
        >
    )
    # Linker flags for device code linking
    target_link_options(gipc PRIVATE -fgpu-rdc)
else()
    # Set CUDA-specific compiler options
    target_compile_options(gipc PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:  # Apply only to CUDA files
            --use_fast_math          # Use fast math operations
            -lineinfo                # Generate line information for profiling
            --ptxas-options=-allow-expensive-optimizations=true,-v  # PTX assembler options
            --extended-lambda        # Enable extended lambda support
            --expt-relaxed-constexpr # Enable relaxed constexpr support
            --default-stream=per-thread  # Use per-thread default stream
        >
    )
endif()

# CUDA 13.x CCCL headers require MSVC's standard-conforming preprocessor.
# Pass /Zc:preprocessor both to cl.exe (C++ files) and to nvcc's host
# compiler pass (CUDA files) to avoid fatal error C1189.
if(MSVC)
    target_compile_options(gipc PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# =============================================================================
# IDE Support and Organization
# =============================================================================

# Organize source files in IDE projects
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_HEADER})

# =============================================================================
# Configuration Summary
# =============================================================================

# Display configuration information
message(STATUS "Project: gpu_ipc")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(USE_HIP)
    message(STATUS "Backend: HIP/ROCm")
    message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
    message(STATUS "HIP standard: ${CMAKE_HIP_STANDARD}")
else()
    message(STATUS "Backend: CUDA")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
endif()
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
list(LENGTH GIPC_SOURCE GIPC_SOURCE_COUNT)
list(LENGTH GIPC_HEADER GIPC_HEADER_COUNT)
message(STATUS "Source files: ${GIPC_SOURCE_COUNT}")
message(STATUS "Header files: ${GIPC_HEADER_COUNT}")
