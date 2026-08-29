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

# Initialize project with C++ and CUDA languages
project(gpu_ipc LANGUAGES CXX CUDA)
include(CTest)

# =============================================================================
# Language Standards Configuration
# =============================================================================

# Set C++17 standard with strict compliance
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA 17 standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# =============================================================================
# Dependency Management
# =============================================================================

# Find CUDA Toolkit with required components
find_package(CUDAToolkit REQUIRED)

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

# Set CUDA-specific target properties
set_target_properties(gipc PROPERTIES
    CUDA_EXTENSIONS OFF                    # Disable CUDA compiler extensions
    CUDA_SEPARABLE_COMPILATION ON          # Enable separate compilation
    CUDA_RESOLVE_DEVICE_SYMBOLS ON         # Resolve device symbols at link time
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} # Target CUDA architecture
)

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
)

# =============================================================================
# Library Linking
# =============================================================================

# Link CUDA libraries
target_link_libraries(gipc PRIVATE
    CUDA::cudart      # CUDA runtime library
    CUDA::cublas      # CUDA BLAS library
    CUDA::cusolver    # CUDA solver library
    CUDA::cusparse    # CUDA sparse matrix library
    CUDA::cufft       # CUDA FFT library
)

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

if(BUILD_TESTING)
    add_executable(device_buffer_growth_tests
        tests/device_buffer_growth_tests.cu)
    target_include_directories(device_buffer_growth_tests PRIVATE "GPU_IPC/")
    target_compile_features(device_buffer_growth_tests PRIVATE cxx_std_17 cuda_std_17)
    target_link_libraries(device_buffer_growth_tests PRIVATE CUDA::cudart)
    set_target_properties(device_buffer_growth_tests PROPERTIES
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(device_buffer_growth_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_test(NAME device-buffer-required-headroom
             COMMAND device_buffer_growth_tests)
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
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
list(LENGTH GIPC_SOURCE GIPC_SOURCE_COUNT)
list(LENGTH GIPC_HEADER GIPC_HEADER_COUNT)
message(STATUS "Source files: ${GIPC_SOURCE_COUNT}")
message(STATUS "Header files: ${GIPC_HEADER_COUNT}")
