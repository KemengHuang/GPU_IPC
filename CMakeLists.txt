cmake_minimum_required(VERSION 3.18)    
# support relative path, cuda_std_17, archive manipulation
#set(CMAKE_CUDA_ARCHITECTURES compute_SM_86)

# project meta
# ============
# settings
set(CUDA_ARCHITECTURES 86)
set(CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
endif()

# project
project(gpu_ipc LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED COMPONENTS cufft REQUIRED)

if(UNIX)
    if(CMAKE_BUILD_TYPE EQUAL "Release")
        set(CMAKE_CXX_FLAGS "-O3")
    endif()
endif()


# find packages
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)


add_executable(gipc)
target_compile_features(gipc PRIVATE cuda_std_17)

target_include_directories(gipc PRIVATE "GPU_IPC/")

file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS 
    "GPU_IPC/*.cu" "GPU_IPC/*.cpp")
file(GLOB_RECURSE GIPC_HANDER CONFIGURE_DEPENDS 
    "GPU_IPC/*.h" "GPU_IPC/*.cuh" "GPU_IPC/*.hpp" "GPU_IPC/*.inl" "GPU_IPC/*.inc")
target_sources(gipc PRIVATE ${GIPC_SOURCE} ${GIPC_HANDER})

# add assets and output directory, that we can easily access them in the code
target_compile_definitions(gipc PRIVATE GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/")
target_compile_definitions(gipc PRIVATE GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/")

target_compile_definitions(gipc PRIVATE USE_SNK SVD_DOUBLE ADAPTIVE_KAPPA)

# link cuda libraries
target_link_libraries(gipc PRIVATE cusparse cublas cusolver cudart_static)
target_link_libraries(gipc PRIVATE GLUT::GLUT GLEW::GLEW OpenGL::GLU Eigen3::Eigen)

target_compile_features(gipc PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(gipc PUBLIC CUDA::cuda_driver CUDA::cudart)

set_target_properties(gipc PROPERTIES
    CUDA_EXTENSIONS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 60)

target_compile_options(gipc
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: 
        ${CMAKE_CUDA_FLAGS} 
        --use_fast_math 
        -lineinfo 
        --ptxas-options=-allow-expensive-optimizations=true 
        -Xcudafe 
        "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
    >
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: 
        --extended-lambda 
        --expt-relaxed-constexpr 
        --default-stream=per-thread
    >
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HANDER})