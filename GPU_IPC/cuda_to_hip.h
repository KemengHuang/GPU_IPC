//
// cuda_to_hip.h
// CUDA-to-HIP compatibility header for GPU_IPC ROCm port
//
// This header provides CUDA-to-HIP symbol mappings when building with HIP.
// On CUDA, it simply includes the standard CUDA runtime header.
//

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// Memory management
#define cudaMalloc                hipMalloc
#define cudaFree                  hipFree
#define cudaMemcpy                hipMemcpy
#define cudaMemcpyAsync           hipMemcpyAsync
#define cudaMemset                hipMemset
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice

// Device management
#define cudaSetDevice             hipSetDevice
#define cudaDeviceSynchronize     hipDeviceSynchronize
#define cudaDeviceReset           hipDeviceReset

// Error handling
#define cudaError_t               hipError_t
#define cudaError                 hipError_t
#define cudaSuccess               hipSuccess
#define cudaGetErrorString        hipGetErrorString
#define cudaGetLastError          hipGetLastError

// Events
#define cudaEvent_t               hipEvent_t
#define cudaEventCreate           hipEventCreate
#define cudaEventRecord           hipEventRecord
#define cudaEventDestroy          hipEventDestroy
#define cudaEventSynchronize      hipEventSynchronize
#define cudaEventElapsedTime      hipEventElapsedTime

// Streams
#define cudaStream_t              hipStream_t
#define cudaStreamCreate          hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy         hipStreamDestroy
#define cudaStreamSynchronize     hipStreamSynchronize
#define cudaStreamNonBlocking     hipStreamNonBlocking

// Warp intrinsics: see GPU_IPC/details/device_utils.inl for the pinned-width-32
// translation of CUDA's *_sync forms. That translation is implemented directly there
// rather than as macros here, because HIP's own headers (hip_bf16.h in particular)
// declare genuine overloaded functions named __shfl_sync and friends, and a
// function-like macro of the same name mis-parses those declarations wherever they are
// transitively included (e.g. via <thrust/...>).

#else

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#endif
