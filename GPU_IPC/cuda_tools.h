//
// cuda_tool.h
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang and Jiming Ruan. All rights reserved.
//

#pragma once
#ifndef _CUDA_TOOLS_H
#define _CUDA_TOOLS_H

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#define CUDA_SAFE_CALL(err)     cuda_safe_call_(err, __FILE__, __LINE__)
const static int default_threads = 256;
//#define CUDA_KERNEL_CHECK(err)  cuda_kernel_check_(err, __FILE__, __LINE__)

inline void cuda_safe_call_(cudaError err, const char* file_name, const int num_line)
{
    if (cudaSuccess != err)
    {
        std::cerr << file_name << "[" << num_line << "]: "
            << "CUDA Running API error[" << (int)err << "]: "
            << cudaGetErrorString(err) << std::endl;

        std::abort();
    }
}
#endif
