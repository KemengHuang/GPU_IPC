#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace gpuipc
{
// Keep all CUB kernel instantiations in one CUDA translation unit. This avoids
// duplicate registrations with separable compilation and compute-sanitizer.
struct DeviceScan
{
    void exclusive_sum(const int* input, int* output, int count) const;
    void exclusive_sum(const uint32_t* input, uint32_t* output, int count) const;
};

struct DeviceRadixSort
{
    void sort_pairs(const uint64_t* keys_in,
                    uint64_t*       keys_out,
                    const uint32_t* values_in,
                    uint32_t*       values_out,
                    int             count) const;
};

struct DeviceReduce
{
    double  sum(const double* input, int count) const;
    double  max(const double* input, int count) const;
    double2 component_max(const double2* input, int count) const;
};
}  // namespace gpuipc
