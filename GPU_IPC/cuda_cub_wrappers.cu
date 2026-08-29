#include "cuda_cub_wrappers.h"

#include "device_buffer.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace gpuipc
{
namespace
{
class PersistentCubWorkspace
{
    void*       m_data     = nullptr;
    std::size_t m_capacity = 0;

  public:
    PersistentCubWorkspace() = default;
    PersistentCubWorkspace(const PersistentCubWorkspace&)            = delete;
    PersistentCubWorkspace& operator=(const PersistentCubWorkspace&) = delete;

    ~PersistentCubWorkspace()
    {
        if(m_data)
            cudaFree(m_data);
    }

    void* get(std::size_t bytes)
    {
        if(bytes <= m_capacity)
            return m_data;

        const std::size_t growth = std::max<std::size_t>(bytes / 2, 1);
        if(bytes > std::numeric_limits<std::size_t>::max() - growth)
        {
            std::cerr << "CUB redundant workspace overflow: required=" << bytes
                      << std::endl;
            std::abort();
        }
        const std::size_t new_capacity = bytes + growth;

        void* replacement = nullptr;
        GPUIPC_CHECK_CUDA(cudaMalloc(&replacement, new_capacity));
        if(m_data)
            GPUIPC_CHECK_CUDA(cudaFree(m_data));
        m_data     = replacement;
        m_capacity = new_capacity;
        return m_data;
    }
};

void* cub_workspace(std::size_t bytes)
{
    static thread_local PersistentCubWorkspace workspace;
    return workspace.get(bytes);
}

DeviceBuffer<double>& scalar_output()
{
    static thread_local DeviceBuffer<double> output(1);
    return output;
}

DeviceBuffer<double2>& pair_output()
{
    static thread_local DeviceBuffer<double2> output(1);
    return output;
}

struct ComponentMax
{
    __host__ __device__ double2 operator()(double2 lhs, double2 rhs) const
    {
        return make_double2(lhs.x > rhs.x ? lhs.x : rhs.x,
                            lhs.y > rhs.y ? lhs.y : rhs.y);
    }
};

template <typename T>
void exclusive_sum_impl(const T* input, T* output, int count)
{
    if(count <= 0)
        return;
    std::size_t bytes = 0;
    GPUIPC_CHECK_CUDA(
        cub::DeviceScan::ExclusiveSum(nullptr, bytes, input, output, count));
    GPUIPC_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
        cub_workspace(bytes), bytes, input, output, count));
}
}  // namespace

void DeviceScan::exclusive_sum(const int* input, int* output, int count) const
{
    exclusive_sum_impl(input, output, count);
}

void DeviceScan::exclusive_sum(const uint32_t* input,
                               uint32_t*       output,
                               int             count) const
{
    exclusive_sum_impl(input, output, count);
}

void DeviceRadixSort::sort_pairs(const uint64_t* keys_in,
                                 uint64_t*       keys_out,
                                 const uint32_t* values_in,
                                 uint32_t*       values_out,
                                 int             count) const
{
    if(count <= 0)
        return;
    std::size_t bytes = 0;
    GPUIPC_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, keys_in, keys_out, values_in, values_out, count));
    GPUIPC_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(cub_workspace(bytes),
                                                       bytes,
                                                       keys_in,
                                                       keys_out,
                                                       values_in,
                                                       values_out,
                                                       count));
}

double DeviceReduce::sum(const double* input, int count) const
{
    if(count <= 0)
        return 0.0;
    auto& output = scalar_output();
    std::size_t bytes = 0;
    GPUIPC_CHECK_CUDA(cub::DeviceReduce::Sum(
        nullptr, bytes, input, output.data(), count));
    GPUIPC_CHECK_CUDA(cub::DeviceReduce::Sum(
        cub_workspace(bytes), bytes, input, output.data(), count));
    double result = 0.0;
    GPUIPC_CHECK_CUDA(cudaMemcpy(
        &result, output.data(), sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}

double DeviceReduce::max(const double* input, int count) const
{
    if(count <= 0)
        return std::numeric_limits<double>::lowest();
    auto& output = scalar_output();
    std::size_t bytes = 0;
    GPUIPC_CHECK_CUDA(cub::DeviceReduce::Max(
        nullptr, bytes, input, output.data(), count));
    GPUIPC_CHECK_CUDA(cub::DeviceReduce::Max(
        cub_workspace(bytes), bytes, input, output.data(), count));
    double result = 0.0;
    GPUIPC_CHECK_CUDA(cudaMemcpy(
        &result, output.data(), sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}

double2 DeviceReduce::component_max(const double2* input, int count) const
{
    const double lowest = std::numeric_limits<double>::lowest();
    if(count <= 0)
        return make_double2(lowest, lowest);
    auto& output = pair_output();
    std::size_t bytes = 0;
    const double2 initial = make_double2(lowest, lowest);
    GPUIPC_CHECK_CUDA(cub::DeviceReduce::Reduce(
        nullptr, bytes, input, output.data(), count, ComponentMax{}, initial));
    GPUIPC_CHECK_CUDA(cub::DeviceReduce::Reduce(cub_workspace(bytes),
                                                bytes,
                                                input,
                                                output.data(),
                                                count,
                                                ComponentMax{},
                                                initial));
    double2 result{};
    GPUIPC_CHECK_CUDA(cudaMemcpy(
        &result, output.data(), sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}
}  // namespace gpuipc
