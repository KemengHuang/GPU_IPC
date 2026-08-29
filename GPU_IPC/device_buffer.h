#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace gpuipc
{
inline void check_cuda(cudaError_t error, const char* expression, const char* file, int line)
{
    if(error == cudaSuccess)
        return;
    std::cerr << file << '[' << line << "]: CUDA error in " << expression << ": "
              << cudaGetErrorString(error) << std::endl;
    std::abort();
}

#define GPUIPC_CHECK_CUDA(expr) ::gpuipc::check_cuda((expr), #expr, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer
{
    static_assert(!std::is_reference_v<T>);

    T*          m_data     = nullptr;
    std::size_t m_size     = 0;
    std::size_t m_capacity = 0;

    // Every real growth reserves 50% beyond the latest requirement. Basing
    // headroom on `required` also covers sudden jumps far above old capacity.
    static std::size_t grown_capacity(std::size_t current, std::size_t required)
    {
        if(required <= current)
            return current;
        const std::size_t growth = std::max<std::size_t>(required / 2, 1);
        if(required > std::numeric_limits<std::size_t>::max() - growth)
        {
            std::cerr << "DeviceBuffer redundant-capacity overflow: required="
                      << required << std::endl;
            std::abort();
        }
        return required + growth;
    }

    static T* allocate(std::size_t count)
    {
        if(count == 0)
            return nullptr;
        if(count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            std::cerr << "DeviceBuffer allocation overflow: count=" << count
                      << ", element_size=" << sizeof(T) << std::endl;
            std::abort();
        }
        T* pointer = nullptr;
        const std::size_t bytes = count * sizeof(T);
        cudaError_t error = cudaMalloc(&pointer, bytes);
        if(error != cudaSuccess)
        {
            std::cerr << "DeviceBuffer allocation failed: count=" << count
                      << ", bytes=" << bytes << ", CUDA error="
                      << cudaGetErrorString(error) << std::endl;
            std::abort();
        }
        return pointer;
    }

    void reallocate(std::size_t capacity, bool preserve)
    {
        T* replacement = allocate(capacity);
        if(preserve && replacement && m_data && m_size > 0)
        {
            const std::size_t copy_count = std::min(m_size, capacity);
            GPUIPC_CHECK_CUDA(cudaMemcpy(replacement,
                                         m_data,
                                         copy_count * sizeof(T),
                                         cudaMemcpyDeviceToDevice));
        }
        if(m_data)
            GPUIPC_CHECK_CUDA(cudaFree(m_data));
        m_data     = replacement;
        m_capacity = capacity;
        if(m_size > capacity)
            m_size = capacity;
    }

  public:
    using value_type = T;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t size) { resize(size); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : m_data(std::exchange(other.m_data, nullptr)),
          m_size(std::exchange(other.m_size, 0)),
          m_capacity(std::exchange(other.m_capacity, 0))
    {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if(this == &other)
            return *this;
        if(m_data)
            cudaFree(m_data);
        m_data     = std::exchange(other.m_data, nullptr);
        m_size     = std::exchange(other.m_size, 0);
        m_capacity = std::exchange(other.m_capacity, 0);
        return *this;
    }

    ~DeviceBuffer()
    {
        if(m_data)
            cudaFree(m_data);
    }

    void resize(std::size_t size)
    {
        if(size > m_capacity)
            reallocate(size, true);
        m_size = size;
    }

    void resize_preserve(std::size_t size)
    {
        if(size > m_capacity)
            reallocate(grown_capacity(m_capacity, size), true);
        m_size = size;
    }

    void resize_discard(std::size_t size)
    {
        if(size > m_capacity)
            reallocate(grown_capacity(m_capacity, size), false);
        m_size = size;
    }

    void reserve(std::size_t capacity)
    {
        if(capacity > m_capacity)
            reallocate(capacity, true);
    }

    void reserve_amortized(std::size_t capacity)
    {
        if(capacity > m_capacity)
            reallocate(grown_capacity(m_capacity, capacity), true);
    }

    void clear() noexcept { m_size = 0; }

    void release()
    {
        if(m_data)
            GPUIPC_CHECK_CUDA(cudaFree(m_data));
        m_data     = nullptr;
        m_size     = 0;
        m_capacity = 0;
    }

    void reset_zero()
    {
        if(m_size > 0)
            GPUIPC_CHECK_CUDA(cudaMemset(m_data, 0, m_size * sizeof(T)));
    }

    void copy_from(const std::vector<T>& host)
    {
        resize(host.size());
        if(!host.empty())
            GPUIPC_CHECK_CUDA(cudaMemcpy(
                m_data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to(std::vector<T>& host) const
    {
        host.resize(m_size);
        if(m_size > 0)
            GPUIPC_CHECK_CUDA(cudaMemcpy(
                host.data(), m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }
    std::size_t size() const noexcept { return m_size; }
    std::size_t capacity() const noexcept { return m_capacity; }

    operator T*() noexcept { return m_data; }
    operator const T*() const noexcept { return m_data; }
};
}  // namespace gpuipc
