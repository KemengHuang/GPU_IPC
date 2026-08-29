#include "device_buffer.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
    if(!condition)
    {
        std::fprintf(stderr, "device-buffer growth test failed: %s\n", message);
        std::abort();
    }
}

__global__ void fill_sequence(int* values, int count, int base)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count)
        values[i] = base + i;
}
}  // namespace

int main()
{
    GPUIPC_CHECK_CUDA(cudaSetDevice(0));

    gpuipc::DeviceBuffer<int> values;
    values.resize_discard(100);
    require(values.capacity() >= 150,
            "initial dynamic allocation lacks required-based headroom");

    values.resize_discard(1'000);  // deliberately exceeds old capacity by far
    require(values.capacity() >= 1'500,
            "large requirement jump was allocated without 50% headroom");
    int* const allocation = values.data();
    const std::size_t retained_capacity = values.capacity();

    values.resize_discard(1'001);
    require(values.data() == allocation && values.capacity() == retained_capacity,
            "follow-up growth reallocated despite retained headroom");
    values.resize_discard(10);
    require(values.capacity() == retained_capacity,
            "logical shrink unexpectedly released capacity");

    gpuipc::DeviceBuffer<int> preserved;
    preserved.resize(4);
    fill_sequence<<<1, 32>>>(preserved.data(), 4, 90);
    GPUIPC_CHECK_CUDA(cudaGetLastError());
    preserved.resize_preserve(1'000);
    require(preserved.capacity() >= 1'500,
            "preserving growth lacks required-based headroom");
    std::vector<int> host;
    preserved.copy_to(host);
    for(int i = 0; i < 4; ++i)
        require(host[i] == 90 + i, "preserving growth lost live values");

    gpuipc::DeviceBuffer<int> reserved;
    reserved.reserve_amortized(2'000);
    require(reserved.size() == 0 && reserved.capacity() >= 3'000,
            "amortized reserve lacks required-based headroom");

    std::puts("device-buffer required-headroom tests passed");
    return 0;
}
