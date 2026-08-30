#pragma once

#if defined(BMOE_HAVE_CUDA)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

namespace bmoe {

struct CudaDeviceTopology {
    int device_id = 0;
    int sm_count = 0;
    int max_threads_per_sm = 0;
    int max_threads_per_block = 0;
    int warp_size = 32;
    int compute_major = 0;
    int compute_minor = 0;
    size_t total_global_mem = 0;
    int memory_bus_width = 0;
    int memory_clock_rate_khz = 0;

    static const CudaDeviceTopology & current(int dev = 0) {
        static CudaDeviceTopology cached[8];
        static bool initialized[8] = {false};

        if (dev < 0 || dev >= 8) dev = 0;
        if (!initialized[dev]) {
            CudaDeviceTopology & t = cached[dev];
            t.device_id = dev;

            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
                t.sm_count = prop.multiProcessorCount;
                t.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
                t.max_threads_per_block = prop.maxThreadsPerBlock;
                t.warp_size = prop.warpSize;
                t.compute_major = prop.major;
                t.compute_minor = prop.minor;
                t.total_global_mem = prop.totalGlobalMem;
                t.memory_bus_width = prop.memoryBusWidth;
                cudaDeviceGetAttribute(&t.memory_clock_rate_khz, cudaDevAttrMemoryClockRate, dev);
            } else {

                // Fallback for RTX 5070 Ti (Blackwell sm_120)
                t.sm_count = 70;
                t.max_threads_per_sm = 1536;
                t.max_threads_per_block = 1024;
                t.compute_major = 12;
                t.compute_minor = 0;
            }
            initialized[dev] = true;
        }
        return cached[dev];
    }

    bool is_blackwell() const {
        return compute_major >= 12;
    }

    // Adaptive grid sizing: saturates SMs without excessive wave quantization tail
    int adaptive_grid_size(int block_size, int work_items) const {
        if (work_items <= 0) return 1;
        const int blocks_needed = (work_items + block_size - 1) / block_size;
        const int target_occupancy_blocks = sm_count * (max_threads_per_sm / block_size);
        return (std::max)(1, (std::min)(blocks_needed, target_occupancy_blocks * 2));
    }
};

inline int cuda_device_sm_count(int dev = 0) {
    return CudaDeviceTopology::current(dev).sm_count;
}

} // namespace bmoe
#endif
