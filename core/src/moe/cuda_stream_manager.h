#pragma once

#include <cstddef>
#include <cstdint>

#if defined(BMOE_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace bmoe {

// Manages dual CUDA streams (Stream 0 compute, Stream 1 transfer) and hardware synchronization events.
class CudaStreamManager {
public:
    CudaStreamManager() = default;
    ~CudaStreamManager() { cleanup(); }

    CudaStreamManager(const CudaStreamManager &) = delete;
    CudaStreamManager & operator=(const CudaStreamManager &) = delete;

    bool init(int device_id = 0) {
#if defined(BMOE_HAVE_CUDA)
        device_id_ = device_id;
        cudaError_t err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) return false;

        // Stream 0: Compute stream (Non-blocking with respect to default stream)
        err = cudaStreamCreateWithFlags(&stream_compute_, cudaStreamNonBlocking);
        if (err != cudaSuccess) return false;

        // Stream 1: Transfer stream (Non-blocking for PCIe HtoD pushes)
        err = cudaStreamCreateWithFlags(&stream_transfer_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            cleanup();
            return false;
        }

        // Timing profiling events
        cudaEventCreate(&event_transfer_start_);
        cudaEventCreate(&event_transfer_stop_);
        cudaEventCreate(&event_compute_start_);
        cudaEventCreate(&event_compute_stop_);

        inited_ = true;
        return true;
#else
        return false;
#endif
    }

    void cleanup() {
#if defined(BMOE_HAVE_CUDA)
        if (inited_) {
            if (stream_compute_) cudaStreamDestroy(stream_compute_);
            if (stream_transfer_) cudaStreamDestroy(stream_transfer_);
            if (event_transfer_start_) cudaEventDestroy(event_transfer_start_);
            if (event_transfer_stop_) cudaEventDestroy(event_transfer_stop_);
            if (event_compute_start_) cudaEventDestroy(event_compute_start_);
            if (event_compute_stop_) cudaEventDestroy(event_compute_stop_);
            stream_compute_ = nullptr;
            stream_transfer_ = nullptr;
            inited_ = false;
        }
#endif
    }

#if defined(BMOE_HAVE_CUDA)
    cudaStream_t compute_stream() const { return stream_compute_; }
    cudaStream_t transfer_stream() const { return stream_transfer_; }
#endif

    bool is_inited() const { return inited_; }
    int device_id() const { return device_id_; }

private:
    bool inited_ = false;
    int device_id_ = 0;

#if defined(BMOE_HAVE_CUDA)
    cudaStream_t stream_compute_ = nullptr;  // Stream 0
    cudaStream_t stream_transfer_ = nullptr; // Stream 1

    cudaEvent_t event_transfer_start_ = nullptr;
    cudaEvent_t event_transfer_stop_ = nullptr;
    cudaEvent_t event_compute_start_ = nullptr;
    cudaEvent_t event_compute_stop_ = nullptr;
#endif
};

} // namespace bmoe
