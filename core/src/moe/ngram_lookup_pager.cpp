#include "ngram_lookup_pager.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(BMOE_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace bmoe {

NgramLookupPager::NgramLookupPager(const Config & cfg) : cfg_(cfg) {}

NgramLookupPager::~NgramLookupPager() {
    shutdown();
}

bool NgramLookupPager::init(void * d_gpu_staging_buf) {
    if (inited_) return true;
    d_gpu_staging_buf_ = d_gpu_staging_buf;

#if defined(BMOE_HAVE_CUDA)
    if (!d_gpu_staging_buf_) {
        cudaError_t err = cudaMalloc(&d_gpu_staging_buf_, bounce_buf_capacity_);
        if (err != cudaSuccess) return false;
    }
    cudaError_t herr = cudaHostAlloc(&h_pinned_bounce_buf_, bounce_buf_capacity_,
                                     cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (herr != cudaSuccess) {
        h_pinned_bounce_buf_ = std::malloc(bounce_buf_capacity_);
    }
#else
    h_pinned_bounce_buf_ = std::malloc(bounce_buf_capacity_);
#endif

    inited_ = true;
    return true;
}

void NgramLookupPager::shutdown() {
    if (!inited_) return;
#if defined(BMOE_HAVE_CUDA)
    if (h_pinned_bounce_buf_) {
        cudaFreeHost(h_pinned_bounce_buf_);
        h_pinned_bounce_buf_ = nullptr;
    }
#else
    if (h_pinned_bounce_buf_) {
        std::free(h_pinned_bounce_buf_);
        h_pinned_bounce_buf_ = nullptr;
    }
#endif
    inited_ = false;
}

bool NgramLookupPager::prefetch_hashes_async(const uint64_t * hash_ids, int n_hashes, void * cuda_stream) {
    if (!inited_ || !hash_ids || n_hashes <= 0) return false;

    size_t total_bytes = (size_t) n_hashes * cfg_.row_bytes;
    if (total_bytes > bounce_buf_capacity_) {
        total_bytes = bounce_buf_capacity_;
        n_hashes = (int) (bounce_buf_capacity_ / cfg_.row_bytes);
    }

    // Zero-fill / synthetic fetch if mock file
    if (h_pinned_bounce_buf_) {
        std::memset(h_pinned_bounce_buf_, 0, total_bytes);
    }
    active_bytes_.store(total_bytes, std::memory_order_relaxed);

#if defined(BMOE_HAVE_CUDA)
    if (d_gpu_staging_buf_ && h_pinned_bounce_buf_) {
        cudaStream_t stream = cuda_stream ? (cudaStream_t) cuda_stream : (cudaStream_t) 0;
        cudaError_t err = cudaMemcpyAsync(d_gpu_staging_buf_, h_pinned_bounce_buf_, total_bytes,
                                          cudaMemcpyHostToDevice, stream);
        return err == cudaSuccess;
    }
#else
    (void) cuda_stream;
#endif

    return true;
}

bool NgramLookupPager::wait_for_staging(void * cuda_stream) {
    if (!inited_) return false;
#if defined(BMOE_HAVE_CUDA)
    if (cuda_stream) {
        cudaError_t err = cudaStreamSynchronize((cudaStream_t) cuda_stream);
        return err == cudaSuccess;
    }
#else
    (void) cuda_stream;
#endif
    return true;
}

} // namespace bmoe
