#pragma once

#include "bmoe/config.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <atomic>

namespace bmoe {

// NgramLookupPager: Asynchronously pages active rows of large N-gram embedding tables
// (e.g. 51B table at Layer 2) from NVMe storage using Direct I/O directly into GPU staging buffers.
class NgramLookupPager {
public:
    struct Config {
        std::string table_path;
        size_t vocab_size = 152064;
        size_t embedding_dim = 2048;
        size_t row_bytes = 2048 * sizeof(uint16_t); // ~4 KB per row
        bool use_direct_io = true;
    };

    explicit NgramLookupPager(const Config & cfg = Config());
    ~NgramLookupPager();

    NgramLookupPager(const NgramLookupPager &) = delete;
    NgramLookupPager & operator=(const NgramLookupPager &) = delete;

    bool init(void * d_gpu_staging_buf = nullptr);
    void shutdown();

    // Enqueue an asynchronous lookup for a batch of hash IDs during early layer (0/1) compute
    bool prefetch_hashes_async(const uint64_t * hash_ids, int n_hashes, void * cuda_stream = nullptr);

    // Wait for the requested hash rows to be staged in the GPU buffer before Layer 2 compute
    bool wait_for_staging(void * cuda_stream = nullptr);

    void * d_buffer() const { return d_gpu_staging_buf_; }
    size_t active_bytes() const { return active_bytes_.load(std::memory_order_relaxed); }

private:
    Config cfg_;
    bool inited_ = false;
    void * d_gpu_staging_buf_ = nullptr;
    void * h_pinned_bounce_buf_ = nullptr;
    size_t bounce_buf_capacity_ = 64 * 1024; // 64 KB staging bounce buffer
    std::atomic<size_t> active_bytes_{0};
    void * file_handle_ = nullptr;
};

} // namespace bmoe
