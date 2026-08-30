#pragma once

#include "cuda_stream_manager.h"
#include "vram_ring_buffer.h"
#include "cuda_occupancy.h"
#include "arc_cache.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <algorithm>

#include "ggml.h"
#include "ggml-backend.h"

namespace bmoe {

// Struct describing an expert slice payload to stage to VRAM
struct ExpertStagingItem {
    const void * h_src = nullptr; // Host memory pointer (pinned host RAM)
    size_t offset_in_slot = 0;   // Offset in the VRAM slot
    size_t size = 0;             // Number of bytes to transfer
};

// Orchestrates Layer N+1 asynchronous staging, VRAM ring slot binding, and pointer swapping.
class CudaExpertStager {
public:
    CudaExpertStager() = default;
    ~CudaExpertStager() { cleanup(); }

    CudaExpertStager(const CudaExpertStager &) = delete;
    CudaExpertStager & operator=(const CudaExpertStager &) = delete;

    bool init(size_t vram_arena_bytes = 2048ull * 1024ull * 1024ull, int num_slots = 4, int device_id = 0) {
#if defined(BMOE_HAVE_CUDA)
        if (!stream_mgr_.init(device_id)) return false;
        if (!ring_buf_.init(vram_arena_bytes, num_slots)) {
            stream_mgr_.cleanup();
            return false;
        }
        slot_capacity_ = vram_arena_bytes / (size_t) num_slots;

        // Allocate a page-locked Write-Combined host bounce buffer for coalesced single-DMA transfers
        if (slot_capacity_ > 0) {
            cudaError_t err = cudaHostAlloc(&pinned_hbuf_, slot_capacity_,
                                            cudaHostAllocWriteCombined | cudaHostAllocMapped);
            if (err == cudaSuccess) {
                pinned_hbuf_cap_ = slot_capacity_;
            } else {
                pinned_hbuf_ = nullptr;
                pinned_hbuf_cap_ = 0;
            }
        }

        inited_ = true;
        return true;
#else
        (void) vram_arena_bytes; (void) num_slots; (void) device_id;
        return false;
#endif
    }

    // Allocate dedicated VRAM buffers for the first N pinned layers (gracefully capped to free VRAM)
    bool init_pinned_layers(int n_pinned, const std::vector<size_t> & layer_sizes) {
#if defined(BMOE_HAVE_CUDA)
        if (n_pinned <= 0 || layer_sizes.empty()) return true;
        cleanup_pinned_layers();

        const int target_pinned = (std::min)(n_pinned, (int) layer_sizes.size());
        d_pinned_layers_.resize(target_pinned, nullptr);
        pinned_layer_sizes_.resize(target_pinned, 0);

        int allocated = 0;
        size_t total_pinned_bytes = 0;

        for (int i = 0; i < target_pinned; ++i) {
            size_t sz = layer_sizes[i];
            if (sz == 0) continue;

            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                // Keep 512 MiB safety headroom
                if (free_bytes < sz + 512ull * 1024ull * 1024ull) {
                    std::fprintf(stderr, "bmoe: VRAM capacity reached (%zu MiB free); capping pinned layers at %d\n",
                                 free_bytes / (1024 * 1024), allocated);
                    break;
                }
            }

            void * d_ptr = nullptr;
            cudaError_t err = cudaMalloc(&d_ptr, sz);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "bmoe: stopped pinning at layer %d (%zu MiB required, error %d)\n",
                             i, sz / (1024 * 1024), (int) err);
                break;
            }
            d_pinned_layers_[i] = d_ptr;
            pinned_layer_sizes_[i] = sz;
            total_pinned_bytes += sz;
            allocated++;
        }
        n_pinned_ = allocated;
        std::fprintf(stderr, "bmoe: hybrid static offload enabled — %d/%d early MoE layers pinned permanently in VRAM (%zu MiB)\n",
                     n_pinned_, target_pinned, total_pinned_bytes / (1024 * 1024));
        return true;
#else
        (void) n_pinned; (void) layer_sizes;
        return false;
#endif
    }

    void set_pinned_layers(int n_pinned) {
        n_pinned_ = n_pinned;
        if (n_pinned_ > 0) {
            std::fprintf(stderr, "bmoe: hybrid static offload enabled — %d early MoE layers pinned permanently in VRAM\n",
                         n_pinned_);
        }
    }

    bool is_layer_pinned(int layer_idx) const {
        return layer_idx >= 0 && layer_idx < n_pinned_;
    }

    void * pinned_layer_ptr(int layer_idx) const {
        if (is_layer_pinned(layer_idx) && layer_idx < (int) d_pinned_layers_.size()) {
            return d_pinned_layers_[layer_idx];
        }
        return nullptr;
    }


    bool upload_pinned_layer(int layer_idx, const void * h_src, size_t size) {
#if defined(BMOE_HAVE_CUDA)
        if (!is_layer_pinned(layer_idx) || !h_src || size == 0) return false;
        if (size > pinned_layer_sizes_[layer_idx]) return false;
        cudaError_t err = cudaMemcpy(d_pinned_layers_[layer_idx], h_src, size, cudaMemcpyHostToDevice);
        return err == cudaSuccess;
#else
        (void) layer_idx; (void) h_src; (void) size;
        return false;
#endif
    }

    void cleanup_pinned_layers() {
#if defined(BMOE_HAVE_CUDA)
        for (void * ptr : d_pinned_layers_) {
            if (ptr) cudaFree(ptr);
        }
        d_pinned_layers_.clear();
        pinned_layer_sizes_.clear();
        n_pinned_ = 0;
#endif
    }

    void cleanup() {
#if defined(BMOE_HAVE_CUDA)
        if (inited_) {
            cleanup_pinned_layers();
            if (pinned_hbuf_) {
                cudaFreeHost(pinned_hbuf_);
                pinned_hbuf_ = nullptr;
                pinned_hbuf_cap_ = 0;
            }
            ring_buf_.cleanup();
            stream_mgr_.cleanup();
            inited_ = false;
        }
#endif
    }

    // Stages a batch of expert weight items for layer_idx on Stream 1 using coalesced contiguous DMA
#if defined(BMOE_HAVE_CUDA)
    bool stage_layer_async(int layer_idx, const std::vector<ExpertStagingItem> & items) {
        if (!inited_ || items.empty()) return false;
        if (is_layer_pinned(layer_idx)) return true; // Already resident in VRAM

        VramRingBuffer::Slot * slot = ring_buf_.acquire_slot_for_transfer(layer_idx, stream_mgr_.transfer_stream());
        if (!slot) return false;

        size_t max_end = 0;
        // Use coalesced single-DMA transfer via pinned host bounce buffer when possible
        if (pinned_hbuf_ && pinned_hbuf_cap_ >= slot->capacity) {
            for (const auto & item : items) {
                size_t aligned_offset = (item.offset_in_slot + 255ull) & ~255ull;
                if (aligned_offset + item.size > slot->capacity) {
                    std::fprintf(stderr, "bmoe: expert staging item exceeds slot capacity (%zu > %zu)\n",
                                 aligned_offset + item.size, slot->capacity);
                    return false;
                }
                std::memcpy((char *) pinned_hbuf_ + aligned_offset, item.h_src, item.size);
                if (aligned_offset + item.size > max_end) {
                    max_end = aligned_offset + item.size;
                }
            }
            if (max_end > 0) {
                cudaError_t err = cudaMemcpyAsync(slot->d_ptr, pinned_hbuf_, max_end,
                                                  cudaMemcpyHostToDevice, stream_mgr_.transfer_stream());
                if (err != cudaSuccess) {
                    std::fprintf(stderr, "bmoe: coalesced cudaMemcpyAsync failed for layer %d (error %d)\n",
                                 layer_idx, (int) err);
                    return false;
                }
            }
        } else {
            // Fallback to sequential aligned transfers
            for (const auto & item : items) {
                size_t aligned_offset = (item.offset_in_slot + 255ull) & ~255ull;
                if (aligned_offset + item.size > slot->capacity) return false;
                char * d_dst = (char *) slot->d_ptr + aligned_offset;
                cudaError_t err = cudaMemcpyAsync(d_dst, item.h_src, item.size,
                                                  cudaMemcpyHostToDevice, stream_mgr_.transfer_stream());
                if (err != cudaSuccess) return false;
                if (aligned_offset + item.size > max_end) max_end = aligned_offset + item.size;
            }
        }

        ring_buf_.mark_transfer_complete(slot, max_end, stream_mgr_.transfer_stream());
        return true;
    }

    // Intercepts the execution seam: rebinds tensor->data to the active VRAM slot or pinned layer
    void * bind_layer_for_compute(int layer_idx, ggml_tensor * tensor, size_t proj_offset = 0) {
        if (!inited_ || !tensor) return nullptr;

        if (is_layer_pinned(layer_idx)) {
            // Pinned layer is permanently resident in native CUDA0 VRAM. Do not rebind.
            return tensor->data;
        }

        TensorBackup backup;
        backup.orig_data = tensor->data;
        backup.orig_buffer = tensor->buffer;

        VramRingBuffer::Slot * slot = ring_buf_.bind_slot_for_compute(layer_idx, stream_mgr_.compute_stream());
        if (!slot) return nullptr;


        // Rebind tensor data to VRAM ring buffer slot at projection offset
        tensor->data = (char *) slot->d_ptr + proj_offset;

        std::lock_guard<std::mutex> lk(swap_mtx_);
        orig_states_[tensor] = backup;
        active_slots_[layer_idx] = slot;
        return tensor->data;
    }

    // Restores original tensor data and buffer pointers after compute kernel is dispatched
    void restore_tensor(int layer_idx, ggml_tensor * tensor) {
        if (!inited_ || !tensor) return;

        std::lock_guard<std::mutex> lk(swap_mtx_);
        auto it = orig_states_.find(tensor);
        if (it != orig_states_.end()) {
            tensor->data = it->second.orig_data;
            tensor->buffer = it->second.orig_buffer;
            orig_states_.erase(it);
        }

        if (!is_layer_pinned(layer_idx)) {
            auto sit = active_slots_.find(layer_idx);
            if (sit != active_slots_.end()) {
                ring_buf_.mark_compute_complete(sit->second, stream_mgr_.compute_stream());
                active_slots_.erase(sit);
            }
        }
    }

    // MoE Hot-Expert LRU / frequency cache in spare VRAM (~1.0 to 1.5 GB)
    bool init_hot_expert_cache(size_t cache_bytes = 1536ull * 1024ull * 1024ull, size_t slice_bytes = 0) {
#if defined(BMOE_HAVE_CUDA)
        if (cache_bytes == 0 || slice_bytes == 0) return false;
        cleanup_hot_expert_cache();

        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            if (free_bytes < cache_bytes + 512ull * 1024ull * 1024ull) {
                cache_bytes = free_bytes > 768ull * 1024ull * 1024ull ? free_bytes - 512ull * 1024ull * 1024ull : 0;
            }
        }
        if (cache_bytes < slice_bytes) return false;

        void * d_ptr = nullptr;
        if (cudaMalloc(&d_ptr, cache_bytes) != cudaSuccess) return false;

        d_hot_cache_ = d_ptr;
        hot_cache_cap_ = cache_bytes;
        hot_slice_bytes_ = slice_bytes;
        const size_t max_hot_experts = cache_bytes / slice_bytes;
        hot_arc_.set_capacity(max_hot_experts);
        hot_slots_.resize(max_hot_experts);
        for (size_t i = 0; i < max_hot_experts; ++i) {
            hot_slots_[i].d_ptr = (char *) d_hot_cache_ + i * slice_bytes;
            hot_slots_[i].key = 0xFFFFFFFF;
            hot_slots_[i].in_use = false;
        }

        std::fprintf(stderr, "bmoe: MoE hot-expert VRAM cache initialized (%zu MiB, %zu slots)\n",
                     cache_bytes / (1024 * 1024), max_hot_experts);
        return true;
#else
        (void) cache_bytes; (void) slice_bytes;
        return false;
#endif
    }

    bool is_expert_hot(int layer_idx, int expert_idx) const {
        if (!d_hot_cache_ || hot_slots_.empty() || is_layer_pinned(layer_idx)) return false;
        uint32_t key = ((uint32_t) layer_idx << 16) | (uint32_t) (expert_idx & 0xFFFF);
        return hot_arc_.is_resident(key);
    }

    void * get_hot_expert_ptr(int layer_idx, int expert_idx) {
        if (!d_hot_cache_ || hot_slots_.empty() || is_layer_pinned(layer_idx)) return nullptr;
        uint32_t key = ((uint32_t) layer_idx << 16) | (uint32_t) (expert_idx & 0xFFFF);
        std::lock_guard<std::mutex> lk(hot_mtx_);
        auto it = hot_key_to_slot_.find(key);
        if (it != hot_key_to_slot_.end()) {
            return hot_slots_[it->second].d_ptr;
        }
        return nullptr;
    }

    void record_expert_access(int layer_idx, int expert_idx, const void * h_src = nullptr, size_t size = 0) {
        if (!d_hot_cache_ || hot_slots_.empty() || is_layer_pinned(layer_idx)) return;
        uint32_t key = ((uint32_t) layer_idx << 16) | (uint32_t) (expert_idx & 0xFFFF);
        auto res = hot_arc_.access(key);

        std::lock_guard<std::mutex> lk(hot_mtx_);
        // Handle evictions
        for (uint32_t evicted_key : res.evicted_resident_keys) {
            auto it = hot_key_to_slot_.find(evicted_key);
            if (it != hot_key_to_slot_.end()) {
                size_t slot_idx = it->second;
                hot_slots_[slot_idx].in_use = false;
                hot_slots_[slot_idx].key = 0xFFFFFFFF;
                hot_key_to_slot_.erase(it);
            }
        }

        // If miss and we have host data, stage hot expert into available slot
        if (!res.hit && h_src && size > 0 && size <= hot_slice_bytes_) {
            for (size_t i = 0; i < hot_slots_.size(); ++i) {
                if (!hot_slots_[i].in_use) {
                    hot_slots_[i].in_use = true;
                    hot_slots_[i].key = key;
                    hot_key_to_slot_[key] = i;
#if defined(BMOE_HAVE_CUDA)
                    cudaMemcpyAsync(hot_slots_[i].d_ptr, h_src, size, cudaMemcpyHostToDevice, stream_mgr_.transfer_stream());
#endif
                    break;
                }
            }
        }
    }

    void cleanup_hot_expert_cache() {
#if defined(BMOE_HAVE_CUDA)
        if (d_hot_cache_) {
            cudaFree(d_hot_cache_);
            d_hot_cache_ = nullptr;
        }
        hot_cache_cap_ = 0;
        hot_slice_bytes_ = 0;
        hot_slots_.clear();
        hot_key_to_slot_.clear();
        hot_arc_.clear();
#endif
    }
#endif

    bool is_inited() const { return inited_; }
    void * vram_arena_ptr() const { return ring_buf_.arena_ptr(); }
    int n_pinned_layers() const { return n_pinned_; }

private:
    struct TensorBackup {
        void * orig_data = nullptr;
        ggml_backend_buffer_t orig_buffer = nullptr;
    };

    struct HotSlot {
        void * d_ptr = nullptr;
        uint32_t key = 0xFFFFFFFF;
        bool in_use = false;
    };

    bool inited_ = false;
    int n_pinned_ = 0;
    size_t slot_capacity_ = 0;
    void * pinned_hbuf_ = nullptr;
    size_t pinned_hbuf_cap_ = 0;
    std::vector<void *> d_pinned_layers_;
    std::vector<size_t> pinned_layer_sizes_;

    CudaStreamManager stream_mgr_;
    VramRingBuffer ring_buf_;

    std::mutex swap_mtx_;
    std::unordered_map<ggml_tensor *, TensorBackup> orig_states_;
    std::unordered_map<int, VramRingBuffer::Slot *> active_slots_;

    // Hot-expert cache state
    void * d_hot_cache_ = nullptr;
    size_t hot_cache_cap_ = 0;
    size_t hot_slice_bytes_ = 0;
    AdaptiveReplacementCache<uint32_t> hot_arc_;
    std::vector<HotSlot> hot_slots_;
    std::unordered_map<uint32_t, size_t> hot_key_to_slot_;
    std::mutex hot_mtx_;
};

} // namespace bmoe
