#pragma once

#include "cuda_stream_manager.h"
#include "vram_ring_buffer.h"
#include "cuda_occupancy.h"

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
#endif

    bool is_inited() const { return inited_; }
    void * vram_arena_ptr() const { return ring_buf_.arena_ptr(); }
    int n_pinned_layers() const { return n_pinned_; }

private:
    struct TensorBackup {
        void * orig_data = nullptr;
        ggml_backend_buffer_t orig_buffer = nullptr;
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
};

} // namespace bmoe
