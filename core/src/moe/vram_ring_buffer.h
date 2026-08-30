#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdio>


#if defined(BMOE_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace bmoe {

// VRAM Ring Buffer for Layer N+1 MoE expert weight staging.
// Preallocates a 2-4 GB device memory arena divided into K slots.
// Implements a strict lifecycle state machine to prevent GPU compute-transfer race conditions.
class VramRingBuffer {
public:
    enum class SlotState {
        Empty,        // Unallocated / Available for staging
        Transferring, // PCIe HtoD transfer in flight on Stream 1
        Ready,        // Transfer complete, ready for compute
        Computing,    // Kernel executing on Stream 0
        Releasable    // Compute complete, safe to reuse
    };

    struct Slot {
        int index = -1;
        int layer_idx = -1;
        void * d_ptr = nullptr;     // Device pointer to the start of this slot
        size_t capacity = 0;        // Byte capacity of this slot
        size_t bytes_used = 0;      // Active byte payload in this slot
        SlotState state = SlotState::Empty;

#if defined(BMOE_HAVE_CUDA)
        cudaEvent_t transfer_done_event = nullptr; // Signaled by Stream 1 when HtoD transfer completes
        cudaEvent_t compute_done_event = nullptr;  // Signaled by Stream 0 when kernel compute completes
#endif
    };

    VramRingBuffer() = default;
    ~VramRingBuffer() { cleanup(); }

    VramRingBuffer(const VramRingBuffer &) = delete;
    VramRingBuffer & operator=(const VramRingBuffer &) = delete;

    // Initializes the arena with total_bytes (e.g. 2 GB - 4 GB) divided into num_slots (e.g. 4)
    bool init(size_t total_bytes, int num_slots = 4) {
        cleanup();
        if (num_slots <= 0 || total_bytes == 0) return false;

        static constexpr size_t kSlotAlign = 4096; // 4KB sector & GGML_CUDA_MAX_ALIGN (256-byte) alignment
        num_slots_ = num_slots;
        slot_capacity_ = (total_bytes / (size_t) num_slots) & ~(kSlotAlign - 1);
        total_capacity_ = slot_capacity_ * (size_t) num_slots;


#if defined(BMOE_HAVE_CUDA)
        cudaError_t err = cudaMalloc(&d_arena_, total_capacity_);
        if (err != cudaSuccess || !d_arena_) {
            std::fprintf(stderr, "bmoe: failed to allocate %zu MiB VRAM arena (error %d)\n",
                         total_capacity_ >> 20, (int) err);
            return false;
        }

        slots_.resize((size_t) num_slots_);
        for (int i = 0; i < num_slots_; ++i) {
            Slot & s = slots_[(size_t) i];
            s.index = i;
            s.layer_idx = -1;
            s.d_ptr = (char *) d_arena_ + (size_t) i * slot_capacity_;
            s.capacity = slot_capacity_;
            s.bytes_used = 0;
            s.state = SlotState::Empty;


            cudaEventCreateWithFlags(&s.transfer_done_event, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&s.compute_done_event, cudaEventDisableTiming);
        }

        inited_ = true;
        std::fprintf(stderr, "bmoe: VRAM ring buffer initialized (%d slots of %zu MiB, total %zu MiB)\n",
                     num_slots_, slot_capacity_ >> 20, total_capacity_ >> 20);
        return true;
#else
        return false;
#endif
    }

    void cleanup() {
#if defined(BMOE_HAVE_CUDA)
        if (inited_) {
            for (Slot & s : slots_) {
                if (s.transfer_done_event) cudaEventDestroy(s.transfer_done_event);
                if (s.compute_done_event) cudaEventDestroy(s.compute_done_event);
                s.transfer_done_event = nullptr;
                s.compute_done_event = nullptr;
            }
            slots_.clear();
            if (d_arena_) cudaFree(d_arena_);
            d_arena_ = nullptr;
            inited_ = false;
        }
#endif
    }

#if defined(BMOE_HAVE_CUDA)
    // Acquires the next available slot for layer_idx to stage weights on Stream 1.
    Slot * acquire_slot_for_transfer(int layer_idx, cudaStream_t stream_transfer) {

        std::lock_guard<std::mutex> lk(mtx_);
        if (!inited_ || slots_.empty()) return nullptr;

        // If a slot is already assigned to this layer_idx, reuse it
        for (Slot & s : slots_) {
            if (s.layer_idx == layer_idx) {
                s.bytes_used = 0;
                s.state = SlotState::Transferring;
                return &s;
            }
        }

        int slot_idx = next_transfer_slot_ % num_slots_;
        Slot & s = slots_[(size_t) slot_idx];

        s.layer_idx = layer_idx;
        s.bytes_used = 0;
        s.state = SlotState::Transferring;
        next_transfer_slot_ = (next_transfer_slot_ + 1) % num_slots_;
        return &s;
    }

    // Marks transfer complete and records the transfer event on Stream 1
    void mark_transfer_complete(Slot * s, size_t bytes_transferred, cudaStream_t stream_transfer) {
        if (!s) return;
        std::lock_guard<std::mutex> lk(mtx_);
        s->bytes_used = bytes_transferred;
        s->state = SlotState::Ready;
        if (s->transfer_done_event) {
            cudaEventRecord(s->transfer_done_event, stream_transfer);
        }
    }

    // Finds the ready slot for layer_idx, ensures transfer is complete, and transitions to Computing.
    Slot * bind_slot_for_compute(int layer_idx, cudaStream_t stream_compute) {
        std::lock_guard<std::mutex> lk(mtx_);
        for (Slot & s : slots_) {
            if (s.layer_idx == layer_idx) {
                if (s.transfer_done_event) {
                    cudaEventSynchronize(s.transfer_done_event);
                }
                s.state = SlotState::Computing;
                return &s;
            }
        }

        return nullptr;
    }

    // Marks compute complete
    void mark_compute_complete(Slot * s, cudaStream_t stream_compute) {
        if (!s) return;
        std::lock_guard<std::mutex> lk(mtx_);
        s->state = SlotState::Releasable;
        cv_.notify_all();
    }

#endif

    bool is_inited() const { return inited_; }
    void * arena_ptr() const { return d_arena_; }
    size_t total_capacity() const { return total_capacity_; }
    size_t slot_capacity() const { return slot_capacity_; }
    int num_slots() const { return num_slots_; }

private:
    bool inited_ = false;
    void * d_arena_ = nullptr;
    size_t total_capacity_ = 0;
    size_t slot_capacity_ = 0;
    int num_slots_ = 0;
    int next_transfer_slot_ = 0;

    std::vector<Slot> slots_;
    std::mutex mtx_;
    std::condition_variable cv_;
};


} // namespace bmoe
