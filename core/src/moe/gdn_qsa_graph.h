#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

#if defined(BMOE_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace bmoe {

// GdnQsaGraphManager: Coordinates memory allocations between recurrent state buffers
// (for 3x Gated DeltaNet layers) and standard KV cache slots (for 1x QSA attention layers).
class GdnQsaGraphManager {
public:
    struct Config {
        int n_layers = 40;
        int gdn_period = 4; // 3 GDN layers per 1 QSA layer (3:1 hybrid ratio)
        size_t state_dim = 128;
        size_t n_heads = 16;
        size_t head_dim = 128;
    };

    explicit GdnQsaGraphManager(const Config & cfg = Config()) : cfg_(cfg) {}
    ~GdnQsaGraphManager() { cleanup(); }

    bool is_gdn_layer(int layer_idx) const {
        return (layer_idx % cfg_.gdn_period) != (cfg_.gdn_period - 1);
    }

    bool is_qsa_layer(int layer_idx) const {
        return !is_gdn_layer(layer_idx);
    }

    bool init(int device_id = 0) {
#if defined(BMOE_HAVE_CUDA)
        cleanup();
        size_t state_bytes_per_layer = cfg_.n_heads * cfg_.head_dim * cfg_.state_dim * sizeof(float);
        d_recurrent_states_.resize(cfg_.n_layers, nullptr);

        for (int il = 0; il < cfg_.n_layers; ++il) {
            if (is_gdn_layer(il)) {
                cudaError_t err = cudaMalloc(&d_recurrent_states_[il], state_bytes_per_layer);
                if (err != cudaSuccess) {
                    cleanup();
                    return false;
                }
                cudaMemset(d_recurrent_states_[il], 0, state_bytes_per_layer);
            }
        }
        inited_ = true;
        return true;
#else
        (void) device_id;
        inited_ = true;
        return true;
#endif
    }

    void reset_states(void * cuda_stream = nullptr) {
#if defined(BMOE_HAVE_CUDA)
        if (!inited_) return;
        size_t state_bytes_per_layer = cfg_.n_heads * cfg_.head_dim * cfg_.state_dim * sizeof(float);
        cudaStream_t stream = cuda_stream ? (cudaStream_t) cuda_stream : (cudaStream_t) 0;
        for (int il = 0; il < cfg_.n_layers; ++il) {
            if (d_recurrent_states_[il]) {
                cudaMemsetAsync(d_recurrent_states_[il], 0, state_bytes_per_layer, stream);
            }
        }
#else
        (void) cuda_stream;
#endif
    }

    void * get_recurrent_state(int layer_idx) const {
        if (layer_idx >= 0 && layer_idx < (int) d_recurrent_states_.size()) {
            return d_recurrent_states_[layer_idx];
        }
        return nullptr;
    }

    void cleanup() {
#if defined(BMOE_HAVE_CUDA)
        for (void * ptr : d_recurrent_states_) {
            if (ptr) cudaFree(ptr);
        }
        d_recurrent_states_.clear();
#endif
        inited_ = false;
    }

private:
    Config cfg_;
    bool inited_ = false;
    std::vector<void *> d_recurrent_states_;
};

} // namespace bmoe
