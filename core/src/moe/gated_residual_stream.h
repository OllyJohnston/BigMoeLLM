#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#if defined(BMOE_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace bmoe {

// GatedResidualStreamManager: Allocates and gates 4-branch parallel residual streams
// for next-generation hybrid architectures.
class GatedResidualStreamManager {
public:
    static constexpr int kNumBranches = 4;

    struct Config {
        size_t hidden_dim = 2048;
        size_t max_seq_len = 4096;
    };

    explicit GatedResidualStreamManager(const Config & cfg = Config()) : cfg_(cfg) {}
    ~GatedResidualStreamManager() { cleanup(); }

    bool init(int device_id = 0) {
#if defined(BMOE_HAVE_CUDA)
        cleanup();
        size_t branch_bytes = cfg_.hidden_dim * sizeof(uint16_t); // BF16/FP16 per token
        for (int i = 0; i < kNumBranches; ++i) {
            cudaError_t err = cudaMalloc(&d_branch_streams_[i], branch_bytes);
            if (err != cudaSuccess) {
                cleanup();
                return false;
            }
            cudaMemset(d_branch_streams_[i], 0, branch_bytes);
        }
        inited_ = true;
        return true;
#else
        (void) device_id;
        inited_ = true;
        return true;
#endif
    }

    void * branch_ptr(int branch_idx) const {
        if (branch_idx >= 0 && branch_idx < kNumBranches) {
            return d_branch_streams_[branch_idx];
        }
        return nullptr;
    }

    void cleanup() {
#if defined(BMOE_HAVE_CUDA)
        for (int i = 0; i < kNumBranches; ++i) {
            if (d_branch_streams_[i]) {
                cudaFree(d_branch_streams_[i]);
                d_branch_streams_[i] = nullptr;
            }
        }
#endif
        inited_ = false;
    }

private:
    Config cfg_;
    bool inited_ = false;
    void * d_branch_streams_[kNumBranches] = {nullptr};
};

} // namespace bmoe
