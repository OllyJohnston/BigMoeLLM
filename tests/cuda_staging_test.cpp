// Unit test for Dual-Stream CUDA Staging and VRAM Ring Buffer
#include "cuda_expert_stager.h"
#include "platform_io.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>
#include <cstring>


#if defined(BMOE_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

using namespace bmoe;

static int failures = 0;

#if defined(BMOE_HAVE_CUDA)

static void test_dual_stream_staging() {
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess || dev_count == 0) {
        std::printf("[SKIP] No CUDA device available for dual-stream test\n");
        return;
    }

    CudaExpertStager stager;
    const size_t arena_bytes = 64 * 1024 * 1024; // 64 MB
    const int num_slots = 4;                     // 16 MB per slot

    if (!stager.init(arena_bytes, num_slots, 0)) {
        std::printf("[FAIL] CudaExpertStager init failed\n");
        ++failures;
        return;
    }

    // Allocate pinned host source data
    const size_t test_payload_sz = 4 * 1024 * 1024; // 4 MB payload
    pio::PinnedAlloc host_alloc;
    if (!pio::pinned_alloc(test_payload_sz, &host_alloc)) {
        std::printf("[FAIL] Pinned host allocation failed\n");
        ++failures;
        return;
    }

    uint8_t * h_data = (uint8_t *) host_alloc.base;
    for (size_t i = 0; i < test_payload_sz; ++i) {
        h_data[i] = (uint8_t) ((i * 13 + 5) & 0xFF);
    }

    // Stage Layer 1
    std::vector<ExpertStagingItem> items;
    ExpertStagingItem item;
    item.h_src = h_data;
    item.offset_in_slot = 0;
    item.size = test_payload_sz;
    items.push_back(item);

    bool stage_ok = stager.stage_layer_async(1, items);
    if (!stage_ok) {
        std::printf("[FAIL] stage_layer_async failed for layer 1\n");
        ++failures;
        pio::pinned_free(&host_alloc);
        return;
    }

    // Create a mock ggml_tensor
    ggml_tensor mock_t{};
    void * orig_magic = (void *) (uintptr_t) 0x12345678;
    mock_t.data = orig_magic;

    // Bind layer 1 for compute
    void * d_vram_ptr = stager.bind_layer_for_compute(1, &mock_t);
    if (!d_vram_ptr || mock_t.data != d_vram_ptr) {
        std::printf("[FAIL] bind_layer_for_compute did not swap tensor data to VRAM pointer\n");
        ++failures;
        pio::pinned_free(&host_alloc);
        return;
    }

    // Verify GPU memory content by copying back to host
    std::vector<uint8_t> h_verify(test_payload_sz);
    err = cudaMemcpy(h_verify.data(), d_vram_ptr, test_payload_sz, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess || std::memcmp(h_data, h_verify.data(), test_payload_sz) != 0) {
        std::printf("[FAIL] GPU VRAM data did not match pinned host source data (error %d)\n", (int) err);
        ++failures;
    } else {
        std::printf("[PASS] Dual-stream staging and GPU VRAM data verification\n");
    }

    // Restore pointer
    stager.restore_tensor(1, &mock_t);
    if (mock_t.data != orig_magic) {
        std::printf("[FAIL] restore_tensor did not restore original tensor data pointer\n");
        ++failures;
    } else {
        std::printf("[PASS] Execution seam tensor pointer restoration\n");
    }


    pio::pinned_free(&host_alloc);
}

#endif

int main() {
#if defined(BMOE_HAVE_CUDA)
    test_dual_stream_staging();
#else
    std::printf("[SKIP] CUDA not compiled in\n");
#endif

    if (failures == 0) {
        std::printf("All CUDA staging tests passed successfully.\n");
        return 0;
    } else {
        std::printf("%d CUDA staging test(s) failed.\n", failures);
        return 1;
    }
}
