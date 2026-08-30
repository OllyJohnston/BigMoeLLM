// Unit tests for Adaptive Replacement Cache (core/src/moe/arc_cache.h)
#include "arc_cache.h"

#include <cstdio>
#include <vector>
#include <numeric>
#include <unordered_map>

using namespace bmoe;

static int failures = 0;

static void test_capacity_and_hits() {
    AdaptiveReplacementCache<int> arc(4); // capacity 4

    // Access 1, 2, 3, 4 -> all in T1
    for (int i = 1; i <= 4; ++i) {
        auto r = arc.access(i);
        if (r.hit) {
            std::printf("[FAIL] Initial insertion should be a miss\n");
            ++failures;
            return;
        }
    }
    if (arc.resident_size() != 4) {
        std::printf("[FAIL] Resident size should be 4, got %zu\n", arc.resident_size());
        ++failures;
        return;
    }

    // Hit on 2 -> moves to T2
    auto r2 = arc.access(2);
    if (!r2.hit || r2.prev_location != AdaptiveReplacementCache<int>::Location::T1) {
        std::printf("[FAIL] Accessing 2 should be a hit in T1\n");
        ++failures;
        return;
    }

    // Access 5 -> should evict LRU of T1 (which is 1) into B1
    auto r5 = arc.access(5);
    if (r5.hit) {
        std::printf("[FAIL] Accessing 5 should be a miss\n");
        ++failures;
        return;
    }
    if (arc.resident_size() != 4) {
        std::printf("[FAIL] Resident size should stay at capacity 4, got %zu\n", arc.resident_size());
        ++failures;
        return;
    }
    if (r5.evicted_resident_keys.empty() || r5.evicted_resident_keys[0] != 1) {
        std::printf("[FAIL] Key 1 should have been evicted from resident set\n");
        ++failures;
        return;
    }

    std::printf("[PASS] ARC capacity and hit transitions\n");
}

static void test_ghost_adaptation() {
    AdaptiveReplacementCache<int> arc(4);

    // Fill with 1, 2, 3, 4
    for (int i = 1; i <= 4; ++i) arc.access(i);

    // Access 2, 3, 4 again to move them to T2 (frequency list)
    arc.access(2);
    arc.access(3);
    arc.access(4);
    // Now T1 has [1], T2 has [4, 3, 2]

    // Access 5 -> T1 has 1 item, so Replace evicts 1 to B1
    arc.access(5);

    double p_before = arc.target_p();
    // Access 1 (hit in B1 ghost list) -> should increase target_p
    auto r1 = arc.access(1);
    if (r1.hit) {
        std::printf("[FAIL] Access in B1 should be reported as cache miss (ghost)\n");
        ++failures;
        return;
    }
    if (r1.prev_location != AdaptiveReplacementCache<int>::Location::B1) {
        std::printf("[FAIL] Previous location of 1 should be B1, got %d\n", (int) r1.prev_location);
        ++failures;
        return;
    }
    if (arc.target_p() <= p_before) {
        std::printf("[FAIL] Target p should have increased on B1 hit (before=%f, after=%f)\n",
                    p_before, arc.target_p());
        ++failures;
        return;
    }

    std::printf("[PASS] ARC ghost adaptation (target_p shift)\n");
}


static void test_pinned_expert_protection() {
    AdaptiveReplacementCache<int> arc(5);

    // Pin expert 100 (e.g. top grammatical router)
    arc.access(100);
    arc.pin(100);

    // Insert 10 items, causing multiple evictions
    for (int i = 1; i <= 10; ++i) {
        arc.access(i);
    }

    // Expert 100 must still be resident
    if (!arc.is_resident(100)) {
        std::printf("[FAIL] Pinned expert 100 was evicted from cache\n");
        ++failures;
        return;
    }

    // Verify top 20% frequency update
    std::unordered_map<int, uint64_t> freqs;
    freqs[10] = 500; // most frequent
    freqs[20] = 400; // second most frequent
    for (int i = 1; i <= 8; ++i) freqs[i] = 10;

    arc.update_top_pinned(freqs, 0.20);
    if (!arc.is_pinned(10) || !arc.is_pinned(20)) {
        std::printf("[FAIL] update_top_pinned failed to pin top 20%% experts (10 and 20)\n");
        ++failures;
        return;
    }

    std::printf("[PASS] ARC pinned expert protection and frequency ranking\n");
}

int main() {
    test_capacity_and_hits();
    test_ghost_adaptation();
    test_pinned_expert_protection();

    if (failures == 0) {
        std::printf("All ARC tests passed successfully.\n");
        return 0;
    } else {
        std::printf("%d ARC test(s) failed.\n", failures);
        return 1;
    }
}
