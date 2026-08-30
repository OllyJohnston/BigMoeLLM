// Unit test for SP-MoE Lookahead Routing and D-CFR (Deferred-Commit Factor Replay)
#include "sp_moe_lookahead.h"
#include "dcfr_verifier.h"
#include "arc_cache.h"

#include <cstdio>
#include <vector>
#include <numeric>

using namespace bmoe;

static int failures = 0;

static void test_sp_moe_lookahead() {
    const int hidden_dim = 4;
    const int n_experts = 8;
    const int top_k = 2;

    // Synthetic draft hidden state (dimension 4)
    std::vector<float> draft_h = {1.0f, 0.5f, -0.5f, 2.0f};

    // Synthetic router gate weights (8 experts x 4 dim)
    std::vector<float> gate_w(n_experts * hidden_dim, 0.0f);
    // Give expert 3 and expert 6 highest dot products
    gate_w[3 * hidden_dim + 0] = 2.0f;
    gate_w[3 * hidden_dim + 3] = 3.0f; // dot = 2*1 + 3*2 = 8.0

    gate_w[6 * hidden_dim + 0] = 1.0f;
    gate_w[6 * hidden_dim + 3] = 2.0f; // dot = 1*1 + 2*2 = 5.0

    std::vector<int32_t> routed = SpMoeLookaheadRouter::route_draft_state(
        draft_h.data(), gate_w.data(), hidden_dim, n_experts, top_k);

    if (routed.size() != 2 || routed[0] != 3 || routed[1] != 6) {
        std::printf("[FAIL] route_draft_state did not predict top experts (3, 6)\n");
        ++failures;
        return;
    }

    // Test lookahead union across 3 draft tokens
    std::vector<std::vector<int32_t>> per_token_topk = {
        {3, 6}, // token t+1
        {1, 3}, // token t+2
        {6, 7}  // token t+3
    };

    AdaptiveReplacementCache<int> arc(16);
    // Pin expert 3 in cache
    arc.access(0 * n_experts + 3); // layer 0, expert 3

    auto pred = SpMoeLookaheadRouter::predict_lookahead_union(per_token_topk, arc, 0, n_experts);

    // Predicted union: {1, 3, 6, 7}
    if (pred.predicted_experts.size() != 4 ||
        pred.predicted_experts != std::vector<int32_t>{1, 3, 6, 7}) {
        std::printf("[FAIL] Lookahead predicted union incorrect\n");
        ++failures;
        return;
    }

    // Missing from ARC: {1, 6, 7} (since 3 is resident)
    if (pred.missing_experts.size() != 3 ||
        pred.missing_experts != std::vector<int32_t>{1, 6, 7}) {
        std::printf("[FAIL] Missing experts subset calculation incorrect\n");
        ++failures;
        return;
    }

    std::printf("[PASS] SP-MoE Lookahead routing and missing expert spooling prediction\n");
}

static void test_dcfr_verification() {
    const int vocab_size = 10;
    const int n_draft = 3;

    std::vector<DcfrDraftToken> drafts(n_draft);
    drafts[0].token_id = 4;
    drafts[1].token_id = 7;
    drafts[2].token_id = 2;

    // Synthetic step logits (4 steps: 3 draft positions + 1 bonus step)
    std::vector<std::vector<float>> logits_storage(n_draft + 1, std::vector<float>(vocab_size, 0.0f));

    // Case 1: Partial acceptance (token 0 matches 4, token 1 matches 7, token 2 diverges from 2 -> target predicts 9)
    logits_storage[0][4] = 10.0f; // matches draft 0
    logits_storage[1][7] = 12.0f; // matches draft 1
    logits_storage[2][9] = 15.0f; // target prefers 9 over draft token 2
    logits_storage[3][1] = 8.0f;

    std::vector<const float *> step_logits = {
        logits_storage[0].data(),
        logits_storage[1].data(),
        logits_storage[2].data(),
        logits_storage[3].data()
    };

    auto res = DcfrVerifier::verify_greedy(step_logits, vocab_size, drafts);
    if (res.accepted_count != 2 || res.commit_tokens != std::vector<int32_t>{4, 7} || res.bonus_token != 9) {
        std::printf("[FAIL] D-CFR verification failed for partial acceptance case\n");
        ++failures;
        return;
    }

    // Case 2: Full acceptance (all 3 draft tokens match, bonus token is 1)
    logits_storage[2][2] = 20.0f; // now matches draft 2
    res = DcfrVerifier::verify_greedy(step_logits, vocab_size, drafts);
    if (res.accepted_count != 3 || res.commit_tokens != std::vector<int32_t>{4, 7, 2} || res.bonus_token != 1) {
        std::printf("[FAIL] D-CFR verification failed for full acceptance case\n");
        ++failures;
        return;
    }

    std::printf("[PASS] D-CFR deferred verification and selective commit\n");
}

int main() {
    test_sp_moe_lookahead();
    test_dcfr_verification();

    if (failures == 0) {
        std::printf("All SP-MoE and D-CFR tests passed successfully.\n");
        return 0;
    } else {
        std::printf("%d SP-MoE/D-CFR test(s) failed.\n", failures);
        return 1;
    }
}
