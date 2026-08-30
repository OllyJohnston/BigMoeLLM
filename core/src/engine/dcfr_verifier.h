#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

namespace bmoe {

// Lightweight draft candidate stored purely in host C++ memory (0 VRAM allocation)
struct DcfrDraftToken {
    int32_t token_id = -1;
    float draft_prob = 0.0f;
    uint32_t step_offset = 0;
};

// Result of batched verification
struct DcfrVerificationResult {
    int accepted_count = 0;             // Number of draft tokens verified and accepted
    std::vector<int32_t> commit_tokens; // The accepted token IDs to commit to KV cache
    int32_t bonus_token = -1;           // The first newly sampled token following the last accepted token
};

// D-CFR: Deferred-Commit Factor Replay Verifier
class DcfrVerifier {
public:
    DcfrVerifier() = default;

    // Evaluates target model's logits against the draft sequence greedily.
    // `target_logits`: 2D array of logits [draft_len + 1, vocab_size]
    // `draft_tokens`: sequence of speculative draft tokens
    static DcfrVerificationResult verify_greedy(const std::vector<const float *> & step_logits,
                                                int vocab_size,
                                                const std::vector<DcfrDraftToken> & draft_tokens) {
        DcfrVerificationResult res;
        if (step_logits.empty() || vocab_size <= 0) return res;

        int n_draft = (int) draft_tokens.size();
        int accepted = 0;

        for (int i = 0; i < n_draft && i < (int) step_logits.size(); ++i) {
            const float * logits = step_logits[(size_t) i];
            if (!logits) break;

            // Find argmax token from target model
            int32_t target_best = 0;
            float max_val = logits[0];
            for (int v = 1; v < vocab_size; ++v) {
                if (logits[v] > max_val) {
                    max_val = logits[v];
                    target_best = (int32_t) v;
                }
            }

            // Check if draft matches target model greedy prediction
            if (target_best == draft_tokens[(size_t) i].token_id) {
                res.commit_tokens.push_back(target_best);
                accepted++;
            } else {
                // Mismatch: reject current and all following draft tokens
                res.bonus_token = target_best; // Emit the target model's true token
                break;
            }
        }

        res.accepted_count = accepted;

        // If all draft tokens were accepted, the bonus token comes from step_logits[n_draft]
        if (accepted == n_draft && (size_t) n_draft < step_logits.size()) {
            const float * last_logits = step_logits[(size_t) n_draft];
            if (last_logits) {
                int32_t target_best = 0;
                float max_val = last_logits[0];
                for (int v = 1; v < vocab_size; ++v) {
                    if (last_logits[v] > max_val) {
                        max_val = last_logits[v];
                        target_best = (int32_t) v;
                    }
                }
                res.bonus_token = target_best;
            }
        }

        return res;
    }
};

} // namespace bmoe
