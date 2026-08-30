#pragma once

#include "arc_cache.h"
#include "../io/platform_io.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace bmoe {

// SP-MoE Lookahead Router: Evaluates drafted hidden states from MTP heads across target router heads
// to predict expert requirements 3-4 tokens in advance and proactively dispatch Direct I/O reads.
class SpMoeLookaheadRouter {
public:
    struct LookaheadPrediction {
        std::vector<int32_t> predicted_experts; // Union of expert IDs predicted for lookahead tokens
        std::vector<int32_t> missing_experts;   // Subset missing from ARC cache needing I/O fetch
    };

    SpMoeLookaheadRouter() = default;

    // Evaluates a draft hidden state vector of dimension `hidden_dim` against router gate weights
    // `gate_weights` (shape: [n_experts, hidden_dim]) and finds top_k experts.
    static std::vector<int32_t> route_draft_state(const float * draft_hidden_state,
                                                  const float * gate_weights,
                                                  int hidden_dim,
                                                  int n_experts,
                                                  int top_k) {
        if (!draft_hidden_state || !gate_weights || hidden_dim <= 0 || n_experts <= 0 || top_k <= 0) {
            return {};
        }

        std::vector<std::pair<float, int32_t>> logits(n_experts);
        for (int e = 0; e < n_experts; ++e) {
            const float * w = gate_weights + (size_t) e * hidden_dim;
            float dot = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                dot += w[d] * draft_hidden_state[d];
            }
            logits[e] = {dot, (int32_t) e};
        }

        std::partial_sort(logits.begin(), logits.begin() + std::min(top_k, n_experts), logits.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });

        std::vector<int32_t> selected;
        selected.reserve((size_t) top_k);
        for (int i = 0; i < top_k && i < n_experts; ++i) {
            selected.push_back(logits[i].second);
        }
        return selected;
    }

    // Calculates the union of predicted experts across lookahead tokens (e.g. t+1 .. t+4)
    // and determines which are missing from the ARC cache.
    template <typename Key>
    static LookaheadPrediction predict_lookahead_union(const std::vector<std::vector<int32_t>> & per_token_topk,
                                                       const AdaptiveReplacementCache<Key> & arc_cache,
                                                       int layer_idx,
                                                       int n_experts) {
        LookaheadPrediction pred;
        std::unordered_set<int32_t> union_set;

        for (const auto & topk : per_token_topk) {
            for (int32_t exp_id : topk) {
                union_set.insert(exp_id);
            }
        }

        for (int32_t exp_id : union_set) {
            pred.predicted_experts.push_back(exp_id);
            // Global expert key in ARC cache is (layer_idx * n_experts + exp_id)
            Key global_key = (Key) (layer_idx * n_experts + exp_id);
            if (!arc_cache.is_resident(global_key)) {
                pred.missing_experts.push_back(exp_id);
            }
        }

        std::sort(pred.predicted_experts.begin(), pred.predicted_experts.end());
        std::sort(pred.missing_experts.begin(), pred.missing_experts.end());
        return pred;
    }
};

} // namespace bmoe
