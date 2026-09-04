// OpenAI-wire request parsing for bmoe-server. Lives in its own header so the parser is
// unit-testable (tests/openai_parse_test.cpp) instead of being trapped inside server_main.cpp.
//
// nlohmann/json comes from the vendored copy in third_party/llama.cpp/vendor, which the
// llama-common PUBLIC link already puts on the include path of bmoe_core consumers. The
// hand-rolled splitting this replaces broke on real-world prompts (a '}' inside a code block
// truncated a message object; a substring "role"/"content" in the text corrupted parsing).
#pragma once

#include "bmoe/config.h"
#include "bmoe/session.h"

#include "nlohmann/json.hpp"

#include <string>
#include <utility>
#include <vector>

namespace bmoe {
namespace openai {

// One parsed request body. `messages` (chat) and `prompt` (completion) are mutually exclusive
// by construction: chat bodies fill messages, completion bodies fill prompt.
struct ParsedBody {
    bool is_chat = false;
    std::vector<ChatMessage> messages; // full conversation, last entry = this turn's user message
    std::string prompt;                // raw completion prompt, or the last user message's text
    std::vector<std::string> images;   // image_url urls found in the last user message
    int n_predict = 512;
    bool has_sampling = false;
    SamplingConfig sampling;           // session defaults, overridden by whatever the body carries
    bool stream = false;
    std::string error;                 // set when parse_body() returns false
};

// Parse one request body against the session's sampling defaults. Returns false and fills
// `error` on malformed JSON, a missing messages/prompt, an empty trailing user message, or a
// trailing message whose role is not "user". Never throws. `n_predict_default` is the value a
// request omitting BOTH max_tokens and max_completion_tokens falls back to (the server's
// -n/--n-predict flag feeds it; 512 preserves the historical wire default).
inline bool parse_body(const std::string & body, const SamplingConfig & defaults, ParsedBody & out,
                       int n_predict_default = 512) {
    out.sampling = defaults;

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(body);
    } catch (const nlohmann::json::parse_error &) {
        out.error = "request body is not valid JSON";
        return false;
    }
    if (!root.is_object()) {
        out.error = "request body must be a JSON object";
        return false;
    }

    if (root.contains("messages")) {
        out.is_chat = true;
        const nlohmann::json & msgs = root["messages"];
        if (!msgs.is_array() || msgs.empty()) {
            out.error = "\"messages\" must be a non-empty array";
            return false;
        }
        out.messages.reserve(msgs.size());
        for (const nlohmann::json & m : msgs) {
            if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
                out.error = "each message needs a string \"role\"";
                return false;
            }
            ChatMessage cm;
            cm.role = m["role"].get<std::string>();
            if (m.contains("content")) {
                const nlohmann::json & c = m["content"];
                if (c.is_string()) {
                    cm.content = c.get<std::string>();
                } else if (c.is_array()) {
                    // Multimodal parts: concatenate the text parts, and collect image_url urls.
                    // The engine's req.images is per-turn, so only the LAST user message's
                    // images are reported — exactly the OpenAI shape where the current turn
                    // carries the new images.
                    std::string text;
                    for (const nlohmann::json & part : c) {
                        if (!part.is_object()) continue;
                        if (part.value("type", "") == "text")
                            text += part.value("text", "");
                        else if (part.value("type", "") == "image_url") {
                            const nlohmann::json & iu = part["image_url"];
                            if (iu.is_string())
                                out.images.push_back(iu.get<std::string>());
                            else if (iu.is_object() && iu.contains("url"))
                                out.images.push_back(iu["url"].get<std::string>());
                        }
                    }
                    cm.content = std::move(text);
                } else {
                    out.error = "message \"content\" must be a string or an array of parts";
                    return false;
                }
            }
            out.messages.push_back(std::move(cm));
        }
        if (out.messages.back().role != "user") {
            out.error = "the last message must have role \"user\"";
            return false;
        }
        out.prompt = out.messages.back().content;
        if (out.prompt.empty()) {
            out.error = "the last user message is empty";
            return false;
        }
    } else {
        out.prompt = root.value("prompt", std::string());
        if (out.prompt.empty()) {
            out.error = "no \"messages\" or non-empty \"prompt\" found";
            return false;
        }
    }

    // max_tokens first, then max_completion_tokens (the newer OpenAI name); session default.
    out.n_predict = root.value("max_tokens", 0);
    if (out.n_predict <= 0) out.n_predict = root.value("max_completion_tokens", 0);
    if (out.n_predict <= 0) out.n_predict = n_predict_default;

    out.stream = root.value("stream", false);

    // Sampling overrides. Each one flips has_sampling, so the engine builds a per-request
    // sampler chain instead of using the session's fixed one. Both spellings of the repeat
    // penalty are accepted; "temp" is the bmoe-cli name, "temperature" the OpenAI one.
    auto mark = [&]() { out.has_sampling = true; };
    if (root.contains("temperature")) {
        out.sampling.temp = (float) root.value("temperature", (double) out.sampling.temp);
        mark();
    } else if (root.contains("temp")) {
        out.sampling.temp = (float) root.value("temp", (double) out.sampling.temp);
        mark();
    }
    if (root.contains("top_p")) {
        out.sampling.top_p = (float) root.value("top_p", (double) out.sampling.top_p);
        mark();
    }
    if (root.contains("top_k")) {
        out.sampling.top_k = root.value("top_k", out.sampling.top_k);
        mark();
    }
    if (root.contains("min_p")) {
        out.sampling.min_p = (float) root.value("min_p", (double) out.sampling.min_p);
        mark();
    }
    if (root.contains("repeat_penalty")) {
        out.sampling.repeat_penalty = (float) root.value("repeat_penalty", (double) out.sampling.repeat_penalty);
        mark();
    } else if (root.contains("repetition_penalty")) {
        out.sampling.repeat_penalty = (float) root.value("repetition_penalty", (double) out.sampling.repeat_penalty);
        mark();
    }
    if (root.contains("presence_penalty")) {
        out.sampling.presence_penalty = (float) root.value("presence_penalty", (double) out.sampling.presence_penalty);
        mark();
    }
    if (root.contains("frequency_penalty")) {
        out.sampling.frequency_penalty = (float) root.value("frequency_penalty", (double) out.sampling.frequency_penalty);
        mark();
    }

    return true;
}

} // namespace openai
} // namespace bmoe