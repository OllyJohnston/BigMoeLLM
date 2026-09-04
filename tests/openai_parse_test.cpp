// Unit tests for bmoe::openai::parse_body (cli/openai_body.h) — the OpenAI-wire request
// parser bmoe-server uses. Pure text in, structured fields out: no model, no llama.cpp,
// no sockets, so it runs unconditionally in ctest.
//
// The cases are the failure modes the hand-rolled string splitting had, plus the wire
// formats a real client (Open WebUI, AnythingLLM, a curl script) actually sends:
//   * a '}' inside a code block truncated a message object and dropped the turn;
//   * a literal "role"/"content" in the text corrupted the substring role detection;
//   * escaped characters were only partially unescaped;
//   * multimodal part arrays (text + image_url) were ignored for text extraction.
//
// Checks are explicit (not <cassert>): the Release build defines NDEBUG.

#include "openai_body.h"

#include <cstdio>
#include <string>

using namespace bmoe;

static int failures = 0;

static void expect(bool cond, const char * name) {
    if (cond) {
        std::printf("[PASS] %s\n", name);
    } else {
        std::printf("[FAIL] %s\n", name);
        ++failures;
    }
}

static void expect_error(const char * body, const char * name) {
    openai::ParsedBody pb;
    bool ok = openai::parse_body(body, SamplingConfig{}, pb);
    expect(!ok && !pb.error.empty(), name);
}

static void expect_messages(const char * body, const std::vector<ChatMessage> & want, const char * name) {
    openai::ParsedBody pb;
    bool ok = openai::parse_body(body, SamplingConfig{}, pb);
    if (!ok) {
        std::printf("[FAIL] %s\n  parse error: %s\n", name, pb.error.c_str());
        ++failures;
        return;
    }
    if (!pb.is_chat || pb.messages.size() != want.size()) {
        std::printf("[FAIL] %s\n  expected %zu chat messages, got %zu\n", name, want.size(), pb.messages.size());
        ++failures;
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        if (pb.messages[i].role != want[i].role || pb.messages[i].content != want[i].content) {
            std::printf("[FAIL] %s\n  message %zu: got role='%s' content='%s', want role='%s' content='%s'\n",
                        name, i, pb.messages[i].role.c_str(), pb.messages[i].content.c_str(),
                        want[i].role.c_str(), want[i].content.c_str());
            ++failures;
            return;
        }
    }
    std::printf("[PASS] %s\n", name);
}

int main() {
    // ── The brace-in-code-block case that broke the old '}' splitting ──────────
    expect_messages(
        R"({"messages":[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"Explain this C++ snippet:\nvoid f() { return; }"},
            {"role":"assistant","content":"It returns."},
            {"role":"user","content":"And what about { nested } braces?"}
        ]})",
        {{"system", "You are a helpful assistant."},
         {"user", "Explain this C++ snippet:\nvoid f() { return; }"},
         {"assistant", "It returns."},
         {"user", "And what about { nested } braces?"}},
        "multi-turn chat with braces in content survives");

    // ── Literal "role"/"content" inside the text must not corrupt parsing ──────
    expect_messages(
        R"({"messages":[{"role":"user","content":"I wrote \"role\":\"user\" and \"content\":\"x\" in my prompt"}]})",
        {{"user", "I wrote \"role\":\"user\" and \"content\":\"x\" in my prompt"}},
        "literal role/content strings inside content survive");

    // ── Escaped characters come back unescaped ─────────────────────────────────
    expect_messages(
        R"({"messages":[{"role":"user","content":"line1\nline2\ttab \"quoted\" back\\slash"}]})",
        {{"user", "line1\nline2\ttab \"quoted\" back\\slash"}},
        "escaped newline/tab/quote/backslash are unescaped");

    // ── Multimodal part arrays: text joined, image_url captured ────────────────
    {
        openai::ParsedBody pb;
        const char * body =
            R"({"messages":[{"role":"user","content":[
                {"type":"text","text":"What is in this image? "},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
                {"type":"text","text":"Please describe it."}
            ]}]})";
        bool ok = openai::parse_body(body, SamplingConfig{}, pb);
        expect(ok, "multimodal body parses");
        if (ok) {
            expect(pb.prompt == "What is in this image? Please describe it.",
                   "text parts of the last user message are concatenated");
            expect(pb.images.size() == 1 && pb.images[0] == "data:image/png;base64,AAAA",
                   "image_url url is captured");
            expect(pb.messages.size() == 1 && pb.messages[0].content == "What is in this image? Please describe it.",
                   "message content holds the joined text");
        }
    }

    // ── A completion body stays a completion ───────────────────────────────────
    {
        openai::ParsedBody pb;
        SamplingConfig dflt;
        dflt.temp = 0.0f; // greedy session default
        bool ok = openai::parse_body(R"({"prompt":"Hello","max_tokens":100,"temperature":0.7,"stream":true})", dflt, pb);
        expect(ok && !pb.is_chat && pb.prompt == "Hello", "completion body is not chat");
        expect(pb.n_predict == 100, "max_tokens is read");
        expect(pb.stream, "stream flag is read");
        expect(pb.has_sampling && pb.sampling.temp == 0.7f, "temperature overrides the session default");
        expect(pb.sampling.top_k == dflt.top_k, "absent sampling fields keep the session default");
    }

    // ── max_completion_tokens fallback and the max_tokens precedence ───────────
    {
        openai::ParsedBody pb;
        openai::parse_body(R"({"prompt":"Hi","max_completion_tokens":321})", SamplingConfig{}, pb);
        expect(pb.n_predict == 321, "max_completion_tokens is read when max_tokens is absent");
        openai::parse_body(R"({"prompt":"Hi","max_tokens":50,"max_completion_tokens":321})", SamplingConfig{}, pb);
        expect(pb.n_predict == 50, "max_tokens wins over max_completion_tokens");
        openai::parse_body(R"({"prompt":"Hi"})", SamplingConfig{}, pb);
        expect(pb.n_predict == 512, "512 fallback when no token cap is given");
        // Explicit n_predict_default (the server's -n/--n-predict) replaces the fallback.
        openai::parse_body(R"({"prompt":"Hi"})", SamplingConfig{}, pb, 20000);
        expect(pb.n_predict == 20000, "n_predict_default applies when no token cap is given");
        openai::parse_body(R"({"prompt":"Hi","max_tokens":77})", SamplingConfig{}, pb, 20000);
        expect(pb.n_predict == 77, "max_tokens still wins over n_predict_default");
    }

    // ── Sampling override spellings ────────────────────────────────────────────
    {
        openai::ParsedBody pb;
        SamplingConfig dflt;
        dflt.top_k = 40;
        dflt.top_p = 0.95f;
        dflt.min_p = 0.05f;
        dflt.repeat_penalty = 1.1f;
        const char * body =
            R"({"prompt":"Hi","temp":0.9,"top_p":0.8,"top_k":10,"min_p":0.1,"repetition_penalty":1.3,"presence_penalty":0.5,"frequency_penalty":0.2})";
        bool ok = openai::parse_body(body, dflt, pb);
        expect(ok, "sampling override body parses");
        if (ok) {
            expect(pb.sampling.temp == 0.9f && pb.sampling.top_p == 0.8f && pb.sampling.top_k == 10 &&
                       pb.sampling.min_p == 0.1f && pb.sampling.repeat_penalty == 1.3f &&
                       pb.sampling.presence_penalty == 0.5f && pb.sampling.frequency_penalty == 0.2f,
                   "all sampling overrides are applied");
        }
        openai::parse_body(R"({"prompt":"Hi","repeat_penalty":1.2})", dflt, pb);
        expect(pb.sampling.repeat_penalty == 1.2f && pb.has_sampling,
               "repeat_penalty spelling is accepted");
    }

    // ── Rejections ─────────────────────────────────────────────────────────────
    expect_error("{not json", "malformed JSON is rejected");
    expect_error("[1,2,3]", "a non-object body is rejected");
    expect_error(R"({})", "a body with neither messages nor prompt is rejected");
    expect_error(R"({"messages":[]})", "an empty messages array is rejected");
    expect_error(R"({"messages":[{"role":"assistant","content":"hi"}]})",
                 "a trailing assistant message is rejected");
    expect_error(R"({"messages":[{"role":"user"}]})", "a message without content leaves an empty trailing user message, rejected");
    expect_error(R"({"messages":[{"role":"user","content":""}]})", "an empty last user message is rejected");
    expect_error(R"({"prompt":""})", "an empty prompt is rejected");
    expect_messages(R"({"messages":[{"role":"assistant"},{"role":"user","content":"hi"}]})",
                    {{"assistant", ""}, {"user", "hi"}},
                    "a non-trailing message without content is accepted as empty");

    if (failures == 0) {
        std::printf("openai_parse: all checks passed\n");
        return 0;
    }
    std::printf("openai_parse: %d check(s) failed\n", failures);
    return 1;
}