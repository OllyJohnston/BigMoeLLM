// bmoe-server — HTTP server mode for BigMoeOnEdge.
//
// Loads a model once (like --session) and serves inferences over HTTP on a configurable
// port. Exposes an OpenAI-compatible REST API:
//
//   POST /v1/completions       text completion (raw prompt)
//   POST /v1/chat/completions  chat completion (message array -> chat template)
//   GET  /v1/models            model metadata
//
// Streaming via server-sent events (stream=true) mirrors the --progress protocol.
// The expert cache and model stay loaded between requests — the same amortisation
// the --session mode provides.
//
// Usage: bmoe-server -m <model.gguf> [--port N] [--host ADDR] [options]
//
// All bmoe-cli streaming/flags work the same way (--moe-stream, --cache-mb, etc.)
// except --prompt and --session.
#include "bmoe/config.h"
#include "bmoe/runtime.h"
#include "bmoe/session.h"
#include "bmoe/recipe.h"
#include "bmoe/metrics.h"
#include "bmoe/version.h"

#include "openai_body.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
using socklen_t = int;
#define close_socket(s) closesocket(s)
#define IS_INVALID_SOCKET(s) ((s) == INVALID_SOCKET)
#else
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
using socket_t = int;
#define close_socket(s) close(s)
#define IS_INVALID_SOCKET(s) ((s) < 0)
#define INVALID_SOCKET (-1)
#endif

using namespace bmoe;

// ── Socket helpers ───────────────────────────────────────────────────────────


// ── JSON helpers ─────────────────────────────────────────────────────────────
// json_escape writes RESPONSES (engine-owned strings into our own JSON), so it stays
// hand-rolled. Parsing REQUESTS is nlohmann's job — see openai_body.h, which is unit-
// tested (tests/openai_parse_test.cpp) against the failure modes the old string
// splitting had: a '}' inside a code block truncated a message object, and a literal
// "role"/"content" in the text corrupted the substring matching.

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            if ((unsigned char) c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof(b), "\\u%04x", c);
                o += b;
            } else {
                o += c;
            }
        }
    }
    return o;
}


struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::string body;
    std::string content_type;
    std::string authorization; // raw Authorization header value, empty when absent
    bool keep_alive = false;
    size_t content_length = 0;
};

static bool parse_http_request(const std::string & raw, HttpRequest & req) {
    size_t eol = raw.find("\r\n");
    if (eol == std::string::npos) return false;

    std::string reqline = raw.substr(0, eol);
    size_t sp1 = reqline.find(' ');
    if (sp1 == std::string::npos) return false;
    size_t sp2 = reqline.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return false;

    req.method = reqline.substr(0, sp1);

    std::string full_path = reqline.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t qm = full_path.find('?');
    if (qm != std::string::npos) {
        req.path = full_path.substr(0, qm);
        req.query = full_path.substr(qm + 1);
    } else {
        req.path = full_path;
    }

    // Parse headers
    size_t hdr_start = eol + 2;
    while (true) {
        size_t hdr_end = raw.find("\r\n", hdr_start);
        if (hdr_end == std::string::npos || hdr_end == hdr_start) break;
        std::string hdr = raw.substr(hdr_start, hdr_end - hdr_start);
        size_t colon = hdr.find(':');
        if (colon != std::string::npos) {
            std::string key = hdr.substr(0, colon);
            std::string val = hdr.substr(colon + 1);
            val.erase(0, val.find_first_not_of(" \t"));

            std::string lkey;
            lkey.resize(key.size());
            std::transform(key.begin(), key.end(), lkey.begin(), ::tolower);

            if (lkey == "content-type") req.content_type = val;
            if (lkey == "authorization") req.authorization = val;
            if (lkey == "connection") {
                std::string lv;
                lv.resize(val.size());
                std::transform(val.begin(), val.end(), lv.begin(), ::tolower);
                req.keep_alive = (lv == "keep-alive");
            }
            if (lkey == "content-length") req.content_length = (size_t) std::atoll(val.c_str());
        }
        hdr_start = hdr_end + 2;
    }

    // Body follows the blank line (\r\n\r\n ends the headers)
    size_t body_start = raw.find("\r\n\r\n");
    if (body_start != std::string::npos) body_start += 4;
    if (body_start < raw.size() && req.content_length > 0) {
        req.body = raw.substr(body_start, req.content_length);
    }

    return true;
}

// Write all bytes to a socket.
static void http_write(socket_t fd, const std::string & s) {
    size_t off = 0;
    while (off < s.size()) {
        int n = send(fd, s.data() + off, (int) (s.size() - off), 0);
        if (n < 0) {
#if defined(_WIN32)
            if (WSAGetLastError() == WSAEINTR) continue;
#else
            if (errno == EINTR) continue;
#endif
            return;
        }
        off += (size_t) n;
    }
}

// Send a complete HTTP response.
static void send_response(socket_t fd,
                          int status,
                          const char * status_text,
                          const std::string & content_type,
                          const std::string & body,
                          bool keep_alive) {
    char buf[128];
    std::string resp;
    std::snprintf(buf, sizeof(buf), "HTTP/1.1 %d %s\r\n", status, status_text);
    resp += buf;
    resp += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    resp += "Content-Type: " + content_type + "\r\n";
    resp += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    resp += "Access-Control-Allow-Origin: *\r\n";
    resp += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    resp += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
    resp += "\r\n";
    resp += body;
    http_write(fd, resp);
}

// Send SSE response headers (no Content-Length — streamed body).
static void send_sse_headers(socket_t fd) {
    // OpenAI-compatible SDKs (OpenAI/JS, OpenAI/Python) use fetch() and expect
    // Transfer-Encoding: chunked for streaming. Connection: close without
    // chunked encoding causes the SDK to read the entire body before parsing,
    // which deadlocks on single-token streams.
    std::string resp = "HTTP/1.1 200 OK\r\n"
                       "Connection: close\r\n"
                       "Transfer-Encoding: chunked\r\n"
                       "Content-Type: text/event-stream\r\n"
                       "Cache-Control: no-cache\r\n"
                       "Access-Control-Allow-Origin: *\r\n"
                       "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                       "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                       "\r\n";
    http_write(fd, resp);
}

// Send an SSE data chunk with proper HTTP chunked transfer encoding.
static void send_sse(socket_t fd, const std::string & data) {
    std::string chunk = "data: " + data + "\n\n";
    char size_buf[16];
    std::snprintf(size_buf, sizeof(size_buf), "%zx\r\n", chunk.size());
    http_write(fd, size_buf);
    http_write(fd, chunk);
    http_write(fd, "\r\n");
}

// Send the terminating zero-size chunk.
static void send_sse_done(socket_t fd) {
    send_sse(fd, "[DONE]");
    http_write(fd, "0\r\n\r\n");
}

static void send_json_error(socket_t fd, int status, const char * msg, bool ka) {
    std::string body = "{\"error\":{\"message\":\"" + json_escape(msg) + "\",\"type\":\"api_error\"}}";
    const char * text = status >= 500   ? "Internal Server Error"
                        : status == 400 ? "Bad Request"
                        : status == 404 ? "Not Found"
                                        : "Error";
    send_response(fd, status, text, "application/json", body, ka);
}


// ── Server state ─────────────────────────────────────────────────────────────

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    int max_connections = 32;
    bool disable_think = false;
    std::string mmproj_path;  // path to multimodal projector (mmproj.gguf) for vision models
    // Shared secret enforced via `Authorization: Bearer <key>` on every endpoint (CORS
    // OPTIONS preflight exempt). Empty = authentication disabled — acceptable on loopback
    // only; a non-loopback bind with no key is refused with a warning.
    std::string api_key;
};

struct ServerState {
    std::unique_ptr<Session> session;
    SessionConfig session_cfg;
    ServerConfig srv_cfg;
    std::atomic<bool> running{true};
    std::mutex generate_mtx;
};


// ── Request handlers ─────────────────────────────────────────────────────────

static void handle_completions(socket_t fd, const HttpRequest & req, ServerState & state, bool chat);

// Request authentication. No api_key configured = open (loopback deployments). With a key set,
// every endpoint except the CORS preflight below must present `Authorization: Bearer <key>`.
static bool auth_ok(const HttpRequest & req, const ServerState & state) {
    if (state.srv_cfg.api_key.empty()) return true;
    if (req.authorization.rfind("Bearer ", 0) != 0) return false;
    std::string tok = req.authorization.substr(7);
    // Trim trailing CR/LF/space a header line may carry; exact match otherwise.
    while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\r' || tok.back() == '\n')) tok.pop_back();
    return tok == state.srv_cfg.api_key;
}

static void handle_request(socket_t fd, const HttpRequest & req, ServerState & state) {
    const bool ka = req.keep_alive;
    std::fprintf(stderr, "bmoe-server: HTTP %s %s (body=%zu bytes)\n", req.method.c_str(), req.path.c_str(), req.body.size());

    // CORS preflight. Must pass WITHOUT authentication: browsers send OPTIONS before the real
    // request and carry no Authorization header on it.
    if (req.method == "OPTIONS") {
        send_response(fd, 204, "No Content", "text/plain", "", ka);
        return;
    }

    // API key gate. Only reached when api_key is configured; Open WebUI / AnythingLLM / curl
    // all send Authorization on the real request, so a missing or wrong key is a flat 401.
    if (!auth_ok(req, state)) {
        const char * body = "{\"error\":{\"message\":\"Invalid or missing API key\",\"type\":\"authentication_error\"}}";
        send_response(fd, 401, "Unauthorized", "application/json", body, ka);
        return;
    }

    // GET /
    if (req.method == "GET" && (req.path == "/" || req.path == "")) {
        std::string body = "{\"name\":\"bmoe-server\","
                           "\"version\":\"" BMOE_VERSION "\","
                           "\"description\":\"BigMoeOnEdge streaming inference server\"}";
        send_response(fd, 200, "OK", "application/json", body, ka);
        return;
    }

    // GET /health — liveness probe. Never blocks on generation; answers from the accept-loop
    // thread pool the moment the connection is read.
    if (req.method == "GET" && req.path == "/health") {
        const char * body = "{\"status\":\"ok\"}";
        send_response(fd, 200, "OK", "application/json", body, ka);
        return;
    }

    // GET /v1/models or /models
    if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
        if (!state.session) {
            send_json_error(fd, 500, "Model not loaded", ka);
            return;
        }
        std::string model_id = "model";
        const std::string & mp = state.session_cfg.model_path;
        size_t slash = mp.rfind('/');
        size_t bslash = mp.rfind('\\');
        size_t sep = (slash != std::string::npos) ? slash : bslash;
        if (sep != std::string::npos && sep + 1 < mp.size()) model_id = mp.substr(sep + 1);

        std::string body = "{\"object\":\"list\",\"data\":[{"
                           "\"id\":\"" +
                           json_escape(model_id) +
                           "\","
                           "\"object\":\"model\","
                           "\"created\":0,"
                           "\"owned_by\":\"bmoe\","
                           "\"meta\":{"
                           "\"arch\":\"" +
                           json_escape(state.session->arch()) +
                           "\","
                           "\"n_ctx\":" +
                           std::to_string(state.session->n_ctx()) +
                           ","
                           "\"n_expert_used\":" +
                           std::to_string(state.session->n_expert_used()) + "}}]}";
        send_response(fd, 200, "OK", "application/json", body, ka);
        return;
    }

    // POST /v1/chat/completions or /chat/completions
    if (req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/chat/completions")) {
        handle_completions(fd, req, state, true);
        return;
    }

    // POST /v1/completions or /completions
    if (req.method == "POST" && (req.path == "/v1/completions" || req.path == "/completions")) {
        handle_completions(fd, req, state, false);
        return;
    }

    send_json_error(fd, 404, "Not found", ka);
}

static void print_lmstudio_telemetry(const RunSummary & sum, double ttft_ms, double decode_sec, double total_sec) {
    char lbuf[64];
    char rbuf[64];

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "┌──────────────────────────────┬──────────────────────────────┐\n");
    std::fprintf(stderr, "│ Generation Statistics                                       │\n");
    std::fprintf(stderr, "├──────────────────────────────┼──────────────────────────────┤\n");

    // Time to First Token (TTFT)
    if (ttft_ms <= 0.0 && sum.prefill_seconds > 0.0) {
        ttft_ms = sum.prefill_seconds * 1000.0;
    }
    std::snprintf(lbuf, sizeof(lbuf), "Time to First Token (TTFT)");
    std::snprintf(rbuf, sizeof(rbuf), "%.2f ms", ttft_ms);
    std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);

    // Prompt Processing (Prefill)
    double prefill_toks = (sum.prefill_seconds > 0.0) ? ((double) sum.n_prompt / sum.prefill_seconds) : 0.0;
    std::snprintf(lbuf, sizeof(lbuf), "Prompt Processing (Prefill)");
    std::snprintf(rbuf, sizeof(rbuf), "%d tokens @ %.2f tok/s", sum.n_prompt, prefill_toks);
    std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);

    // Token Generation (Decode)
    double decode_toks = (decode_sec > 0.0) ? ((double) sum.n_generated / decode_sec) : 0.0;
    std::snprintf(lbuf, sizeof(lbuf), "Token Generation (Decode)");
    std::snprintf(rbuf, sizeof(rbuf), "%d tokens @ %.2f tok/s", sum.n_generated, decode_toks);
    std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);

    // Speculative Acceptance & Mean Draft Length (when active / drafted > 0)
    if (sum.mtp_drafted > 0) {
        double accept_pct = (100.0 * (double) sum.mtp_accepted) / (double) sum.mtp_drafted;
        std::snprintf(lbuf, sizeof(lbuf), "Speculative Acceptance");
        std::snprintf(rbuf, sizeof(rbuf), "%.2f%% (%lld accepted / %lld)", accept_pct, (long long) sum.mtp_accepted, (long long) sum.mtp_drafted);
        std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);

        double avg_len = (sum.mtp_decodes > 0) ? ((double) sum.n_generated / (double) sum.mtp_decodes) : 1.0;
        std::snprintf(lbuf, sizeof(lbuf), "Mean Draft Length");
        std::snprintf(rbuf, sizeof(rbuf), "%.2f tokens/step", avg_len);
        std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);
    }

    // Compact Rollback (only when checkpoints were taken; zero on the cr_depth=-1 default)
    if (sum.mtp_ckpt_saves > 0) {
        std::snprintf(lbuf, sizeof(lbuf), "Compact Rollback");
        std::snprintf(rbuf, sizeof(rbuf), "%lld ckpts, %lld replays, %lld host fb",
                      (long long) sum.mtp_ckpt_saves, (long long) sum.mtp_replays,
                      (long long) sum.mtp_host_fallback);
        std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);
    }

    // Total Response Time
    int total_tokens = sum.n_prompt + sum.n_generated;
    std::snprintf(lbuf, sizeof(lbuf), "Total Response Time");
    if (total_sec < 60.0) {
        std::snprintf(rbuf, sizeof(rbuf), "%.2f s (%d tokens total)", total_sec, total_tokens);
    } else {
        std::snprintf(rbuf, sizeof(rbuf), "%.1f m (%d tokens total)", total_sec / 60.0, total_tokens);
    }
    std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);

    // MoE VRAM ARC Cache Hit Rate
    if (sum.cache_hit_pct >= 0.0) {
        std::snprintf(lbuf, sizeof(lbuf), "MoE VRAM ARC Cache Hit Rate");
        std::snprintf(rbuf, sizeof(rbuf), "%.1f%%", sum.cache_hit_pct);
        std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);
    }

    std::fprintf(stderr, "└──────────────────────────────┴──────────────────────────────┘\n\n");
    std::fflush(stderr);
}

static void handle_completions(socket_t fd, const HttpRequest & req, ServerState & state, bool chat) {

    if (!state.session) {
        send_json_error(fd, 500, "Model not loaded", false);
        return;
    }

    // Parse the OpenAI wire body. /v1/chat/completions (messages) and /v1/completions
    // (prompt) both land here; the parser is nlohmann-based and unit-tested, so code
    // blocks full of '}' or literal "role"/"content" in the text can no longer corrupt
    // the message boundary scan (tests/openai_parse_test.cpp).
    bmoe::openai::ParsedBody pb;
    if (!bmoe::openai::parse_body(req.body, state.session_cfg.sampling, pb, state.session_cfg.n_predict)) {
        send_json_error(fd, 400, pb.error.c_str(), false);
        return;
    }
    // Cap n_predict so web UI requests (e.g. AnythingLLM default 4096) never exceed session context size
    const int max_allowed = (std::max)(32, state.session->n_ctx() - 128);
    if (pb.n_predict > max_allowed) pb.n_predict = max_allowed;

    const bool stream = pb.stream;

    // Build generate request
    GenerateRequest greq;
    greq.prompt = pb.prompt;
    greq.messages = std::move(pb.messages);
    greq.images = std::move(pb.images);
    greq.n_predict = pb.n_predict;
    greq.render_text = !stream || chat; // a chat stream needs the parsed text/reasoning deltas
    greq.think = !state.srv_cfg.disable_think;
    greq.has_sampling = pb.has_sampling || pb.sampling.temp > 0.0f;
    greq.sampling = pb.sampling;
    // Chat is multi-turn: hand the engine the FULL conversation and keep the KV, so the
    // n_common prefix match serves this turn from the previous one's cache — the README's
    // "multi-turn KV reuse", which the old last-user-message extraction made unreachable.
    // A divergent history (new chat, rewind) is trimmed back to the shared prefix by the
    // engine. Raw completions stay stateless: clear_kv stays true, each prompt independent.
    greq.clear_kv = !pb.is_chat;
    long created = static_cast<long>(std::time(nullptr));


    // Serialize inference: the session runs one generation at a time (the engine itself is not
    // thread-safe across generate() calls). try_lock so a second generation request while one is
    // streaming gets a clean 429 instead of blocking this worker — metadata endpoints never
    // reach this mutex and answer immediately regardless.
    std::unique_lock<std::mutex> gen_lock(state.generate_mtx, std::try_to_lock);
    if (!gen_lock.owns_lock()) {
        send_json_error(fd, 429, "Another generation is in progress; try again shortly", req.keep_alive);
        return;
    }

    auto req_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point first_token_time;
    bool got_first_token = false;

    if (!stream) {
        auto on_token_probe = [&](const TokenMetrics &) {
            if (!got_first_token) {
                first_token_time = std::chrono::steady_clock::now();
                got_first_token = true;
            }
        };

        auto result = state.session->generate(greq, on_token_probe, nullptr);
        auto req_end = std::chrono::steady_clock::now();
        double total_sec = std::chrono::duration<double>(req_end - req_start).count();
        double ttft_ms = got_first_token
                             ? std::chrono::duration<double, std::milli>(first_token_time - req_start).count()
                             : (result.summary.prefill_seconds * 1000.0);
        double decode_sec = got_first_token
                                ? std::chrono::duration<double>(req_end - first_token_time).count()
                                : (total_sec - (ttft_ms / 1000.0));
        if (decode_sec < 0.0) decode_sec = 0.0;

        if (!result) {
            send_json_error(fd, 500, result.error.c_str(), false);
            return;
        }

        print_lmstudio_telemetry(result.summary, ttft_ms, decode_sec, total_sec);

        std::string reply_content = result.generated_text;
        // generated_text already has the reasoning stripped by the engine's chat parser.
        // NEVER fall back to reasoning_text here: shoving the thinking span into content
        // is exactly the leak a chat UI would show as visible answer text.

        std::string id_prefix = chat ? "chatcmpl" : "cmpl";
        std::string object = chat ? "chat.completion" : "text_completion";

        std::string body;
        if (chat) {
            body = "{\"id\":\"" + id_prefix + "-" + std::to_string(result.summary.n_prompt) +
                   "\","
                   "\"object\":\"" +
                   object +
                   "\","
                   "\"created\":" +
                   std::to_string(created) +
                   ","
                   "\"model\":\"bmoe\","
                   "\"choices\":[{"
                   "\"index\":0,"
                   "\"message\":{"
                   "\"role\":\"assistant\","
                   "\"content\":\"" +
                   json_escape(reply_content) +
                   "\""
                   + (result.reasoning_text.empty()
                          ? ""
                          : ",\"reasoning_content\":\"" + json_escape(result.reasoning_text) + "\"")
                   + "},"
                   "\"finish_reason\":\"stop\""
                   "}],"
                   "\"usage\":{"
                   "\"prompt_tokens\":" +
                   std::to_string(result.summary.n_prompt) +
                   ","
                   "\"completion_tokens\":" +
                   std::to_string(result.summary.n_generated) +
                   ","
                   "\"total_tokens\":" +
                   std::to_string(result.summary.n_prompt + result.summary.n_generated) + "}}";
        } else {
            body = "{\"id\":\"" + id_prefix + "-" + std::to_string(result.summary.n_prompt) +
                   "\","
                   "\"object\":\"" +
                   object +
                   "\","
                   "\"created\":" +
                   std::to_string(created) +
                   ","
                   "\"model\":\"bmoe\","
                   "\"choices\":[{"
                   "\"text\":\"" +
                   json_escape(reply_content) +
                   "\","
                   "\"index\":0,"
                   "\"finish_reason\":\"stop\","
                   "\"logprobs\":null"
                   "}],"
                   "\"usage\":{"
                   "\"prompt_tokens\":" +
                   std::to_string(result.summary.n_prompt) +
                   ","

                   "\"completion_tokens\":" +
                   std::to_string(result.summary.n_generated) +
                   ","
                   "\"total_tokens\":" +
                   std::to_string(result.summary.n_prompt + result.summary.n_generated) + "}}";
        }
        send_response(fd, 200, "OK", "application/json", body, false);
        return;
    }

    // ── Streaming (SSE) ─────────────────────────────────────────────────
    send_sse_headers(fd);

    std::string id_prefix = chat ? "chatcmpl" : "cmpl";
    std::string object = chat ? "chat.completion.chunk" : "text_completion";

    // For chat, send the role first
    if (chat) {
        std::string data = "{\"id\":\"" + id_prefix + "-" + std::to_string(created) +
                           "\","
                           "\"object\":\"" +
                           object +
                           "\","
                           "\"created\":" +
                           std::to_string(created) +
                           ","
                           "\"model\":\"bmoe\","
                           "\"choices\":[{"
                           "\"index\":0,"
                           "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                           "\"finish_reason\":null"
                           "}]}";
        send_sse(fd, data);
    }

    // How much of each cumulative field has already been streamed. The engine re-parses the
    // whole generation on every token (render_text=true), so m.text/m.reasoning grow monotonically;
    // each SSE chunk carries only the not-yet-sent suffix, and the reasoning markers the parser
    // consumed never appear in either stream.
    std::string sent_content;
    std::string sent_reasoning;

    auto on_token = [&](const TokenMetrics & m) {
        if (!got_first_token) {
            first_token_time = std::chrono::steady_clock::now();
            got_first_token = true;
        }

        std::string data = "{\"id\":\"" + id_prefix + "-" + std::to_string(created) +
                           "\","
                           "\"object\":\"" +
                           object +
                           "\","
                           "\"created\":" +
                           std::to_string(created) +
                           ","
                           "\"model\":\"bmoe\","
                           "\"choices\":[{";

        if (chat) {
            std::string new_reasoning = m.reasoning.size() > sent_reasoning.size()
                                            ? m.reasoning.substr(sent_reasoning.size())
                                            : std::string();
            std::string new_content = m.text.size() > sent_content.size()
                                          ? m.text.substr(sent_content.size())
                                          : std::string();
            sent_reasoning = m.reasoning;
            sent_content = m.text;

            data += "\"index\":0,\"delta\":{";
            bool first = true;
            if (!new_reasoning.empty()) {
                data += "\"reasoning_content\":\"" + json_escape(new_reasoning) + "\"";
                first = false;
            }
            if (!new_content.empty()) {
                if (!first) data += ",";
                data += "\"content\":\"" + json_escape(new_content) + "\"";
                first = false;
            }
            if (first) data += "\"content\":\"\"";
            data += "},\"finish_reason\":null";
        } else {
            data += "\"index\":0,\"delta\":{\"text\":\"" + json_escape(m.piece) + "\"},\"finish_reason\":null";
        }

        data += "}]}";
        send_sse(fd, data);
    };

    auto result = state.session->generate(greq, on_token, nullptr);
    auto req_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(req_end - req_start).count();
    double ttft_ms = got_first_token
                         ? std::chrono::duration<double, std::milli>(first_token_time - req_start).count()
                         : (result.summary.prefill_seconds * 1000.0);
    double decode_sec = got_first_token
                            ? std::chrono::duration<double>(req_end - first_token_time).count()
                            : (total_sec - (ttft_ms / 1000.0));
    if (decode_sec < 0.0) decode_sec = 0.0;

    if (result) {
        print_lmstudio_telemetry(result.summary, ttft_ms, decode_sec, total_sec);

        // Final chunk with usage and finish_reason
        std::string data = "{\"id\":\"" + id_prefix + "-" + std::to_string(created) +
                           "\","
                           "\"object\":\"" +
                           object +
                           "\","
                           "\"created\":" +
                           std::to_string(created) +
                           ","
                           "\"model\":\"bmoe\","
                           "\"choices\":[{"
                           "\"index\":0,"
                           "\"delta\":{},"
                           "\"finish_reason\":\"stop\""
                           "}],"
                           "\"usage\":{"
                           "\"prompt_tokens\":" +
                           std::to_string(result.summary.n_prompt) +
                           ","
                           "\"completion_tokens\":" +
                           std::to_string(result.summary.n_generated) +
                           ","
                           "\"total_tokens\":" +
                           std::to_string(result.summary.n_prompt + result.summary.n_generated) + "}}";
        send_sse(fd, data);
        send_sse_done(fd);
    } else {
        // A decode/prefill failure after the stream already started: terminate the SSE stream
        // with a proper error event, NOT a bare JSON body mid-stream and NOT a false "stop". The
        // OpenAI chunk shape is kept (delta + finish_reason) with an explicit error object so a
        // strict client can distinguish a completed run from a failed one; finish_reason "error"
        // is what OpenAI-compatible probes listen for.
        std::string err_data = "{\"id\":\"" + id_prefix + "-" + std::to_string(created) +
                               "\","
                               "\"object\":\"" +
                               object +
                               "\","
                               "\"created\":" +
                               std::to_string(created) +
                               ","
                               "\"model\":\"bmoe\","
                               "\"choices\":[{"
                               "\"index\":0,"
                               "\"delta\":{},"
                               "\"finish_reason\":\"error\""
                               "}],"
                               "\"error\":{\"message\":\"" +
                               json_escape(result.error) +
                               "\",\"type\":\"api_error\"}}";
        send_sse(fd, err_data);
        send_sse_done(fd);
    }
}

// ── Connection handling ──────────────────────────────────────────────────────

// Read the full HTTP request from a blocking socket: headers + body.
// Returns false if the connection closed or the request was too large.
// The body cap is sized for multimodal payloads: a single base64 image routinely lands
// in the 2-8 MiB range, and the old 1 MiB limit silently dropped those connections
// before vision processing could begin.
static constexpr size_t MAX_BODY_SIZE = 64 * 1024 * 1024;

static bool read_request(socket_t fd, std::string & raw) {
    char buf[65536];
    while (true) {
        int n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
#if defined(_WIN32)
            if (n < 0 && WSAGetLastError() == WSAEINTR) continue;
#else
            if (n < 0 && errno == EINTR) continue;
#endif
            return false; // connection closed or error
        }
        raw.append(buf, (size_t) n);

        // Check if we have the full headers
        size_t hdr_end = raw.find("\r\n\r\n");
        if (hdr_end == std::string::npos) {
            if (raw.size() > 65536) return false; // headers too large
            continue;                             // need more data
        }

        // Parse Content-Length from headers
        size_t body_start = hdr_end + 4;
        std::string headers = raw.substr(0, hdr_end);
        size_t cl_pos = headers.find("Content-Length:");
        if (cl_pos == std::string::npos) cl_pos = headers.find("content-length:");
        if (cl_pos != std::string::npos) {
            size_t colon = headers.find(':', cl_pos);
            if (colon != std::string::npos) {
                size_t cl = (size_t) std::atoll(headers.c_str() + colon + 1);
                if (raw.size() - body_start >= cl) return true; // full body received
                // Need more body data
                if (raw.size() > MAX_BODY_SIZE) return false; // body too large
                continue;
            } else {
                return true; // malformed header, treat as header-only
            }
        } else {
            // No Content-Length: return what we have
            return true;
        }
    }
}

// Process one HTTP connection (may serve multiple requests if keep-alive).
static void process_connection(socket_t fd, ServerState & state) {
    while (true) {
        std::string raw;
        if (!read_request(fd, raw)) return; // connection closed

        HttpRequest req;

        if (!parse_http_request(raw, req)) {
            send_json_error(fd, 400, "Bad request", false);
            return;
        }

        handle_request(fd, req, state);

        if (!req.keep_alive) return;
        // For keep-alive, loop back for the next request
    }
}

// ── Server lifecycle ─────────────────────────────────────────────────────────

static void print_usage(const char * argv0) {
    std::printf("usage: %s -m <model.gguf> [options]\n"
                "\n"
                "  -m, --model PATH        gguf model (required)\n"
                "      --mmproj PATH       multimodal projector (mmproj.gguf) for vision models\n"
                "      --no-mmproj-offload disable GPU offload for multimodal projector (keep in RAM)\n"
                "      --mmproj-offload [on|off] enable/disable GPU offload for mmproj (default: on)\n"
                "      --port N            HTTP server port (default 8080)\n"
                "      --host ADDR         bind address (default 127.0.0.1; use 0.0.0.0 for\n"
                "                          remote access)\n"
                "      --api-key SECRET    require Authorization: Bearer SECRET on every endpoint\n"
                "                          (mandatory when binding a non-loopback interface)\n"
                "\n"
                "  All bmoe-cli streaming and decoding flags are supported:\n"
                "  -t, --threads, -ngl, --n-gpu-layers, -nkqv, --no-offload-kqv, -c, --ctx-size\n"
                "  --ubatch, --batch-size, --moe-stream, --cache-mb, --cache-floor-mb, --cache-ceil-mb,\n"
                "  --io-threads, --no-odirect, --dense-weights,\n"
                "  --prefetch, --predict-prefetch, --drop-cold-experts,\n"
                "  --overlap, --io-two-wave, --route-ahead,\n"
                "  --temp, --top-k, --top-p, --seed,\n"
                "  --mtp, --ngram, --draft, --mtp-p-min, --spec-mtp-cr-depth, --spec-draft-adaptive, --ngram-min-match,\n"
                "  --model-draft FILE, --n-gpu-layers-draft N, --spec-type draft-mtp|draft,\n"
                "  --n-expert-used, --load-all\n"
                "  --no-think           disable model thinking\n"
                "\n"
                "  -h, --help              show this text and exit\n"
                "      --version           print the engine version and exit\n"
                "\n"
                "API endpoints:\n"
                "  GET  /v1/models           list loaded model\n"
                "  POST /v1/completions      text completion (OpenAI-compatible)\n"
                "  POST /v1/chat/completions chat completion (OpenAI-compatible)\n"
                "\n"
                "  Both POST endpoints accept stream=true for SSE token streaming.\n"
                "\n"
                "Environment:\n"
                "  BMOE_SERVER_PORT  override --port\n"
                "  BMOE_SERVER_HOST  override --host\n"
                "  BMOE_SERVER_API_KEY  override --api-key\n"
                "  All BMOE_* env vars from bmoe-cli also apply\n",
                argv0);
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(hOut, &mode)) {
            SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT);
        }
    }
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    if (hErr != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(hErr, &mode)) {
            SetConsoleMode(hErr, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT);
        }
    }
#endif

    RunConfig cfg;
    ServerConfig srv;
    bool n_predict_set = false; // -n/--n-predict explicitly given: feeds the wire fallback

    std::set<std::string> seen;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        seen.insert(a);
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(1);
            }
            return argv[++i];
        };

        if (a == "-m" || a == "--model")
            cfg.model_path = next("-m");
        else if (a == "--mmproj")
            cfg.mmproj_path = next("--mmproj");
        else if (a == "--no-mmproj-offload")
            cfg.mmproj_offload = false;
        else if (a == "--mmproj-offload") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string v = argv[++i];
                cfg.mmproj_offload = (v == "1" || v == "true" || v == "on");
            } else {
                cfg.mmproj_offload = true;
            }
        }
        else if (a == "--port")
            srv.port = std::atoi(next("--port"));
        else if (a == "--host")
            srv.host = next("--host");
        else if (a == "--api-key")
            srv.api_key = next("--api-key");
        else if (a == "-p" || a == "--prompt") {
            next("-p"); // ignored in server mode
        } else if (a == "-n" || a == "--n-predict") {
            cfg.n_predict = std::atoi(next("-n"));
            n_predict_set = true;
        }
        else if (a == "-t" || a == "--threads")
            cfg.n_threads = std::atoi(next("-t"));
        else if (a == "-ngl" || a == "--n-gpu-layers")
            cfg.n_gpu_layers = std::atoi(next("-ngl"));
        else if (a == "-c" || a == "--ctx-size")
            cfg.n_ctx = std::atoi(next("-c"));

        else if (a == "--ubatch")
            cfg.n_ubatch = std::atoi(next("--ubatch"));
        else if (a == "--batch-size")
            cfg.n_batch = std::atoi(next("--batch-size"));
        else if (a == "--n-expert-used")
            cfg.n_expert_used = std::atoi(next("--n-expert-used"));
        else if (a == "--temp")
            cfg.sampling.temp = (float) std::atof(next("--temp"));
        else if (a == "--top-k")
            cfg.sampling.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p")
            cfg.sampling.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--min-p")
            cfg.sampling.min_p = (float) std::atof(next("--min-p"));
        else if (a == "--repeat-penalty" || a == "--penalty-repeat")
            cfg.sampling.repeat_penalty = (float) std::atof(next(a.c_str()));
        else if (a == "--presence-penalty" || a == "--penalty-present")
            cfg.sampling.presence_penalty = (float) std::atof(next(a.c_str()));
        else if (a == "--frequency-penalty" || a == "--penalty-freq")
            cfg.sampling.frequency_penalty = (float) std::atof(next(a.c_str()));
        else if (a == "--repeat-last-n" || a == "--penalty-last-n")
            cfg.sampling.repeat_last_n = std::atoi(next(a.c_str()));
        else if (a == "--cpu-moe")
            cfg.moe.cpu_moe = true;
        else if (a == "--n-cpu-moe") {
            cfg.moe.cpu_moe = true;
        } else if (a == "--load-mode") {
            std::string lm = next("--load-mode");
            if (lm == "mmap") {
                cfg.moe.cpu_moe = true;
                cfg.moe.dense_weights = DenseWeightsMode::Mmap;
            }
        }
        else if (a == "--seed")
            cfg.sampling.seed = (uint32_t) std::strtoul(next("--seed"), nullptr, 10);
        else if (a == "--mtp" || a == "--ngram") {
            const DraftSource want = a == "--mtp" ? DraftSource::mtp : DraftSource::ngram;
            if (cfg.spec.enabled() && cfg.spec.source != want) {
                std::fprintf(stderr, "bmoe-server: --mtp and --ngram are exclusive; choose one.\n");
                return 2;
            }
            cfg.spec.source = want;
        } else if (a == "--draft")
            cfg.spec.draft_max = std::atoi(next("--draft"));
        else if (a == "--mtp-p-min")
            cfg.spec.draft_p_min = (float) std::atof(next("--mtp-p-min"));
        else if (a == "--spec-mtp-cr-depth")
            cfg.spec.cr_depth = std::atoi(next("--spec-mtp-cr-depth"));
        else if (a == "--spec-draft-adaptive")
            cfg.spec.draft_adaptive = true;
        else if (a == "-md" || a == "--model-draft") {
            cfg.spec.model_draft = next("-md");
            if (cfg.spec.source == DraftSource::none)
                cfg.spec.source = DraftSource::mtp;
        } else if (a == "-ngld" || a == "--n-gpu-layers-draft")
            cfg.spec.n_gl_draft = std::atoi(next("-ngld"));
        else if (a == "--spec-type") {
            const std::string st = next("--spec-type");
            if (st != "draft-mtp" && st != "draft") {
                std::fprintf(stderr, "bmoe-server: --spec-type must be 'draft-mtp' or 'draft'\n");
                return 2;
            }
            if (cfg.spec.enabled() && cfg.spec.source != DraftSource::mtp) {
                std::fprintf(stderr, "bmoe-server: --spec-type conflicts with --ngram; choose one.\n");
                return 2;
            }
            cfg.spec.source = DraftSource::mtp;
            cfg.spec.driver = st == "draft" ? SpecDriver::simple : SpecDriver::mtp;
        }
        else if (a == "--ngram-min-match")
            cfg.spec.ngram_min_match = std::atoi(next("--ngram-min-match"));
        else if (a == "-ctk" || a == "--cache-type-k")
            cfg.cache_type_k = next("-ctk");
        else if (a == "-ctv" || a == "--cache-type-v")
            cfg.cache_type_v = next("-ctv");
        else if (a == "-fa" || a == "--flash-attn")
            cfg.flash_attn = true;
        else if (a == "--no-flash-attn" || a == "--no-fa")
            cfg.flash_attn = false;
        else if (a == "-nkqv" || a == "--no-offload-kqv" || a == "--no-kv-offload" || a == "-nkvo")
            cfg.no_kv_offload = true;
        else if (a == "--no-think") {
            cfg.think = false;
            srv.disable_think = true;
        }

        // --chat is now always enabled (chat template applied to messages)
        else if (a == "--moe-stream")
            cfg.moe.enabled = true;
        else if (a == "--cache-mb") {
            const std::string v = next("--cache-mb");
            if (v == "auto")
                cfg.moe.cache_auto = true;
            else
                cfg.moe.cache_mb = std::atoi(v.c_str());
        } else if (a == "--cache-floor-mb")
            cfg.moe.cache_floor_mb = std::atoi(next("--cache-floor-mb"));
        else if (a == "--cache-ceil-mb")
            cfg.moe.cache_ceil_mb = std::atoi(next("--cache-ceil-mb"));
        else if (a == "--io-threads")
            cfg.moe.io_threads = std::atoi(next("--io-threads"));
        else if (a == "--no-odirect")
            cfg.moe.o_direct = false;
        else if (a == "--n-pinned-layers" || a == "--pinned-layers")
            cfg.moe.n_pinned_layers = std::atoi(next(a.c_str()));
        else if (a == "--dense-weights") {

            const std::string m = next("--dense-weights");
            if (m == "mmap")
                cfg.moe.dense_weights = DenseWeightsMode::Mmap;
            else if (m == "warm")
                cfg.moe.dense_weights = DenseWeightsMode::Warmed;
            else if (m == "anon")
                cfg.moe.dense_weights = DenseWeightsMode::Anonymous;
            else if (m == "ahwb")
                cfg.moe.dense_weights = DenseWeightsMode::Pinned;
            else {
                std::fprintf(stderr, "bmoe-server: --dense-weights expects mmap|warm|anon|ahwb\n");
                return 2;
            }
        } else if (a == "--load-all")
            cfg.moe.load_all = true;
        else if (a == "--force-cache")
            cfg.moe.force_cache = true;
        else if (a == "--overlap")
            cfg.moe.overlap = true;
        else if (a == "--io-two-wave")
            cfg.moe.io_two_wave = true;
        else if (a == "--prefetch")
            cfg.moe.prefetch_layers = std::atoi(next("--prefetch"));
        else if (a == "--prefetch-sync")
            cfg.moe.prefetch_sync = true;
        else if (a == "--drop-cold-experts")
            cfg.moe.drop_cold_frac = (float) std::atof(next("--drop-cold-experts"));
        else if (a == "--drop-no-renorm")
            cfg.moe.drop_renorm = false;
        else if (a == "--drop-in-prefill")
            cfg.moe.drop_prefill = true;
        else if (a == "--route-ahead")
            cfg.moe.route_ahead = std::atoi(next("--route-ahead"));
        else if (a == "--predict-prefetch")
            cfg.moe.predict_prefetch = true;
        else if (a == "--predict-spec-max")
            cfg.moe.predict_spec_max = std::atoi(next("--predict-spec-max"));
        else if (a == "--no-cg" || a == "--no-cuda-graphs") {
#if defined(_WIN32)
            _putenv("GGML_CUDA_DISABLE_GRAPHS=1");
#else
            setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
        }
        else if (a == "--list-archs") {

            std::printf("supported MoE architectures:\n");
            for (int k = 0; k < n_moe_recipes(); ++k)
                std::printf("  %s\n", moe_recipe_at(k)->arch);
            return 0;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--version") {
            std::printf("%s\n", bmoe::version());
            return 0;
        } else {
            std::fprintf(stderr, "bmoe-server: unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Env overrides
    const char * env_port = std::getenv("BMOE_SERVER_PORT");
    if (env_port && *env_port) srv.port = std::atoi(env_port);
    const char * env_host = std::getenv("BMOE_SERVER_HOST");
    if (env_host && *env_host) srv.host = env_host;
    const char * env_key = std::getenv("BMOE_SERVER_API_KEY");
    if (env_key && *env_key && srv.api_key.empty()) srv.api_key = env_key;

    // Binding a non-loopback interface without a key is how the GPU ends up exposed to the
    // local network; refuse to be quiet about it even when authentication is on.
    const bool loopback = srv.host == "127.0.0.1" || srv.host == "localhost" || srv.host == "::1";
    if (!loopback && srv.api_key.empty()) {
        std::fprintf(stderr, "bmoe-server: WARNING: binding %s with no --api-key — every client on the network\n"
                             "             can drive inference and read responses. Set --api-key to require auth.\n",
                     srv.host.c_str());
    }

    // bmoe-cli env overrides also apply
    if (!seen.count("--cache-mb")) {
        const char * v = std::getenv("BMOE_CACHE_MB");
        if (v && *v) cfg.moe.cache_mb = std::atoi(v);
    }
    if (!seen.count("--io-threads")) {
        const char * v = std::getenv("BMOE_IO_THREADS");
        if (v && *v) cfg.moe.io_threads = std::atoi(v);
    }

    if (cfg.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // OpenAI-compatible servers always serve chat-format requests (pi sends
    // messages arrays). Always enable chatml so the model's chat template is
    // applied to the conversation, regardless of whether --chat was passed.
    cfg.chatml = true;

    ValidationResult vr = validate(cfg);
    if (!vr) {
        std::fprintf(stderr, "config error: %s\n", vr.error.c_str());
        return 1;
    }

    // ── Open the session ──────────────────────────────────────────────
    std::fprintf(stderr, "bmoe-server: loading model %s ...\n", cfg.model_path.c_str());

    const SessionConfig sc_base = session_config_from(cfg);
    SessionConfig sc = sc_base;
    if (n_predict_set) sc.n_predict = cfg.n_predict; // -n only when explicitly given (RunConfig default 128 is CLI semantics)
    std::string error;
    std::unique_ptr<Session> session = Session::open(sc, error, nullptr, nullptr, nullptr);
    if (!session) {
        std::fprintf(stderr, "bmoe-server: failed to load model: %s\n", error.c_str());
        return 1;
    }

    std::fprintf(stderr, "bmoe-server: model loaded: arch=%s, n_ctx=%d, think_ctl=%s, n_expert_used=%d\n",
                 session->arch().c_str(), session->n_ctx(), think_control_name(session->think_control()),
                 session->n_expert_used());
    std::fprintf(stderr, "bmoe-server: listening on http://%s:%d\n", srv.host.c_str(), srv.port);

    // ── Create the listening socket ───────────────────────────────────
#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::fprintf(stderr, "bmoe-server: WSAStartup failed\n");
        return 1;
    }
#endif

    socket_t listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (IS_INVALID_SOCKET(listen_fd)) {
#if defined(_WIN32)
        std::fprintf(stderr, "bmoe-server: socket() failed with error %d\n", WSAGetLastError());
        WSACleanup();
#else
        std::fprintf(stderr, "bmoe-server: socket() failed: %s\n", std::strerror(errno));
#endif
        return 1;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *) &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) srv.port);

    if (srv.host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, srv.host.c_str(), &addr.sin_addr) != 1) {
            std::fprintf(stderr, "bmoe-server: invalid host: %s\n", srv.host.c_str());
            close_socket(listen_fd);
#if defined(_WIN32)
            WSACleanup();
#endif
            return 1;
        }
    }

    if (bind(listen_fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
#if defined(_WIN32)
        std::fprintf(stderr, "bmoe-server: bind(%s:%d) failed with error %d\n", srv.host.c_str(), srv.port,
                     WSAGetLastError());
#else
        std::fprintf(stderr, "bmoe-server: bind(%s:%d) failed: %s\n", srv.host.c_str(), srv.port,
                     std::strerror(errno));
#endif
        close_socket(listen_fd);
#if defined(_WIN32)
        WSACleanup();
#endif
        return 1;
    }

    if (listen(listen_fd, srv.max_connections) < 0) {
#if defined(_WIN32)
        std::fprintf(stderr, "bmoe-server: listen() failed with error %d\n", WSAGetLastError());
#else
        std::fprintf(stderr, "bmoe-server: listen() failed: %s\n", std::strerror(errno));
#endif
        close_socket(listen_fd);
#if defined(_WIN32)
        WSACleanup();
#endif
        return 1;
    }

    // ── Multi-threaded server loop ─────────────────────────────────────
    // Each accepted connection runs in its own detached worker thread, so a long SSE
    // generation on one client no longer blocks /v1/models, /health or other clients.
    // The inference path is serialized inside handle_completions via generate_mtx
    // (try_lock -> 429 when busy); metadata endpoints never touch it and stay fast.
    ServerState state;
    state.session = std::move(session);
    state.session_cfg = sc;
    state.srv_cfg = srv;

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        socket_t client_fd = accept(listen_fd, (struct sockaddr *) &client_addr, &client_len);
        if (IS_INVALID_SOCKET(client_fd)) {
#if defined(_WIN32)
            if (WSAGetLastError() == WSAEINTR) continue;
            std::fprintf(stderr, "bmoe-server: accept() error %d\n", WSAGetLastError());
#else
            if (errno == EINTR) continue;
            std::fprintf(stderr, "bmoe-server: accept() error: %s\n", std::strerror(errno));
#endif
            continue;
        }

        // Detached worker: `state` lives for the process lifetime (this loop never exits),
        // so the reference stays valid for the connection. The thread owns the socket and
        // closes it when done.
        std::thread([client_fd, &state] {
            try {
                process_connection(client_fd, state);
            } catch (const std::exception & ex) {
                std::fprintf(stderr, "bmoe-server: exception during connection: %s\n", ex.what());
            } catch (...) {
                std::fprintf(stderr, "bmoe-server: unknown exception during connection\n");
            }
            close_socket(client_fd);
        }).detach();
    }

    close_socket(listen_fd);
#if defined(_WIN32)
    WSACleanup();
#endif
    std::fprintf(stderr, "bmoe-server: shutting down\n");
    return 0;
}

