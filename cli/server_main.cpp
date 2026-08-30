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


// ── Minimal JSON utilities (hand-rolled, dependency-free) ────────────────────

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

static size_t json_find_key(const std::string & json, const char * key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t k = json.find(pat);
    if (k == std::string::npos) return std::string::npos;
    size_t c = json.find(':', k + pat.size());
    if (c == std::string::npos) return std::string::npos;
    return c + 1;
}

static std::string json_extract_string(const std::string & json, const char * key, const std::string & dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    if (p >= json.size() || json[p] != '"') return dflt;
    ++p;
    std::string raw;
    for (; p < json.size(); ++p) {
        if (json[p] == '\\' && p + 1 < json.size()) {
            raw += json[p];
            raw += json[p + 1];
            ++p;
        } else if (json[p] == '"') {
            break;
        } else {
            raw += json[p];
        }
    }
    // Unescape
    std::string out;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            switch (raw[++i]) {
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            default:
                out += raw[i];
                break;
            }
        } else {
            out += raw[i];
        }
    }
    return out;
}

static int json_extract_int(const std::string & json, const char * key, int dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    return std::atoi(json.c_str() + p);
}

static double json_extract_double(const std::string & json, const char * key, double dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    return std::atof(json.c_str() + p);
}

static bool json_extract_bool(const std::string & json, const char * key, bool dflt) {
    size_t p = json_find_key(json, key);
    if (p == std::string::npos) return dflt;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n'))
        ++p;
    return json.compare(p, 4, "true") == 0;
}

// Extract the last user message content from a chat messages array.
// Handles both string content ("content":"text") and array content.
static std::string extract_last_user_message(const std::string & body) {
    size_t msgs = body.find("\"messages\"");
    if (msgs == std::string::npos) return "";

    std::string last_content;
    size_t pos = msgs;
    while (true) {
        size_t obj_start = body.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end = body.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj = body.substr(obj_start, obj_end - obj_start + 1);

        size_t role_pos = obj.find("\"role\"");
        bool is_user = false;
        if (role_pos != std::string::npos) {
            size_t r_colon = obj.find(':', role_pos + 6);
            if (r_colon != std::string::npos) {
                if (obj.find("\"user\"", r_colon) != std::string::npos) {
                    is_user = true;
                }
            }
        }

        size_t content_pos = obj.find("\"content\"");
        if (content_pos != std::string::npos) {
            size_t cp = obj.find(':', content_pos + 9);
            if (cp != std::string::npos) {
                ++cp;
                while (cp < obj.size() && (obj[cp] == ' ' || obj[cp] == '\t' || obj[cp] == '\n' || obj[cp] == '\r'))
                    ++cp;
                if (cp < obj.size()) {
                    if (obj[cp] == '"') {
                        ++cp;
                        std::string raw;
                        for (; cp < obj.size(); ++cp) {
                            if (obj[cp] == '\\' && cp + 1 < obj.size()) {
                                raw += obj[cp];
                                raw += obj[cp + 1];
                                ++cp;
                            } else if (obj[cp] == '"') {
                                break;
                            } else {
                                raw += obj[cp];
                            }
                        }
                        std::string content;
                        for (size_t i = 0; i < raw.size(); ++i) {
                            if (raw[i] == '\\' && i + 1 < raw.size()) {
                                switch (raw[++i]) {
                                case 'n': content += '\n'; break;
                                case 'r': content += '\r'; break;
                                case 't': content += '\t'; break;
                                case '"': content += '"'; break;
                                case '\\': content += '\\'; break;
                                default: content += raw[i]; break;
                                }
                            } else {
                                content += raw[i];
                            }
                        }
                        if (is_user) last_content = content;
                    }
                }
            }
        }

        pos = obj_end + 1;
    }
    return last_content;
}


// Extract image URLs from the messages array (OpenAI-compatible format).
// Returns a vector of image URLs (data URLs or HTTPS URLs).
static std::vector<std::string> extract_images(const std::string & body) {
    std::vector<std::string> images;
    size_t msgs = body.find("\"messages\"");
    if (msgs == std::string::npos) return images;

    size_t pos = msgs;
    while (true) {
        // Find "type":"image_url"
        size_t type_pos = body.find("\"type\"", pos);
        if (type_pos == std::string::npos) break;
        size_t type_val = type_pos + 7; // skip "type"
        while (type_val < body.size() &&
               (body[type_val] == ' ' || body[type_val] == ':' || body[type_val] == '\t' || body[type_val] == '\n'))
            ++type_val;

        bool is_image = body.compare(type_val, 12, "\"image_url\"") == 0;

        if (is_image) {
            // Find the "url" field within this image_url object
            size_t url_pos = body.find("\"url\"", type_pos);
            if (url_pos != std::string::npos) {
                size_t url_val = url_pos + 6; // skip "url"
                while (url_val < body.size() &&
                       (body[url_val] == ' ' || body[url_val] == ':' || body[url_val] == '\t' || body[url_val] == '\n'))
                    ++url_val;
                if (url_val < body.size() && body[url_val] == '"') {
                    ++url_val;
                    std::string url;
                    for (; url_val < body.size(); ++url_val) {
                        if (body[url_val] == '\\' && url_val + 1 < body.size()) {
                            url += body[url_val];
                            url += body[url_val + 1];
                            ++url_val;
                        } else if (body[url_val] == '"') {
                            break;
                        } else {
                            url += body[url_val];
                        }
                    }
                    if (!url.empty()) {
                        images.push_back(url);
                    }
                }
            }
        }
        pos = type_pos + 7;
    }
    return images;
}

// ── HTTP primitives ──────────────────────────────────────────────────────────

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::string body;
    std::string content_type;
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

static void handle_request(socket_t fd, const HttpRequest & req, ServerState & state) {
    const bool ka = req.keep_alive;
    std::fprintf(stderr, "bmoe-server: HTTP %s %s (body=%zu bytes)\n", req.method.c_str(), req.path.c_str(), req.body.size());

    // CORS preflight
    if (req.method == "OPTIONS") {
        send_response(fd, 204, "No Content", "text/plain", "", ka);
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

static void print_lmstudio_telemetry(const RunSummary & sum, double ttft_ms, double total_sec) {
    char lbuf[64];
    char rbuf[64];

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "┌─────────────────────────────────────────────────────────────┐\n");
    std::fprintf(stderr, "│ Generation Statistics                                       │\n");
    std::fprintf(stderr, "├──────────────────────────────┬──────────────────────────────┤\n");

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
    double decode_toks = (sum.gen_seconds > 0.0) ? ((double) sum.n_generated / sum.gen_seconds) : sum.tokens_per_second;
    std::snprintf(lbuf, sizeof(lbuf), "Token Generation (Decode)");
    std::snprintf(rbuf, sizeof(rbuf), "%d tokens @ %.2f tok/s", sum.n_generated, decode_toks);
    std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);

    // Total Response Time
    int total_tokens = sum.n_prompt + sum.n_generated;
    std::snprintf(lbuf, sizeof(lbuf), "Total Response Time");
    if (total_sec < 60.0) {
        std::snprintf(rbuf, sizeof(rbuf), "%.2f s (%d tokens total)", total_sec, total_tokens);
    } else {
        std::snprintf(rbuf, sizeof(rbuf), "%.1f m (%d tokens total)", total_sec / 60.0, total_tokens);
    }
    std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);

    // Speculative Acceptance (MTP)
    if (sum.mtp_drafted > 0) {
        double accept_pct = (100.0 * (double) sum.mtp_accepted) / (double) sum.mtp_drafted;
        double avg_len = (sum.mtp_decodes > 0) ? ((double) sum.n_generated / (double) sum.mtp_decodes) : 1.0;
        std::snprintf(lbuf, sizeof(lbuf), "Speculative Acceptance (MTP)");
        std::snprintf(rbuf, sizeof(rbuf), "%.1f%% (%lld/%lld, avg len: %.1f)", accept_pct, (long long) sum.mtp_accepted, (long long) sum.mtp_drafted, avg_len);
        std::fprintf(stderr, "│ %-28s │ %-28s │\n", lbuf, rbuf);
    }

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

    // Build the prompt — accept both `messages` (chat) and `prompt` (completion) formats
    std::string prompt;
    std::vector<std::string> images;
    if (req.body.find("\"messages\"") != std::string::npos) {
        prompt = extract_last_user_message(req.body);
        images = extract_images(req.body);
    } else {
        prompt = json_extract_string(req.body, "prompt", "");
    }
    if (prompt.empty()) {
        send_json_error(fd, 400, "No user message or prompt found", false);
        return;
    }
    int n_predict = json_extract_int(req.body, "max_tokens", 0);
    if (n_predict <= 0) n_predict = json_extract_int(req.body, "max_completion_tokens", 0);
    if (n_predict <= 0) n_predict = 512;
    // Cap n_predict so web UI requests (e.g. AnythingLLM default 4096) never exceed session context size
    const int max_allowed = (std::max)(32, state.session->n_ctx() - 128);
    if (n_predict > max_allowed) n_predict = max_allowed;


    SamplingConfig sc = state.session_cfg.sampling;
    bool has_custom_sampling = false;

    if (req.body.find("\"temperature\"") != std::string::npos) {
        sc.temp = (float) json_extract_double(req.body, "temperature", sc.temp);
        has_custom_sampling = true;
    } else if (req.body.find("\"temp\"") != std::string::npos) {
        sc.temp = (float) json_extract_double(req.body, "temp", sc.temp);
        has_custom_sampling = true;
    }

    if (req.body.find("\"top_p\"") != std::string::npos) {
        sc.top_p = (float) json_extract_double(req.body, "top_p", sc.top_p);
        has_custom_sampling = true;
    }

    if (req.body.find("\"top_k\"") != std::string::npos) {
        sc.top_k = json_extract_int(req.body, "top_k", sc.top_k);
        has_custom_sampling = true;
    }

    if (req.body.find("\"min_p\"") != std::string::npos) {
        sc.min_p = (float) json_extract_double(req.body, "min_p", sc.min_p);
        has_custom_sampling = true;
    }

    if (req.body.find("\"repeat_penalty\"") != std::string::npos) {
        sc.repeat_penalty = (float) json_extract_double(req.body, "repeat_penalty", sc.repeat_penalty);
        has_custom_sampling = true;
    } else if (req.body.find("\"repetition_penalty\"") != std::string::npos) {
        sc.repeat_penalty = (float) json_extract_double(req.body, "repetition_penalty", sc.repeat_penalty);
        has_custom_sampling = true;
    }

    if (req.body.find("\"presence_penalty\"") != std::string::npos) {
        sc.presence_penalty = (float) json_extract_double(req.body, "presence_penalty", sc.presence_penalty);
        has_custom_sampling = true;
    }

    if (req.body.find("\"frequency_penalty\"") != std::string::npos) {
        sc.frequency_penalty = (float) json_extract_double(req.body, "frequency_penalty", sc.frequency_penalty);
        has_custom_sampling = true;
    }

    bool stream = json_extract_bool(req.body, "stream", false);

    // Build generate request
    GenerateRequest greq;
    greq.prompt = prompt;
    greq.images = images;
    greq.n_predict = n_predict;
    greq.render_text = !stream; // Render full text for non-streaming completions
    greq.think = !state.srv_cfg.disable_think;
    greq.has_sampling = has_custom_sampling || sc.temp > 0.0f;
    greq.sampling = sc;
    long created = static_cast<long>(std::time(nullptr));


    // Global generation lock: serialize requests to prevent GPU stream/context collision
    std::unique_lock<std::mutex> gen_lock(state.generate_mtx);

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

        if (!result) {
            send_json_error(fd, 500, result.error.c_str(), false);
            return;
        }

        print_lmstudio_telemetry(result.summary, ttft_ms, total_sec);

        std::string reply_content = result.generated_text;
        if (reply_content.empty() && !result.reasoning_text.empty()) {
            reply_content = result.reasoning_text;
        }

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
                   "},"
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
            data += "\"index\":0,\"delta\":{\"content\":\"" + json_escape(m.piece) + "\"},\"finish_reason\":null";
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

    if (result) {
        print_lmstudio_telemetry(result.summary, ttft_ms, total_sec);

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
        // Stream an error then done
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
                               "\"finish_reason\":\"stop\""
                               "}]}";
        send_sse(fd, err_data);
        send_sse_done(fd);
    }
}

// ── Connection handling ──────────────────────────────────────────────────────

// Read the full HTTP request from a blocking socket: headers + body.
// Returns false if the connection closed or the request was too large.
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
                if (raw.size() > 1024 * 1024) return false; // body too large (>1MB)
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
                "      --port N            HTTP server port (default 8080)\n"
                "      --host ADDR         bind address (default 127.0.0.1; use 0.0.0.0 for\n"
                "                          remote access)\n"
                "\n"
                "  All bmoe-cli streaming and decoding flags are supported:\n"
                "  -t, --threads, -ngl, --n-gpu-layers, -c, --ctx-size, --ubatch, --batch-size\n"
                "  --moe-stream, --cache-mb, --cache-floor-mb, --cache-ceil-mb,\n"
                "  --io-threads, --no-odirect, --dense-weights,\n"
                "  --prefetch, --predict-prefetch, --drop-cold-experts,\n"
                "  --overlap, --io-two-wave, --route-ahead,\n"
                "  --temp, --top-k, --top-p, --seed,\n"
                "  --mtp, --ngram, --draft, --mtp-p-min, --ngram-min-match,\n"
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
                "  All BMOE_* env vars from bmoe-cli also apply\n",
                argv0);
}

int main(int argc, char ** argv) {
    RunConfig cfg;
    ServerConfig srv;

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
        else if (a == "--port")
            srv.port = std::atoi(next("--port"));
        else if (a == "--host")
            srv.host = next("--host");
        else if (a == "-p" || a == "--prompt") {
            next("-p"); // ignored in server mode
        } else if (a == "-n" || a == "--n-predict")
            cfg.n_predict = std::atoi(next("-n"));
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

    const SessionConfig sc = session_config_from(cfg);
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

    // ── Simple single-threaded server loop ────────────────────────────
    // One connection at a time; good enough for on-device use.
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

        try {
            process_connection(client_fd, state);
        } catch (const std::exception & ex) {
            std::fprintf(stderr, "bmoe-server: exception during connection: %s\n", ex.what());
        } catch (...) {
            std::fprintf(stderr, "bmoe-server: unknown exception during connection\n");
        }
        close_socket(client_fd);

    }

    close_socket(listen_fd);
#if defined(_WIN32)
    WSACleanup();
#endif
    std::fprintf(stderr, "bmoe-server: shutting down\n");
    return 0;
}

