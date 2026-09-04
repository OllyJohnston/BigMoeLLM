# BigMoeLLM

<p align="center"><b>Run massive Mixture-of-Experts (MoE) models on consumer GPUs and memory-constrained hardware.</b></p>

<p align="center">
  <a href="https://github.com/OllyJohnston/BigMoeLLM/releases/latest"><img src="https://img.shields.io/github/v/release/OllyJohnston/BigMoeLLM" alt="Latest release"></a>
  <a href="https://github.com/OllyJohnston/BigMoeLLM/actions/workflows/ci.yml"><img src="https://github.com/OllyJohnston/BigMoeLLM/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/OllyJohnston/BigMoeLLM" alt="License"></a>
</p>

---

> [!WARNING]
> **Experimental Research Project**: BigMoeLLM is an experimental research and development engine actively exploring zero-fork hybrid VRAM pinning, NVMe streaming, and fast CPU MoE offload architectures for next-generation Mixture-of-Experts models. APIs, CLI flags, and internal scheduling heuristics are subject to rapid iteration.

BigMoeLLM is a high-performance inference engine built to run large Mixture-of-Experts (MoE) language models that exceed your GPU's physical VRAM on ordinary consumer graphics cards (12 GB – 16 GB GPUs like RTX 4070 / 4080 / 5070 Ti) and desktop workstations.

By combining **Hybrid VRAM Pinning**, **Zero-Copy Host MMAP Offloading**, **Continuous CUDA Graph Capture**, and **Direct I/O NVMe Expert Streaming**, BigMoeLLM achieves **64+ tok/s** on models like Qwen3.6-35B-A3B on a single 16 GB GPU—outperforming standard hybrid runtime splits while requiring zero in-tree patches to upstream `llama.cpp`.

---

## Key Highlights

- **64+ tok/s on Consumer GPUs**: Run 35B+ MoE models at interactive speeds on a single RTX 5070 Ti (16 GB VRAM) using hybrid pinned execution.
- **Hybrid VRAM Pinning (`--n-pinned-layers N`)**: Retain dense attention, normalization layers, and the first $N$ critical MoE blocks in fast GPU VRAM (~8.2 GB), while unpinned experts reside in system RAM or flash.
- **CUDA Graph Reuse with Fixed Pointers**: Unlike dynamic swappers that break graph capture, unpinned weights maintain fixed memory addresses across forward passes, allowing full CUDA Graph recording and execution on CUDA Stream 0 with near-zero CPU launch latency.
- **Built-in OpenAI Server (`bmoe-server`)**: Full OpenAI-compatible HTTP API (`/v1/chat/completions` and `/v1/completions`) with SSE streaming, dynamic per-query sampling overrides (`temp`, `top_p`, `min_p`, `top_k`, repetition penalties), multi-turn KV reuse, multi-modal image support, and reasoning spans isolated into `reasoning_content` (never leaked into the answer) for AnythingLLM, Open WebUI, and custom apps. Threaded request handling keeps `/v1/models`, `/health` and CORS preflights responsive while another client streams a long generation, and `--api-key` authentication protects non-loopback binds.
- **Hardware-Adaptive NVMe Streaming (`--moe-stream`)**: For >RAM setups, stream active Top-K expert slices on demand directly from NVMe flash via Win32 / POSIX Direct I/O (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) and asynchronous CUDA staging.
- **Blackwell `sm_120` & Multi-Gen GPU Support**: Built for NVIDIA Blackwell (`compute_120` / `sm_120a`), Ada Lovelace (`sm_89`), Ampere (`sm_86`/`sm_80`), and Apple Silicon / CPU fallback.
- **KV Cache Quantization & Flash Attention**: Native support for `-ctk q8_0 -ctv q8_0` and `-fa` to cut KV cache VRAM footprint by up to 50%.
- **Zero Upstream Forks**: Interacts cleanly through `llama.cpp`'s public API layer (`llama.h`, `ggml.h`, `ggml-backend.h`).

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │              Incoming Request                 │
                      │       (bmoe-server / OpenAI SSE API)          │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │       llama.cpp Graph Scheduler               │
                      └───────┬───────────────────────────────┬───────┘
                              │                               │
             ┌────────────────▼─────────────┐   ┌─────────────▼─────────────┐
             │       CUDA0 VRAM (8.2 GB)     │   │       Host RAM (mmap)     │
             ├──────────────────────────────┤   ├───────────────────────────┤
             │ • Embeddings & Output LM Head│   │ • Unpinned MoE Expert     │
             │ • Attention & Norm Weights   │   │   Weights (Layers 12..39) │
             │ • Pinned MoE Layers (0..11)  │   │ • Evaluated across 12 CPU │
             │ • KV Cache (Q8_0 Quantized)  │   │   Threads (AVX2/AVX-512)  │
             │ • CUDA Graph Capture Replay  │   │ • Zero PCIe Weight DMA    │
             └──────────────────────────────┘   └───────────────────────────┘
```

### Execution Modes

1. **Hybrid CPU-MoE Mode (`--cpu-moe --n-pinned-layers 12`) [Recommended for Desktop Workstations]**:
   - Early layers ($0 \dots 11$) and all dense attention/norm tensors live in CUDA0 VRAM (~8.2 GB).
   - Remaining MoE expert weights stay mapped in system RAM via mmap.
   - Unpinned FFN operations execute directly across 12 CPU worker threads, passing only intermediate activation vectors across PCIe.
   - Pointers remain 100% static, enabling continuous **CUDA Graph Capture Reuse** for maximum throughput (**64 tok/s** on RTX 5070 Ti).

2. **NVMe Expert Streaming Mode (`--moe-stream --n-pinned-layers 16`) [For Extreme >RAM Models]**:
   - Pinned layers stay resident in VRAM.
   - Unpinned expert slices are streamed on demand from NVMe SSD directly into GPU ring buffers on CUDA Stream 1 via asynchronous Direct I/O and staged before layer computation on CUDA Stream 0.
   - Adaptive Replacement Cache (ARC) keeps the most frequently routed experts resident in VRAM/RAM.

---

## Supported Models

| Architecture | Example Models | Parameters | Footprint | Recommended Mode |
|---|---|---:|---:|---|
| `qwen35moe` | Qwen3.6-35B-A3B (MTP / Base) | 35.5B (3.3B active) | ~22.3 GB | Hybrid `--cpu-moe -ngl 99 --n-pinned-layers 12` |
| `qwen3moe` | Qwen3-30B-A3B | 30B (3B active) | ~18.5 GB | Hybrid `--cpu-moe -ngl 99 --n-pinned-layers 12` |
| `qwen4exp` | Qwen3.8-Flash-Next (Qwen4 Preview) | 125B (6B active) | ~82 GB | Streaming `--moe-stream --cache-mb auto` |
| `deepseek4` | DeepSeek V4 Flash 0731 | 284B (13B active) | ~91 GB | Streaming `--moe-stream --cache-mb auto` |
| `gpt-oss` | gpt-oss-120b | 120B (12B active) | ~60 GB | Streaming `--moe-stream --overlap` |
| `gemma4` | Gemma-4-26B-A4B | 26B (4B active) | ~17.0 GB | Hybrid `--cpu-moe -ngl 99 --n-pinned-layers 10` |

---

## Benchmarks

*All benchmarks measured on: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, Blackwell `sm_120`), 64 GB DDR5 RAM, NVMe Gen4 SSD, Windows 11.*

### Qwen3.6-35B-A3B (Q4_K_M / Q6_K / APEX MTP)

| Runtime / Engine | Configuration | VRAM Usage | Tok/s (Decode) |
|---|---|---:|---:|
| Standard llama.cpp / LM Studio | Default Hybrid Split (CPU/GPU split offload) | ~14.2 GB | ~54.0 tok/s |
| **BigMoeLLM (`bmoe-server`)** | **Hybrid `--cpu-moe --n-pinned-layers 12 -ctk q8_0 -ctv q8_0 -fa`** | **8.2 GB** | **64.2 tok/s** |
| BigMoeLLM (`bmoe-cli`) | NVMe Streaming (`--moe-stream --cache-mb 24000 --ngram`) | 8.2 GB | 38.5 tok/s |

---

## Quickstart

### Building from Source

#### Prerequisites
- **CUDA Toolkit**: CUDA 12.8 (or 12.x / 13.x) with NVCC
- **CMake**: Version 3.21 or later
- **C++17 Compiler**: Visual Studio 2022 (MSVC) on Windows or GCC 11+ / Clang 14+ on Linux

#### Build Commands

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/OllyJohnston/BigMoeLLM.git
cd BigMoeLLM

# Configure with CMake (Blackwell sm_120 targeted by default)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="120"

# Build bmoe-server and bmoe-cli
cmake --build build --config Release --target bmoe-server bmoe-cli -j 8
```

The executables will be located in `build/cli/Release/` (Windows) or `build/cli/` (Linux).

---

## Running the OpenAI Server (`bmoe-server`)

Launch `bmoe-server` with your model:

```bash
build/cli/Release/bmoe-server.exe \
  -m "D:\AI_Models\Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf" \
  --port 10000 \
  --n-gpu-layers 99 \
  --n-pinned-layers 12 \
  -ctk q8_0 \
  -ctv q8_0 \
  -fa \
  --cpu-moe \
  -t 12
```

> **Concurrency**: each connection runs on its own worker thread. Inference is serialized — a second
> generation request while one is active returns `HTTP 429` (`"Another generation is in progress; try
> again shortly"`) instead of blocking, and `/v1/models`, `/health` and CORS `OPTIONS` preflights
> answer immediately even mid-stream.
>
> **Authentication**: binding a non-loopback interface (`--host 0.0.0.0`) without `--api-key` prints a
> warning. Set `--api-key <secret>` (or `BMOE_SERVER_API_KEY`) to require
> `Authorization: Bearer <secret>` on every endpoint (CORS `OPTIONS` preflight exempt); requests
> without a matching key get `HTTP 401`.

### Server Flags Reference

| Flag | Description | Default |
|---|---|---|
| `-m, --model <path>` | Path to model GGUF file | Required |
| `--port <int>` | HTTP server port | `10000` |
| `--host <ip>` | Listening host interface | `127.0.0.1` |
| `--api-key <secret>` | Require `Authorization: Bearer <secret>` on every endpoint (CORS `OPTIONS` exempt); non-loopback binds warn when absent | none |
| `-ngl, --n-gpu-layers <int>` | Layers to offload to GPU | `99` (all) |
| `--n-pinned-layers <int>` | Number of early layers to permanently pin in VRAM | `12` |
| `--cpu-moe` | Enable static host RAM MoE mapping and CPU threadpool execution | `false` |
| `-ctk, --cache-type-k <type>` | KV cache quantization for Key (`q8_0`, `q4_0`, `f16`) | `f16` |
| `-ctv, --cache-type-v <type>` | KV cache quantization for Value (`q8_0`, `q4_0`, `f16`) | `f16` |
| `-fa, --flash-attn` | Enable Flash Attention | `true` |
| `-t, --threads <int>` | CPU worker threads for unpinned computation | `12` |
| `--temp <float>` | Default sampling temperature (<=0 for greedy) | `0.7` |
| `--top-p <float>` | Default nucleus sampling cutoff | `0.95` |
| `--top-k <int>` | Default Top-K cutoff | `40` |
| `--min-p <float>` | Default Min-P cutoff | `0.05` |
| `--repeat-penalty <float>` | Repetition penalty | `1.1` |

### Connecting to Web UIs

#### AnythingLLM
1. Open AnythingLLM Settings -> **AI Providers** -> **Generic OpenAI**.
2. Set **Base URL** to `http://127.0.0.1:10000/v1`.
3. Enter any API Key (e.g. `bmoe`); if the server runs with `--api-key`, use that exact key.
4. Select or enter Model Name `bmoe`.

#### Python / OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:10000/v1",
    api_key="bmoe"  # any string when the server has no --api-key; otherwise the configured key
)

response = client.chat.completions.create(
    model="bmoe",
    messages=[{"role": "user", "content": "Explain how hybrid Mixture-of-Experts execution works."}],
    temperature=0.7,
    top_p=0.95,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

---

## Running the Command Line Interface (`bmoe-cli`)

For one-off prompts and benchmarks:

```bash
build/cli/Release/bmoe-cli.exe \
  -m "D:\AI_Models\Qwen3.6-35B-A3B.gguf" \
  --n-gpu-layers 99 \
  --n-pinned-layers 12 \
  -ctk q8_0 -ctv q8_0 -fa \
  --cpu-moe \
  -t 12 \
  -n 256 \
  -p "Explain the mathematical intuition behind Sparse MoE routing."
```

---

## Command-Line Options & Memory Management Reference

For custom inference engines and forks, users and downstream tools need clear documentation of non-standard or essential memory management switches.

### Command-Line Options

| Flag | Description | Default |
|---|---|---|
| `-m, --model <path>` | Path to model GGUF file. | Required |
| `--port <port>` | HTTP listening port for the server. | `8080` (or `10000`) |
| `-ngl, --n-gpu-layers <N>` | Number of layers to offload to GPU (`99` for full offload). | `0` |
| `-c, --ctx-size <N>` | Context window size in tokens. | `4096` |
| `-ctk, --cache-type-k <type>` | KV cache data type for keys (`f16`, `q8_0`, `q4_0`, etc.). | `f16` |
| `-ctv, --cache-type-v <type>` | KV cache data type for values (`f16`, `q8_0`, `q4_0`, etc.). | `f16` |
| `-nkqv, --no-offload-kqv` | Keep the KV cache in system RAM instead of VRAM. | Off |
| `--no-mmproj-offload` | Keep multimodal projector (CLIP/ViT) weights in system RAM. | Off |
| `--mmproj-offload [on\|off]` | Enable/disable GPU offloading for multimodal projector. | `on` |
| `-fa, --flash-attn` | Enable Flash Attention kernels. | Off (or `on` when quantizing KV) |
| `-b, --batch-size <N>` | Logical batch size for prompt processing. | `512` |
| `-ub, --ubatch-size <N>` | Physical micro-batch size for GPU compute steps. | `512` |
| `-t, --threads <N>` | Number of CPU compute threads to allocate. | Auto |

### Key Flags Explained

- **`-nkqv` / `--no-offload-kqv` / `--no-kv-offload`**: Keeps the entire KV cache allocated in host DDR4/DDR5 system memory while allowing all model layer weights to remain fully resident in VRAM (`-ngl 99`). This allows running 64k to 100k+ contexts on consumer GPUs without running out of VRAM.
- **`--no-mmproj-offload`**: Leaves vision projector tensors in system RAM, freeing 1.0 to 2.5 GB of VRAM for larger context windows and model weights when running multimodal models.
- **`-ctk q8_0 -ctv q8_0`**: Quantizes the key-value cache to 8-bit precision, cutting KV VRAM usage roughly in half with near-lossless attention quality. Use `q4_0` to reduce cache memory footprint by 75%.

---

## Change History

### Recent Improvements & Milestones

- **2026-09-04: Predict-Prefetch Crash Fix with Hybrid Static Offload (engine 0.27.1)**: Fixed a crash on `--predict-prefetch` + `--n-pinned-layers`: the speculative-read path assumed every layer index exists in the host `lbuf_` ring, but pinned layers have a null host buffer (their weights are GPU-resident). The lane staging loop then shipped read jobs with a null destination — `VirtualAlloc(NULL, MEM_COMMIT)` silently succeeds at a system address, so a lane's `FileReader::read` jumped to 0x0. Proven with cdb (`memcpy_repmovs` WRITE@0x0 ← `drain_spec` ← `io_worker`). Pinned layers are always resident, so speculation is a no-op for them: `prefetch`, `enqueue_predicted_ids` and the staging loop now early-return on `is_layer_pinned`. Crash only reproduced live (gate models use unpinned tiny-moe), so the gates could not catch it; suite still 16/16. Debug tooling (cdb/dumpbin/DbgHelp/readiness-probe) documented in `AGENTS.md`.
- **2026-09-03: Active (Lane-Direct) Prediction Prefetch (engine 0.27.0)**: `--predict-prefetch` now publishes the stale-gate predictor's speculated misses straight to the idle I/O lanes via a bounded queue, so reads start up to a full layer earlier instead of waiting on the eval-thread graph callback. All LRU mutation stays on the eval thread; the worker→lane and eval→prefetch paths dedup through the shared residency guards. Verified: byte-identity gates G10a/G5c/S2 pass, suite 16/16.
- **2026-09-03: Expert-Streaming Crash Fix & First Fully Green Gate Suite (`f9cd370`, engine 0.26.1)**: Fixed the long-standing "session-20" segfault (`0xC0000005` in the CUDA driver during the first decode of certain gate configs — the one failure that had dogged the byte-identity suite for weeks). Root cause: the scheduler's MoE used-expert copy (`ggml-backend.cpp` `copy_experts`) pads each consecutive group of used experts with up to 512 bytes *into the next expert's slice* to keep CUDA MMQ free of NaNs at group boundaries — safe for upstream's fully-mapped buffers, but the streamer's LRU cache backs expert tensors with lazily-committed `vm_reserve` ranges, so that pad read faulted whenever the next expert was not resident. `commit_proj_pages` now also commits the first page of the next expert's slice (one 4 KiB page per committed expert; `VirtualAlloc(MEM_COMMIT)` is a no-op on committed pages; residency accounting untouched). Two related correctness fixes batched in: `SamplingConfig::temp` default restored to greedy (`0.0`) — the `0.7` default made every `RunConfig{}` sample stochastically, which surfaced as the "byte divergence" in the predictor gates (G10a/G13a), the predictor itself always being observation-only — and `cudaDeviceSynchronize()` drains in session teardown and reopen so an in-flight kernel never races the device-handle release. **All 16 byte-identity gates now pass — the first fully green suite in the project's history.**
- **2026-09-01: Batched IQ Panel GEMM for CPU Prefill (`d5e576b6` submodule, engine 0.26.0)**: Ported upstream `ggml-org/llama.cpp` PR #27402 to `bmoe/expert-ready-hook` — new `iqp.cpp`/`iqp.h` decode grid-IQ quantized weights (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL) into per-thread int8 panels and run a batched integer GEMM for large-batch prompt processing, up to 8-10x faster CPU prefill / ~2x on CPU MoE layers. Dispatches after the src1->q8_K barrier for batches >= 8; N=1 decode keeps the lightweight GEMV path; `GGML_NO_IQ_PANEL` disables it for A/B. Verified: 883/883 CPU MUL_MAT/MUL_MAT_ID backend-ops, gates at baseline.
- **2026-09-01: Bounded QSA Decode Attention for Flash-Next (`f3ad45b9a` submodule)**: Ported upstream PR #27977 to `bmoe/expert-ready-hook`. Qwen3.8-Flash-Next decode previously masked a full cache-window attention matrix — the Flash Attention skip never fired on single-token batches, so every token walked all `n_kv` K/V tiles. The new `build_qsa_gather` path gathers exactly the indexer-selected cells per query (`ggml_get_rows`, queries on the stream axis) and dispatch picks it when `flash_attn && 4*n_tps*width < n_kv`, keeping decode compute bounded to the top-k budget as context grows (engine version 0.25.0).
- **2026-09-01: CUDA MoE Fusion for Speculative Batches (`68cad06`)**: The fused `mul_mat_vec_q_moe` kernel and the `topk_moe` router now serve multi-token verification batches (N = 2..8), so speculative decoding (MTP / n-gram) on MoE architectures no longer falls back to per-token, un-fused kernel launches. Ported from upstream `ggml-org/llama.cpp` PR #27621 (omitting `SWIGLU_CLAMP`) onto `bmoe/expert-ready-hook`; the submodule now points at the public `OllyJohnston/llama.cpp` fork, pinned at `0fe77575e` on upstream base `b10680`. Byte-identity gates extended with multi-token and topk boundary cases: 900/900 `MUL_MAT_VEC_FUSION` and 416/416 `TOPK_MOE` pass on CUDA (engine version 0.24.0).
- **2026-09-01: MTP Compact Rollback & Adaptive Draft Sizing**: `--spec-mtp-cr-depth N` shrinks the MTP target context's recurrent snapshot table from `draft_max` to `N` recurrent-state snapshots; a rejected draft deeper than `N` restores a partial recurrent-state checkpoint (ON_DEVICE with host fallback) and re-decodes the accepted prefix in one extra decode, instead of leaving `n_rs_seq = draft_max` always in memory. `--spec-draft-adaptive` sizes each draft from measured acceptance. New summary counters `mtp_ckpt_saves` / `mtp_replays` / `mtp_host_fallback` report how often each path fired. Measured on a 35B-A3B MTP model: 251.25 MiB recurrent at 3 snapshots vs 125.62 MiB at 1, with ~half of deep drafts at `N=1` taking the replay path. See `docs/mtp.md`.
- **2026-08-31: Threaded Server, API-Key Auth & Multi-Turn KV Reuse (`dbc2cdf`)**: `bmoe-server` now serves each connection on its own worker thread (the accept loop no longer blocks for the duration of a stream), serializes generations with a clean `HTTP 429` busy response, and adds `GET /health`. `--api-key` / `BMOE_SERVER_API_KEY` enforces `Authorization: Bearer <secret>` with `HTTP 401` on unauthorized access (CORS `OPTIONS` exempt; non-loopback binds warn when no key is set). Chat requests hand the engine the full `messages` array (`GenerateRequest::messages`, `clear_kv=false`), so the `n_common` prefix match serves a continued conversation from the previous turn's KV cache; reasoning spans are streamed to `delta.reasoning_content` and reported in `message.reasoning_content`, never leaked into `content`. Request parsing moved from hand-rolled string splitting to vendored nlohmann/json (a `}` in a code block no longer truncates a message) with the body cap raised to 64 MiB for multimodal payloads; speculative decoding alongside `temp > 0` is now accepted as heuristic mode (see `docs/mtp.md`).
- **2026-08-31: Multimodal Projector Offload Control (`c228eac`)**: Added `--no-mmproj-offload` and `--mmproj-offload [on|off]` flags to `bmoe-cli` and `bmoe-server` to keep vision encoder / CLIP weights in host system RAM instead of device VRAM.
- **2026-08-31: KV Cache Offload Control & Speculative Telemetry (`a684f97`)**: Added `-nkqv` / `--no-offload-kqv` / `--no-kv-offload` flags to keep the KV cache in system RAM; corrected decode rate calculations to true wall-clock time; added MTP / speculative decoding acceptance and draft length rows to the `Generation Statistics` box.
- **2026-08-31: Experimental Status & Documentation Update (`22ead20`)**: Added experimental project status disclaimer and detailed milestone change history.
- **2026-08-30: Batch Size Support & Submodule Synchronization (`f2141a5`)**: Added `-b` / `--batch` / `--batch-size` flag support to `bmoe-cli` for flexible prompt batching, and synchronized `llama.cpp` upstream submodule for latest architectural fixes.
- **2026-08-30: Windows Console & Telemetry Display (`2de7578`, `4c3005e`)**: Enabled UTF-8 console code page and ANSI Virtual Terminal Processing on Windows; implemented real-time performance telemetry summaries (prompt eval tok/s, decode tok/s, TTFT, and memory bandwidth).
- **2026-08-30: Critical Decode Path Acceleration (`75b531b`)**: Bypassed ARC caching bookkeeping and asynchronous prefetch worker triggers when hot-caching is inactive or when MoE layers are permanently pinned to VRAM, eliminating overhead on dense/MTP models.
- **2026-08-30: Hot-Expert ARC Cache (`68a4678`)**: Integrated Adaptive Replacement Cache (ARC) for hot MoE experts with CUDA staging and VRAM ring buffer management.
- **2026-08-30: High-Priority Polling Threadpool (`dfe3bbb`)**: Integrated a dedicated high-priority CPU worker threadpool with spin-polling to minimize CPU dispatch latency during `--cpu-moe` passes.
- **2026-08-30: Host MMAP Memory Preservation (`724ba68`, `8ea7555`)**: Enforced direct system RAM mmap weight mappings and protected massive models (such as Qwen3.8-Flash-Next and DeepSeek-V4) from VRAM exhaustion by overriding dense gather tables (`per_layer_token_embd`) to host-resident memory.
- **2026-08-30: Zero-Split Hybrid CUDA Graph Capture (`0d5fc7f`, `7776819`)**: Implemented unified CUDA dummy buffer bindings across hybrid CPU/GPU MoE boundaries, eliminating graph splits and enabling continuous CUDA graph reuse on CUDA Stream 0.
- **2026-08-30: KV Cache Quantization & Flash Attention Matrix (`db02fdb`)**: Expanded support for KV cache quantization (`-ctk q8_0`/`q4_0`, `-ctv q8_0`/`q4_0`) and enforced Flash Attention (`-fa`) integration across all supported models.
- **2026-08-30: OpenAI-Compatible Server Engine (`96502a9`)**: Built high-performance `bmoe-server` featuring SSE streaming, dynamic sampling parameters, multi-turn KV reuse, and multi-modal image support.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
