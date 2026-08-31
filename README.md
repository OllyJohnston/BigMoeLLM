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
- **Built-in OpenAI Server (`bmoe-server`)**: Full OpenAI-compatible HTTP API (`/v1/chat/completions` and `/v1/completions`) with SSE streaming, dynamic per-query sampling overrides (`temp`, `top_p`, `min_p`, `top_k`, repetition penalties), multi-turn KV reuse, and multi-modal image support for AnythingLLM, Open WebUI, and custom apps.
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

### Server Flags Reference

| Flag | Description | Default |
|---|---|---|
| `-m, --model <path>` | Path to model GGUF file | Required |
| `--port <int>` | HTTP server port | `10000` |
| `--host <ip>` | Listening host interface | `127.0.0.1` |
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
3. Enter any API Key (e.g. `bmoe`).
4. Select or enter Model Name `bmoe`.

#### Python / OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:10000/v1",
    api_key="bmoe"
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

## Change History

### Recent Improvements & Milestones

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
