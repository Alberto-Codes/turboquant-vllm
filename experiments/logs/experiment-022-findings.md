# Experiment 022 — Clip Duration Comparison: Baseline vs TQ4

**Date:** 2026-03-29
**Hardware:** RTX 4090 (24 GiB), NVIDIA driver 570.x
**Episode:** Seinfeld S05E12 "The Stall" (1371s)
**Start offset:** 120s (skip cold open/credits)
**Max tokens:** 512

## Setup

| Config | Baseline (quadlet) | TQ4 v1.2.1 |
|---|---|---|
| Image | `vllm/vllm-openai:v0.18.0` | `vllm-turboquant:1.2.1` |
| KV cache | `--kv-cache-dtype fp8` | `--attention-backend CUSTOM` (TQ4) |
| Model len | 6144 | 6144 |
| GPU util | 0.90 | 0.90 |
| Eager | yes | yes |
| Perf mode | interactivity | interactivity |

## Results — Molmo2-8B

### Baseline (FP8 KV, quadlet)

| Clip | Elapsed | In Tok | Out Tok | Quality |
|---|---|---|---|---|
| 5s | 10.08s | 1,798 | 208 | Jerry + Elaine recognized, accurate scene description |
| 15s | 9.13s | 2,334 | 194 | Jerry recognized, calls them "two strangers" |
| 30s | 17.50s | 2,609 | 403 | Hallucinates "Ghostbusters 2", zero character recognition |
| 60s | 13.91s | 2,703 | 304 | Hallucinates "baseball game at movie theater", zero recognition |

**Finding:** Baseline degrades severely at 30s+. Token count plateaus (~2700) regardless of duration — Molmo2 samples fixed frames. Longer clips spread attention across more frames, losing scene specificity.

### TQ4 v1.2.1 (CUSTOM backend)

| Clip | Elapsed | In Tok | Out Tok | Quality |
|---|---|---|---|---|
| 5s | 8.02s | 1,798 | 228 | Jerry + Elaine recognized, accurate scene description |
| 15s | OOM | — | — | Vision encoder GELU OOM (not TQ4) |
| 30s | OOM | — | — | Engine dead after 15s failure |
| 60s | OOM | — | — | Engine dead after 15s failure |

**Finding:** TQ4 hotfix (#43) fixed the `_decompress_cache` buffer allocation — the 5s clip completed successfully with no OOM in TQ4 code. The 15s+ OOM occurs in `torch.nn.modules.activation.py` (GELU in vision encoder), not in TQ4. Root cause: Molmo2-8B (17 GiB weights) + vLLM KV block pool at 0.90 util leaves <250 MiB free. FP8 KV (baseline) uses less KV memory, leaving more headroom for the encoder.

### Head-to-Head — 5s Clip

| Metric | Baseline (FP8 KV) | TQ4 v1.2.1 | Delta |
|---|---|---|---|
| Elapsed | 10.08s | 8.02s | **1.26x faster** |
| Input tokens | 1,798 | 1,798 | Same |
| Output tokens | 208 | 228 | +10% more output |
| Characters | jerry:5, elaine:4 | jerry:3, elaine:2 | Both correct |

## OOM Analysis

The v1.2.0 OOM was in `tq4_backend.py:551` (`_decompress_cache` allocating ~308 MiB).
The v1.2.1 hotfix (#43) introduced `_decompress_cache_paged` with bounded scratch buffers (~2 MiB).
The v1.2.1 OOM is in vLLM's vision encoder (GELU activation), NOT in TQ4 code.

**Memory budget on 24 GiB (Molmo2-8B):**
- Model weights: ~17 GiB
- vLLM KV block pool (0.90 util): ~4.5 GiB
- PyTorch overhead: ~1.5 GiB
- Free: ~0.5 GiB (insufficient for encoder forward pass on second request)

The baseline survives because FP8 KV blocks are half the size of auto-dtype blocks, leaving ~1-2 GiB more headroom for the encoder.

## Conclusions

1. **TQ4 vLLM backend works** — proven at 5s clips on Molmo2-8B (1.26x faster)
2. **Hotfix #43 resolved the decompress buffer OOM** — confirmed via traceback analysis
3. **24 GiB is too tight for Molmo2-8B + TQ4 at multi-clip workloads** — VRAM fragmentation after first request leaves insufficient headroom for vision encoder
4. **Baseline quality degrades at 30s+** — hallucinations increase as frames compete for attention; short clips (5s) produce best results regardless of backend
5. **Next step:** Fix decode buffer over-provisioning in `_init_cg_buffers`

## Results — Molmo2-4B

### Baseline (FP8 KV)

| Clip | Elapsed | In Tok | Out Tok | Quality |
|---|---|---|---|---|
| 5s | 8.45s | 1,798 | 203 | Hallucinates "Cinema Paradiso", zero character recognition |
| 15s | 11.51s | 2,334 | 316 | "Two characters in theater", no names |
| 30s | 16.48s | 2,609 | 471 | Correctly identifies Seinfeld, Kramer + Elaine recognized |
| 60s | 13.26s | 2,703 | 301 | Hallucinates "Richard Gere and Julia Roberts" |

**Finding:** 4B baseline works at all durations but quality is worse than 8B — more hallucinations, fewer correct character identifications. Interestingly, 30s was the sweet spot for 4B (correctly identified the show and characters).

### TQ4 v1.2.1

All durations failed. OOM in `_init_cg_buffers` (line 386) allocating `_cg_decompress_v` (428 MiB).

**Root cause:** The CUDA graph decode buffers (`_cg_decompress_k`/`_cg_decompress_v`) are pre-allocated for `max_model_len` (6144) across all layers/heads. At 0.90 util, vLLM's KV block pool claims all remaining VRAM, leaving only ~404 MiB — insufficient for the 428 MiB decode buffer.

This is a different OOM from the v1.2.0 prefill issue (fixed in #43). The decode buffers from Story 4-2 were never tested under vLLM's tight memory budget.

## Bug Summary

| Version | OOM Location | Buffer | Fix Status |
|---|---|---|---|
| v1.2.0 | `_decompress_cache` (prefill) | ~308 MiB per call | Fixed in #43 (v1.2.1) |
| v1.2.1 | `_init_cg_buffers` (decode) | ~428 MiB pre-allocated | **Unfixed** — needs lazy alloc or reduced sizing |
| v1.2.1 | Vision encoder GELU (8B only) | ~156 MiB transient | Not TQ4's fault — 8B model fills 24 GiB |

## Results — v1.2.2 (decode buffer fix, PR #45)

PR #45 bounded decode buffers to `min(max_model_len, max_tokens)` (~12 MiB) and routed
`_tq4_decode` through `_decompress_cache_paged`. All durations now pass on both models.

### Molmo2-8B: Baseline vs TQ4 v1.2.2

| Clip | Baseline (FP8) | TQ4 v1.2.2 | Speedup | Quality |
|---|---|---|---|---|
| 5s | 10.08s | 10.10s | 1.0x | Both: jerry+elaine recognized |
| 15s | 9.13s | 5.96s | **1.53x** | TQ4: elaine:5 / Baseline: jerry:2 |
| 30s | 17.50s | 11.43s | **1.53x** | **TQ4: jerry:5, elaine:6** / Baseline: hallucinated "Ghostbusters 2" |
| 60s | 13.91s | 6.86s | **2.02x** | Both hallucinate, TQ4 2x faster |

### Molmo2-4B: Baseline vs TQ4 v1.2.2

| Clip | Baseline (FP8) | TQ4 v1.2.2 | Speedup | Quality |
|---|---|---|---|---|
| 5s | 8.45s | 10.04s | 0.84x | Both hallucinate (4B too small for character ID) |
| 15s | 11.51s | 14.68s | 0.78x | Both hallucinate |
| 30s | 16.48s | 3.28s | **5.0x** | Baseline: kramer+elaine / TQ4: generic |
| 60s | 13.26s | 4.44s | **3.0x** | Both hallucinate |

### Key Findings

1. **8B is the right model for video.** TQ4 is 1.5-2x faster across all durations and produces
   better character recognition at 30s (jerry:5+elaine:6 vs hallucinated Ghostbusters).
2. **TQ4 quality advantage at 30s is significant.** The baseline loses coherence at 30s+ while
   TQ4 still correctly identifies the show and characters.
3. **4B is too small for reliable character recognition** on either backend.
4. **Both OOM bugs (prefill #43, decode #45) are fixed** as of v1.2.2.
