# Experiment 024 — Zero-Change Model Probe: Llama 3.1 8B + Mistral 7B

**Date:** 2026-03-30
**Hardware:** RTX 4090 (24 GiB), NVIDIA driver 595.x
**vLLM:** v0.18.0, TQ4 via vllm-turboquant:1.2.2
**Models:** meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3

## Goal

Determine if TQ4 works on non-Molmo2 models with zero code changes, validate quality across short prompts, long passage comprehension, and multi-turn conversation, and measure the KV cache capacity advantage.

## Setup

| Config | Baseline | TQ4 v1.2.2 |
|---|---|---|
| Image | `vllm/vllm-openai:v0.18.0` | `vllm-turboquant:1.2.2` |
| KV cache | `--kv-cache-dtype fp8` | `--attention-backend CUSTOM` (TQ4) |
| GPU util | 0.85 | 0.85 |
| Eager | yes | yes |

Both models require `HF_TOKEN` (gated repos). Llama OOMs at `gpu_util=0.90` during profiling — 0.85 is the working config for 8B text models.

## Results — Quality (6 tests per config)

### Test Suite

| Test | What it measures | Token scale |
|---|---|---|
| 4 short prompts | Factual, reasoning, creative, long-form | 18-66 in |
| Long passage | 5 factual questions about a ~960-token article | ~960 in |
| 5-turn conversation | Conversational memory and coherence | 50→1,224 in (growing) |

### Llama 3.1 8B

| Test | Baseline FP8 | TQ4 v1.2.2 | Match? |
|---|---|---|---|
| Short prompts | 4/4 | 4/4 | identical |
| Haiku | "Lines of code entwined / Errors hidden, patience worn / Logic's gentle dance" | **identical** | exact match |
| Reasoning (sheep) | 9 (correct) | 9 (correct) | exact match |
| Long passage facts | 5/5 | 5/5 | exact match |
| Multi-turn coherence | Kyoto + temples + summary | Kyoto + temples + summary | equivalent |
| **Total** | **6/6 PASS** | **6/6 PASS** | |
| Avg short latency | 2.5s | 4.9s | TQ4 ~2x slower |

### Mistral 7B

| Test | Baseline FP8 | TQ4 v1.2.2 | Match? |
|---|---|---|---|
| Short prompts | 4/4 | 4/4 | identical |
| Haiku | "Errors flash / In the labyrinth of lines / Peace in each fix" | **identical** | exact match |
| Reasoning (sheep) | 9 (correct) | 9 (correct) | exact match |
| Long passage facts | 5/5 | 5/5 | exact match |
| Multi-turn coherence | Kyoto + temples + summary | Kyoto + temples + summary | equivalent |
| **Total** | **6/6 PASS** | **6/6 PASS** | |
| Avg short latency | 2.6s | 4.6s | TQ4 ~1.8x slower |

**Both models: zero code changes, quality-lossless through 1,200+ tokens of compressed KV cache.**

## Results — KV Cache Capacity (Llama 3.1 8B)

| Config | Baseline FP8 | TQ4 v1.2.2 | TQ4 Advantage |
|---|---|---|---|
| 4K / 0.90 util | **OOM** (profiler) | Works | TQ4 enables config baseline can't start |
| 8K / 0.85 util | 71,824 tok (8.8x concurrency) | 135,104 tok (16.5x concurrency) | **1.88x more KV cache** |
| 16K / 0.80 util | 52,560 tok (3.2x concurrency) | 98,832 tok (6.0x concurrency) | **1.88x more KV cache** |

**Consistent 1.88x KV capacity advantage** across configs. The ratio is 3.76x TQ4 compression / 2x FP8 compression = 1.88x net advantage over the FP8 baseline.

At 16K context length, baseline can serve **3 concurrent requests** while TQ4 serves **6** — same hardware, same model, same quality.

## Results — Latency

| Model | Baseline Avg | TQ4 Avg | Overhead |
|---|---|---|---|
| Llama 3.1 8B | 2.5s | 4.9s | 1.96x |
| Mistral 7B | 2.6s | 4.6s | 1.77x |

TQ4 decode latency overhead is ~1.8-2x on short text prompts. This is expected — TQ4's value is at long contexts (video, document QA) where the 3.76x KV compression enables workloads that don't fit otherwise, not at short text where FP8 is already fast enough.

## OOM Analysis

| Config | Baseline | TQ4 | Root Cause |
|---|---|---|---|
| Llama 8B, 4K, 0.90 util | OOM | Works | vLLM profiler allocates attention intermediates proportional to max_model_len — model weights (~15 GiB) + profiler overhead exceeds 24 GiB at 0.90 |
| Llama 8B, 8K, 0.90 util | OOM | OOM | Same profiler bottleneck — TQ4 doesn't help with attention intermediates |
| Llama 8B, 16K, 0.90 util | OOM | OOM | Same |

The profiler OOM is independent of KV compression — it's the model's forward pass during vLLM warmup that needs VRAM proportional to `max_model_len` for attention intermediate tensors. Both backends hit the same ceiling. TQ4's advantage manifests in the KV block allocation that happens after profiling.

## Conclusions

1. **Both Llama 3.1 8B and Mistral 7B work with TQ4, zero code changes.** No kernel modifications, no model-specific configs. Just `--attention-backend CUSTOM`.

2. **Quality is lossless.** Temperature=0 outputs are identical (same haiku text) or equivalent (same factual answers, same conversational coherence) across baseline and TQ4 through 1,200+ tokens of KV cache.

3. **1.88x KV cache capacity advantage over FP8 baseline.** Consistent across 8K and 16K configs. At 16K context, TQ4 serves 6 concurrent requests vs baseline's 3.

4. **Story 5-1 is a validation exercise, not development.** Both models work out of the box. The story should focus on documenting the results, adding to the compatibility matrix, and updating the README.

5. **The 3.76x compression is real, but FP8 baseline already compresses 2x.** The net advantage vs FP8 is 1.88x, not 3.76x. Vs FP16 baseline (no `--kv-cache-dtype fp8`), the full 3.76x advantage applies.

6. **~2x latency overhead on short text** is the tradeoff. TQ4's value proposition is capacity (more concurrent requests, longer contexts) not per-request speed on short prompts.
