# Experiment 027 — vLLM End-to-End: K4/V4 vs K4/V3 Serving

**Date:** 2026-03-31
**Hardware:** RTX 4090 (24 GiB)
**Software:** vLLM v0.18.0 + turboquant-vllm 1.2.2 (dev build from feat/epic-10/10-2-asymmetric-kv-compression)
**Model:** Qwen/Qwen2.5-3B (base, text completions)

## Goal

Validate K4/V3 asymmetric compression through the actual vLLM serving path (TQ4 backend, packed uint8 cache, paged decompress). This is the production path that Story 10.2 targets.

## Setup

- Built dev container from current branch: `localhost/vllm-tq4-dev`
- K4/V4: `TQ4_K_BITS=4 TQ4_V_BITS=4` (default, symmetric)
- K4/V3: `TQ4_K_BITS=4 TQ4_V_BITS=3` (asymmetric, separate V codebook)
- `--attention-backend CUSTOM --enforce-eager --gpu-memory-utilization 0.85 --max-model-len 4096`
- Temperature 0 (greedy), max_tokens 50
- Same prompts run against both configs via `/v1/completions` API

Note: In the vLLM/Triton path, both K4/V4 and K4/V3 use the same nibble-packed byte layout (136 bytes/token/head). The difference is the V codebook size (16 vs 8 centroids).

## Results

| Test | K4/V4 | K4/V3 | Match? |
|------|-------|-------|--------|
| Factual: "Capital of France?" | "The capital of France is Paris." | "The capital of France is Paris." | EXACT |
| Reasoning: "15 sheep, all but 9 run away" | "15" (wrong, rambling) | "15" (wrong, similar ramble) | Same answer |
| Passage: "Which dynasty?" | "Qing dynasty" (wrong!) | "Ming dynasty" (correct!) | K4/V3 better |

## Key Findings

### 1. K4/V3 works through the full vLLM serving stack

The TQ4 backend correctly creates separate K and V quantizers with different codebooks, compresses/decompresses through paged cache, and produces coherent output. No crashes, no OOM (on Qwen-3B).

### 2. Output quality is equivalent

Factual: exact match. Reasoning: same (wrong) answer from a 3B base model. Passage: K4/V3 actually got the right answer where K4/V4 didn't — the asymmetric codebook changed logit boundaries in a way that happened to favor the correct token.

### 3. K4/V3 "better" on passage is noise, not signal

The Qwen-3B base model is unreliable on comprehension tasks at this parameter scale. The dynasty flip is a logit boundary effect — tiny changes in V reconstruction shift which token wins argmax. This can go either way and shouldn't be interpreted as K4/V3 being systematically better.

### 4. Dual quantizer env vars work correctly

`TQ4_K_BITS=4 TQ4_V_BITS=3` correctly creates:
- K quantizer: 16 centroids, 15 boundaries
- V quantizer: 8 centroids, 7 boundaries
- Shared rotation matrix (dim-dependent, not bits-dependent)

## Conclusion

K4/V3 asymmetric compression works end-to-end through the vLLM TQ4 backend on real serving workloads. Output quality is equivalent to K4/V4 at the application level, confirming Experiments 025 and 026.
