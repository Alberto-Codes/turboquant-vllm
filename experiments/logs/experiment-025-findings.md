# Experiment 025 — AC3 Asymmetric Quality Matrix (Real Activations)

**Date:** 2026-03-31
**Hardware:** RTX 4090 (24 GiB), NVIDIA driver 595.x
**Software:** turboquant-vllm @ feat/epic-10/10-2-asymmetric-kv-compression
**Models:** 8 validated families (all cached locally)

## Goal

Validate AC3 from Story 10.2 (Asymmetric K/V Compression) using real model forward passes — not random Gaussian data. Determine whether K4/V3 meets the 0.99 compression quality threshold with real KV activations.

## Hypothesis

Real model activations have lower entropy than random Gaussian data, so cosine similarity under compression should meet or exceed the random-data baseline. The AC3 gate requires K4/V3 >= 0.99 cosine on all 8 validated models.

## Setup

- 128-token technical English prefill (real model forward pass)
- KV tensors captured via `_KVCaptureCache` hook (clones before delegation)
- Compress/decompress with `TurboQuantCompressorMSE` at each bit config
- Per-layer cosine similarity: `cos(ref_k, decomp_k)`, `cos(ref_v, decomp_v)`
- Models loaded one at a time, VRAM freed between models

## Results — Quality Matrix

| Model | head_dim | layers | K4/V4 min | K4/V3 min | K4/V3 V min | K4/V2 min |
|-------|----------|--------|-----------|-----------|-------------|-----------|
| Qwen2.5-3B | 128 | 36 | 0.9935 | 0.9823 | 0.9823 | 0.9390 |
| Gemma-2-2b | 256 | 26 | 0.9948 | 0.9823 | 0.9823 | 0.9388 |
| Molmo2-4B | 128 | 36 | 0.9943 | 0.9821 | 0.9821 | 0.9393 |
| Gemma-3-4b-it | 256 | 34 | 0.9911 | 0.9794 | 0.9794 | 0.9345 |
| Phi-3-mini | 96 | 32 | 0.9950 | 0.9827 | 0.9827 | 0.9400 |
| Mistral-7B | 128 | 32 | 0.9947 | 0.9825 | 0.9825 | 0.9395 |
| Llama-3.1-8B | 128 | 32 | 0.9947 | 0.9823 | 0.9823 | 0.9392 |
| Phi-4 | 128 | 40 | 0.9945 | 0.9824 | 0.9824 | 0.9395 |

## Key Findings

### 1. K4/V4 passes 0.99 on all 8 models (baseline confirmed)

All models exceed 0.99 at symmetric 4-bit. Gemma-3 is the tightest at 0.9911 (consistent with its head_dim=256 which has more dimensions to reconstruct).

### 2. K4/V3 consistently achieves ~0.98, NOT 0.99

The V component at 3-bit produces cosine ~0.982 across all models. This is NOT noise — it's the inherent reconstruction ceiling of 8-centroid Lloyd-Max quantization (3-bit = 8 levels vs 4-bit = 16 levels).

**The 0.99 AC3 gate for K4/V3 is physically unreachable at 3-bit V.**

The K component remains at ~0.994 (still at 4-bit). Only the V component drops.

### 3. The result is consistent across all model families

V cosine at 3-bit is remarkably stable: 0.9794–0.9827 across 8 diverse model families (Llama, Mistral, Qwen, Phi, Gemma, Molmo). This confirms the floor is determined by the quantizer (8 centroids), not by model-specific activation distributions.

### 4. K4/V2 data: 4 centroids for V gives ~0.94 cosine

V at 2-bit (4 centroids) drops to ~0.94. Still potentially usable for some applications but a meaningful quality degradation.

### 5. Phi-3-mini required `get_usable_length` shim

Phi-3's attention implementation calls `cache.get_usable_length()` which was removed in newer `transformers`. A shim patching it to `get_seq_length()` resolved the issue. This is a known Phi-3 compatibility quirk, not a TurboQuant bug.

## Implications for AC3 Gate

The original AC3 requirement — "K4/V3 exceeds 0.99 cosine on all validated models" — was set based on the 4-bit compression quality tier. **3-bit V quantization cannot physically achieve 0.99 per-layer cosine similarity.**

**Recommended gate adjustment:**

| Config | Gate | Threshold | Rationale |
|--------|------|-----------|-----------|
| K4/V4 | PASS/FAIL | >= 0.99 | 16 centroids, proven achievable |
| K4/V3 | PASS/FAIL | >= 0.97 | 8 centroids for V, 0.98 achievable ceiling |
| K4/V2 | Data only | N/A | 4 centroids, exploratory |

The per-layer cosine metric measures single-token reconstruction fidelity. Real-world output quality (perplexity, BLEU, factual accuracy) degrades more gracefully than per-layer cosine suggests — the softmax attention weighting averages out per-token errors. Experiment 024 showed zero output quality degradation at K4/V4 through 1,200+ tokens.

## Observations

- **Real activations ≈ random Gaussian** for quality measurement: contrary to the hypothesis, real activations did NOT produce higher cosine than random data. Both converge to the same floor set by the codebook size. The ~0.98 floor is a quantizer property, not data-dependent.
- **Gemma-3 is the tightest model**: 0.9911 at K4/V4, 0.9794 at K4/V3. Its head_dim=256 means more dimensions to reconstruct per token, slightly diluting the codebook's effectiveness.
- **Phi-3-mini at head_dim=96 performs best**: 0.9950 at K4/V4, 0.9827 at K4/V3. Smaller dimensions compress better per-token.

## Next Steps

1. **Adjust AC3 gate** to 0.97 for K4/V3 (reflect physical ceiling of 3-bit V)
2. **Run end-to-end quality test** (like Experiment 024's text comparison) with K4/V3 to validate no output quality degradation at the application level
3. **Consider K5/V3 config** — 5-bit K (32 centroids) + 3-bit V (8 centroids) might hit 0.99 on both components while still saving storage vs K4/V4
