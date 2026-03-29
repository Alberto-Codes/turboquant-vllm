# Experiment 021: INT8 Fused Paged TQ4 Prefill Benchmark

**Date:** 2026-03-29T21:07:35.199966+00:00
**GPU:** NVIDIA GeForce RTX 4090
**Config:** Molmo2 (28Q/4KV, D=128, block_size=16)

## Correctness Gate

Mean cosine: 0.999701, min cosine: 0.998349 — **PASS**

## Kernel Benchmark (paths a vs b)

| Prefill | Decompress-all (μs) | INT8 (μs) | Speedup | Tput base (tok/s) | Tput INT8 (tok/s) | Projection | Status |
|--------:|--------------------:|----------:|--------:|------------------:|------------------:|-----------:|-------:|
| 128 | 273 | 150 | 1.81x | 469,621 | 850,611 | 1.0-1.3x | above |
| 512 | 277 | 198 | 1.40x | 1,846,842 | 2,590,438 | 1.1-1.5x | within |
| 1,024 | 348 | 388 | 0.90x | 2,945,915 | 2,636,050 | 1.3-1.8x | below |
| 2,048 | 641 | 1,074 | 0.60x | 3,196,654 | 1,906,571 | 1.3-2.0x | below |

## INT8 Prefill Gate Recommendation

**Recommendation:** `disable-by-default`

INT8 prefill shows 1.15x median speedup (min 0.60x, max 1.81x) — marginal overall. Keep TQ4_USE_INT8_PREFILL disabled by default; users can opt in.

## Notes

- Values are p50 (median) from CUDA event timing.
- Decompress-all: gather pages → tq4_decompress → SDPA (causal, enable_gqa).
- INT8 prefill: fused_paged_tq4_int8_prefill() (IMMA Q@K^T + FP16 P@V).
- Experiment 017 showed ~0.99x INT8/FP16 ratio with naive single-stage kernels. Speedup depends on autotune optimization (Task 3).
- Warmup: 10, timed: 50 iterations per condition.
