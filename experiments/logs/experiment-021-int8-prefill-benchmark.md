# Experiment 021: INT8 Fused Paged TQ4 Prefill Benchmark

**Date:** 2026-03-29T21:05:49.335635+00:00
**GPU:** NVIDIA GeForce RTX 4090
**Config:** Molmo2 (28Q/4KV, D=128, block_size=16)

## Correctness Gate

Mean cosine: 0.999701, min cosine: 0.998349 — **PASS**

## Kernel Benchmark (paths a vs b)

| Prefill | Decompress-all (μs) | INT8 (μs) | Speedup | Tput base (tok/s) | Tput INT8 (tok/s) | Projection | Status |
|--------:|--------------------:|----------:|--------:|------------------:|------------------:|-----------:|-------:|
| 128 | 292 | 147 | 1.98x | 438,837 | 867,855 | 1.0-1.3x | above |
| 512 | 297 | 190 | 1.56x | 1,723,848 | 2,688,228 | 1.1-1.5x | above |
| 1,024 | 519 | 515 | 1.01x | 1,972,379 | 1,990,127 | 1.3-1.8x | below |
| 2,048 | 855 | 1,188 | 0.72x | 2,396,639 | 1,723,587 | 1.3-2.0x | below |

## INT8 Prefill Gate Recommendation

**Recommendation:** `disable-by-default`

INT8 prefill shows 1.29x median speedup (min 0.72x, max 1.98x) — marginal overall. Keep TQ4_USE_INT8_PREFILL disabled by default; users can opt in.

## Notes

- Values are p50 (median) from CUDA event timing.
- Decompress-all: gather pages → tq4_decompress → SDPA (causal, enable_gqa).
- INT8 prefill: fused_paged_tq4_int8_prefill() (IMMA Q@K^T + FP16 P@V).
- Experiment 017 showed ~0.99x INT8/FP16 ratio with naive single-stage kernels. Speedup depends on autotune optimization (Task 3).
- Warmup: 10, timed: 100 iterations per condition.
