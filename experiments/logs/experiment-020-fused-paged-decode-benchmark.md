# Experiment 020: Fused Paged TQ4 Decode Benchmark

**Date:** 2026-03-29T21:04:47.560712+00:00
**GPU:** NVIDIA GeForce RTX 4090
**Config:** Molmo2 (28Q/4KV, D=128, block_size=16)

## Correctness Gate

Mean cosine: 0.999998, min cosine: 0.999994 — **PASS**

## Kernel Benchmark (paths a vs b)

| Context | Decompress-all (μs) | Fused (μs) | Speedup | Projection | Status |
|--------:|--------------------:|-----------:|--------:|-----------:|-------:|
| 1,024 | 292 | 173 | 1.70x | 1.2-1.5x | above |
| 4,096 | 295 | 374 | 0.79x | 1.5-2.0x | below |
| 8,192 | 295 | 630 | 0.47x | 1.8-2.5x | below |
| 16,384 | 286 | 1,219 | 0.23x | 2.0-2.8x | below |
| 32,768 | 406 | 2,050 | 0.20x | 2.5-3.0x | below |

## Buffer Downsizing VRAM Savings

With fused decode enabled, decompress buffers (pre-allocated for CUDA graph compatibility) are downsized from full-cache capacity to max-prefill capacity, since decode no longer decompresses.

| Metric | Without fused | With fused |
|--------|--------------|------------|
| Buffer sizing | 32,768 tokens (full cache) | 2,048 tokens (max prefill) |
| K+V decompress buffers | 64.0 MiB | 4.0 MiB |
| Savings | — | 60.0 MiB (94%) |

## Analysis

Values are p50 (median) from CUDA event timing.
Decompress-all path: gather pages → tq4_decompress → SDPA (enable_gqa).
Fused path: fused_paged_tq4_decode() (pre-rotate → in-tile decompress + attention → post-rotate).

Warmup: 20, timed: 200 iterations per condition.
