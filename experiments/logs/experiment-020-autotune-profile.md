# Experiment 020: Fused Paged TQ4 Decode Benchmark

**Date:** 2026-03-29T21:07:15.138006+00:00
**GPU:** NVIDIA GeForce RTX 4090
**Config:** Molmo2 (28Q/4KV, D=128, block_size=16)

## Correctness Gate

Mean cosine: 0.999999, min cosine: 0.999999 — **PASS**

## Kernel Benchmark (paths a vs b)

| Context | Decompress-all (μs) | Fused (μs) | Speedup | Projection | Status |
|--------:|--------------------:|-----------:|--------:|-----------:|-------:|
| 1,024 | 192 | 136 | 1.42x | 1.2-1.5x | within |
| 8,192 | 189 | 573 | 0.33x | 1.8-2.5x | below |
| 32,768 | 290 | 2,093 | 0.14x | 2.5-3.0x | below |

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

Warmup: 10, timed: 50 iterations per condition.
