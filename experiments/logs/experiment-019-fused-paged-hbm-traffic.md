# Experiment 019: Fused Paged TQ4 HBM Traffic Analysis

**Phase:** D9 Phase 3a (FP16 decode)
**Date:** 2026-03-28
**Story:** 6.2 -- Fused Paged TQ4 Decode Kernel

## Summary

The fused paged TQ4 decode kernel reads compressed blocks directly from
vLLM's page table and decompresses in SRAM, eliminating the HBM
round-trip for decompressed FP16 data.

## Per-Token HBM Traffic

| Path | Read (B) | Write (B) | Total (B) |
|------|----------|-----------|-----------|
| **Fused paged** | 136 | 0 | **136** |
| Reference (decompress-all) | 136 | 512 | 1,160* |

*Reference total includes re-read of decompressed FP16 data by Flash Attention.

**Bandwidth reduction: 8.5x** (136 vs 1,160 bytes/token)

## Byte Breakdown (Molmo2: 4 KV heads, D=128)

### Fused path (single kernel, no HBM temp buffer)
- K compressed read: 4 heads * 68 bytes = 272 bytes (indices + norms)
- V compressed read: 4 heads * 68 bytes = 272 bytes
- Total per token: 544 bytes across all heads
- Per K or V per head: 68 bytes (64 nibble-packed + 4 fp32 norm)

### Reference path (three serial HBM operations)
1. **Read** compressed KV from HBM: 136 bytes/token/head-pair
2. **Write** decompressed FP16 KV to HBM temp: 512 bytes/token
3. **Read** decompressed FP16 KV for Flash Attention: 512 bytes/token

## Why This Matters

Decode attention on RTX 4090 is deeply memory-bound:
- Arithmetic intensity: 27.2 FLOPs/byte
- Ridge point: 327 FLOPs/byte
- Gap: 12x below compute-bound regime

The 8.5x HBM bandwidth reduction directly translates to throughput
improvement because the kernel is bandwidth-limited, not compute-limited.

## Validation

- Cache parity (fused vs decompress-all): >0.999 cosine across all layers
- Kernel correctness (fused vs contiguous ref): >0.998 cosine
- 36-layer composition: PASS (1024 prefill + 32 gen tokens)
- GQA configs tested: Molmo2 (28Q/4KV), Llama (32Q/8KV), Mistral (32Q/8KV)
