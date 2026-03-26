# TurboQuant Consumer — Roadmap

Implementation status and path forward for TurboQuant KV cache compression
on consumer GPUs (RTX 4090, 24 GB VRAM) with Molmo2 vision-language models.

**Paper:** arXiv 2504.19874 — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)

---

## Completed

### Layer 1: Core Quantization Algorithm

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Lloyd-Max codebook solver | Done | 8 | Gaussian approx for d >= 64, `@lru_cache`d |
| `TurboQuantMSE` (Stage 1) | Done | 6 | Rotation + scalar quantize, ~95% cosine sim at 3-bit |
| `TurboQuantProd` (Stage 2) | Done | 5 | MSE + QJL correction, unbiased inner products |

### Layer 2a: Production Compressors

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `CompressedKeys` / `CompressedValues` | Done | — | Dataclass containers |
| `TurboQuantCompressorV2` (keys) | Done | 6 | Includes `asymmetric_attention_scores()` |
| `TurboQuantCompressorMSE` (values) | Done | 2 | MSE-only for value reconstruction |

### Layer 2b: Compressed KV Cache (real VRAM savings)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `TurboQuantKVCache` (accuracy-only) | Done | 7 | Compress-decompress round-trip, no VRAM savings |
| `CompressedDynamicCache` | Done | 13 | uint8 indices + fp32 norms, 1.94x compression |
| Benchmark harness (`--compressed`) | Done | — | A/B testing with Molmo2, VRAM + quality metrics |

### Experiments

| # | Date | Result | Key Finding |
|---|------|--------|-------------|
| 001 | 2026-03-25 | Failed | TurboQuantProd (2-bit MSE + 1-bit QJL) garbled output — QJL wasted in drop-in mode |
| 002 | 2026-03-25 | Passed | MSE-only fix: identical text output, coherent video (1.3x overhead) |
| 003 | 2026-03-25 | Passed | CompressedDynamicCache: coherent output, 1.94x compression, fp16 norms bug found and fixed |

---

## Next Up

### P1: Long-sequence regression test

**Goal:** Catch precision bugs like the fp16 norms issue in CI, without needing a GPU or real model.

**Approach:** Synthetic test that creates a multi-layer cache (e.g., 36 layers, 1000+ tokens), compresses/decompresses, and verifies that the accumulated error doesn't exceed a threshold. Compare `CompressedDynamicCache` output vs `TurboQuantKVCache` output across many layers.

### P2: TQ4 nibble packing (3.76x compression)

**Goal:** Switch from TQ3 (3-bit, uint8) to TQ4 (4-bit, nibble-packed) for nearly 2x better compression with *better* quality and trivial packing logic.

**Why TQ4 over TQ3 packing:**

| Config | Bytes/block (128 indices) | Compression | Quality | Packing Complexity |
|--------|--------------------------|-------------|---------|-------------------|
| TQ3 uint8 (current) | 132 | 1.94x | ~95% cosine sim | None |
| **TQ4 nibble-packed** | **68** | **3.76x** | **~97% cosine sim** | **Trivial** |
| TQ3 bit-packed | 52 | 4.92x | ~95% cosine sim | Hard (3-bit byte-crossing) |

TQ4 nibble packing wins on every axis that matters right now:
- **Better quality** — 16 centroids vs 8, ~97% vs ~95% cosine similarity
- **Nearly 2x better compression** — 3.76x vs 1.94x
- **Trivial packing** — `(a << 4) | b` to pack, `>> 4` and `& 0xF` to unpack
- **No Triton kernel needed** — pure PyTorch bit-shift ops

3-bit packing is genuinely hard (bits cross byte boundaries) and no PyTorch/Triton implementation exists. The only working 3-bit packer is in C/CUDA (ik_llama.cpp). The extra 30% compression (4.92x vs 3.76x) isn't worth a Triton kernel at this stage.

**Research backing:**
- PyTorch `torch.uint1`-`torch.uint7` are placeholder dtypes only — no ops
- TorchAO has int4 packing but it's weight-quant-specific, not reusable
- Every Python TurboQuant implementation (Dejan.ai, tonbistudio) uses wasteful uint8
- The ik_llama.cpp 3-bit layout (52 bytes/block) is the gold standard but C-only

**Approach:** Add `--bits 4` support to `CompressedDynamicCache` with nibble-packed uint8 storage. Pack pairs of 4-bit indices into single bytes. The Lloyd-Max solver already supports arbitrary bit widths.

**Projected VRAM for Molmo2-4B (36 layers, 8 KV heads, 11K tokens):**

| Mode | KV Cache Size | Savings vs FP16 |
|------|--------------|-----------------|
| FP16 baseline | 1,639 MiB | — |
| TQ3 uint8 (current) | 845 MiB | 794 MiB (1.94x) |
| **TQ4 nibble** | **436 MiB** | **1,203 MiB (3.76x)** |

---

## Future Work

### P3: Molmo2-8B validation

**Goal:** Confirm CompressedDynamicCache works with the larger model. The 8B model recognizes character names (e.g., "Elaine", "Kramer") which 4B cannot.

**Approach:** Run benchmark with `--model allenai/Molmo2-8B --compressed`. May need `bitsandbytes` 4-bit weight quantization to fit model + compressed cache in 24 GB.

**When:** After TQ4 nibble packing is validated on 4B. Using 4B for experiments is faster to set up and tear down.

### P4: Triton fused dequant-attention kernel

**Goal:** Fuse dequantization with attention computation to avoid materializing full decompressed tensors.

**Why:** Current `CompressedDynamicCache` dequantizes the ENTIRE cache at every layer at every generation step (11K+ vectors through a 128x128 matrix multiply). This is the source of the 2.35x overhead. A fused kernel would:
1. Read nibble-packed indices + fp32 norms directly
2. Compute centroid lookup + rotation + scaling inline
3. Compute Q @ K^T without materializing the full K tensor
4. Reduce both latency AND peak VRAM

**Complexity:** Medium-high. Requires Triton kernel development and correctness validation.

### P5: TQ3 bit-packing (research, nice-to-have)

**Goal:** Pack 3-bit indices at the theoretical optimum (48 bytes per 128 indices, 4.92x compression).

**Why deferred:** 3-bit indices cross byte boundaries, making parallel pack/unpack non-trivial. No PyTorch/Triton implementation exists — only C/CUDA (ik_llama.cpp). The 30% improvement over TQ4 nibble (4.92x vs 3.76x) doesn't justify the complexity until the easier wins are shipped.

### P6: vLLM native integration

**Goal:** Run TurboQuant as a vLLM KV cache backend, enabling compressed caching in production serving.

**Status:** vLLM currently supports FP8 KV cache only. No integer sub-byte support. The KV Offloading Connector API (Jan 2026) handles offloading tiers, not quantization. Integrating TurboQuant would require attention backend changes, not just connector API.

**Our path:** Monitor upstream. When vLLM ships TurboQuant support (expected Q2-Q3 2026), we adopt on day one with confidence from our validation work. Our codebase could also serve as a reference implementation for a vLLM contribution.

---

## Hardware Context

| Component | Spec | Relevance |
|-----------|------|-----------|
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) | All benchmarks run here |
| CPU | AMD 7800X3D | Codebook solving, data loading |
| RAM | 128 GB DDR5 | Model offloading when needed |
| Target model | Molmo2-4B (experiments) / 8B (future) | Vision-language model for video analysis |
| Target workload | Seinfeld clip analysis | 11K+ visual tokens at 2fps |
| Production stack | vLLM in Podman (CDI GPU) | Currently FP8 KV cache |

---

## Key Lessons

1. **FP16 norms are a trap.** At 10K+ token sequences across 36 layers, fp16 norm precision loss compounds and flips low-confidence logits. Always use fp32 for norms.

2. **QJL is invisible in drop-in mode.** Standard attention does `Q @ K.T` on decompressed keys. QJL correction only helps with `estimate_inner_product()` (custom kernel). Using QJL in drop-in mode wastes 1 bit of MSE resolution for nothing.

3. **Peak VRAM != KV cache size.** On Molmo2-4B with 11K tokens, forward-pass activations dominate peak VRAM (~90%). KV cache compression savings are real but invisible to `max_memory_allocated()`. They matter for max_model_len budgeting, not peak measurement.

4. **PyTorch treats uint8 as boolean masks.** Fancy indexing with uint8 tensors triggers boolean masking, not integer indexing. Always cast to `.long()` before centroid lookup.

5. **Don't fight byte alignment.** TQ4 nibble packing (2 values per byte) is trivial and gives 3.76x compression. TQ3 bit-packing (3-bit byte-crossing) is hard and only 30% better. Work with the hardware, not against it.

6. **No PyTorch sub-byte ecosystem.** `torch.uint3` etc. are placeholders with no ops. TorchAO packing is weight-quant-specific. Every KV cache implementation rolls its own Triton kernels. Plan accordingly.
