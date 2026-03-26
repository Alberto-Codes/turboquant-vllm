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
| 005 | 2026-03-25 | Passed | Incremental dequant: 3.76x compression with 1.78x overhead (down from 3.36x) |

---

### P1: Long-sequence regression test

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Multi-layer precision test | Done | 4 | 36 layers, 1024 prefill + 32 gen steps, >0.999 cosine sim |
| TQ4 regression at scale | Done | 1 | Same scale test for 4-bit nibble-packed path |

### P2: TQ4 nibble packing (3.76x compression)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `_nibble_pack` / `_nibble_unpack` | Done | 1 | Bit-shift pack/unpack, exact round-trip verified |
| `CompressedDynamicCache` bits=4 | Done | 7 | Auto-enabled at bits=4, transparent to callers |
| `_CompressedLayer.packed` flag | Done | — | Tracks packing format through cat/stats |

### P3: Incremental dequantization (1.78x overhead)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Incremental dequant in `CompressedDynamicCache` | Done | — | Dequantize only new tokens, maintain running buffer |

**Experiment 005 results:** 3.76x compression with 1.78x overhead (down from 3.36x full-cache dequant). Near-identical output quality preserved.

### Compression Summary

| Mode | Bytes/block | Compression | Quality | Overhead | Status |
|------|-------------|-------------|---------|----------|--------|
| FP16 baseline | 256 | 1.0x | — | — | — |
| TQ3 uint8 | 132 | 1.94x | ~95% cosine | 2.35x | Done |
| TQ4 full-cache dequant | 68 | 3.76x | ~97% cosine | 3.36x | Done |
| **TQ4 incremental dequant** | **68** | **3.76x** | **~97% cosine** | **1.78x** | **Done** |
| TQ3 bit-packed | 52 | 4.92x | ~95% cosine | — | Deferred (P5) |

**Projected VRAM for Molmo2-4B (36 layers, 8 KV heads, 11K tokens):**

| Mode | KV Cache Size | Savings vs FP16 |
|------|--------------|-----------------|
| FP16 baseline | 1,639 MiB | — |
| TQ3 uint8 | 845 MiB | 794 MiB (1.94x) |
| **TQ4 nibble** | **436 MiB** | **1,203 MiB (3.76x)** |

---

## In Progress

### P3b: Fused Triton Q@K^T kernel (validated, needs Flash Attention fusion)

| Component | Status | Notes |
|-----------|--------|-------|
| Fused Q@K^T Triton kernel | **Done** | Nibble unpacking + pre-rotation trick, 17.8x speedup |
| Micro-benchmark (11K tokens) | **Done** | 1.0 cosine similarity vs unfused reference |
| Single-layer Molmo2-4B integration | **Done** | Correct output with fused kernel |
| Multi-layer integration | **Blocked** | Needs Flash Attention-style fusion (see below) |

**Key finding:** A fused Q@K^T-only kernel does not match SDPA precision when composed across all 36 transformer layers. The fp32 kernel scores differ from bf16 SDPA scores by 0.023 cosine per layer, which compounds into degenerate output. Full Flash Attention-style fusion (Q@K^T + online softmax + V matmul in one kernel) is required for multi-layer correctness. This is a 1-2 week project using the Triton Flash Attention tutorial as scaffold.

---

## Future Work

### P7: AMD ROCm platform support (Radeon 890M / gfx1150)

**Goal:** Enable TurboQuant development and validation on AMD integrated GPUs, starting with Radeon 890M (RDNA 3.5, gfx1150) on a Ryzen AI 9 HX 370 laptop running Bazzite (immutable Fedora).

**Context:** gfx1150 lacks official ROCm support as of March 2026. The `HSA_OVERRIDE_GFX_VERSION=11.0.0` workaround enables PyTorch GPU detection in containerized environments. Core algorithm is device-agnostic and works on CPU without modification.

**Research:** See `_bmad-output/planning-artifacts/research/technical-rocm-amd-igpu-pytorch-inference-research-2026-03-26.md` for full feasibility assessment.

#### Phase 0 — Smoke Test

| Step | Action | Success Criteria |
|------|--------|-----------------|
| 0.1 | Pull ROCm PyTorch container via Podman | Container runs |
| 0.2 | Set `HSA_OVERRIDE_GFX_VERSION=11.0.0` | `torch.cuda.is_available()` → True |
| 0.3 | Run `torch.cuda.get_device_name(0)` | Returns Radeon 890M identifier |
| 0.4 | Run simple matmul on GPU | Correct result, no crash |
| 0.5 | Run test suite on CPU inside container | All 62 tests pass |
| 0.6 | Run test suite on GPU inside container | Assess pass rate |

#### Phase 1 — Dev Environment

| Step | Action | Depends On |
|------|--------|-----------|
| 1.1 | Create `Containerfile` for dev environment (ROCm + project deps) | Phase 0 |
| 1.2 | Mount project sources as volumes | 1.1 |
| 1.3 | Add cross-device validation test fixtures (CPU vs GPU) | 1.1 |
| 1.4 | Verify `uv sync` works inside container with ROCm PyTorch | 1.1 |

#### Phase 2 — Core Algorithm on AMD

| Work Stream | GPU Required? | Priority |
|------------|--------------|----------|
| Improve codebook solver convergence | No | High |
| Extend test coverage beyond 85% | No | High |
| Add support for additional bit widths (2-bit, 5-bit) | No | Medium |
| Benchmark CompressedDynamicCache API ergonomics | No | Medium |
| Explore `torch.compile(mode="default")` on ROCm | Yes | Medium |

#### Phase 3 — End-to-End Validation (if GPU works)

| Step | Action | Notes |
|------|--------|-------|
| 3.1 | Download Molmo2-4B weights (~8 GB) | HuggingFace auth required |
| 3.2 | Run baseline inference (no compression) on GPU | Correctness baseline |
| 3.3 | Run TQ4 compressed inference on GPU | Compare output quality |
| 3.4 | Cross-validate: same inputs on CPU vs GPU | Catch HSA override precision issues |
| 3.5 | Benchmark actual throughput on 890M | Establish performance baseline |

**Known limitations:**
- ~11-16x slower than RTX 4090 (DDR5 ~90 GB/s vs GDDR6X ~1 TB/s)
- `torch.compile` `reduce-overhead` mode broken on ROCm — must use `mode="default"`
- Fused Triton kernel (P3b/P5) is NVIDIA-only; requires porting or alternative approach
- `hipMallocManaged()` not supported on gfx1150 as of ROCm 7.2

**Decision framework:**
```
Phase 0 Smoke Test
       │
       ├─ GPU detected + tests pass → Phase 1 → Phase 2 + 3
       │
       └─ GPU NOT detected → CPU-only development (still valuable)
                               └─ Monitor ROCm releases for gfx1150 support
```

---

### P4: Molmo2-8B validation

**Goal:** Confirm CompressedDynamicCache works with the larger model. The 8B model recognizes character names (e.g., "Elaine", "Kramer") which 4B cannot.

**Approach:** Run benchmark with `--model allenai/Molmo2-8B --compressed`. May need `bitsandbytes` 4-bit weight quantization to fit model + compressed cache in 24 GB.

**When:** After P3 (incremental dequant) eliminates the decode overhead.

### P5: Flash Attention-style fused kernel (research)

**Goal:** Fuse the full attention computation (Q@K^T + online softmax + V matmul) into a single Triton kernel that reads nibble-packed indices directly.

**Why:** The P3b Q@K^T-only kernel achieves 17.8x on the micro-benchmark but can't maintain SDPA precision across 36 layers. A full Flash Attention-style kernel would match SDPA's online softmax behavior while reading compressed keys.

**Complexity:** 1-2 weeks. Use the Triton Flash Attention tutorial as scaffold, inject centroid gather + nibble unpack into the inner tile loop.

### P6: TQ3 bit-packing (research, nice-to-have)

**Goal:** Pack 3-bit indices at the theoretical optimum (48 bytes per 128 indices, 4.92x compression).

**Why deferred:** 3-bit indices cross byte boundaries, making parallel pack/unpack non-trivial. No PyTorch/Triton implementation exists — only C/CUDA (ik_llama.cpp). The 30% improvement over TQ4 nibble (4.92x vs 3.76x) doesn't justify the complexity until the easier wins are shipped.

### P6: vLLM native integration

**Goal:** Run TurboQuant as a vLLM KV cache backend, enabling compressed caching in production serving.

**Status:** vLLM currently supports FP8 KV cache only. No integer sub-byte support. The KV Offloading Connector API (Jan 2026) handles offloading tiers, not quantization. Integrating TurboQuant would require attention backend changes, not just connector API.

**Our path:** Monitor upstream. When vLLM ships TurboQuant support (expected Q2-Q3 2026), we adopt on day one with confidence from our validation work. Our codebase could also serve as a reference implementation for a vLLM contribution.

---

## Hardware Context

### Primary — Desktop (RTX 4090)

| Component | Spec | Relevance |
|-----------|------|-----------|
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) | All benchmarks run here |
| CPU | AMD 7800X3D | Codebook solving, data loading |
| RAM | 128 GB DDR5 | Model offloading when needed |
| Target model | Molmo2-4B (experiments) / 8B (future) | Vision-language model for video analysis |
| Target workload | Seinfeld clip analysis | 11K+ visual tokens at 2fps |
| Production stack | vLLM in Podman (CDI GPU) | Currently FP8 KV cache |

### Secondary — Laptop (Radeon 890M iGPU)

| Component | Spec | Relevance |
|-----------|------|-----------|
| GPU | AMD Radeon 890M (32 GB shared VRAM, gfx1150 RDNA 3.5) | ROCm via HSA override (P7) |
| CPU | AMD Ryzen AI 9 HX 370 (12C/24T, 5.16 GHz) | Algorithm dev, CPU-path testing |
| NPU | AMD XDNA (Strix) | Future exploration (incompatible with custom cache ops) |
| RAM | 64 GB DDR5 | Shared with iGPU VRAM |
| OS | Bazzite 43 (immutable Fedora) | Podman-native, no system package installs |
| Dev environment | Podman + ROCm container | `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Bandwidth | ~90 GB/s (DDR5) vs ~1 TB/s (4090 GDDR6X) | 11-16x slower end-to-end |

---

## Key Lessons

1. **FP16 norms are a trap.** At 10K+ token sequences across 36 layers, fp16 norm precision loss compounds and flips low-confidence logits. Always use fp32 for norms.

2. **QJL is invisible in drop-in mode.** Standard attention does `Q @ K.T` on decompressed keys. QJL correction only helps with `estimate_inner_product()` (custom kernel). Using QJL in drop-in mode wastes 1 bit of MSE resolution for nothing.

3. **Peak VRAM != KV cache size.** On Molmo2-4B with 11K tokens, forward-pass activations dominate peak VRAM (~90%). KV cache compression savings are real but invisible to `max_memory_allocated()`. They matter for max_model_len budgeting, not peak measurement.

4. **PyTorch treats uint8 as boolean masks.** Fancy indexing with uint8 tensors triggers boolean masking, not integer indexing. Always cast to `.long()` before centroid lookup.

5. **Don't fight byte alignment.** TQ4 nibble packing (2 values per byte) is trivial and gives 3.76x compression. TQ3 bit-packing (3-bit byte-crossing) is hard and only 30% better. Work with the hardware, not against it.

6. **No PyTorch sub-byte ecosystem.** `torch.uint3` etc. are placeholders with no ops. TorchAO packing is weight-quant-specific. Every KV cache implementation rolls its own Triton kernels. Plan accordingly.
