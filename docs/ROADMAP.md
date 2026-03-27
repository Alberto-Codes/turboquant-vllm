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
| AMD ROCm validation | **Done** | Triton HIP backend, 1.0 cosine, 0.31 ms/call on 890M (experiment 008) |
| Multi-layer integration | **Blocked** | Needs Flash Attention-style fusion (see below) |

**Key finding:** A fused Q@K^T-only kernel does not match SDPA precision when composed across all 36 transformer layers. The fp32 kernel scores differ from bf16 SDPA scores by 0.023 cosine per layer, which compounds into degenerate output. Full Flash Attention-style fusion (Q@K^T + online softmax + V matmul in one kernel) is required for multi-layer correctness. This is a 1-2 week project using the Triton Flash Attention tutorial as scaffold.

**Cross-platform:** The Triton kernel works on both NVIDIA (CUDA) and AMD (ROCm/HIP) with zero code changes. The multi-layer precision issue is platform-independent — P5 fix applies to both.

---

## Future Work

### P7: AMD ROCm platform support (Radeon 890M / gfx1150)

**Goal:** Enable TurboQuant development and validation on AMD integrated GPUs, starting with Radeon 890M (RDNA 3.5, gfx1150) on a Ryzen AI 9 HX 370 laptop running Bazzite (immutable Fedora).

**Context:** gfx1150 lacks official ROCm support as of March 2026. The `HSA_OVERRIDE_GFX_VERSION=11.0.0` workaround enables PyTorch GPU detection in containerized environments. Core algorithm is device-agnostic and works on CPU without modification.

**Research:** See `_bmad-output/planning-artifacts/research/technical-rocm-amd-igpu-pytorch-inference-research-2026-03-26.md` for full feasibility assessment.

#### Phase 0 — Smoke Test (2026-03-26, COMPLETED)

| Step | Action | Result |
|------|--------|--------|
| 0.1 | Pull ROCm PyTorch container via Podman | ✅ `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` |
| 0.2 | Set `HSA_OVERRIDE_GFX_VERSION=11.0.0` | ✅ `torch.cuda.is_available()` → True |
| 0.3 | Run `torch.cuda.get_device_name(0)` | ✅ "AMD Radeon Graphics", 32 GB, reports gfx1100 |
| 0.4 | Run simple matmul on GPU | ✅ 1000x1000 and 2000x2000 matmul, CPU/GPU match atol=1e-4 |
| 0.5 | Run test suite on CPU inside container | ✅ **62/62 tests pass** (12.4s) |
| 0.6 | Run TurboQuant ops on GPU, cross-validate vs CPU | ✅ Bit-identical quantization, 0.995 cache cosine, no NaN/Inf |

**Findings:** Initial attempts crashed with `Memory critical error — Memory in use` on all HSA override values (`11.0.0`, `11.0.1`, `11.0.2`, `11.5.0`, `11.5.1`). Root cause: **SELinux label enforcement** on Bazzite blocks `hipMalloc` inside Podman containers. Fix: `--security-opt=label=disable`.

With SELinux labels disabled + `HSA_OVERRIDE_GFX_VERSION=11.0.0`:
- GPU compute fully functional (1000x1000 and 2000x2000 matmul)
- CPU/GPU agreement within atol=1e-4 (max diff 0.004 — normal fp divergence)
- Memory allocation working (71.6 MB allocated, 86.0 MB reserved)
- All 62 TurboQuant tests pass on CPU inside container
- ROCm 7.11 preview also provides native gfx1150 wheels at `https://repo.amd.com/rocm/whl/gfx1150/`

**Working Podman command:**
```bash
podman run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  --security-opt=label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -v ~/Projects/turboquant-consumer:/workspace:z \
  -w /workspace \
  docker.io/rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

#### Phase 1 — Dev Environment (COMPLETED 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 1.1 | Create `Containerfile` for dev environment (ROCm + project deps) | ✅ `infra/Containerfile.rocm` + `infra/run-rocm.sh` |
| 1.2 | Mount project sources as volumes | ✅ Handled by `run-rocm.sh` (+ HF cache mount) |
| 1.3 | Add cross-device validation test fixtures (CPU vs GPU) | ✅ 21 tests parametrized, 84/84 pass on AMD GPU |
| 1.4 | Verify `uv sync` works inside container with ROCm PyTorch | ⚠️ `uv sync` installs CUDA torch from PyPI — use `PYTHONPATH=/workspace/src` instead |

**Phase 1.3 Results — Cross-Device Test Parametrization (2026-03-26):**

Spike audit found **zero source changes needed** — all internal state tensors (`codebook.centroids`, `codebook.boundaries`, `self.rotation`, `self.qjl_matrix`) already use `.to(input.device)` before operations.

Implementation: Added `device` fixture in `conftest.py` parametrized with `["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]`. 21 tests across `test_lloyd_max.py`, `test_quantizer.py`, and `test_compressors.py` now run on both CPU and GPU. GPU tests skip gracefully when CUDA is unavailable.

Validated inside ROCm container on Radeon 890M (gfx1150): **84/84 tests passed** with no tolerance relaxation — all existing `atol`, cosine similarity, and correlation thresholds hold on AMD GPU.

#### Phase 2 — Core Algorithm on AMD (COMPLETED 2026-03-26)

**Session 1 — Quick wins (gates Phase 3):**

| Step | Action | GPU? | Status |
|------|--------|------|--------|
| 2.1 | `torch.compile(mode="default")` spike on ROCm | Yes | ✅ Both `default` and `reduce-overhead` pass, 1.17x speedup, perfect eager parity |
| 2.2 | KV cache test parametrization — add `device` fixture to `test_kv_cache.py` (basic update, nibble packing, VRAM savings) | Yes | ✅ 11 tests parametrized, 95/95 pass on AMD GPU |

**Session 2 — Coverage & hardening:**

| Step | Action | GPU? | Status |
|------|--------|------|--------|
| 2.3 | Push test coverage above 90% — identify uncovered paths | No | ✅ 99% coverage (338/340 stmts), threshold raised to 95% |
| 2.4 | Codebook solver convergence — check edge cases (low/high dims, extreme bit widths) | No | ✅ Added exact Beta path, 6-bit (64 levels), and boundary edge case tests |

**Session 3 — Optional research:**

| Step | Action | GPU? | Status |
|------|--------|------|--------|
| 2.5 | 2-bit and 5-bit support — extend `solve_lloyd_max` and tests | No | ✅ Code already generic; added tests for bits 2-5 across quantizer, codebook, and KV cache |
| 2.6 | CompressedDynamicCache API ergonomics review | No | ✅ API is clean — consistent constructors, well-structured exports, no sharp edges |

#### Phase 3 — End-to-End Validation (COMPLETED 2026-03-27)

| Step | Action | Status |
|------|--------|--------|
| 3.1 | Verify Molmo2-4B weights accessible (~8 GB) | ✅ Model accessible, 4 shards fetched |
| 3.2 | Run baseline inference (no compression) on GPU | ✅ 5.0 tok/s, 10,052 MiB peak, coherent output |
| 3.3 | Run TQ4 compressed inference on GPU | ✅ 4.3 tok/s, 3.76x KV compression, coherent output |
| 3.4 | Cross-validate: same inputs on CPU vs GPU | ✅ **16/16 tokens match** (100%) — HSA override is safe |
| 3.5 | Benchmark actual throughput on 890M | ✅ 5.0 tok/s baseline, 0.86x TQ4 overhead |

**Validation script:** `experiments/experiment_007_e2e_amd_validation.py`

```bash
# All steps in one run (inside ROCm container):
./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py

# Skip slow CPU cross-validation:
./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py --skip-cross-validate

# Custom settings:
./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py \
    --model allenai/Molmo2-4B --bits 4 --max-new-tokens 64
```

Results: `experiments/logs/experiment-007-e2e-amd-validation.json`

**Phase 3 key findings:**
- HSA override (gfx1150 → gfx1100) introduces zero token-level precision errors
- TQ4 compression overhead is only ~15% at short sequences (0.86x throughput)
- 890M throughput (~5 tok/s) is ~10-12x slower than 4090 (~50-60 tok/s), matching bandwidth ratio prediction
- ROCm SDPA warnings (`Mem Efficient` / `Flash Efficient` experimental) — output correct despite warnings

**Known limitations:**
- ~11-16x slower than RTX 4090 (DDR5 ~90 GB/s vs GDDR6X ~1 TB/s)
- `torch.compile` works on ROCm 7.1 (both `default` and `reduce-overhead` modes pass with TurboQuant ops, 1.17x speedup)
- Fused Triton kernel (P3b) works on ROCm via HIP backend (experiment 008) — multi-layer precision issue (P5) is platform-independent
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

**Platform:** Triton HIP backend confirmed working for the Q@K^T kernel (experiment 008), so P5 should also work cross-platform (NVIDIA + AMD ROCm).

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
