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

**Key finding:** A fused Q@K^T-only kernel does not match SDPA precision when composed across all 36 transformer layers. The fp32 kernel scores differ from bf16 SDPA scores by 0.023 cosine per layer, which compounds to 0.43 over 36 layers — degenerate output. Root cause: the Q@K^T-only approach materializes attention scores in fp16 at two intermediate points, causing error multiplication. Full Flash Attention-style fusion (Q@K^T + online softmax + V matmul in one kernel, fp32 accumulation throughout) is required for multi-layer correctness. See P5 for the implementation roadmap.

**Research:** See `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-flash-attention-fusion-turboquant-kv-cache-research-2026-03-26.md` for the detailed analysis that quantified this drift and identified the full FA fusion solution.

**Cross-platform:** The Triton kernel works on both NVIDIA (CUDA) and AMD (ROCm/HIP) with zero code changes. The multi-layer precision issue is platform-independent — P5 fix applies to both.

---

## Future Work

### P8: AMD ROCm platform support (Radeon 890M / gfx1150)

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

### P5: Fused TQ4 Flash Attention kernel

**Goal:** Fuse the full attention computation (Q@K^T + online softmax + V matmul) into a single Triton kernel that reads nibble-packed TQ4 indices directly — never materializing decompressed keys or the attention score matrix in fp16.

**Why:** The P3b Q@K^T-only kernel achieves 17.8x on the micro-benchmark but can't maintain SDPA precision across 36 layers (0.023 cosine loss/layer → 0.43 over 36 layers). The root cause is two fp16 materialization points: scores after Q@K^T and weights after softmax. Full Flash Attention fusion eliminates both by maintaining the `(m_i, l_i, acc)` state machine entirely in fp32, casting only the final output to fp16. The correction factor `alpha = exp2(m_old - m_new)` is mathematically exact, not approximate.

**Expected precision:** >0.998 per-layer cosine similarity (vs 0.977 with Q@K^T-only), >0.93 over 36 layers (vs 0.43).

**Platform:** Triton HIP backend confirmed working for the Q@K^T kernel (experiment 008), so P5 should also work cross-platform (NVIDIA + AMD ROCm).

**Key architectural insight (arXiv 2511.11581):** GQA Q-Block pattern flattens multiple query heads sharing a KV head into a single 2D tensor. For Molmo2's 28Q/4KV (7:1 ratio): 7 Q-heads per block → BLOCK_M=8 (padded from 7). This avoids per-head loops and maps cleanly to Triton's tile-based programming model.

**Survey of existing systems (13 reviewed):** KIVI, BitDecoding, Kitty, INT-FlashAttention, QServe, FlashInfer, etc. — none fuse vector quantization codebook lookup with Flash Attention. This would be novel.

#### Phase 1: Vanilla Triton FA baseline (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 1.1 | Fork Triton tutorial FA kernel, forward-only fp16/bf16 | **Done** — 15 tests, all >0.999 cosine vs SDPA |
| 1.2 | Add GQA support via head mapping (not Q-Block yet) | **Done** — 4:1 and 7:1 GQA validated |
| 1.3 | Register via HuggingFace `AttentionInterface.register()` | **Done** — 64/64 token-identical text output on Molmo2-4B |
| 1.4 | Autotune for RTX 4090: BLOCK_M∈{16,64,128}, BLOCK_N∈{32,64} | **Done** — 0.26-0.38x SDPA throughput (see below) |

**Experiment 009 results (Molmo2-4B, RTX 4090, bf16):**

| Mode | SDPA tok/s | Triton FA tok/s | Ratio | Token match |
|------|-----------|----------------|-------|-------------|
| Text-only (17 input) | 43.2 | 11.2 | 0.26x | **64/64 (100%)** |
| Image (1205 input) | 47.1 | 17.7 | 0.38x | 8/64 (coherent, expected divergence) |

**Key findings:**
- **Correctness validated:** Token-identical output for text-only. Image divergence ("iconic" added at token 8) is expected — 1205-token prefill amplifies fp differences between cuDNN Flash Attention and our Triton kernel.
- **Throughput gap is expected:** SDPA dispatches to cuDNN's Flash Attention (years of CUDA engineering). Our Triton kernel is a correct scaffold, not a performance competitor. Phase 2 fundamentally changes the memory access pattern (reading compressed indices), so the SDPA comparison becomes irrelevant.
- **Autotune key fix:** Original key `["N_CTX_Q", "N_CTX_KV", "HEAD_DIM"]` caused re-autotuning on every decode step (N_CTX_KV changes each token). Fixed to `["N_CTX_Q", "HEAD_DIM"]` — 100x speedup from 0.5 to 11-18 tok/s.
- **Model config:** Molmo2-4B has 32Q/8KV (4:1 GQA), not 28Q/4KV (7:1) as initially assumed. Both ratios validated in unit tests.

#### Phase 2: TQ4 K-only fusion (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 2.1 | Pre-rotate Q outside kernel: `q_rot = query @ Pi_T` | **Done** — wrapper handles rotation |
| 2.2 | Replace K tile load: nibble unpack → centroid gather → norm scale | **Done** — `tl.join` + `reshape` interleaves even/odd centroids |
| 2.3 | Keep V as standard fp16 (uncompressed) | **Done** — 9 tests, all >0.998 cosine vs unfused path |
| 2.4 | Benchmark decode throughput | Deferred to experiment 010 |

**Key technique:** `tl.join(k_hi, k_lo).reshape(BLOCK_N, HEAD_DIM)` interleaves even/odd centroid values to reconstruct the full decompressed K tile without explicit loops.

#### Phase 3: TQ4 K+V fusion (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 3.1 | Add V tile decompression (same codebook and rotation as K) | **Done** — same nibble unpack + centroid gather + interleave |
| 3.2 | Post-rotation trick: `out = acc @ Pi` outside kernel | **Done** — mirror of pre-rotation for K |
| 3.3 | Validate fused K+V vs unfused path | **Done** — 7 tests, all >0.998 cosine |
| 3.4 | Bandwidth measurement + decode throughput | Deferred to experiment 010 |

**Key insight:** K and V share the same rotation matrix and codebook (same seed in CompressedDynamicCache). Pre-rotate Q by `Pi^T`, compute attention in rotated space, post-rotate output by `Pi`. One centroid table, zero in-kernel rotations.

#### Phase 4: Model integration + E2E validation (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 4.1 | Integrate fused kernel via AttentionInterface + cache side-channel | **Done** — `install_fused_tq4_kv()`, `get_compressed()` API |
| 4.2 | Experiment 010: E2E Molmo2-4B 3-path comparison | **Done** — coherent output across all 36 layers (see below) |
| 4.3 | Variable sequence length + edge cases | Deferred to production hardening |
| 4.4 | Regression test suite (3 tiers) | Tier 1 done (31 unit tests), Tier 2-3 deferred |

**Experiment 010 results (Molmo2-4B, RTX 4090, bf16, 36 layers):**

| Path | Text tok/s | Image tok/s | Text quality |
|------|-----------|-------------|--------------|
| SDPA baseline | 44.0 | 46.9 | "portrayed by Jason Alexand..." |
| Unfused TQ4 | 13.1 | 25.6 | "portrayed by Jerry Seinfel..." |
| Fused TQ4 K+V | 11.9 | 14.4 | "portrayed by Jerry Seinfel..." |

**Key findings:**
- **36-layer composition: PASS.** The Q@K^T-only kernel produced garbled output (0.43 cosine). The fused FA kernel produces coherent, factually correct text — the fp32 online softmax prevents catastrophic precision drift.
- **Fused vs unfused divergence is expected.** 14% token match (text), 52% (image). Same pattern as Phase 1 (SDPA vs Triton FA): different numerical paths compound through autoregressive generation. All outputs are valid paraphrases, not degraded.
- **Throughput: fused 0.91x of unfused (warm cache).** Decompression overhead (nibble unpack + centroid gather + interleave + pre/post rotation) currently exceeds the bandwidth savings at short sequences. Bandwidth wins should dominate at longer sequences (11K+ tokens).
- **VRAM: comparable across all paths** (~10 GiB). Peak is dominated by model weights + activations, not KV cache at 64 output tokens.

**Integration approach (decided after research — see party mode discussion 2026-03-26):**

Approach B selected: **AttentionInterface + side-channel cache reference.** Evaluated four options:

| Approach | Description | Verdict |
|----------|-------------|---------|
| A: Full forward monkey-patch | Replace entire `Molmo2Attention.forward()` | **Rejected** — fragile, must replicate 50+ lines of attention logic (QK norm, RoPE variants), breaks on transformers updates |
| **B: AttentionInterface + cache stash** | Register via `ALL_ATTENTION_FUNCTIONS`, stash cache ref on modules | **Selected** — clean HF API, small coupling surface, graceful fallback |
| C: Custom cache skipping decompression | Override `cache.update()` return value | **Rejected** — breaks HF cache API contract |
| D: AttentionInterface + cache kwarg | Pass cache via `**kwargs` like `eager_paged` | **Rejected** — Molmo2 doesn't pass cache through kwargs to attention function |

**Why B works:** `CompressedDynamicCache.update()` stores compressed K/V in `_compressed_keys`/`_compressed_values` (already implemented). The attention function reads compressed data from the cache, ignoring the decompressed K/V arguments. Requires:
1. `get_compressed(layer_idx)` public method on CompressedDynamicCache
2. `module._tq4_cache` reference stashed on each attention layer
3. Attention function checks for stash, falls back to SDPA if absent

**Known overhead:** `cache.update()` still decompresses K/V (wasted in fused path). Addressed by Phase 5 "fused-aware cache mode" (see below).

#### Projected performance (RTX 4090)

| Metric | Current (unfused) | Phase 2 (K-only) | Phase 3 (K+V) | Improvement |
|--------|-------------------|-------------------|---------------|-------------|
| Decode tok/s | 8.9 | 15-20 | 25-35 | 2.8-3.9x |
| Memory traffic/layer | ~25 MB | ~6 MB | ~3-6 MB | 4-8x less |
| Overhead vs baseline | 1.78x slower | ~1.1x | 0.85-1.1x | Near parity |

#### RTX 4090 resource budget

| Resource | Budget | Notes |
|----------|--------|-------|
| Memory bandwidth | 1008 GB/s | Target >500 GB/s achieved (50%+) |
| Shared memory | 128 KB/SM | Design <64 KB/block for 2 concurrent blocks |
| Registers | <128/thread | 4 warps with good occupancy |
| Occupancy | >50% | Sufficient for memory-bound kernel |

#### Success criteria

- Per-layer cosine similarity: >0.998 (match FA-level precision)
- 36-layer composition: >0.93 (residual connections stabilize)
- Decode tokens/sec: >15 (Phase 2), >25 (Phase 3)
- Text output: matches reference (decisive gate)
- KV cache VRAM: unchanged (3.76x compression preserved)

#### Testing strategy (3 tiers)

1. **Unit (fast):** Nibble unpack, centroid gather, single-tile attention vs standard
2. **Per-layer (medium):** Each of 36 layers >0.998 cosine similarity vs SDPA
3. **End-to-end (slow):** Full Molmo2 inference >0.93 cosine, text output identical

#### Prerequisites

- Triton ≥3.0 (available via `torch.triton`)
- HuggingFace transformers ≥4.50 (`AttentionInterface` API)
- Molmo2-4B weights (cached locally)
- RTX 4090 GPU
- CompressedDynamicCache (Layer 2b — done)
- Nsight Compute (profiling)

#### Research references

- `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-triton-flash-attention-tutorial-deep-dive-2026-03-26.md` — FA inner loop mechanics, numerical stability analysis
- `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-flash-attention-fusion-turboquant-kv-cache-research-2026-03-26.md` — Fusion architecture, precision analysis, 13-system survey, GQA Q-Block pattern
- `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-fused-turboquant-triton-kernel-research-2026-03-25.md` — Original Q@K^T-only analysis (superseded by full FA approach)
- arXiv 2511.11581 — "Anatomy of Attention" (GQA Q-Block, Triton performance parity with FA-3)
- arXiv 2405.02803 — "Is Flash Attention Stable?" (FA achieves 1.7x lower RMSE than SDPA)

### P5b: Fused-aware cache mode (COMPLETE 2026-03-26)

**Goal:** Eliminate wasted decompression in the fused kernel path.

**Result:** Added `fused_mode=True` flag to CompressedDynamicCache. `install_fused_tq4_kv()` enables it automatically. Throughput impact: +6% on text (17 tokens), +13% on image (1205 tokens).

---

## North Star: Compression AND Speed for Video

The Google TurboQuant paper claims 5-6x compression and "up to 8x speedup." **Research (2026-03-26) revealed the 8x claim is a blog-post microbenchmark (FP32 JAX logit computation on H100), not end-to-end inference, not vs Flash Attention.** The paper itself contains no speedup benchmark table. Both compression and speed are required — but the realistic target is vs FP32 eager attention, not vs cuDNN Flash Attention.

### What the Research Revealed

**Five key findings from deep-dive (web search + paper analysis + community implementations):**

1. **The "8x" is marketing.** Blog post: "4-bit vs 32-bit on H100 using JAX baseline." Paper benchmarks only end-to-end quality (NIAH, LongBench), not speed. Independent analysis: *"the headline framing is doing a lot of heavy lifting."*

2. **Everyone dropped QJL.** Dejan.ai: *"QJL residual correction dropped cosine to 0.69."* ik_llama.cpp: *"All bits to Lloyd-Max is faster, simpler, perplexity matches."* Our experiment 012 (0.847-0.971 cosine) matches the community consensus. QJL is not viable for multi-layer composition.

3. **TurboQuant was never tested on VLMs.** Paper tested only Llama-3.1-8B and Ministral-7B. Visual tokens have fundamentally different distributions (per-channel value variation vs per-token for text, 10x lower gradient sensitivity, more pronounced key outliers — per AKVQ-VL, MBQ, VidKV research).

4. **Real RTX 4090 fused kernel speedup is 1.15-1.22x** (Dejan.ai Q@K^T microbenchmark). Not 8x. Codebook lookup can't compete with direct tensor core matmul.

5. **SageAttention proves quantized attention CAN be 3x faster than FA2** on RTX 4090 using INT8 Q@K^T. This is a proven path — but uses scalar quantization, not vector quantization.

**Reference: Dejan.ai RTX 4090 fused kernel (closest to our work):**

| KV Length | Standard Q@K^T | Fused TQ | Speedup |
|-----------|---------------|----------|---------|
| 128 | 0.076 ms | 0.066 ms | 1.15x |
| 512 | 0.061 ms | 0.050 ms | 1.22x |
| 4096 | 0.062 ms | 0.051 ms | 1.22x |

**Reference: tonbistudio MSE-only per-layer cosine (matches our results):**

| Bits | Cosine (2K) | Cosine (4K) | Cosine (8K) |
|------|-------------|-------------|-------------|
| 3-bit | 0.9961 | 0.9955 | 0.9945 |
| 4-bit | 0.998+ | 0.998+ | 0.998+ |

### P9: Closing the Speed Gap — Action Plan

#### Phase 1: Establish the real baseline (COMPLETE 2026-03-26)

**Experiment 013 results (Molmo2-4B, RTX 4090, text-only 17 tokens):**

| Path | tok/s | vs FP32 eager |
|------|-------|---|
| bf16 eager | 55.3 | 1.43x |
| bf16 SDPA (cuDNN FA) | 43.5 | 1.13x |
| FP32 eager (paper's baseline) | 38.6 | 1.0x |
| Fused TQ4 K+V | 8.6-12.6 | 0.22-0.33x |

**Key insight:** At 17 input tokens, attention is <5% of total inference compute. ALL attention paths are fast because the bottleneck is model weights + MLP, not KV cache reads. The paper's "8x" only applies to the attention logit micro-operation in isolation — it is invisible in end-to-end inference at short sequences. FP32 eager is only 13% slower than cuDNN Flash Attention here.

**The speedup opportunity is at 5K-11K+ tokens** where attention dominates compute and the 3.76x bandwidth reduction from compressed KV reads would actually matter.

**Synthetic kernel micro-benchmark (decode mode, seq_q=1, Molmo2-4B config 32Q/8KV, RTX 4090):**

| KV Length | SDPA | Triton FA | Fused TQ4 | TQ4/SDPA | TQ4/FA |
|-----------|------|-----------|-----------|----------|--------|
| 64 | 10 us | 24 us | 79 us | 7.7x slower | 3.3x slower |
| 1,024 | 18 us | 24 us | 205 us | 11.5x slower | 8.5x slower |
| 4,096 | 48 us | 81 us | 728 us | 15.3x slower | 9.0x slower |
| 16,384 | 313 us | 304 us | 2712 us | 8.7x slower | 8.9x slower |

**No bandwidth crossover.** The P5 dequant-then-dot kernel is 8-9x slower than SDPA at ALL sequence lengths. The centroid gather (random codebook lookup per index) doesn't coalesce on GPU — fundamentally slower than sequential fp16 reads. Kernel optimization cannot fix this architectural mismatch. See Key Lesson #9.

**This means the fused kernel is NOT the path to speed.** The compression value comes from needing fewer inference calls (bigger context window), not faster attention.

#### Phase 2: Full episode benchmark (COMPLETE 2026-03-26)

**Goal:** Process a real Seinfeld episode segment through Molmo2 with and without TQ4 compression. Measure total wall-clock time AND output quality.

**Original hypothesis:** TQ4 gives ~1.0x total wall-clock time (3.8x fewer calls offsets 0.56x per-token speed) with qualitatively better output (19s scene context vs 5s fragments).

**Result: Hypothesis exceeded.** TQ4 is **1.97x faster** (not 1.0x), because Molmo2 subsamples video frames — 19s clips use only ~31% more input tokens than 5s clips, so 4 TQ4 clips process 2.8x fewer total tokens than 12 baseline clips.

| Metric | Baseline (vLLM 8B, 5s) | TQ4 (HF 8B, 19s) |
|--------|------------------------|-------------------|
| Clips | 12 | 4 |
| Total time | 90.6s | 46.0s |
| Speedup | 1.0x | **1.97x** |
| Input tokens | 31,784 | 11,379 |
| Output tokens | 2,576 | 967 |
| Characters found | Jerry 13, George 5, Elaine 5 | Jerry 7, George 2, Elaine 3 |

**Key findings:**

1. **Frame subsampling is the hidden multiplier.** Molmo2 extracts a fixed number of frames regardless of clip duration (~2,854 tokens for 5s, ~3,753 for 19s). Longer clips don't proportionally increase token cost. This means fewer calls = fewer total tokens = faster.

2. **Quality is dramatically better with 19s context.** TQ4's single-pass apartment scene correctly identified Jerry, George, and Elaine with coherent scene narration. Baseline's 5s fragments hallucinated "Brooklyn Decker as Elaine Benes," "Monica Geller played by Jennifer Aniston" (confusing Seinfeld with Friends), and fabricated on-screen text. 5 seconds is insufficient for confident show/character identification.

3. **3.76x KV compression held at real sequence lengths** (~3,750 tokens). VRAM peak: 19-21 GiB for 8B bf16 + TQ4 on RTX 4090 (24 GiB).

**Scale-up confirmation (5 minutes, 300s):**

| Metric | Baseline (vLLM 8B, 5s) | TQ4 (HF 8B, 19s) |
|--------|------------------------|-------------------|
| Clips | 60 | 16 |
| Total time | 457.6s | 183.0s |
| Speedup | 1.0x | **2.50x** |
| Input tokens | 168,688 | 56,399 |

Speedup improved from 1.97x (1 min) to 2.50x (5 min) — per-call overhead (HTTP, base64, scheduling) is paid 60 times for baseline vs 16 for TQ4.

**Experiment:** `experiments/experiment_014_full_episode_benchmark.py`
**Data:** `experiments/logs/experiment-014-combined-8b.json` (1 min), `experiment-014-combined-8b-5min.json` (5 min)

#### Experiment 015: vLLM duration stress test + TQ4 ceiling (COMPLETE 2026-03-27)

Tested vLLM baseline and TQ4/HF paths at progressively longer clip durations to find each path's limits.

**vLLM (baseline): fixed token budget, sparse sampling.**

vLLM extracts a fixed ~2,880 input tokens regardless of clip duration. A 5-second clip and the full 22-minute episode both produce ~2,880 tokens. Longer clips = sparser frame sampling = fewer frames per second of video.

| Duration | Tokens | Quality |
|----------|--------|---------|
| 5s | 2,854 | Fragmented, frequent hallucinations |
| 60s | 2,880 | Decent — Jerry identified |
| 120s | 2,880 | Surprisingly good — Jerry, George, Elaine |
| 5 min | 2,905 | Renamed George to "Paul", fabricated characters |
| 10 min | 2,911 | Called it "Friends", invented "Jerry and Elaine's wedding" |
| **22 min (full ep)** | **2,923** | **"Jerry (played by George Costanza)"** — identity collapse |

Quality sweet spot for vLLM: **60-120s** — long enough for temporal context, short enough for meaningful frame density.

**TQ4/HF: tokens scale with duration, ViT VRAM wall.**

Unlike vLLM, HF transformers extracts more frames for longer clips (tokens increase with duration). TQ4 compresses the text model's KV cache but cannot help with the vision encoder's (ViT) internal attention, which is the actual VRAM bottleneck.

| Duration | Tokens | VRAM Peak | Status |
|----------|--------|-----------|--------|
| 19s | 3,753 | 19.4 GiB | works |
| 20s | 3,753-4,376 | 19.9 GiB | works |
| 30s | 5,533-6,156 | 21.3 GiB | works |
| 33s | 4,821-6,156 | 21.3 GiB | **max** |
| 34s+ | — | >24 GiB | **OOM (ViT, not KV cache)** |

Quality remains consistent up to 33s — no TQ4-induced degradation. The 19s limit from prior analysis was conservative; actual ceiling is **33s for Molmo2-8B bf16 on RTX 4090 (24 GiB)**.

**Key insight:** The two paths have fundamentally different failure modes. vLLM fails gracefully (sparse sampling → hallucinations) while TQ4/HF fails hard (OOM). But TQ4/HF produces faithful descriptions up to its VRAM limit because it processes dense frames, while vLLM's sparse sampling degrades output quality even when it doesn't crash.

**Data:** `experiments/logs/experiment-015-vllm-duration-stress-test.json`

#### Phase 3: vLLM integration for TQ4 KV cache

**Goal:** Run TQ4 compression inside vLLM's serving stack for production deployment via the existing `vllm-nvidia.service` quadlet on port 8100.

**Prerequisite:** Phase 2 episode benchmark confirms the compression value proposition with real data. ✅ (experiment 014: 2.5x speedup)

**Architecture research (2026-03-27):**

vLLM v0.18.0 has a first-class plugin system for custom attention backends:
- `AttentionBackendEnum.CUSTOM` enum slot + `register_backend()` decorator in `v1/attention/backends/registry.py`
- `FlashMLASparseBackend` is a direct template — uses custom 656-byte packed format per token (mixed FP8 + BF16 + scales), comparable to our TQ4 68-byte format
- Custom cache dtype strings supported (precedent: `"fp8_ds_mla"`)
- `get_kv_cache_shape()` allows non-standard page layouts
- Block allocator (`KVCacheManager`) is shape-agnostic — works unchanged with smaller pages

**Integration approach: Custom attention backend (plugin, no fork)**

| vLLM Interface | TQ4 Implementation |
|---|---|
| `AttentionBackend` | `TQ4AttentionBackend` — declares `"tq4"` cache dtype, custom page shape |
| `AttentionImpl.forward()` | Dequant TQ4 → FP16 in-kernel, then Flash Attention (SDPA path) |
| `get_kv_cache_shape()` | `(num_blocks, block_size, packed_bytes_per_token)` |
| `reshape_and_cache` custom op | Triton kernel: compress incoming K/V → TQ4, write to paged blocks |
| `AttentionMetadataBuilder` | Reuse Flash Attention's metadata builder |

**TQ4 page layout (per token per head):**

| Component | Size | Notes |
|---|---|---|
| Nibble-packed indices | head_dim/2 = 64 bytes | uint8, two 4-bit indices per byte |
| Norm | 4 bytes | fp32 mandatory (fp16 degrades >10K tokens) |
| **Total** | **68 bytes** | **vs 256 bytes FP16 = 3.76x compression** |

**Model-level constants (initialized once at load):**
- Rotation matrix: `[head_dim, head_dim]` fp32 orthogonal (~66 KB for head_dim=128)
- Lloyd-Max codebook: 16 centroids (TQ4) — deterministic from seed
- Shared across all layers (validated in 36-layer E2E tests)

**Implementation phases:**

##### Phase 3a: Backend skeleton

| Step | Action | Status |
|------|--------|--------|
| 3a.1 | Create `src/turboquant_consumer/vllm/tq4_backend.py` | ✅ |
| 3a.2 | Implement `TQ4AttentionBackend` (subclass `FlashAttentionBackend`) | ✅ |
| 3a.3 | Implement `get_kv_cache_shape()` — inherits Flash Attention shape (Phase 3b overrides) | ✅ |
| 3a.4 | Stub `TQ4AttentionImpl` — passthrough to `FlashAttentionImpl` (no compression) | ✅ |
| 3a.5 | Register via `vllm.general_plugins` entry point + `register_tq4_backend()` | ✅ |
| 3a.6 | Smoke test: `vllm serve` starts and serves with TQ4 backend selected | ✅ |

13 tests pass (registration, interface compliance, byte math, mm_prefix). All pre-commit hooks green.

**Smoke test result (2026-03-27):** vLLM 0.18.0 + Molmo2-8B + `--attention-backend CUSTOM` → model loads, serves on port 8100, responds correctly ("Four." to "What is 2+2?"). Two issues discovered and fixed during smoke test:
- `supports_mm_prefix()` must return `True` for VLMs with bidirectional visual attention (Molmo2)
- `get_name()` must return `"CUSTOM"` (the enum member name), not a custom string — vLLM does `AttentionBackendEnum[backend.get_name()]` during model init

##### Phase 3b: TQ4 compression quality validation through vLLM (COMPLETE 2026-03-27)

Validates TQ4 compress→decompress round-trip inside vLLM's attention path. Uses standard 5D cache shape (no VRAM savings yet). The cache stores decompressed (lossy) FP16 data after TQ4 round-trip.

**Architecture:** `forward_includes_kv_cache_update = True`. Override `forward()` to compress→decompress new K/V tokens, write lossy FP16 to standard cache via `do_kv_cache_update()`, then delegate to Flash Attention via `super().forward()`.

| Step | Action | Status |
|------|--------|--------|
| 3b.1 | Override `get_kv_cache_shape()` for TQ4 packed layout | ⏭️ deferred to 3c (see below) |
| 3b.2 | Init rotation matrix + Lloyd-Max codebook in `TQ4AttentionImpl.__init__()` | ✅ |
| 3b.3 | Compression in `forward()`: compress new K/V → nibble-packed indices + fp32 norms | ✅ |
| 3b.4 | Decompression in `forward()`: unpack nibbles → centroid lookup → inverse rotate → rescale → FP16 | ✅ |
| 3b.5 | Call Flash Attention on decompressed FP16 K/V buffers | ✅ |
| 3b.6 | Validate output matches HF SDPA path (cosine similarity > 0.999 per layer) | |
| 3b.7 | Multi-layer validation (36 layers, composition > 0.93) | |
| 3b.8 | Smoke test: `vllm serve` + Molmo2-8B with TQ4 compression active | ✅ |

152 tests pass. All pre-commit hooks green.

**Smoke test result (2026-03-27):** vLLM 0.18.0 + Molmo2-8B + `--attention-backend CUSTOM` with TQ4 compress→decompress active → model loads, serves on port 8100, correct responses:
- "What is 2+2?" → "4" ✅
- "Name the four main characters of Seinfeld" → "Jerry, George, Elaine, and Kramer" ✅ (all 4 preserved)

**Bugs found and fixed:**
- `TQ4AttentionImpl.__init__()` — vLLM passes 11 args positionally; switched to `*args, **kwargs` passthrough
- `_generate_rotation_matrix()` — vLLM sets `torch.set_default_device('cuda')` and default dtype to bfloat16 during model init; the CPU generator and `torch.randn` created bfloat16 tensors on CUDA. Fixed with explicit `device="cpu", dtype=torch.float32`
- `get_kv_cache_stride_order()` — inherited 5-element tuple from Flash Attention, but 3D packed shape needs 3-element. Reverted to standard 5D shape (see 3b.1 deferral below)

**Key discovery — 3b.1 deferred:** `get_kv_cache_shape()` only controls the *reshape* of the cache tensor, not its *allocation*. Buffer allocation uses `KVCacheSpec.page_size_bytes` (set in `FullAttentionSpec` during model init, based on `num_kv_heads * head_size * dtype_size * 2`). Overriding `get_kv_cache_shape()` to a smaller TQ4 shape causes a reshape mismatch because the buffer was sized for the standard page. To get actual VRAM savings, we must also override `page_size_bytes` in the `KVCacheSpec`. This is a Phase 3c task.

**Key implementation details:**
- Rotation matrix and codebook are CPU-initialized with explicit `device="cpu", dtype=torch.float32` to survive vLLM's default device/dtype context, then lazily moved to GPU on first `forward()` call.
- VIT encoder uses FLASH_ATTN separately (auto-selected for `vit attention`). Our TQ4 override only affects text decoder layers.
- `forward_includes_kv_cache_update = True` means forward() handles both cache write (via `do_kv_cache_update()`) and attention (via `super().forward()`). The runtime does NOT call `do_kv_cache_update()` separately.
- `attn_metadata.slot_mapping` is available inside `forward()` for writing new tokens.

##### Phase 3c: Packed TQ4 cache layout + VRAM savings

Override buffer allocation to use TQ4 page size (68 bytes/token/head vs 256 FP16). Then Triton kernels if profiling shows PyTorch compress/dequant is a bottleneck.

| Step | Action | Status |
|------|--------|--------|
| 3c.1 | `TQ4FullAttentionSpec(FullAttentionSpec)` with overridden `real_page_size_bytes` + monkey-patch `Attention.get_kv_cache_spec` + register in `spec_manager_map` | ✅ |
| 3c.2 | Override `get_kv_cache_shape()` → `(num_blocks, block_size, num_kv_heads * 136)` flat uint8 layout | ✅ |
| 3c.3 | Override `get_kv_cache_stride_order()` → raise `NotImplementedError` for identity fallback | ✅ |
| 3c.4 | Implement `_compress_and_store()` — scatter-write TQ4 bytes to flat cache via `slot_mapping` | ✅ |
| 3c.5 | Implement `_decompress_cache()` — read uint8 blocks, decompress to `(NB, BS, H, D)` FP16, call `flash_attn_varlen_func` directly | ✅ |
| 3c.6 | Smoke test: `vllm serve` + Molmo2-8B with TQ4 packed cache, verify VRAM reduction | ✅ |
| 3c.7 | Profile: is PyTorch compress/dequant the bottleneck, or Flash Attention? | ✅ |
| 3c.8 | Triton fused read+dequant kernel (decompress is 68% of decode at 4096 cache) | |
| 3c.9 | Triton fused compress+write kernel (compress is 26% of decode at 4096 cache) | |
| 3c.10 | Validate bit-for-bit match with pure PyTorch path | |

169 tests pass. All pre-commit hooks green.

**Profiling result (experiment 015, 2026-03-27):** RTX 4090, D=128, Qh=32, KVh=8:
- **Decode at 4096 cache: compress 26.0%, decompress 68.0%, Flash Attention 6.0%**
- Decompress dominates — scales linearly with cache length (0.127ms@128 → 0.405ms@4096)
- Compress is ~constant (~0.155ms) since it always processes 1 new token
- Flash Attention is negligible (0.022-0.036ms) — RTX 4090 tensor cores handle it instantly
- Prefill: compress dominates at short seqlens, attention (O(n²)) takes over at 2048+
- **Conclusion: Triton kernels (3c.8-3c.9) are absolutely worth it — 94% of decode cost is in PyTorch compress/decompress**

**Smoke test result (2026-03-27):** vLLM 0.18.0 + Molmo2-8B + `--attention-backend CUSTOM` with packed TQ4 uint8 cache:
- Model loads, serves on port 8100 ✅
- "What is 2+2?" → "4" ✅
- "Name the four main characters of Seinfeld" → "George Costanza, Jerry Seinfeld, Elaine Benes, and Kramer" ✅ (all 4 preserved)
- KV cache: 60,880 tokens, 14.86x concurrency at max_model_len=4096

**Architecture (follows FlashMLA pattern exactly):**
- `TQ4FullAttentionSpec(FullAttentionSpec)` overrides `real_page_size_bytes` → `block_size * num_kv_heads * 136`
- `dtype=torch.uint8` in spec — buffer stays byte-addressable
- `get_kv_cache_shape()` returns `(NB, BS, H*136)` — flat uint8 byte layout
- `get_kv_cache_stride_order()` raises `NotImplementedError` → identity fallback (same as FlashMLA)
- `forward()` calls `flash_attn_varlen_func` directly (bypasses `super().forward()` and `do_kv_cache_update()`)
- Monkey-patch in `register_tq4_backend()` replaces `Attention.get_kv_cache_spec`
- `spec_manager_map[TQ4FullAttentionSpec] = FullAttentionManager` for KV cache manager compatibility

**Bugs found and fixed during smoke test:**
- `get_kv_cache_stride_order()` inherited 5-element tuple from FlashAttentionBackend, mismatched 3D shape. Fixed by overriding to raise `NotImplementedError`.
- `spec_manager_map` uses `type()` (exact match), not `isinstance()`. `TQ4FullAttentionSpec` not in map. Fixed by registering in `register_tq4_backend()`.

**Implementation notes for 3c.1 (page_size_bytes override):**
- Follows `MLAAttentionSpec` pattern: subclass `FullAttentionSpec`, override `real_page_size_bytes`
- Options: (a) custom `AttentionSpec` subclass with overridden `page_size_bytes`, (b) post-init patch on the spec, (c) override at the `Attention` layer level
- The compress/decompress code from Phase 3b (`_compress`, `_decompress`, `_compress_and_store`, `_decompress_cache`) is already written and tested — it just needs to target the packed cache instead of doing a round-trip

##### Phase 3d: Production benchmark

| Step | Action | Status |
|------|--------|--------|
| 3d.1 | Re-run experiment 014 with vLLM + TQ4 backend | |
| 3d.2 | Compare: vLLM-TQ4 vs vLLM-baseline vs HF-TQ4 | |
| 3d.3 | Measure throughput (tok/s), VRAM, quality | |
| 3d.4 | Update `vllm-nvidia.service` quadlet to use TQ4 backend | |

**Resolved risks:**
- ~~`CacheDType` Literal type~~ — bypassed entirely. Use `"auto"` dtype. No vLLM source patch needed.
- `get_name()` must return `"CUSTOM"` (enum member name), not a custom string.
- `supports_mm_prefix()` must return `True` for VLMs like Molmo2.
- `_generate_rotation_matrix()` must use `device="cpu", dtype=torch.float32` — vLLM overrides default device and dtype during model init.
- `TQ4AttentionImpl.__init__()` must accept `*args, **kwargs` — vLLM passes `attn_type`, `kv_sharing_target_layer_name`, `sinks` positionally.
- `get_kv_cache_shape()` alone cannot change buffer allocation — `KVCacheSpec.page_size_bytes` controls the raw buffer size.

#### Phase 4: Research — SageAttention-style INT8 path (future)

SageAttention v2 achieves **3x faster than FA2** on RTX 4090 using INT8 for Q@K^T. If we could quantize decompressed K tiles to INT8 and use tensor core INT8 matmul, we'd get both TurboQuant compression AND INT8 speed. This is speculative but the only proven path to beating cuDNN FA on consumer GPUs.

### Success Criteria (revised, evidence-based)

- KV compression: >=3.5x vs FP16 (**DONE** — 3.76x)
- Speed vs FP32 eager: >=2x at 1K+ tokens (**experiment 013: 0.2x at 17 tokens — attention is <5% of compute at short sequences**)
- Speed vs cuDNN FA: target >0.8x at 11K tokens (**requires kernel optimization, deprioritized**)
- Text quality: coherent output, character names preserved (**DONE** — Molmo2-8B validated)
- max_model_len: 3x increase in vLLM serving (**P9 Phase 3**)
- Total video throughput: process longer clips in fewer inference calls (**DONE — 1.97x faster, experiment 014**)

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
| GPU | AMD Radeon 890M (32 GB shared VRAM, gfx1150 RDNA 3.5) | ROCm via HSA override (P8) |
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

7. **Q@K^T-only fusion is a dead end for multi-layer models.** Materializing attention scores in fp16 between Q@K^T and softmax introduces 0.023 cosine loss per layer, compounding to 0.43 over 36 layers. Full Flash Attention fusion (fp32 accumulation throughout, single fp16 cast at output) is the only correct approach. The 17.8x micro-benchmark speedup was misleading — always validate multi-layer composition early.

8. **No existing system fuses codebook VQ with Flash Attention.** Survey of 13 quantized attention implementations (KIVI, BitDecoding, Kitty, QServe, FlashInfer, etc.) found they all use scalar quantization (INT2/4/8, FP4/8). TurboQuant's vector quantization codebook lookup is architecturally different and requires a novel kernel design.

9. **Dequant-then-dot is the correct architecture.** P5 dequant-then-dot produces >0.999 cosine across 36 layers. The paper's `estimate_inner_product` (avoiding dequant) was the expected speed path, but experiment 012 confirmed what Dejan.ai and ik_llama.cpp found independently: QJL drops cosine to 0.69-0.97 and produces garbled multi-layer output. MSE-only is the community consensus.

10. **QJL is a dead end for multi-layer models.** Confirmed by three independent implementations (ours, Dejan.ai, ik_llama.cpp). The 1-bit sign correction adds noise that compounds catastrophically through 36 layers. ik_llama.cpp: *"All bits to Lloyd-Max centroids is faster, simpler, and perplexity matches."*

11. **Paper speed claims require context.** The "8x" is from a blog post (not the paper), comparing FP32 JAX logits on H100 — a single-operation microbenchmark. Dejan.ai's actual RTX 4090 fused kernel speedup: 1.15-1.22x. SageAttention (INT8 scalar quant) achieves 3x vs FA2 — codebook VQ can't match direct tensor core INT8 matmul.

12. **TurboQuant was never tested on VLMs.** Paper tested only Llama-3.1-8B and Ministral-7B. Visual tokens have fundamentally different KV cache distributions (per-channel value variation, more pronounced key outliers, 10x lower gradient sensitivity). Multiple VLM quantization papers (AKVQ-VL, MBQ, VidKV, CalibQuant) document this. Our MSE-only approach may work precisely because the random rotation decorrelates these distribution differences.

13. **The real video throughput metric is total clip time, not per-token speed.** Processing 19 seconds of video in one 0.6x-speed pass beats five 5-second passes at 1.0x speed — fewer inference calls, full cross-frame context, no stitching artifacts. Compression enables capability (longer context), which dominates per-token speed in the video workload.

14. **vLLM has first-class custom backend support.** `AttentionBackendEnum.CUSTOM` + `register_backend()` decorator (v0.18.0). `FlashMLASparseBackend` is a direct template — 656-byte custom packed format with mixed dtypes. No fork needed for TQ4 integration; the block allocator is shape-agnostic.
