## Experiment: 008 — Fused Triton Q@K^T Kernel on AMD ROCm

**Date:** 2026-03-27
**Hardware:** Radeon 890M iGPU (32 GB shared DDR5, gfx1150 RDNA 3.5)
**Container:** `turboquant-rocm` (ROCm 7.1 + PyTorch 2.8.0 + Triton 3.4.0+rocm7.1.0)
**Override:** `HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1150 -> gfx1100)
**Result:** **PASS** — fused kernel works on ROCm with zero code changes

### Hypothesis

The existing Triton fused Q@K^T kernel (`_fused_qk_nibble_kernel`) — previously assumed NVIDIA-only — should work on ROCm via Triton's HIP backend without modification, since it uses standard Triton operations (load, store, bitshift, gather) with no CUDA-specific intrinsics.

**Verdict:** Confirmed. The kernel produces exact results (1.000000 cosine, 0.0 max diff vs reference) on AMD Radeon 890M.

### Setup

```bash
./infra/run-rocm.sh python experiments/experiment_008_triton_fused_rocm.py
```

**Configuration:** batch=1, q_heads=32, kv_heads=8, kv_len=1024, head_dim=128, bits=4 (nibble-packed)

### Results

| Check | Result |
|-------|--------|
| Triton version | 3.4.0+rocm7.1.0 |
| Triton backend | HIP |
| Basic Triton kernel (vector add) | PASS (exact match) |
| Fused Q@K^T kernel launch | PASS (no errors) |
| Output shape | [1, 32, 1, 1024] (correct) |
| NaN/Inf | None |
| Cosine similarity vs reference | **1.000000** |
| Max absolute difference | **0.000000** |
| Avg time per call (1K tokens) | **0.31 ms** |

### Key Findings

1. **Triton HIP backend is fully functional on gfx1150.** The `_fused_qk_nibble_kernel` required zero code changes — all Triton operations (bitshift nibble unpack, centroid gather, GQA head mapping, accumulation) work identically on the HIP backend. The roadmap's "NVIDIA-only" note is outdated.

2. **Autotuning works on ROCm.** The 5 autotuned kernel configurations (varying BLOCK_S, BLOCK_D, num_warps) all compiled and ran. Triton selected the optimal config automatically.

3. **Performance is reasonable for iGPU.** 0.31 ms per Q@K^T call at 1K tokens is usable. The 890M won't match the 4090's throughput, but the kernel overhead is small relative to the rest of the inference pipeline.

4. **Multi-layer precision issue is platform-independent.** The kernel's known limitation — 0.023 cosine error per layer compounding across 36 layers — is a numerical issue (fp32 kernel vs bf16 SDPA), not a platform issue. The fix (P5: Flash Attention-style fusion) is needed on both NVIDIA and AMD.

### Next Steps

- Update ROADMAP.md: change P3b from "NVIDIA-only" to "cross-platform (NVIDIA + AMD ROCm)"
- P5 (Flash Attention fusion) remains the critical path for multi-layer deployment on both platforms
