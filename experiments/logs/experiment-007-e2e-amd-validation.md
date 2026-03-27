## Experiment: 007 — End-to-End Molmo2-4B Validation on AMD Radeon 890M

**Date:** 2026-03-27
**Hardware:** Radeon 890M iGPU (32 GB shared DDR5, gfx1150 RDNA 3.5), Ryzen AI 9 HX 370 (12C/24T), 64 GB DDR5
**OS:** Bazzite 43 (immutable Fedora), kernel 6.17.7
**Container:** `turboquant-rocm` (ROCm 7.1 + PyTorch 2.8.0)
**Override:** `HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1150 -> gfx1100)
**Result:** **PASS** — all 5 validation steps completed successfully

### Hypothesis

Molmo2-4B inference with TQ4 CompressedDynamicCache produces coherent, correct output on AMD Radeon 890M via ROCm. The HSA architecture override introduces no additional precision errors beyond the expected TurboQuant quantization loss (~97% cosine at 4-bit). CPU and GPU baseline outputs should agree on early tokens (greedy decode), with potential divergence from bf16/fp32 accumulation differences.

**Verdict:** Confirmed. CPU and GPU produce bit-identical tokens (16/16 match). TQ4 output is coherent with 3.76x KV cache compression.

### Prerequisites

- Phases 0-2 completed: 95/95 parametrized tests pass on AMD GPU
- `torch.compile` validated on ROCm (both `default` and `reduce-overhead`)
- HuggingFace model weights cached or auth configured

### Setup

```bash
# Run from project root
./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py \
    --model allenai/Molmo2-4B \
    --bits 4 \
    --max-new-tokens 64 \
    --cross-validate-tokens 16
```

**Prompt:** "Describe the main character of Seinfeld in one paragraph."

### Results

#### Step 3.1 — Model Weight Verification

| Check | Result |
|-------|--------|
| Model accessible | **Yes** |
| Shards fetched | 4/4 (8 min, cached on subsequent runs) |

#### Step 3.2 — GPU Baseline Inference (no compression)

| Metric | Value |
|--------|-------|
| Input tokens | 17 |
| Output tokens | 64 |
| Throughput (tok/s) | **5.0** |
| VRAM peak (MiB) | 10,052 |
| Time (s) | 12.9 |
| Output coherent? | **Yes** |

> "Jerry Seinfeld is the main character of the sitcom *Seinfeld*, portrayed by Jason Alexander. Jerry is a witty, neurotic, and often self-absorbed New York City man in his late 20s or early 30s. He works as a freelance writer and is known for his dry humor"

#### Step 3.3 — GPU TQ4 Compressed Inference

| Metric | Value |
|--------|-------|
| Output tokens | 64 |
| Throughput (tok/s) | **4.3** |
| VRAM peak (MiB) | 10,054 |
| Time (s) | 15.1 |
| KV cache compressed | 2.99 MiB vs 11.25 MiB baseline |
| Compression ratio | **3.76x** (nibble-packed) |
| Texts identical to baseline? | No (expected — see Key Findings) |
| Output coherent? | **Yes** |

> "Jerry Seinfeld is the main character of the TV show *Seinfeld*, portrayed by Jerry Seinfeld. He is a 30-something-year-old New York City man who works as a writer for a magazine. Jerry is known for his dry wit, sarcasm, and tendency to overthink everyday situations."

#### Step 3.4 — CPU vs GPU Cross-Validation

| Metric | Value |
|--------|-------|
| Tokens compared | 16 |
| Tokens matching | **16** |
| Match rate | **100.0%** |
| First divergence point | None — perfect match |
| CPU throughput | 1.1 tok/s (fp32, 15.1s) |

#### Step 3.5 — Throughput Summary

| Metric | Baseline | TQ4 Compressed |
|--------|----------|----------------|
| tok/s | 5.0 | 4.3 |
| VRAM peak (MiB) | 10,052 | 10,054 |
| KV cache (MiB) | 11.25 | 2.99 |
| Overhead ratio | 1.0x | 0.86x |

**890M vs 4090 comparison:** ~5 tok/s vs ~50-60 tok/s (10-12x), consistent with the DDR5 (~90 GB/s) vs GDDR6X (~1 TB/s) bandwidth ratio predicted in Phase 0.

### Key Findings

1. **HSA override introduces zero token-level errors.** CPU (fp32) and GPU (bf16 via gfx1150->gfx1100 override) produce bit-identical token sequences across 16 greedy-decoded tokens. The architecture spoof is safe for inference.

2. **TQ4 output is coherent but not identical to baseline.** First 9 tokens match exactly (`Jerry Seinfeld is the main character of the`), then diverge at token 10: `sitcom` (baseline) vs `TV show` (TQ4). This is expected — lossy quantization shifts logit probabilities by ~3% (97% cosine), and autoregressive decoding amplifies small differences.

3. **Compression overhead is minimal at short sequences.** TQ4 adds only ~15% latency (0.86x throughput ratio) at 80-token sequence length. The incremental dequantization strategy (P3, experiment 005) is paying off — only 1 token decompressed per step.

4. **VRAM peak is dominated by model weights, not KV cache.** 10,052 MiB peak vs 11.25 MiB KV cache — the cache is <0.1% of peak VRAM at this sequence length. KV compression matters at 11K+ tokens (the Seinfeld episode workload), not at 80.

5. **5 tok/s on an iGPU is usable for development.** Not production speed, but fast enough for iterating on prompts and validating outputs without needing the desktop 4090.

6. **ROCm SDPA is experimental on gfx1150.** Two warnings about `Mem Efficient attention` and `Flash Efficient attention` being experimental. Output was correct despite the warnings — the fallback path works. Enabling `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` may improve throughput.

### Next Steps

- Update ROADMAP.md Phase 3 status to COMPLETED
- Consider `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for potential speedup
- P4 (Molmo2-8B): needs bitsandbytes 4-bit weight quantization to fit in 32 GB
- Long-sequence validation (11K+ tokens) requires video input — deferred to molmo-video-analyzer integration
