# Project Overview — turboquant-vllm

## Summary

TurboQuant KV cache compression as a drop-in vLLM plugin. Implements Google's **TurboQuant** algorithm ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) for compressing transformer KV caches to 3-4 bits per coordinate on consumer GPUs.

Achieves **3.76x KV cache compression with near-identical output quality** via TQ4 nibble packing + incremental dequantization, at 1.78x overhead. Validated on Molmo2-4B vision-language models processing 11K-token video clips on an RTX 4090.

First open-source TurboQuant implementation — paper to working vLLM plugin in 72 hours.

---

## Project Identity

| Field | Value |
|-------|-------|
| **Name** | turboquant-vllm |
| **Version** | 0.1.0 |
| **Type** | Python library + vLLM plugin (pip-installable) |
| **License** | Apache 2.0 |
| **Author** | Alberto-Codes |
| **Repository** | [github.com/Alberto-Codes/turboquant-vllm](https://github.com/Alberto-Codes/turboquant-vllm) |
| **PyPI** | [turboquant-vllm](https://pypi.org/project/turboquant-vllm/) |

---

## Technology Stack

| Category | Technology | Version | Notes |
|----------|-----------|---------|-------|
| **Language** | Python | >=3.12 | Type hints, `from __future__ import annotations` |
| **Build System** | uv (uv_build) | >=0.11.1 | Lockfile: `uv.lock` |
| **Deep Learning** | PyTorch | >=2.6 | Core tensor ops, CUDA/ROCm |
| **GPU Kernels** | Triton | (via PyTorch) | Fused Flash Attention, TQ4 compress/decompress |
| **ML Framework** | HuggingFace transformers | >=4.57, <5.0 | DynamicCache integration, model loading |
| **Math** | scipy | >=1.10 | `integrate.quad` for Lloyd-Max codebook |
| **Tensor Ops** | einops | >=0.8 | Tensor rearrangement |
| **Serving** | vLLM | >=0.18 (optional) | TQ4 attention backend plugin |
| **Linting** | ruff | >=0.15 | Format, lint, isort |
| **Type Check** | ty | >=0.0.1a33 | Type safety |
| **Doc Check** | docvet | >=1.13.0 | Google-style docstring enforcement |
| **Testing** | pytest | >=9.0 | + asyncio, cov, mock, randomly |
| **Security** | uv-secure | >=0.17.0 | Dependency vulnerability scanning |
| **CI/CD** | GitHub Actions | — | OIDC trusted publishing to PyPI |

---

## Architecture Type

**Layered library** with strict DAG dependency flow:

```
lloyd_max → quantizer → compressors → kv_cache → benchmark
                 ↓
            triton/ (parallel branch: fused GPU kernels)
                 ↓
            vllm/ (serving integration plugin)
```

No circular dependencies. The `triton/` module is a parallel branch with its own internal layering (vanilla FA → fused TQ4 K → fused TQ4 K+V).

---

## Repository Structure

- **Type**: Monolith (single cohesive package)
- **Layout**: src-layout (`src/turboquant_vllm/`)
- **Source Modules**: 16 Python files
- **Test Files**: 9 files, ~180+ tests
- **Experiment Scripts**: 10 GPU validation experiments
- **Research Docs**: 4 technical research documents

---

## Compression Results

| Mode | KV Cache | Compression | Quality | Overhead |
|------|----------|-------------|---------|----------|
| FP16 baseline | 1,639 MiB | 1.0x | -- | -- |
| TQ3 (3-bit uint8) | 845 MiB | 1.94x | ~95% cosine | 2.35x slower |
| TQ4 full-cache dequant | 435 MiB | 3.76x | ~97% cosine | 3.36x slower |
| **TQ4 incremental dequant** | **435 MiB** | **3.76x** | **~97% cosine** | **1.78x slower** |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| MSE-only for drop-in mode | QJL correction invisible to standard `Q @ K.T` attention |
| TQ4 nibble packing over TQ3 bit-packing | Trivial pack/unpack, 3.76x compression, ~97% quality |
| fp32 norms, not fp16 | fp16 precision loss compounds across 36 layers at 10K+ tokens |
| Non-invasive monkey-patching | Avoids subclassing DynamicCache across transformers versions |
| `@lru_cache` on Lloyd-Max | 64 compressor instances share one codebook computation |
| Incremental dequantization | Only new tokens dequantized per decode step |
| Pre/post rotation trick | Applies rotation to Q (O(tokens)) not cache (O(cache_len)) |

---

## Links to Detailed Documentation

- [Architecture](./ARCHITECTURE.md) -- Module map, dependency DAG, data flow diagrams
- [Roadmap](./ROADMAP.md) -- Implementation status, experiments, next steps
- [Source Tree Analysis](./source-tree-analysis.md) -- Annotated directory structure
- [Development Guide](./development-guide.md) -- Setup, build, test, lint commands
- [Research](./research/) -- Technical research documents informing design
