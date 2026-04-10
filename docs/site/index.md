# turboquant-vllm

Reference implementation for TurboQuant KV cache compression in HuggingFace `DynamicCache`, with verification tooling for model compatibility and an optional vLLM plugin bridge. **3.76x compression, near-identical output quality, and a clear split between reference workflows here vs native vLLM upstream.**

> Implements Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), the first KV cache quantization method with provably near-optimal distortion rates.

> Native vLLM TurboQuant is converging upstream in [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479). This repo is the HuggingFace/reference path: research workflows, verification, and architecture validation.

## Choose the Right Path

- Use `turboquant-vllm` for HuggingFace cache compression, model validation, multimodal experiments, and architecture research.
- Use upstream native vLLM TurboQuant when you want the in-tree serving path in vLLM.
- Use the plugin path here only when you specifically need the out-of-tree bridge (`--attention-backend CUSTOM`).

## Install

=== "pip"

    ```bash
    pip install turboquant-vllm
    ```

=== "uv"

    ```bash
    uv add turboquant-vllm
    ```

Optional vLLM plugin extras:

```bash
pip install turboquant-vllm[vllm]
uv add turboquant-vllm --extra vllm
```

## Quick Start

### HuggingFace (primary workflow)

```python
from transformers import DynamicCache
from turboquant_vllm import CompressedDynamicCache

cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)

# Pass cache (not the wrapper) to model.generate()
# Compression happens transparently on every cache.update()
```

### Verify a model before deeper integration

```bash
python -m turboquant_vllm.verify --model allenai/Molmo2-4B --bits 4
```

### Optional vLLM plugin bridge

```bash
vllm serve allenai/Molmo2-8B --attention-backend CUSTOM
```

The TQ4 attention backend registers automatically via vLLM's plugin system. KV cache pages are compressed to 68 bytes/token/head (vs 256 bytes FP16).

## Benchmark Results

Molmo2-4B (bfloat16, 36 layers) on RTX 4090 — 11K visual tokens from 2fps video + 256 generation tokens:

| Mode | KV Cache | Compression | Output Quality | Overhead |
|------|----------|-------------|----------------|----------|
| FP16 baseline | 1,639 MiB | 1.0x | -- | -- |
| TQ3 (3-bit) | 845 MiB | 1.94x | ~95% cosine similarity | 2.35x |
| TQ4 (full dequant) | 435 MiB | 3.76x | ~97% cosine similarity | 3.36x |
| **TQ4 (incremental)** | **435 MiB** | **3.76x** | **~97% cosine, 100+ matching tokens** | **1.78x** |

## How It Works

1. **Random orthogonal rotation** maps each KV vector onto coordinates that follow a known Beta distribution
2. **Lloyd-Max scalar quantization** finds optimal centroids for that distribution at 3-4 bits per coordinate
3. **Nibble packing** stores two 4-bit indices per byte for 3.76x compression
4. **Incremental dequantization** only decompresses new tokens each decode step, keeping overhead at 1.78x

## Next Steps

- [vLLM Plugin Guide](usage/vllm.md) — serving with TQ4 compression
- [Container Deployment](usage/container.md) — pre-built image with turboquant-vllm baked in
- [HuggingFace Guide](usage/huggingface.md) — direct integration with DynamicCache
- [API Reference](reference/index.md) — auto-generated from docstrings
