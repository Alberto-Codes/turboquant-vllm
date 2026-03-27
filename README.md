[![PyPI](https://img.shields.io/pypi/v/turboquant-vllm)](https://pypi.org/project/turboquant-vllm/)
[![Python](https://img.shields.io/pypi/pyversions/turboquant-vllm)](https://pypi.org/project/turboquant-vllm/)
[![License](https://img.shields.io/pypi/l/turboquant-vllm)](https://github.com/Alberto-Codes/turboquant-vllm/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![docs vetted](https://img.shields.io/badge/docs%20vetted-docvet-purple)](https://github.com/Alberto-Codes/docvet)

# turboquant-vllm

TurboQuant KV cache compression as a drop-in vLLM plugin. **3.76x compression, near-identical output quality, one env var to enable.**

> First open-source TurboQuant implementation — paper to working vLLM plugin in 72 hours.

## Install

```bash
pip install turboquant-vllm[vllm]
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add turboquant-vllm --extra vllm
```

## Quick Start (vLLM)

The TQ4 attention backend registers automatically via vLLM's plugin system:

```bash
vllm serve allenai/Molmo2-4B --attention-backend CUSTOM
```

No code changes required. The plugin compresses KV cache pages to 68 bytes/token/head (vs 256 bytes FP16).

## Quick Start (HuggingFace)

```python
from transformers import DynamicCache
from turboquant_vllm import CompressedDynamicCache

cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)

# Pass cache (not the wrapper) to model.generate()
# Compression happens transparently on every cache.update()
```

## Benchmark Results

Molmo2-4B (bfloat16, 36 layers) on RTX 4090 — 11K visual tokens from 2fps video + 256 generation tokens:

| Mode | KV Cache | Compression | Output Quality | Overhead |
|------|----------|-------------|----------------|----------|
| FP16 baseline | 1,639 MiB | 1.0x | -- | -- |
| TQ3 (3-bit) | 845 MiB | 1.94x | ~95% cosine similarity | 2.35x |
| TQ4 (full dequant) | 435 MiB | 3.76x | ~97% cosine similarity | 3.36x |
| **TQ4 (incremental)** | **435 MiB** | **3.76x** | **~97% cosine, 100+ matching tokens** | **1.78x** |

## How It Works

Implements Google's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm (ICLR 2026):

1. **Random orthogonal rotation** maps each KV vector onto coordinates that follow a known Beta distribution
2. **Lloyd-Max scalar quantization** finds optimal centroids for that distribution at 3-4 bits per coordinate
3. **Nibble packing** stores two 4-bit indices per byte for 3.76x compression
4. **Incremental dequantization** only decompresses new tokens each decode step, keeping overhead at 1.78x

## What Gets Compressed

| Data | Compressed | Format |
|------|-----------|--------|
| Key cache vectors | Yes | uint8 nibble-packed indices + fp32 norms |
| Value cache vectors | Yes | uint8 nibble-packed indices + fp32 norms |
| Rotation matrices | No | Generated once per layer from fixed seed |
| Lloyd-Max codebook | No | Computed once, shared across all layers |

## Roadmap

- [x] Core TurboQuant algorithm (Lloyd-Max, MSE quantizer, compressors)
- [x] CompressedDynamicCache with incremental dequantization
- [x] vLLM TQ4 attention backend plugin
- [x] Fused Triton kernels (17.8x Q@K^T speedup, Flash Attention fusion)
- [ ] Container image with turboquant-vllm baked in
- [ ] Full Flash Attention fusion with fp32 online softmax
- [ ] SageAttention-style INT8 path

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- Module map, dependency DAG, data flow diagrams
- [Roadmap](docs/ROADMAP.md) -- Detailed implementation status and experiment results
- [Development Guide](docs/development-guide.md) -- Setup, build, test, lint commands

## Citation

```bibtex
@inproceedings{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Han, Insu and Daliri, Majid and Karbasi, Amin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## License

[Apache 2.0](https://github.com/Alberto-Codes/turboquant-vllm/blob/main/LICENSE)
