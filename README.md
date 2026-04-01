[![PyPI](https://img.shields.io/pypi/v/turboquant-vllm)](https://pypi.org/project/turboquant-vllm/)
[![Python](https://img.shields.io/pypi/pyversions/turboquant-vllm)](https://pypi.org/project/turboquant-vllm/)
[![License](https://img.shields.io/pypi/l/turboquant-vllm)](https://github.com/Alberto-Codes/turboquant-vllm/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![docs vetted](https://img.shields.io/badge/docs%20vetted-docvet-purple)](https://github.com/Alberto-Codes/docvet)

# turboquant-vllm

TurboQuant KV cache compression as a drop-in vLLM plugin. **3.76x KV cache compression with asymmetric K/V support, validated across 8 model families.**

> Implements Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — the first KV cache quantization method with provably near-optimal distortion rates.

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
vllm serve meta-llama/Llama-3.1-8B-Instruct --attention-backend CUSTOM
```

No code changes required. The plugin compresses KV cache pages to 68 bytes/token/head (vs 256 bytes FP16). For asymmetric K/V compression:

```bash
TQ4_K_BITS=4 TQ4_V_BITS=3 vllm serve meta-llama/Llama-3.1-8B-Instruct --attention-backend CUSTOM
```

## Quick Start (HuggingFace)

```python
from transformers import DynamicCache
from turboquant_vllm import CompressedDynamicCache

cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, k_bits=4, v_bits=3)

# Pass cache (not the wrapper) to model.generate()
# Compression happens transparently on every cache.update()
```

## Compression Quality

Per-layer minimum cosine similarity on real model activations (128-token prefill, RTX 4090):

| Model | head_dim | K4/V4 cosine | K4/V3 cosine |
|-------|----------|-------------|-------------|
| Llama 3.1 8B | 128 | 0.9947 | 0.9823 |
| Qwen2.5 3B | 128 | 0.9935 | 0.9823 |
| Mistral 7B | 128 | 0.9947 | 0.9825 |
| Phi-3-mini | 96 | 0.9950 | 0.9827 |
| Phi-4 | 128 | 0.9945 | 0.9824 |
| Gemma 2 2B | 256 | 0.9948 | 0.9823 |
| Gemma 3 4B | 256 | 0.9911 | 0.9794 |
| Molmo2 4B | 128 | 0.9943 | 0.9821 |

Validate any model yourself with the verify CLI:

```bash
python -m turboquant_vllm.verify --model meta-llama/Llama-3.1-8B --bits 4
python -m turboquant_vllm.verify --model meta-llama/Llama-3.1-8B --k-bits 4 --v-bits 3
```

## Serving Performance

Llama-3.1-8B-Instruct on RTX 4090, 200 concurrent requests ([Exp 029](experiments/logs)):

| Metric | Baseline | TQ4 (K4/V4) | Delta |
|--------|----------|-------------|-------|
| Request throughput | 8.14 req/s | 7.55 req/s | -7.3% |
| Output tok/s | 1,042 | 967 | -7.3% |
| Median TTFT | 9,324 ms | 6,977 ms | **-25.2%** |
| Median TPOT | 47.6 ms | 143.6 ms | +201% |

TQ4 reduces time-to-first-token by 25% (smaller cache pages = faster prefill) but increases per-token decode latency ~3x due to online decompression. Net throughput impact is -7% at high concurrency. Best suited for memory-bound workloads: long contexts, high batch sizes, or limited VRAM.

## How It Works

Implements Google's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm (ICLR 2026):

1. **Random orthogonal rotation** maps each KV vector onto coordinates that follow a known Beta distribution
2. **Lloyd-Max scalar quantization** finds optimal centroids for that distribution at 3-4 bits per coordinate
3. **Nibble packing** stores two 4-bit indices per byte for 3.76x compression
4. **Incremental dequantization** only decompresses new tokens each decode step, keeping overhead at 1.78x

## What Gets Compressed

| Data | Compressed | Format |
|------|-----------|--------|
| Key cache vectors | Yes (k_bits, default 4) | uint8 nibble-packed indices + fp32 norms |
| Value cache vectors | Yes (v_bits, default 4) | uint8 nibble-packed indices + fp32 norms |
| Rotation matrices | No | Generated once per layer from fixed seed |
| Lloyd-Max codebook | No | Computed once, shared across all layers |

## Roadmap

- [x] Core TurboQuant algorithm (Lloyd-Max, MSE quantizer, compressors)
- [x] CompressedDynamicCache with incremental dequantization
- [x] vLLM TQ4 attention backend plugin
- [x] Fused Triton kernels (4.5x compress, 4x decompress speedup)
- [x] Fused paged TQ4 decode with 8.5x HBM bandwidth reduction
- [x] INT8 Q@K^T prefill path
- [x] CUDA graph compatibility (buffer pre-allocation)
- [x] Multi-model validation (8 families, head_dim 64/96/128/256)
- [x] Sliding window attention bypass (Gemma 2/3)
- [x] Asymmetric K/V compression (k_bits/v_bits)
- [ ] Sparse V decompression for decode acceleration
- [ ] Container image with turboquant-vllm baked in
- [ ] Full Flash Attention fusion with fp32 online softmax

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
