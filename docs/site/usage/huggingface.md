# HuggingFace Integration

This is the primary workflow for `turboquant-vllm`: use HuggingFace's `DynamicCache` path for research, benchmarking, model validation, and architecture experimentation.

## Install

```bash
pip install turboquant-vllm
```

## Usage

```python
from transformers import DynamicCache
from turboquant_vllm import CompressedDynamicCache

cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)

# Pass cache (not the wrapper) to model.generate()
# Compression happens transparently on every cache.update()
```

## Bit Width Options

| Bits | Compression | Quality | Use Case |
|------|-------------|---------|----------|
| 2 | ~8x | ~75% cosine | Experimental |
| 3 | 1.94x | ~95% cosine | Memory-constrained |
| **4** | **3.76x** | **~97% cosine** | **Recommended** |
| 5 | ~1.6x | ~99% cosine | Quality-critical |

!!! tip "TQ4 (bits=4) is the sweet spot"
    Nibble packing at 4-bit gives 3.76x compression with ~97% cosine similarity. 3-bit gives only 1.94x because indices are stored as full bytes (no cross-byte packing).

## Accuracy-Only Mode

For measuring compression quality without VRAM savings:

```python
from turboquant_vllm import TurboQuantKVCache

cache = DynamicCache()
wrapper = TurboQuantKVCache(cache, head_dim=128, bits=4)

# Keys and values are compressed then immediately decompressed
# No VRAM savings, but measures the quality impact of compression
```

## Compression Stats

```python
stats = compressed.compression_stats()
# {
#   'num_layers': 36,
#   'seq_len': 1024,
#   'num_heads': 8,
#   'head_dim': 128,
#   'bits': 4,
#   'nibble_packed': True,
#   'compression_ratio': 3.76,
#   'savings_mib': 150.2,
# }
```

## Benchmark CLI

Run A/B comparisons on Molmo2 models:

```bash
uv run python -m turboquant_vllm.benchmark \
    --model allenai/Molmo2-4B \
    --bits 4 --compressed \
    --video /path/to/clip.mp4 \
    --max-new-tokens 256
```
