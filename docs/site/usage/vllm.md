# vLLM Plugin

turboquant-vllm registers as a custom attention backend via vLLM's plugin system. No code changes are needed — just install and pass a CLI flag.

## Install

```bash
pip install turboquant-vllm[vllm]
```

## Serve

```bash
vllm serve allenai/Molmo2-8B --attention-backend CUSTOM
```

The `tq4_backend` entry point registers automatically on import. vLLM will log:

```
INFO [cuda.py:257] Using AttentionBackendEnum.CUSTOM backend.
```

## How It Works

The TQ4 attention backend replaces vLLM's default KV cache page format:

| | FP16 (default) | TQ4 (turboquant-vllm) |
|---|---|---|
| Bytes per token per KV head | 256 | 68 |
| Compression ratio | 1.0x | 3.76x |
| Storage format | float16 | uint8 nibble-packed + fp32 norms |

On each attention step, the backend:

1. Decompresses TQ4 blocks back to float16
2. Delegates to Flash Attention for the actual attention computation
3. Only decompresses new tokens incrementally (not the full cache)

## Configuration

The plugin uses sensible defaults. No additional configuration is needed beyond `--attention-backend CUSTOM`.

| vLLM Flag | Recommended | Notes |
|-----------|-------------|-------|
| `--attention-backend CUSTOM` | Required | Enables TQ4 |
| `--enforce-eager` | Recommended | CUDA graphs not yet validated with TQ4 |
| `--max-model-len` | Model-specific | Unchanged from standard vLLM |
| `--gpu-memory-utilization` | 0.85-0.90 | Unchanged from standard vLLM |

## Manual Registration

If you need to register the backend programmatically (e.g., in a custom launcher):

```python
from turboquant_vllm.vllm import register_tq4_backend

register_tq4_backend()
```

## Supported Models

Any model supported by vLLM should work with the TQ4 backend. Validated on:

- **Molmo2-4B** — 11K visual tokens, video inference
- **Molmo2-8B** — 6K context, video + text inference
