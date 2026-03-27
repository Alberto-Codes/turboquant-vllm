"""TurboQuant KV cache compression for consumer GPUs.

Implements Google's TurboQuant algorithm (ICLR 2026) for compressing
transformer key-value caches to 3-4 bits per coordinate with near-zero
accuracy loss. Designed for benchmarking on consumer hardware (RTX 4090).

Reference: arXiv 2504.19874 — "TurboQuant: Online Vector Quantization
with Near-optimal Distortion Rate"

Attributes:
    CompressedDynamicCache: KV cache with real VRAM savings (uint8 + fp32).
    TurboQuantKVCache: Accuracy-only KV cache wrapper (no VRAM savings).
    TurboQuantCompressorMSE: Value cache compressor (MSE-optimal).
    TurboQuantCompressorV2: Key cache compressor (QJL-corrected).
    TurboQuantMSE: Stage 1 quantizer (rotation + Lloyd-Max).
    TurboQuantProd: Stage 1 + 2 quantizer (MSE + QJL).
    LloydMaxCodebook: Precomputed optimal scalar quantizer.
    solve_lloyd_max: Factory for Lloyd-Max codebooks (cached).

Examples:
    ```python
    from turboquant_vllm import TurboQuantKVCache

    wrapper = TurboQuantKVCache(cache, head_dim=128, bits=3)
    ```

See Also:
    :mod:`turboquant_vllm.benchmark`: CLI harness for benchmarking.
    :mod:`turboquant_vllm.lloyd_max`: Lloyd-Max codebook solver.
"""

from turboquant_vllm.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)
from turboquant_vllm.kv_cache import CompressedDynamicCache, TurboQuantKVCache
from turboquant_vllm.lloyd_max import LloydMaxCodebook, solve_lloyd_max
from turboquant_vllm.quantizer import TurboQuantMSE, TurboQuantProd

__all__ = [
    "CompressedDynamicCache",
    "LloydMaxCodebook",
    "TurboQuantCompressorMSE",
    "TurboQuantCompressorV2",
    "TurboQuantKVCache",
    "TurboQuantMSE",
    "TurboQuantProd",
    "solve_lloyd_max",
]
