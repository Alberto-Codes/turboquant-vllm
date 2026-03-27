"""Fused Triton kernels for TurboQuant compressed attention.

Phase 1 (P5): Vanilla Flash Attention kernel with GQA support,
serving as the scaffold for TQ4 decompression fusion in Phase 2.

Legacy: Q@K^T-only fused kernel (superseded -- see Key Lesson #7).

Attributes:
    triton_flash_attention: Flash Attention forward with online softmax.
    triton_fa_forward: HF AttentionInterface-compatible wrapper.
    register_triton_fa: Register the ``triton_fa`` backend globally.
    install_triton_fa: Register and activate on a specific model.
    fused_qk_scores: Legacy Q@K^T-only kernel (kept for reference).

Examples:
    Direct kernel usage:

    ```python
    from turboquant_consumer.triton import triton_flash_attention

    out = triton_flash_attention(q, k, v)
    ```

    HuggingFace integration:

    ```python
    from turboquant_consumer.triton import install_triton_fa

    install_triton_fa(model)
    output = model.generate(...)
    ```

See Also:
    :mod:`turboquant_consumer.kv_cache`: CompressedDynamicCache storage layer.
"""

from turboquant_consumer.triton.attention_interface import (
    install_triton_fa,
    register_triton_fa,
    triton_fa_forward,
)
from turboquant_consumer.triton.flash_attention import triton_flash_attention

# Legacy (Q@K^T-only kernel, superseded by full FA fusion)
from turboquant_consumer.triton.fused_qk_attention import fused_qk_scores

__all__ = [
    "triton_flash_attention",
    "triton_fa_forward",
    "register_triton_fa",
    "install_triton_fa",
    "fused_qk_scores",
]
