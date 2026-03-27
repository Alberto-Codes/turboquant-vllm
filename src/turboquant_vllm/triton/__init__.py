"""Fused Triton kernels for TurboQuant compressed attention.

Phase 1 (P5): Vanilla Flash Attention kernel with GQA support.
Phase 2 (P5): Fused TQ4 K decompression inside the FA inner loop.
Phase 3 (P5): Fused TQ4 K+V decompression with post-rotation.
Phase 3c.8: Standalone TQ4 cache decompress kernel for vLLM backend.
Phase 3c.9: Standalone TQ4 compress kernel for vLLM backend.

Legacy: Q@K^T-only fused kernel (superseded -- see Key Lesson #7).

Attributes:
    triton_flash_attention: Vanilla FA forward with online softmax.
    triton_flash_attention_tq4: Fused TQ4 FA with compressed K tiles.
    triton_flash_attention_tq4_kv: Fused TQ4 FA with compressed K+V tiles.
    triton_fa_forward: HF AttentionInterface-compatible wrapper.
    register_triton_fa: Register the ``triton_fa`` backend globally.
    install_triton_fa: Register and activate vanilla FA on a model.
    install_fused_tq4_kv: Activate fused TQ4 K+V with cache side-channel.
    uninstall_fused_tq4_kv: Remove fused attention and restore SDPA.
    tq4_compress: Fused TQ4 compress (norm+rotate+quantize+pack).
    tq4_decompress: Fused TQ4 decompress (unpack+gather+scale).
    fused_qk_scores: Legacy Q@K^T-only kernel (kept for reference).

Examples:
    Direct kernel usage:

    ```python
    from turboquant_vllm.triton import triton_flash_attention

    out = triton_flash_attention(q, k, v)
    ```

    HuggingFace integration:

    ```python
    from turboquant_vllm.triton import install_triton_fa

    install_triton_fa(model)
    output = model.generate(...)
    ```

See Also:
    :mod:`turboquant_vllm.kv_cache`: CompressedDynamicCache storage layer.
"""

from turboquant_vllm.triton.attention_interface import (
    install_fused_tq4_kv,
    install_triton_fa,
    register_triton_fa,
    triton_fa_forward,
    triton_fa_tq4_kv_forward,
    uninstall_fused_tq4_kv,
)
from turboquant_vllm.triton.flash_attention import triton_flash_attention
from turboquant_vllm.triton.flash_attention_tq4 import triton_flash_attention_tq4
from turboquant_vllm.triton.flash_attention_tq4_kv import (
    triton_flash_attention_tq4_kv,
)
from turboquant_vllm.triton.fused_qk_attention import fused_qk_scores
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

__all__ = [
    "triton_flash_attention",
    "triton_flash_attention_tq4",
    "triton_flash_attention_tq4_kv",
    "triton_fa_forward",
    "triton_fa_tq4_kv_forward",
    "register_triton_fa",
    "install_triton_fa",
    "install_fused_tq4_kv",
    "uninstall_fused_tq4_kv",
    "tq4_compress",
    "tq4_decompress",
    "fused_qk_scores",
]
