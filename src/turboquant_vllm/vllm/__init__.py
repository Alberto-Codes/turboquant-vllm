"""TQ4 compressed KV cache backend for vLLM.

Registers a custom attention backend that stores KV cache pages in
TurboQuant 4-bit format (68 bytes/token/head vs 256 bytes FP16 = 3.76x
compression).

Attributes:
    TQ4AttentionBackend: Custom attention backend registered as CUSTOM.
    TQ4AttentionImpl: Attention implementation (passthrough in Phase 3a).
    register_tq4_backend: Callable to register the backend manually.

See Also:
    :mod:`turboquant_vllm.kv_cache`: CompressedDynamicCache for HF transformers.

Usage:
    The backend registers automatically via the ``vllm.general_plugins``
    entry point when turboquant-vllm is installed with the ``vllm``
    extra::

        pip install turboquant-vllm[vllm]
        vllm serve <model> --attention-backend CUSTOM

    Or register manually before starting vLLM::

        from turboquant_vllm.vllm import register_tq4_backend

        register_tq4_backend()
"""

from turboquant_vllm.vllm.tq4_backend import (
    TQ4AttentionBackend,
    TQ4AttentionImpl,
    TQ4FullAttentionSpec,
    register_tq4_backend,
)

__all__ = [
    "TQ4AttentionBackend",
    "TQ4AttentionImpl",
    "TQ4FullAttentionSpec",
    "register_tq4_backend",
]
