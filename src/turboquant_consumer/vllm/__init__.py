"""TQ4 compressed KV cache backend for vLLM.

Registers a custom attention backend that stores KV cache pages in
TurboQuant 4-bit format (68 bytes/token/head vs 256 bytes FP16 = 3.76x
compression).

Attributes:
    TQ4AttentionBackend: Custom attention backend registered as CUSTOM.
    TQ4AttentionImpl: Attention implementation (passthrough in Phase 3a).
    register_tq4_backend: Callable to register the backend manually.

See Also:
    :mod:`turboquant_consumer.kv_cache`: CompressedDynamicCache for HF transformers.

Usage:
    The backend registers automatically via the ``vllm.general_plugins``
    entry point when turboquant-consumer is installed with the ``vllm``
    extra::

        uv pip install turboquant-consumer[vllm]
        vllm serve <model> --attention-backend CUSTOM

    Or register manually before starting vLLM::

        from turboquant_consumer.vllm import register_tq4_backend

        register_tq4_backend()
"""

from turboquant_consumer.vllm.tq4_backend import (
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
