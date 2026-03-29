"""Shared TQ4AttentionImpl factory for vLLM cache tests.

Extracted from ``test_vllm_cache.py`` (Test Maturity Priority 1)
to consolidate duplicated ``_make_impl`` and ``_make_cache`` helpers
across ``test_vllm_cache.py`` and ``test_vllm_cache_cudagraph.py``.
"""

from __future__ import annotations

import torch

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.vllm.tq4_backend import (
    TQ4AttentionImpl,
    _tq4_bytes_per_token_kv,
)

# Constants matching Molmo2-8B GQA config
NUM_KV_HEADS = 8
NUM_HEADS = 32
HEAD_SIZE = 128
BLOCK_SIZE = 16


def make_impl(quantizer: TurboQuantMSE) -> TQ4AttentionImpl:
    """Create a TQ4AttentionImpl without full vLLM init.

    Args:
        quantizer: TurboQuantMSE instance (from conftest ``tq4_quantizer``).
    """
    from turboquant_vllm.vllm.tq4_backend import TQ4_NORM_BYTES

    impl = object.__new__(TQ4AttentionImpl)
    impl.head_size = HEAD_SIZE
    impl.num_kv_heads = NUM_KV_HEADS
    impl.num_heads = NUM_HEADS

    impl._tq4_rotation = quantizer.rotation.clone()
    impl._tq4_centroids = quantizer.codebook.centroids.clone()
    impl._tq4_boundaries = quantizer.codebook.boundaries.clone()
    rot_t = quantizer.rotation.T.contiguous()
    impl._tq4_rot_T_even = rot_t[:, 0::2].contiguous()
    impl._tq4_rot_T_odd = rot_t[:, 1::2].contiguous()
    impl._cg_buffers_ready = False
    impl._fused_paged_available = False
    impl._max_prefill_len = 2048

    half_D = HEAD_SIZE // 2
    impl._half_D = half_D
    impl._k_idx_end = NUM_KV_HEADS * half_D
    impl._k_norm_end = impl._k_idx_end + NUM_KV_HEADS * TQ4_NORM_BYTES
    impl._v_idx_end = impl._k_norm_end + NUM_KV_HEADS * half_D
    impl._total_bytes = impl._v_idx_end + NUM_KV_HEADS * TQ4_NORM_BYTES

    return impl


def make_cache(num_blocks: int) -> torch.Tensor:
    """Create an empty TQ4 paged cache.

    Returns:
        ``(num_blocks, BLOCK_SIZE, total_bytes)`` uint8 tensor.
    """
    total_bytes = NUM_KV_HEADS * _tq4_bytes_per_token_kv(HEAD_SIZE)
    return torch.zeros(num_blocks, BLOCK_SIZE, total_bytes, dtype=torch.uint8)
