"""Tests for TQ4 vLLM backend registration, interface, and byte calculations.

Phase 3a tests: plugin wiring, registration, class hierarchy, interface compliance.
Requires vLLM to be installed.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

from turboquant_vllm.vllm.tq4_backend import (  # noqa: E402
    TQ4AttentionBackend,
    TQ4AttentionImpl,
    _tq4_bytes_per_token,
    _tq4_bytes_per_token_kv,
    register_tq4_backend,
)


class TestTQ4Registration:
    """Backend registration and discovery."""

    def test_register_overrides_custom_enum(self):
        register_tq4_backend()
        cls = AttentionBackendEnum.CUSTOM.get_class()
        assert cls is TQ4AttentionBackend

    def test_register_is_idempotent(self):
        register_tq4_backend()
        register_tq4_backend()
        cls = AttentionBackendEnum.CUSTOM.get_class()
        assert cls is TQ4AttentionBackend


class TestTQ4AttentionBackend:
    """Backend class interface compliance."""

    def test_name_matches_enum(self):
        assert TQ4AttentionBackend.get_name() == "CUSTOM"

    def test_impl_cls(self):
        assert TQ4AttentionBackend.get_impl_cls() is TQ4AttentionImpl

    def test_builder_cls(self):
        assert TQ4AttentionBackend.get_builder_cls() is FlashAttentionMetadataBuilder

    def test_subclasses_flash_attention(self):
        assert issubclass(TQ4AttentionBackend, FlashAttentionBackend)

    def test_forward_includes_kv_cache_update(self):
        assert TQ4AttentionBackend.forward_includes_kv_cache_update is True

    def test_compression_ratio_math(self):
        """TQ4 byte layout gives 3.76x compression vs FP16."""
        num_kv_heads, head_size = 8, 128
        tq4_bytes = 2 * num_kv_heads * _tq4_bytes_per_token(head_size)
        fp16_bytes = 2 * num_kv_heads * head_size * 2
        ratio = fp16_bytes / tq4_bytes
        assert abs(ratio - 3.76) < 0.01

    def test_supported_dtypes(self):
        assert torch.float16 in TQ4AttentionBackend.supported_dtypes
        assert torch.bfloat16 in TQ4AttentionBackend.supported_dtypes

    def test_supports_mm_prefix(self):
        assert TQ4AttentionBackend.supports_mm_prefix() is True

    def test_packed_kv_cache_shape(self):
        """Phase 3c: packed uint8 layout (NB, BS, total_bytes)."""
        shape = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )
        expected_bytes = 8 * _tq4_bytes_per_token_kv(128)  # 8 * 136 = 1088
        assert shape == (100, 16, expected_bytes)
        assert expected_bytes == 1088

    def test_packed_shape_not_5d(self):
        """Phase 3c shape is 3D, not the standard 5D."""
        shape = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=50,
            block_size=16,
            num_kv_heads=4,
            head_size=64,
        )
        assert len(shape) == 3

    def test_packed_shape_varies_with_heads(self):
        """More KV heads = more bytes per token."""
        shape_4h = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_size=128,
        )
        shape_8h = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=10,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )
        assert shape_8h[2] == 2 * shape_4h[2]


class TestTQ4AttentionImpl:
    """Impl class hierarchy."""

    def test_subclasses_flash_impl(self):
        assert issubclass(TQ4AttentionImpl, FlashAttentionImpl)


class TestTQ4ByteCalculation:
    """TQ4 page layout math."""

    def test_bytes_per_token_head_128(self):
        assert _tq4_bytes_per_token(128) == 68

    def test_bytes_per_token_head_64(self):
        assert _tq4_bytes_per_token(64) == 36

    def test_bytes_per_token_head_256(self):
        assert _tq4_bytes_per_token(256) == 132

    def test_bytes_per_token_kv_128(self):
        assert _tq4_bytes_per_token_kv(128) == 136

    def test_bytes_per_token_kv_64(self):
        assert _tq4_bytes_per_token_kv(64) == 72
