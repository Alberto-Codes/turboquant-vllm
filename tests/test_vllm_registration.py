"""Tests for TQ4 vLLM backend registration and interface.

Phase 3a tests: plugin wiring, registration, class hierarchy, interface compliance.
Requires vLLM to be installed.
"""

from __future__ import annotations

from collections.abc import Generator

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

pytestmark = [pytest.mark.unit]


class TestTQ4Registration:
    """Backend registration and discovery."""

    @pytest.fixture(autouse=True)
    def _restore_tq4_registration(self) -> Generator[None, None, None]:
        """Save and restore all globals mutated by register_tq4_backend()."""
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.attention.backends.registry import register_backend
        from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

        from turboquant_vllm.vllm import tq4_backend as tq4_mod
        from turboquant_vllm.vllm.tq4_backend import TQ4FullAttentionSpec

        # Snapshot state
        try:
            orig_custom_cls = AttentionBackendEnum.CUSTOM.get_class()
        except ValueError:
            orig_custom_cls = None
        had_tq4_spec = TQ4FullAttentionSpec in spec_manager_map
        orig_get_kv = Attention.get_kv_cache_spec
        orig_mod_global = tq4_mod._original_get_kv_cache_spec

        yield

        # Restore in reverse order
        tq4_mod._original_get_kv_cache_spec = orig_mod_global
        Attention.get_kv_cache_spec = orig_get_kv
        if not had_tq4_spec:
            spec_manager_map.pop(TQ4FullAttentionSpec, None)
        if orig_custom_cls is not None:
            register_backend(
                AttentionBackendEnum.CUSTOM,
                f"{orig_custom_cls.__module__}.{orig_custom_cls.__qualname__}",
            )

    def test_register_overrides_custom_enum(self) -> None:
        register_tq4_backend()
        cls = AttentionBackendEnum.CUSTOM.get_class()
        assert cls is TQ4AttentionBackend

    def test_register_is_idempotent(self) -> None:
        register_tq4_backend()
        register_tq4_backend()
        cls = AttentionBackendEnum.CUSTOM.get_class()
        assert cls is TQ4AttentionBackend


class TestTQ4AttentionBackend:
    """Backend class interface compliance."""

    def test_name_matches_enum(self) -> None:
        assert TQ4AttentionBackend.get_name() == "CUSTOM"

    def test_impl_cls(self) -> None:
        assert TQ4AttentionBackend.get_impl_cls() is TQ4AttentionImpl

    def test_builder_cls(self) -> None:
        from turboquant_vllm.vllm.tq4_backend import TQ4MetadataBuilder

        assert TQ4AttentionBackend.get_builder_cls() is TQ4MetadataBuilder
        assert issubclass(TQ4MetadataBuilder, FlashAttentionMetadataBuilder)

    def test_subclasses_flash_attention(self) -> None:
        assert issubclass(TQ4AttentionBackend, FlashAttentionBackend)

    def test_forward_includes_kv_cache_update(self) -> None:
        assert TQ4AttentionBackend.forward_includes_kv_cache_update is True

    def test_compression_ratio_math(self) -> None:
        """TQ4 byte layout gives 3.76x compression vs FP16."""
        num_kv_heads, head_size = 8, 128
        tq4_bytes = 2 * num_kv_heads * _tq4_bytes_per_token(head_size)
        fp16_bytes = 2 * num_kv_heads * head_size * 2
        ratio = fp16_bytes / tq4_bytes
        assert abs(ratio - 3.76) < 0.01

    def test_supported_dtypes(self) -> None:
        assert torch.float16 in TQ4AttentionBackend.supported_dtypes
        assert torch.bfloat16 in TQ4AttentionBackend.supported_dtypes

    def test_supports_mm_prefix(self) -> None:
        assert TQ4AttentionBackend.supports_mm_prefix() is True

    def test_packed_kv_cache_shape(self) -> None:
        """Phase 3c: packed uint8 layout (NB, BS, padded_bytes).

        Padded slot: next_power_of_2(136) = 256. Total = 8 * 256 = 2048.
        """
        from vllm.utils.math_utils import next_power_of_2

        shape = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )
        padded_slot = next_power_of_2(_tq4_bytes_per_token_kv(128))
        expected_bytes = 8 * padded_slot  # 8 * 256 = 2048
        assert shape == (100, 16, expected_bytes)
        assert expected_bytes == 2048

    def test_packed_shape_not_5d(self) -> None:
        """Phase 3c shape is 3D, not the standard 5D."""
        shape = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=50,
            block_size=16,
            num_kv_heads=4,
            head_size=64,
        )
        assert len(shape) == 3

    def test_packed_shape_varies_with_heads(self) -> None:
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

    def test_subclasses_flash_impl(self) -> None:
        assert issubclass(TQ4AttentionImpl, FlashAttentionImpl)
