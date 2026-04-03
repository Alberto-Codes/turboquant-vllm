"""Tests for TurboQuant KV cache wrapper integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch

from turboquant_vllm.kv_cache import CompressedDynamicCache, TurboQuantKVCache

from .conftest import BITS, DIM


@pytest.mark.unit
class TestKVCache:
    """Validate KV cache wrapper integration with HuggingFace DynamicCache."""

    def test_basic_update(self, device: torch.device) -> None:
        """Wrapped cache should accept and return correct tensor shapes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 8, 1, DIM).to(device)
        values = torch.randn(1, 8, 1, DIM).to(device)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape[-1] == DIM
        assert out_v.shape[-1] == DIM

    def test_disable_passthrough(self) -> None:
        """Disabling compression should pass through unmodified tensors."""
        from transformers import DynamicCache

        head_dim = 64
        cache = DynamicCache()
        tq_cache = TurboQuantKVCache(cache, head_dim=head_dim, bits=BITS)

        tq_cache.disable()
        keys = torch.randn(1, 4, 1, head_dim)
        values = torch.randn(1, 4, 1, head_dim)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        torch.testing.assert_close(out_k[:, :, -1:, :], keys)
        torch.testing.assert_close(out_v[:, :, -1:, :], values)

    def test_restore_unwraps(self) -> None:
        """Restore should remove the wrapper entirely."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update
        tq_cache = TurboQuantKVCache(cache, head_dim=64, bits=BITS)
        assert cache.update != original_update

        tq_cache.restore()
        assert cache.update == original_update

    def test_keys_only_mode(self) -> None:
        """compress_values=False should only compress keys."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS, compress_values=False)

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        # Values should be unchanged (no compression)
        torch.testing.assert_close(out_v[:, :, -1:, :], values)
        # Keys should be different (compressed then decompressed)
        assert not torch.allclose(out_k[:, :, -1:, :], keys, atol=1e-6)

    def test_values_only_mode(self) -> None:
        """compress_keys=False should only compress values."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS, compress_keys=False)

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        # Keys should be unchanged
        torch.testing.assert_close(out_k[:, :, -1:, :], keys)
        # Values should be different (compressed then decompressed)
        assert not torch.allclose(out_v[:, :, -1:, :], values, atol=1e-6)

    def test_multi_layer_accumulation(self, device: torch.device) -> None:
        """Cache should accumulate tokens across multiple update calls."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        for _ in range(3):
            keys = torch.randn(1, 4, 1, DIM).to(device)
            values = torch.randn(1, 4, 1, DIM).to(device)
            out_k, out_v = cache.update(keys, values, layer_idx=0)

        assert out_k.shape[2] == 3
        assert out_v.shape[2] == 3

    def test_enable_after_disable(self) -> None:
        """Re-enabling compression after disable should resume compressing."""
        from transformers import DynamicCache

        cache = DynamicCache()
        tq_cache = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        # Disable, add uncompressed token
        tq_cache.disable()
        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        # Re-enable, add compressed token
        tq_cache.enable()
        out_k, out_v = cache.update(
            torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0
        )

        assert out_k.shape[2] == 2  # Both tokens accumulated


@pytest.mark.unit
class TestCompressedDynamicCache:
    """Validate CompressedDynamicCache with real VRAM savings."""

    def test_basic_update(self, device: torch.device) -> None:
        """Wrapped cache should accept and return correct tensor shapes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 8, 1, DIM).to(device)
        values = torch.randn(1, 8, 1, DIM).to(device)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape == (1, 8, 1, DIM)
        assert out_v.shape == (1, 8, 1, DIM)

    def test_multi_token_prefill(self, device: torch.device) -> None:
        """Prefill with multiple tokens should compress and dequantize all."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        seq_len = 64
        keys = torch.randn(1, 4, seq_len, DIM).to(device)
        values = torch.randn(1, 4, seq_len, DIM).to(device)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape == (1, 4, seq_len, DIM)
        assert out_v.shape == (1, 4, seq_len, DIM)

    def test_output_dtype_matches_input(self, device: torch.device) -> None:
        """Decompressed output should match the input dtype."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 4, 1, DIM, dtype=torch.bfloat16).to(device)
        values = torch.randn(1, 4, 1, DIM, dtype=torch.bfloat16).to(device)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.dtype == torch.bfloat16
        assert out_v.dtype == torch.bfloat16

    def test_stores_uint8_indices(self) -> None:
        """Compressed storage should use uint8 for indices."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        ck = cc._compressed_keys[0]
        cv = cc._compressed_values[0]
        assert ck is not None
        assert cv is not None
        assert ck.indices.dtype == torch.uint8
        assert cv.indices.dtype == torch.uint8

    def test_stores_fp32_norms(self) -> None:
        """Compressed storage should use fp32 for norms (fp16 causes degradation)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        ck = cc._compressed_keys[0]
        cv = cc._compressed_values[0]
        assert ck is not None
        assert cv is not None
        assert ck.norms.dtype == torch.float32
        assert cv.norms.dtype == torch.float32

    def test_vram_savings(self, device: torch.device) -> None:
        """Compressed storage should be smaller than FP16 baseline."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Simulate 100 tokens across 4 layers
        for layer in range(4):
            cache.update(
                torch.randn(1, 8, 100, DIM).to(device),
                torch.randn(1, 8, 100, DIM).to(device),
                layer_idx=layer,
            )

        compressed_bytes = cc.vram_bytes()
        baseline_bytes = cc.baseline_vram_bytes()

        # uint8 (1 byte) + fp32 norm (4 bytes per vector) vs fp16 (2 bytes per element)
        # 132 bytes vs 256 bytes per token per head → ~1.94x compression
        assert compressed_bytes < baseline_bytes
        ratio = baseline_bytes / compressed_bytes
        assert ratio > 1.9, f"Expected >1.9x compression, got {ratio:.2f}x"

    def test_compression_stats(self) -> None:
        """compression_stats() should return valid statistics."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(
            torch.randn(1, 4, 50, DIM), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        stats = cc.compression_stats()
        assert stats["num_layers"] == 1
        assert stats["seq_len"] == 50
        assert stats["head_dim"] == DIM
        assert stats["bits"] == BITS
        assert stats["compression_ratio"] > 1.9
        assert stats["savings_mib"] > 0

    def test_multi_layer_accumulation(self) -> None:
        """Cache should accumulate tokens across multiple update calls."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        for _ in range(3):
            keys = torch.randn(1, 4, 1, DIM)
            values = torch.randn(1, 4, 1, DIM)
            out_k, out_v = cache.update(keys, values, layer_idx=0)

        assert out_k.shape[2] == 3
        assert out_v.shape[2] == 3

    def test_get_seq_length(self) -> None:
        """get_seq_length() should reflect compressed storage."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        assert cache.get_seq_length() == 0

        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=0
        )
        assert cache.get_seq_length() == 10

        cache.update(torch.randn(1, 4, 5, DIM), torch.randn(1, 4, 5, DIM), layer_idx=0)
        assert cache.get_seq_length() == 15

    def test_disable_passthrough(self) -> None:
        """Disabling compression should pass through unmodified tensors."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cc.disable()
        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        torch.testing.assert_close(out_k[:, :, -1:, :], keys)
        torch.testing.assert_close(out_v[:, :, -1:, :], values)

    def test_restore_unwraps(self) -> None:
        """Restore should remove the wrapper entirely."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update
        original_get_seq = cache.get_seq_length
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
        assert cache.update != original_update

        cc.restore()
        assert cache.update == original_update
        assert cache.get_seq_length == original_get_seq

    def test_accuracy_parity_with_accuracy_only(self, device: torch.device) -> None:
        """Compressed cache should produce same values as accuracy-only cache.

        Both compress with TurboQuantCompressorMSE at the same bit-width,
        so the decompressed outputs should be numerically identical.
        """
        from transformers import DynamicCache

        keys = torch.randn(1, 4, 10, DIM).to(device)
        values = torch.randn(1, 4, 10, DIM).to(device)

        # Accuracy-only mode
        cache_a = DynamicCache()
        _ = TurboQuantKVCache(cache_a, head_dim=DIM, bits=BITS)
        out_k_a, out_v_a = cache_a.update(keys.clone(), values.clone(), layer_idx=0)

        # Compressed mode
        cache_c = DynamicCache()
        _ = CompressedDynamicCache(cache_c, head_dim=DIM, bits=BITS)
        out_k_c, out_v_c = cache_c.update(keys.clone(), values.clone(), layer_idx=0)

        # Both go through the same compressor, so reconstructions should match.
        # Small difference due to uint8/fp16 storage precision in compressed mode.
        cos_k = torch.nn.functional.cosine_similarity(
            out_k_a.flatten(), out_k_c.flatten(), dim=0
        )
        cos_v = torch.nn.functional.cosine_similarity(
            out_v_a.flatten(), out_v_c.flatten(), dim=0
        )
        assert cos_k > 0.999, f"Key cosine similarity {cos_k:.4f} too low"
        assert cos_v > 0.999, f"Value cosine similarity {cos_v:.4f} too low"

    def test_odd_head_dim_raises_with_4bit(self) -> None:
        """bits=4 with odd head_dim should raise ValueError."""
        from transformers import DynamicCache

        cache = DynamicCache()
        with pytest.raises(ValueError, match="even head_dim"):
            CompressedDynamicCache(cache, head_dim=127, bits=4)

    def test_compression_stats_empty(self) -> None:
        """compression_stats() on empty cache should return empty dict."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
        assert cc.compression_stats() == {}

    def test_get_seq_length_when_disabled(self) -> None:
        """get_seq_length should use original method when disabled."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=0
        )
        cc.disable()
        seq_len = cache.get_seq_length(0)
        assert seq_len == 10

    def test_enable_after_disable(self) -> None:
        """Re-enabling should resume compression on CompressedDynamicCache."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cc.disable()
        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        cc.enable()
        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        assert len(cc._compressed_keys) == 1

    def test_decompressed_buffers_accumulate(self) -> None:
        """Decompressed buffers persist across layers for incremental dequant."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Layer 0
        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=0
        )
        l0_keys = cache.layers[0].keys
        assert l0_keys is not None
        assert l0_keys.numel() > 0

        # Layer 1 — layer 0's buffer should still exist (incremental dequant)
        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=1
        )
        l0_after = cache.layers[0].keys
        l1_keys = cache.layers[1].keys
        assert l0_after is not None
        assert l0_after.numel() > 0  # Not freed
        assert l1_keys is not None
        assert l1_keys.numel() > 0

        # Compressed storage should also exist
        assert len(cc._compressed_keys) == 2


@pytest.mark.unit
class TestLazyInitCompat:
    """Validate lazy_initialization compatibility with transformers 4.x and 5.x."""

    def test_lazy_init_two_args_accepted(self) -> None:
        """When lazy_initialization accepts two args (transformers 5.x), both are passed."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        # Mock lazy_initialization to accept two args and record calls
        calls: list[tuple[torch.Tensor, ...]] = []

        def two_arg_lazy_init(
            self_layer: Any, k: torch.Tensor, v: torch.Tensor
        ) -> None:
            calls.append((k, v))
            self_layer.dtype, self_layer.device = k.dtype, k.device
            self_layer.keys = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.values = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.is_initialized = True

        with patch(
            "transformers.cache_utils.DynamicLayer.lazy_initialization",
            two_arg_lazy_init,
        ):
            cache.update(keys, values, layer_idx=0)

        assert len(calls) == 1
        assert calls[0][0].shape == keys.shape
        assert calls[0][1].shape == values.shape

    def test_lazy_init_one_arg_fallback(self) -> None:
        """When lazy_initialization rejects two args (transformers 4.x), falls back to one."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        # Mock lazy_initialization to reject two args (transformers 4.x behavior)
        calls: list[tuple[torch.Tensor, ...]] = []

        def one_arg_lazy_init(self_layer: Any, k: torch.Tensor) -> None:
            calls.append((k,))
            self_layer.dtype, self_layer.device = k.dtype, k.device
            self_layer.keys = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.values = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.is_initialized = True

        with patch(
            "transformers.cache_utils.DynamicLayer.lazy_initialization",
            one_arg_lazy_init,
        ):
            cache.update(keys, values, layer_idx=0)

        assert len(calls) == 1
        assert calls[0][0].shape == keys.shape

    def test_lazy_init_fused_mode_two_args(self) -> None:
        """Fused mode path also passes both args when accepted."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
        cc.fused_mode = True

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        calls: list[tuple[torch.Tensor, ...]] = []

        def two_arg_lazy_init(
            self_layer: Any, k: torch.Tensor, v: torch.Tensor
        ) -> None:
            calls.append((k, v))
            self_layer.dtype, self_layer.device = k.dtype, k.device
            self_layer.keys = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.values = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.is_initialized = True

        with patch(
            "transformers.cache_utils.DynamicLayer.lazy_initialization",
            two_arg_lazy_init,
        ):
            cache.update(keys, values, layer_idx=0)

        assert len(calls) == 1
        assert calls[0][0].shape == keys.shape
        assert calls[0][1].shape == values.shape

    def test_lazy_init_fused_mode_one_arg_fallback(self) -> None:
        """Fused mode path falls back to one arg when two are rejected."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
        cc.fused_mode = True

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        calls: list[tuple[torch.Tensor, ...]] = []

        def one_arg_lazy_init(self_layer: Any, k: torch.Tensor) -> None:
            calls.append((k,))
            self_layer.dtype, self_layer.device = k.dtype, k.device
            self_layer.keys = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.values = torch.tensor([], dtype=k.dtype, device=k.device)
            self_layer.is_initialized = True

        with patch(
            "transformers.cache_utils.DynamicLayer.lazy_initialization",
            one_arg_lazy_init,
        ):
            cache.update(keys, values, layer_idx=0)

        assert len(calls) == 1
        assert calls[0][0].shape == keys.shape
