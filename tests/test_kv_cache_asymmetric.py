"""Tests for asymmetric K/V bit-width support in CompressedDynamicCache.

Validates that keys and values can be compressed at different bit-widths
(e.g., k_bits=4, v_bits=3) while maintaining backward compatibility with
the symmetric `bits` shorthand.
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.kv_cache import CompressedDynamicCache

from .conftest import BITS, DIM


@pytest.mark.unit
class TestAsymmetricConstruction:
    """Validate asymmetric k_bits/v_bits parameter handling."""

    def test_symmetric_shorthand_backward_compat(self) -> None:
        """bits=4 alone should set both k_bits and v_bits to 4."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=4)

        assert cc.k_bits == 4
        assert cc.v_bits == 4

    def test_asymmetric_construction_k4_v3(self) -> None:
        """k_bits=4, v_bits=3 should create compressors with different bits."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=3)

        assert cc.k_bits == 4
        assert cc.v_bits == 3
        assert cc.key_compressor.bits == 4
        assert cc.value_compressor.bits == 3

    def test_asymmetric_construction_k4_v2(self) -> None:
        """k_bits=4, v_bits=2 should work for research configs."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=2)

        assert cc.k_bits == 4
        assert cc.v_bits == 2

    def test_bits_fallback_when_kv_none(self) -> None:
        """Bits should apply to both when k_bits/v_bits are not set."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        assert cc.k_bits == BITS
        assert cc.v_bits == BITS

    def test_no_bits_no_kv_bits_raises(self) -> None:
        """Omitting all bit-width parameters should raise ValueError."""
        from transformers import DynamicCache

        cache = DynamicCache()
        with pytest.raises(ValueError, match="bit"):
            CompressedDynamicCache(cache, head_dim=DIM, bits=None)

    def test_k_bits_4_odd_head_dim_raises(self) -> None:
        """k_bits=4 with odd head_dim should raise (nibble packing requires even)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        with pytest.raises(ValueError, match="even head_dim"):
            CompressedDynamicCache(cache, head_dim=127, k_bits=4, v_bits=3)

    def test_v_bits_3_odd_head_dim_ok(self) -> None:
        """v_bits=3 with odd head_dim should NOT raise (no nibble packing)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        # k_bits=3 too (no nibble packing for either), odd head_dim is fine
        cc = CompressedDynamicCache(cache, head_dim=127, k_bits=3, v_bits=3)
        assert cc.k_bits == 3
        assert cc.v_bits == 3

    def test_v_bits_4_odd_head_dim_raises(self) -> None:
        """v_bits=4 with odd head_dim should raise (nibble packing requires even)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        with pytest.raises(ValueError, match="even head_dim"):
            CompressedDynamicCache(cache, head_dim=127, k_bits=3, v_bits=4)


@pytest.mark.unit
class TestAsymmetricCompression:
    """Validate asymmetric compress/decompress round-trip."""

    def test_asymmetric_update_shapes(self) -> None:
        """Asymmetric CDC should produce correct output shapes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=3)

        keys = torch.randn(1, 4, 10, DIM)
        values = torch.randn(1, 4, 10, DIM)
        out_k, out_v = cache.update(keys, values, layer_idx=0)

        assert out_k.shape == (1, 4, 10, DIM)
        assert out_v.shape == (1, 4, 10, DIM)

    def test_k4_nibble_packed_v3_unpacked(self) -> None:
        """K at 4-bit should be nibble-packed, V at 3-bit should be unpacked."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=3)

        cache.update(
            torch.randn(1, 4, 1, DIM),
            torch.randn(1, 4, 1, DIM),
            layer_idx=0,
        )

        ck = cc._compressed_keys[0]
        cv = cc._compressed_values[0]
        assert ck is not None
        assert cv is not None
        # K nibble-packed: head_dim // 2
        assert ck.packed is True
        assert ck.indices.shape[-1] == DIM // 2
        # V unpacked: head_dim
        assert cv.packed is False
        assert cv.indices.shape[-1] == DIM

    def test_asymmetric_seq_accumulation(self) -> None:
        """Asymmetric CDC should accumulate tokens correctly."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=3)

        for _ in range(5):
            cache.update(
                torch.randn(1, 4, 1, DIM),
                torch.randn(1, 4, 1, DIM),
                layer_idx=0,
            )

        assert cache.get_seq_length() == 5


@pytest.mark.unit
class TestAsymmetricVRAM:
    """Validate VRAM calculations for asymmetric configurations."""

    def test_vram_bytes_asymmetric(self) -> None:
        """Asymmetric config should have correct VRAM calculation."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=3)

        cache.update(
            torch.randn(1, 1, 1, DIM),
            torch.randn(1, 1, 1, DIM),
            layer_idx=0,
        )

        vram = cc.vram_bytes()
        # K: 64 nibble-packed + 4 norm = 68 bytes
        # V: 128 unpacked + 4 norm = 132 bytes
        # Total: 200 bytes per token per head per layer
        expected = 68 + 132
        assert vram == expected, f"Expected {expected}, got {vram}"

    def test_vram_bytes_symmetric_unchanged(self) -> None:
        """Symmetric bits=4 should give same VRAM as before (backward compat)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=4)

        cache.update(
            torch.randn(1, 1, 1, DIM),
            torch.randn(1, 1, 1, DIM),
            layer_idx=0,
        )

        vram = cc.vram_bytes()
        # K: 64 + 4 = 68, V: 64 + 4 = 68, total = 136
        assert vram == 136


@pytest.mark.unit
class TestAsymmetricStats:
    """Validate compression_stats() for asymmetric configurations."""

    def test_stats_reports_k_bits_v_bits(self) -> None:
        """compression_stats() should report separate k_bits and v_bits."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=3)

        cache.update(
            torch.randn(1, 4, 50, DIM),
            torch.randn(1, 4, 50, DIM),
            layer_idx=0,
        )

        stats = cc.compression_stats()
        assert stats["k_bits"] == 4
        assert stats["v_bits"] == 3

    def test_stats_vram_at_context_lengths(self) -> None:
        """compression_stats() should include VRAM estimates at representative lengths."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, k_bits=4, v_bits=3)

        cache.update(
            torch.randn(1, 4, 50, DIM),
            torch.randn(1, 4, 50, DIM),
            layer_idx=0,
        )

        stats = cc.compression_stats()
        assert "vram_estimate" in stats
        estimates = stats["vram_estimate"]
        assert 4096 in estimates
        assert 16384 in estimates
        assert 32768 in estimates

    def test_stats_backward_compat_symmetric(self) -> None:
        """Symmetric bits should still appear in stats for backward compat."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(
            torch.randn(1, 4, 50, DIM),
            torch.randn(1, 4, 50, DIM),
            layer_idx=0,
        )

        stats = cc.compression_stats()
        assert stats["k_bits"] == BITS
        assert stats["v_bits"] == BITS
        # bits key should still be present for backward compat
        assert stats["bits"] == BITS

    def test_stats_compression_ratio_asymmetric(self) -> None:
        """Asymmetric K4/V3 should have different ratio than symmetric K4/V4."""
        from transformers import DynamicCache

        # Symmetric K4/V4
        cache_sym = DynamicCache()
        cc_sym = CompressedDynamicCache(cache_sym, head_dim=DIM, bits=4)
        cache_sym.update(
            torch.randn(1, 4, 100, DIM),
            torch.randn(1, 4, 100, DIM),
            layer_idx=0,
        )
        ratio_sym = cc_sym.compression_stats()["compression_ratio"]

        # Asymmetric K4/V3
        cache_asym = DynamicCache()
        cc_asym = CompressedDynamicCache(cache_asym, head_dim=DIM, k_bits=4, v_bits=3)
        cache_asym.update(
            torch.randn(1, 4, 100, DIM),
            torch.randn(1, 4, 100, DIM),
            layer_idx=0,
        )
        ratio_asym = cc_asym.compression_stats()["compression_ratio"]

        # K4/V3 (2.56x) < K4/V4 (3.76x)
        assert ratio_asym < ratio_sym


@pytest.mark.unit
class TestPackedSizeHelper:
    """Validate _packed_size() helper for byte layout calculations."""

    def test_packed_size_4bit(self) -> None:
        """4-bit should use nibble packing (head_dim // 2)."""
        from turboquant_vllm.kv_cache import _packed_size

        assert _packed_size(4, 128) == 64
        assert _packed_size(4, 256) == 128

    def test_packed_size_3bit(self) -> None:
        """3-bit should use unpacked (head_dim)."""
        from turboquant_vllm.kv_cache import _packed_size

        assert _packed_size(3, 128) == 128
        assert _packed_size(3, 64) == 64

    def test_packed_size_2bit(self) -> None:
        """2-bit should use unpacked (head_dim)."""
        from turboquant_vllm.kv_cache import _packed_size

        assert _packed_size(2, 128) == 128
