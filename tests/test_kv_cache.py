"""Tests for TurboQuant KV cache wrapper integration."""

import pytest
import torch
import torch.nn.functional as F

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

        assert cc._compressed_keys[0].indices.dtype == torch.uint8
        assert cc._compressed_values[0].indices.dtype == torch.uint8

    def test_stores_fp32_norms(self) -> None:
        """Compressed storage should use fp32 for norms (fp16 causes degradation)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        assert cc._compressed_keys[0].norms.dtype == torch.float32
        assert cc._compressed_values[0].norms.dtype == torch.float32

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
class TestBitWidthSupport:
    """Validate CompressedDynamicCache at non-default bit widths."""

    @pytest.mark.parametrize("bits", [2, 5])
    def test_compress_decompress_quality(self, bits: int) -> None:
        """2-bit and 5-bit should produce valid output with expected quality ordering."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=bits)

        keys = torch.randn(1, 4, 32, DIM)
        values = torch.randn(1, 4, 32, DIM)
        out_k, out_v = cache.update(keys, values, layer_idx=0)

        assert out_k.shape == keys.shape
        cos = torch.nn.functional.cosine_similarity(
            keys.flatten(), out_k.flatten(), dim=0
        )
        if bits == 2:
            # 4 quantization levels — coarse, flattened cosine varies with data
            assert cos > 0.75, f"2-bit cosine {cos:.4f} too low"
        else:
            # 32 levels — near-lossless per-vector but flattened cosine is noisier
            assert cos > 0.90, f"5-bit cosine {cos:.4f} too low"

    def test_5bit_better_quality_than_3bit(self) -> None:
        """5-bit (32 levels) should beat 3-bit (8 levels) in reconstruction."""
        from transformers import DynamicCache

        original = torch.randn(1, 4, 50, DIM)

        cache3 = DynamicCache()
        _ = CompressedDynamicCache(cache3, head_dim=DIM, bits=3)
        out3, _ = cache3.update(
            original.clone(), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        cache5 = DynamicCache()
        _ = CompressedDynamicCache(cache5, head_dim=DIM, bits=5)
        out5, _ = cache5.update(
            original.clone(), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        cos3 = torch.nn.functional.cosine_similarity(
            original.flatten(), out3.flatten(), dim=0
        )
        cos5 = torch.nn.functional.cosine_similarity(
            original.flatten(), out5.flatten(), dim=0
        )
        assert cos5 > cos3, f"5-bit ({cos5:.4f}) should beat 3-bit ({cos3:.4f})"


BITS_4 = 4


@pytest.mark.unit
class TestNibblePacking:
    """Validate TQ4 nibble-packed storage in CompressedDynamicCache."""

    def test_basic_update_4bit(self, device: torch.device) -> None:
        """4-bit compressed cache should accept and return correct shapes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        keys = torch.randn(1, 8, 1, DIM).to(device)
        values = torch.randn(1, 8, 1, DIM).to(device)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape == (1, 8, 1, DIM)
        assert out_v.shape == (1, 8, 1, DIM)

    def test_indices_are_nibble_packed(self) -> None:
        """At bits=4, indices should be half the head_dim (packed pairs)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=0
        )

        # Packed: head_dim // 2 = 64 instead of 128
        assert cc._compressed_keys[0].indices.shape == (1, 4, 10, DIM // 2)
        assert cc._compressed_keys[0].indices.dtype == torch.uint8
        assert cc._compressed_keys[0].packed is True

    def test_nibble_pack_unpack_roundtrip(self, device: torch.device) -> None:
        """Pack then unpack should recover exact original indices."""
        indices = torch.randint(0, 16, (2, 4, 8, DIM), dtype=torch.uint8).to(device)

        packed = CompressedDynamicCache._nibble_pack(indices)
        assert packed.shape == (2, 4, 8, DIM // 2)

        unpacked = CompressedDynamicCache._nibble_unpack(packed)
        assert unpacked.shape == (2, 4, 8, DIM)
        torch.testing.assert_close(unpacked, indices.long())

    def test_compression_ratio_4bit(self, device: torch.device) -> None:
        """4-bit nibble-packed should achieve ~3.7x compression."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        for layer in range(4):
            cache.update(
                torch.randn(1, 8, 100, DIM).to(device),
                torch.randn(1, 8, 100, DIM).to(device),
                layer_idx=layer,
            )

        ratio = cc.baseline_vram_bytes() / cc.vram_bytes()
        # 68 bytes per block vs 256 → ~3.76x
        assert ratio > 3.5, f"Expected >3.5x compression, got {ratio:.2f}x"

    def test_4bit_better_quality_than_3bit(self, device: torch.device) -> None:
        """TQ4 should have higher cosine similarity than TQ3."""
        from transformers import DynamicCache

        original = torch.randn(1, 4, 50, DIM).to(device)

        # TQ3
        cache3 = DynamicCache()
        _ = CompressedDynamicCache(cache3, head_dim=DIM, bits=BITS)
        out3, _ = cache3.update(
            original.clone(), torch.randn(1, 4, 50, DIM).to(device), layer_idx=0
        )
        cos3 = torch.nn.functional.cosine_similarity(
            original.flatten(), out3.flatten(), dim=0
        )

        # TQ4
        cache4 = DynamicCache()
        _ = CompressedDynamicCache(cache4, head_dim=DIM, bits=BITS_4)
        out4, _ = cache4.update(
            original.clone(), torch.randn(1, 4, 50, DIM).to(device), layer_idx=0
        )
        cos4 = torch.nn.functional.cosine_similarity(
            original.flatten(), out4.flatten(), dim=0
        )

        assert cos4 > cos3, f"TQ4 ({cos4:.4f}) should beat TQ3 ({cos3:.4f})"

    def test_multi_layer_generation_4bit(self) -> None:
        """Nibble-packed cache should handle multi-layer prefill + gen."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        # Prefill 4 layers
        for layer in range(4):
            cache.update(
                torch.randn(1, 4, 64, DIM),
                torch.randn(1, 4, 64, DIM),
                layer_idx=layer,
            )

        # Generate 5 tokens
        for _ in range(5):
            for layer in range(4):
                out_k, _ = cache.update(
                    torch.randn(1, 4, 1, DIM),
                    torch.randn(1, 4, 1, DIM),
                    layer_idx=layer,
                )

        assert out_k.shape == (1, 4, 69, DIM)  # 64 prefill + 5 gen
        assert cache.get_seq_length(0) == 69

    def test_compression_stats_4bit(self) -> None:
        """Stats should report nibble_packed=True and correct head_dim."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        cache.update(
            torch.randn(1, 4, 50, DIM), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        stats = cc.compression_stats()
        assert stats["bits"] == 4
        assert stats["nibble_packed"] is True
        assert stats["head_dim"] == DIM
        assert stats["compression_ratio"] > 3.5


@pytest.mark.unit
class TestLongSequenceRegression:
    """Regression tests for precision at realistic sequence lengths.

    These tests simulate multi-layer transformer KV cache usage at
    1000+ tokens across 36 layers to catch precision bugs (like the
    fp16 norms issue) that don't manifest in short-sequence unit tests.
    """

    NUM_LAYERS = 36
    NUM_HEADS = 8
    PREFILL_LEN = 1024
    GEN_STEPS = 32

    def _run_prefill_and_generate(
        self,
        wrapper_cls: type,
        bits: int = BITS,
    ) -> list[torch.Tensor]:
        """Simulate prefill + generation through all layers.

        Args:
            wrapper_cls: TurboQuantKVCache or CompressedDynamicCache.
            bits: Quantization bit width.

        Returns:
            List of decompressed key tensors (one per layer) after
            prefill + generation steps, for the final generation step.
        """
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = wrapper_cls(cache, head_dim=DIM, bits=bits)

        # Prefill: push PREFILL_LEN tokens through all layers
        for layer_idx in range(self.NUM_LAYERS):
            k = torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM)
            v = torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM)
            cache.update(k, v, layer_idx=layer_idx)

        # Generation: push 1 token at a time through all layers
        final_keys: list[torch.Tensor] = []
        for step in range(self.GEN_STEPS):
            step_keys = []
            for layer_idx in range(self.NUM_LAYERS):
                k = torch.randn(1, self.NUM_HEADS, 1, DIM)
                v = torch.randn(1, self.NUM_HEADS, 1, DIM)
                out_k, _ = cache.update(k, v, layer_idx=layer_idx)
                step_keys.append(out_k)
            final_keys = step_keys

        return final_keys

    def test_compressed_matches_accuracy_only(self) -> None:
        """CompressedDynamicCache should closely match TurboQuantKVCache.

        Both use TurboQuantCompressorMSE at the same bit-width. The
        only difference is storage format (uint8+fp32 vs fp32 round-trip).
        After 36 layers and 1000+ tokens, the outputs should still be
        nearly identical. A cosine similarity below 0.999 per layer
        would indicate a precision regression.
        """
        torch.manual_seed(42)
        keys_accuracy = self._run_prefill_and_generate(TurboQuantKVCache)

        torch.manual_seed(42)
        keys_compressed = self._run_prefill_and_generate(CompressedDynamicCache)

        assert len(keys_accuracy) == self.NUM_LAYERS
        assert len(keys_compressed) == self.NUM_LAYERS

        for layer_idx in range(self.NUM_LAYERS):
            ka = keys_accuracy[layer_idx]
            kc = keys_compressed[layer_idx]
            assert ka.shape == kc.shape, (
                f"Layer {layer_idx}: shape mismatch {ka.shape} vs {kc.shape}"
            )

            cos = F.cosine_similarity(ka.flatten(), kc.flatten(), dim=0)
            assert cos > 0.999, (
                f"Layer {layer_idx}: cosine similarity {cos:.6f} < 0.999 — "
                f"precision regression detected"
            )

    def test_no_nan_or_inf_at_scale(self) -> None:
        """No NaN or Inf should appear after many layers and tokens."""
        torch.manual_seed(42)
        keys = self._run_prefill_and_generate(CompressedDynamicCache)

        for layer_idx, k in enumerate(keys):
            assert not k.isnan().any(), f"Layer {layer_idx}: NaN in keys"
            assert not k.isinf().any(), f"Layer {layer_idx}: Inf in keys"

    def test_seq_length_tracks_correctly(self) -> None:
        """get_seq_length should return prefill + generation tokens."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Prefill
        for layer_idx in range(self.NUM_LAYERS):
            cache.update(
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                layer_idx=layer_idx,
            )

        assert cache.get_seq_length(0) == self.PREFILL_LEN

        # Generate 10 tokens
        for _ in range(10):
            for layer_idx in range(self.NUM_LAYERS):
                cache.update(
                    torch.randn(1, self.NUM_HEADS, 1, DIM),
                    torch.randn(1, self.NUM_HEADS, 1, DIM),
                    layer_idx=layer_idx,
                )

        assert cache.get_seq_length(0) == self.PREFILL_LEN + 10

    def test_compression_stats_at_scale(self) -> None:
        """Compression stats should reflect realistic cache sizes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        for layer_idx in range(self.NUM_LAYERS):
            cache.update(
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                layer_idx=layer_idx,
            )

        stats = cc.compression_stats()
        assert stats["num_layers"] == self.NUM_LAYERS
        assert stats["seq_len"] == self.PREFILL_LEN
        assert stats["num_heads"] == self.NUM_HEADS
        assert stats["compression_ratio"] > 1.9
        # At 36 layers × 8 heads × 1024 tokens, savings should be substantial
        assert stats["savings_mib"] > 50

    def test_4bit_nibble_at_scale(self) -> None:
        """TQ4 nibble-packed should work at 36 layers with 1024 tokens."""
        torch.manual_seed(42)
        keys_accuracy = self._run_prefill_and_generate(TurboQuantKVCache, bits=4)

        torch.manual_seed(42)
        keys_compressed = self._run_prefill_and_generate(CompressedDynamicCache, bits=4)

        for layer_idx in range(self.NUM_LAYERS):
            cos = F.cosine_similarity(
                keys_accuracy[layer_idx].flatten(),
                keys_compressed[layer_idx].flatten(),
                dim=0,
            )
            assert cos > 0.999, (
                f"Layer {layer_idx}: TQ4 cosine similarity {cos:.6f} < 0.999"
            )
