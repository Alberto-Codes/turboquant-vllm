"""Tests for TQ4 cache spec, byte calculations, and packed cache round-trip.

Phase 3c tests: TQ4FullAttentionSpec, byte layout math, compress_and_store,
decompress_cache round-trip.  Requires vLLM to be installed.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: E402

from turboquant_vllm.vllm.tq4_backend import (  # noqa: E402  # isort: skip
    TQ4AttentionBackend,
    TQ4FullAttentionSpec,
    _tq4_bytes_per_token,
    _tq4_bytes_per_token_kv,
)

from tests.helpers.vllm_impl import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_SIZE,
    NUM_KV_HEADS,
    make_cache,
    make_impl,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Phase 3c: TQ4 byte layout math
# ---------------------------------------------------------------------------


class TestTQ4ByteCalculation:
    """TQ4 page layout math."""

    def test_bytes_per_token_head_128(self) -> None:
        """head_dim=128 yields 68 bytes per token per head."""
        assert _tq4_bytes_per_token(128) == 68

    def test_bytes_per_token_head_64(self) -> None:
        """head_dim=64 yields 36 bytes per token per head."""
        assert _tq4_bytes_per_token(64) == 36

    def test_bytes_per_token_head_256(self) -> None:
        """head_dim=256 yields 132 bytes per token per head."""
        assert _tq4_bytes_per_token(256) == 132

    def test_bytes_per_token_kv_128(self) -> None:
        """head_dim=128 yields 136 bytes per token for K+V combined."""
        assert _tq4_bytes_per_token_kv(128) == 136

    def test_bytes_per_token_kv_64(self) -> None:
        """head_dim=64 yields 72 bytes per token for K+V combined."""
        assert _tq4_bytes_per_token_kv(64) == 72


# ---------------------------------------------------------------------------
# Phase 3c: TQ4FullAttentionSpec
# ---------------------------------------------------------------------------


class TestTQ4FullAttentionSpec:
    """TQ4 cache spec page size override."""

    def test_subclasses_full_attention_spec(self) -> None:
        """TQ4FullAttentionSpec inherits from FullAttentionSpec."""
        assert issubclass(TQ4FullAttentionSpec, FullAttentionSpec)

    def test_page_size_bytes_molmo2_8b(self) -> None:
        """Molmo2-8B: 8 KV heads, head_dim=128, block_size=16.

        Padded slot: next_power_of_2(136) = 256.
        Page = 16 * 8 * 256 = 32,768.
        """
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        # Padded for hybrid model alignment
        assert spec.page_size_bytes == 32_768

    def test_page_size_vs_fp16(self) -> None:
        """TQ4 page is smaller than FP16 (padded for hybrid alignment)."""
        tq4_spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        fp16_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
        ratio = fp16_spec.page_size_bytes / tq4_spec.page_size_bytes
        # With padding: 65536/32768 = 2.0 (less than ideal 3.76x but
        # needed for hybrid model page alignment)
        assert ratio >= 1.5

    def test_dtype_is_uint8(self) -> None:
        """Spec preserves uint8 dtype for packed cache storage."""
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        assert spec.dtype == torch.uint8

    def test_block_size_scales_page_size(self) -> None:
        """Doubling block_size doubles page_size_bytes."""
        spec_16 = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        spec_32 = TQ4FullAttentionSpec(
            block_size=32,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        assert spec_32.page_size_bytes == 2 * spec_16.page_size_bytes

    def test_frozen_dataclass(self) -> None:
        """Frozen dataclass prevents attribute mutation."""
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        with pytest.raises(AttributeError):
            spec.block_size = 32  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Phase 3c: compress_and_store + decompress_cache round-trip
# ---------------------------------------------------------------------------


class TestTQ4PackedCacheRoundTrip:
    """Compress -> store -> decompress round-trip on packed uint8 cache."""

    def test_compress_store_decompress_single_token(self, tq4_quantizer) -> None:
        """Single token compress-store-decompress writes to correct slot."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        slot_mapping = torch.tensor([5])  # slot 5 = block 0, pos 5

        impl._compress_and_store(key, value, kv_cache, slot_mapping)

        # Verify bytes were written (slot 5 should be non-zero)
        flat = kv_cache.view(-1, impl._total_bytes)
        assert flat[5].any(), "Slot 5 should have data"
        assert not flat[0].any(), "Slot 0 should still be zeros"

        # Decompress full cache and check round-trip
        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.float32)
        assert key_cache.shape == (
            4,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )
        assert value_cache.shape == (
            4,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )

        # Check the written slot has non-zero data
        reconstructed_k = key_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)[5]
        reconstructed_v = value_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)[5]
        assert reconstructed_k.any(), "Decompressed K should be non-zero"
        assert reconstructed_v.any(), "Decompressed V should be non-zero"

    def test_round_trip_cosine_similarity(self, tq4_quantizer) -> None:
        """TQ4 round-trip should achieve >0.85 cosine on random data.

        Note: real model KV cache data achieves >0.97 cosine (validated
        in experiment 003/005).  Random Gaussian vectors give lower cosine
        because their distribution differs from actual KV activations.
        """
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=2)

        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        slot_mapping = torch.tensor([0])

        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.float32)

        recon_k = key_cache[0, 0]  # block 0, pos 0
        recon_v = value_cache[0, 0]

        failures = []
        for h in range(NUM_KV_HEADS):
            cos_k = torch.nn.functional.cosine_similarity(
                key[0, h].unsqueeze(0), recon_k[h].unsqueeze(0)
            ).item()
            cos_v = torch.nn.functional.cosine_similarity(
                value[0, h].unsqueeze(0), recon_v[h].unsqueeze(0)
            ).item()
            if cos_k <= 0.85:
                failures.append(f"K head {h} cosine {cos_k:.4f} < 0.85")
            if cos_v <= 0.85:
                failures.append(f"V head {h} cosine {cos_v:.4f} < 0.85")
        assert not failures, "\n".join(failures)

    def test_multi_token_scatter_write(self, tq4_quantizer) -> None:
        """Multiple tokens scattered to different slots."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)
        N = 5

        key = torch.randn(N, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(N, NUM_KV_HEADS, HEAD_SIZE)
        # Scatter to non-contiguous slots across blocks
        slot_mapping = torch.tensor([0, 17, 33, 48, 63])

        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.float32)

        flat_k = key_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)

        failures = []
        for i, slot in enumerate(slot_mapping.tolist()):
            for h in range(NUM_KV_HEADS):
                cos_k = torch.nn.functional.cosine_similarity(
                    key[i, h].unsqueeze(0), flat_k[slot, h].unsqueeze(0)
                ).item()
                if cos_k <= 0.85:
                    failures.append(
                        f"Token {i} slot {slot} head {h}: K cos {cos_k:.4f}"
                    )
        assert not failures, "\n".join(failures)

    def test_empty_slots_decompress_to_zero(self, tq4_quantizer) -> None:
        """Unwritten slots should decompress to zero."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=2)

        # Write only slot 0
        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0]))

        key_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        flat_k = key_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)

        # Slot 0 should have data, slot 1 should be all zeros
        assert flat_k[0].any(), "Slot 0 should have data"
        assert not flat_k[1].any(), "Slot 1 should be zeros"

    def test_overwrite_slot(self, tq4_quantizer) -> None:
        """Writing to the same slot twice should overwrite."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=1)

        key1 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        key2 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)

        impl._compress_and_store(key1, value, kv_cache, torch.tensor([0]))
        impl._compress_and_store(key2, value, kv_cache, torch.tensor([0]))

        key_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        recon_k = key_cache[0, 0]

        # Should be closer to key2 than key1
        cos_k2 = torch.nn.functional.cosine_similarity(
            key2[0, 0].unsqueeze(0), recon_k[0].unsqueeze(0)
        )
        cos_k1 = torch.nn.functional.cosine_similarity(
            key1[0, 0].unsqueeze(0), recon_k[0].unsqueeze(0)
        )
        assert cos_k2 > cos_k1, "Overwrite should match key2, not key1"

    def test_cache_shape_matches_backend(self, tq4_quantizer) -> None:
        """Cache shape from backend matches what decompress expects."""
        shape = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=10,
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
        )
        cache = torch.zeros(*shape, dtype=torch.uint8)
        impl = make_impl(tq4_quantizer)

        key_cache, value_cache = impl._decompress_cache(cache, torch.float32)
        assert key_cache.shape == (
            10,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )
        assert value_cache.shape == (
            10,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )

    def test_bfloat16_output_dtype(self, tq4_quantizer) -> None:
        """Decompress can produce bf16 output."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=1)

        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0]))

        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.bfloat16)
        assert key_cache.dtype == torch.bfloat16
        assert value_cache.dtype == torch.bfloat16

    def test_no_ensure_device_flag(self, tq4_quantizer) -> None:
        """Eager device init removed legacy _tq4_on_device flag (D7 mod 5)."""
        impl = make_impl(tq4_quantizer)
        assert not hasattr(impl, "_tq4_on_device")
        assert not hasattr(impl, "_ensure_device")


# ---------------------------------------------------------------------------
# Story 8.2: kv_cache=None guard
# ---------------------------------------------------------------------------


class TestForwardKvCacheNone:
    """Guard: forward() handles kv_cache=None without crashing."""

    def test_forward_kv_cache_none_returns_zero(self, tq4_quantizer) -> None:
        """forward() with kv_cache=None returns zero-filled output."""
        from types import SimpleNamespace

        impl = make_impl(tq4_quantizer)
        # forward() needs attn_type for the encoder check — set to DECODER
        from vllm.v1.attention.backend import AttentionType

        impl.attn_type = AttentionType.DECODER

        num_tokens = 4
        query = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE)
        key = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE)
        value = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE)
        output = torch.ones(num_tokens, NUM_KV_HEADS * HEAD_SIZE)

        attn_metadata = SimpleNamespace(num_actual_tokens=num_tokens)

        result = impl.forward(
            layer=None,
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=attn_metadata,
            output=output,
        )

        assert result is output
        assert (result == 0).all(), "Output should be zero-filled"

    def test_forward_kv_cache_none_does_not_init_buffers(self, tq4_quantizer) -> None:
        """kv_cache=None should not trigger CUDA graph buffer allocation."""
        from types import SimpleNamespace

        impl = make_impl(tq4_quantizer)
        from vllm.v1.attention.backend import AttentionType

        impl.attn_type = AttentionType.DECODER

        num_tokens = 1
        query = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE)
        key = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE)
        value = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE)
        output = torch.ones(num_tokens, NUM_KV_HEADS * HEAD_SIZE)

        attn_metadata = SimpleNamespace(num_actual_tokens=num_tokens)

        impl.forward(
            layer=None,
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=attn_metadata,
            output=output,
        )

        assert not impl._cg_buffers_ready, "_cg_buffers_ready should remain False"


# TestCUDAGraphBufferPreallocation moved to test_vllm_cache_cudagraph.py
# (Test Maturity Priority 1 — split oversized test file)


# ---------------------------------------------------------------------------
# Story 10.2: Asymmetric K/V byte layout
# ---------------------------------------------------------------------------


class TestTQ4AsymmetricByteLayout:
    """Validate TQ4 backend with asymmetric K/V bit-widths."""

    def test_bytes_per_token_symmetric_default(self) -> None:
        """Default (no bits args) should match original 136 bytes."""
        assert _tq4_bytes_per_token_kv(128) == 136

    def test_asymmetric_compress_decompress_roundtrip(self, tq4_quantizer) -> None:
        """K4/V3 compress-decompress roundtrip should produce valid cosine.

        In the vLLM path, Triton kernels always nibble-pack. K4/V3 uses the
        same byte layout as K4/V4 — the difference is codebook quality, not
        storage.
        """
        from turboquant_vllm.quantizer import TurboQuantMSE

        v_quantizer = TurboQuantMSE(HEAD_SIZE, 3, seed=42)
        impl = make_impl(tq4_quantizer, k_bits=4, v_bits=3, v_quantizer=v_quantizer)
        total_bytes = impl._total_bytes
        kv_cache = torch.zeros(2, BLOCK_SIZE, total_bytes, dtype=torch.uint8)

        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)

        impl._compress_and_store(key, value, kv_cache, torch.tensor([0]))
        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.float32)

        recon_k = key_cache[0, 0]
        recon_v = value_cache[0, 0]

        for h in range(NUM_KV_HEADS):
            cos_k = torch.nn.functional.cosine_similarity(
                key[0, h].unsqueeze(0), recon_k[h].unsqueeze(0)
            ).item()
            cos_v = torch.nn.functional.cosine_similarity(
                value[0, h].unsqueeze(0), recon_v[h].unsqueeze(0)
            ).item()
            assert cos_k > 0.85, f"K head {h} cosine {cos_k:.4f} too low"
            assert cos_v > 0.75, f"V head {h} cosine {cos_v:.4f} too low"

    def test_asymmetric_same_byte_layout(self, tq4_quantizer) -> None:
        """VLLM path: K4/V3 has same byte layout as K4/V4 (Triton always nibble-packs)."""
        from turboquant_vllm.quantizer import TurboQuantMSE

        v_quantizer = TurboQuantMSE(HEAD_SIZE, 3, seed=42)
        impl_sym = make_impl(tq4_quantizer)
        impl_asym = make_impl(
            tq4_quantizer, k_bits=4, v_bits=3, v_quantizer=v_quantizer
        )

        # Triton always nibble-packs, so layout is identical
        assert impl_asym._total_bytes == impl_sym._total_bytes
