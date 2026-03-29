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
    TQ4AttentionImpl,
    TQ4FullAttentionSpec,
    TQ4MetadataBuilder,
    _tq4_bytes_per_token,
    _tq4_bytes_per_token_kv,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Phase 3c: TQ4 byte layout math
# ---------------------------------------------------------------------------


class TestTQ4ByteCalculation:
    """TQ4 page layout math."""

    def test_bytes_per_token_head_128(self) -> None:
        assert _tq4_bytes_per_token(128) == 68

    def test_bytes_per_token_head_64(self) -> None:
        assert _tq4_bytes_per_token(64) == 36

    def test_bytes_per_token_head_256(self) -> None:
        assert _tq4_bytes_per_token(256) == 132

    def test_bytes_per_token_kv_128(self) -> None:
        assert _tq4_bytes_per_token_kv(128) == 136

    def test_bytes_per_token_kv_64(self) -> None:
        assert _tq4_bytes_per_token_kv(64) == 72


# ---------------------------------------------------------------------------
# Phase 3c: TQ4FullAttentionSpec
# ---------------------------------------------------------------------------


class TestTQ4FullAttentionSpec:
    """TQ4 cache spec page size override."""

    def test_subclasses_full_attention_spec(self) -> None:
        assert issubclass(TQ4FullAttentionSpec, FullAttentionSpec)

    def test_page_size_bytes_molmo2_8b(self) -> None:
        """Molmo2-8B: 8 KV heads, head_dim=128, block_size=16."""
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        # 16 tokens * 8 heads * 136 bytes/token/head = 17,408
        assert spec.page_size_bytes == 17_408

    def test_page_size_vs_fp16(self) -> None:
        """TQ4 page is 3.76x smaller than FP16."""
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
        assert abs(ratio - 3.76) < 0.01

    def test_dtype_is_uint8(self) -> None:
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        assert spec.dtype == torch.uint8

    def test_block_size_scales_page_size(self) -> None:
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

    NUM_KV_HEADS = 8
    NUM_HEADS = 32  # GQA: 4:1 Q-to-KV ratio (matches Molmo2-8B)
    HEAD_SIZE = 128
    BLOCK_SIZE = 16

    def _make_impl(self, quantizer):
        """Create a TQ4AttentionImpl without full vLLM init.

        Args:
            quantizer: TurboQuantMSE instance (from conftest tq4_quantizer fixture).
        """
        from turboquant_vllm.vllm.tq4_backend import TQ4_NORM_BYTES

        # Bypass FlashAttentionImpl.__init__ -- only init TQ4 primitives
        impl = object.__new__(TQ4AttentionImpl)
        impl.head_size = self.HEAD_SIZE
        impl.num_kv_heads = self.NUM_KV_HEADS
        impl.num_heads = self.NUM_HEADS

        impl._tq4_rotation = quantizer.rotation.clone()
        impl._tq4_centroids = quantizer.codebook.centroids.clone()
        impl._tq4_boundaries = quantizer.codebook.boundaries.clone()
        rot_t = quantizer.rotation.T.contiguous()
        impl._tq4_rot_T_even = rot_t[:, 0::2].contiguous()
        impl._tq4_rot_T_odd = rot_t[:, 1::2].contiguous()
        impl._cg_buffers_ready = False
        impl._fused_paged_available = False
        impl._max_prefill_len = 2048

        half_D = self.HEAD_SIZE // 2
        impl._half_D = half_D
        impl._k_idx_end = self.NUM_KV_HEADS * half_D
        impl._k_norm_end = impl._k_idx_end + self.NUM_KV_HEADS * TQ4_NORM_BYTES
        impl._v_idx_end = impl._k_norm_end + self.NUM_KV_HEADS * half_D
        impl._total_bytes = impl._v_idx_end + self.NUM_KV_HEADS * TQ4_NORM_BYTES

        return impl

    def _make_cache(self, num_blocks):
        total_bytes = self.NUM_KV_HEADS * _tq4_bytes_per_token_kv(self.HEAD_SIZE)
        return torch.zeros(num_blocks, self.BLOCK_SIZE, total_bytes, dtype=torch.uint8)

    def test_compress_store_decompress_single_token(self, tq4_quantizer) -> None:
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=4)

        key = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
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
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
        )
        assert value_cache.shape == (
            4,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
        )

        # Check the written slot has non-zero data
        reconstructed_k = key_cache.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)[5]
        reconstructed_v = value_cache.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)[5]
        assert reconstructed_k.any(), "Decompressed K should be non-zero"
        assert reconstructed_v.any(), "Decompressed V should be non-zero"

    def test_round_trip_cosine_similarity(self, tq4_quantizer) -> None:
        """TQ4 round-trip should achieve >0.85 cosine on random data.

        Note: real model KV cache data achieves >0.97 cosine (validated
        in experiment 003/005).  Random Gaussian vectors give lower cosine
        because their distribution differs from actual KV activations.
        """
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=2)

        key = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        slot_mapping = torch.tensor([0])

        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.float32)

        recon_k = key_cache[0, 0]  # block 0, pos 0
        recon_v = value_cache[0, 0]

        for h in range(self.NUM_KV_HEADS):
            cos_k = torch.nn.functional.cosine_similarity(
                key[0, h].unsqueeze(0), recon_k[h].unsqueeze(0)
            ).item()
            cos_v = torch.nn.functional.cosine_similarity(
                value[0, h].unsqueeze(0), recon_v[h].unsqueeze(0)
            ).item()
            assert cos_k > 0.85, f"K head {h} cosine {cos_k:.4f} < 0.85"
            assert cos_v > 0.85, f"V head {h} cosine {cos_v:.4f} < 0.85"

    def test_multi_token_scatter_write(self, tq4_quantizer) -> None:
        """Multiple tokens scattered to different slots."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=4)
        N = 5

        key = torch.randn(N, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(N, self.NUM_KV_HEADS, self.HEAD_SIZE)
        # Scatter to non-contiguous slots across blocks
        slot_mapping = torch.tensor([0, 17, 33, 48, 63])

        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.float32)

        flat_k = key_cache.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)

        for i, slot in enumerate(slot_mapping.tolist()):
            for h in range(self.NUM_KV_HEADS):
                cos_k = torch.nn.functional.cosine_similarity(
                    key[i, h].unsqueeze(0), flat_k[slot, h].unsqueeze(0)
                ).item()
                assert cos_k > 0.85, (
                    f"Token {i} slot {slot} head {h}: K cos {cos_k:.4f}"
                )

    def test_empty_slots_decompress_to_zero(self, tq4_quantizer) -> None:
        """Unwritten slots should decompress to zero."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=2)

        # Write only slot 0
        key = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0]))

        key_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        flat_k = key_cache.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)

        # Slot 0 should have data, slot 1 should be all zeros
        assert flat_k[0].any(), "Slot 0 should have data"
        assert not flat_k[1].any(), "Slot 1 should be zeros"

    def test_overwrite_slot(self, tq4_quantizer) -> None:
        """Writing to the same slot twice should overwrite."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=1)

        key1 = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        key2 = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)

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
            block_size=self.BLOCK_SIZE,
            num_kv_heads=self.NUM_KV_HEADS,
            head_size=self.HEAD_SIZE,
        )
        cache = torch.zeros(*shape, dtype=torch.uint8)
        impl = self._make_impl(tq4_quantizer)

        key_cache, value_cache = impl._decompress_cache(cache, torch.float32)
        assert key_cache.shape == (
            10,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
        )
        assert value_cache.shape == (
            10,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
        )

    def test_bfloat16_output_dtype(self, tq4_quantizer) -> None:
        """Decompress can produce bf16 output."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=1)

        key = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0]))

        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.bfloat16)
        assert key_cache.dtype == torch.bfloat16
        assert value_cache.dtype == torch.bfloat16

    def test_no_ensure_device_flag(self, tq4_quantizer) -> None:
        """Eager device init removed _tq4_on_device flag (D7 mod 5)."""
        impl = self._make_impl(tq4_quantizer)
        assert not hasattr(impl, "_tq4_on_device")
        assert not hasattr(impl, "_ensure_device")


# ---------------------------------------------------------------------------
# D7: CUDA graph buffer pre-allocation tests
# ---------------------------------------------------------------------------


class TestCUDAGraphBufferPreallocation:
    """Tests for D7 CUDA graph buffer pre-allocation."""

    NUM_KV_HEADS = 8
    NUM_HEADS = 32
    HEAD_SIZE = 128
    BLOCK_SIZE = 16

    def _make_impl(self, quantizer):
        """Create a TQ4AttentionImpl without full vLLM init."""
        from turboquant_vllm.vllm.tq4_backend import TQ4_NORM_BYTES

        impl = object.__new__(TQ4AttentionImpl)
        impl.head_size = self.HEAD_SIZE
        impl.num_kv_heads = self.NUM_KV_HEADS
        impl.num_heads = self.NUM_HEADS

        impl._tq4_rotation = quantizer.rotation.clone()
        impl._tq4_centroids = quantizer.codebook.centroids.clone()
        impl._tq4_boundaries = quantizer.codebook.boundaries.clone()
        rot_t = quantizer.rotation.T.contiguous()
        impl._tq4_rot_T_even = rot_t[:, 0::2].contiguous()
        impl._tq4_rot_T_odd = rot_t[:, 1::2].contiguous()
        impl._cg_buffers_ready = False
        impl._fused_paged_available = False
        impl._max_prefill_len = 2048

        half_D = self.HEAD_SIZE // 2
        impl._half_D = half_D
        impl._k_idx_end = self.NUM_KV_HEADS * half_D
        impl._k_norm_end = impl._k_idx_end + self.NUM_KV_HEADS * TQ4_NORM_BYTES
        impl._v_idx_end = impl._k_norm_end + self.NUM_KV_HEADS * half_D
        impl._total_bytes = impl._v_idx_end + self.NUM_KV_HEADS * TQ4_NORM_BYTES

        return impl

    def _make_cache(self, num_blocks):
        total_bytes = self.NUM_KV_HEADS * _tq4_bytes_per_token_kv(self.HEAD_SIZE)
        return torch.zeros(num_blocks, self.BLOCK_SIZE, total_bytes, dtype=torch.uint8)

    def test_get_cudagraph_support_returns_single_token_decode(self) -> None:
        """TQ4 builder reports UNIFORM_SINGLE_TOKEN_DECODE (AC 2)."""
        from vllm.v1.attention.backend import AttentionCGSupport

        result = TQ4MetadataBuilder.get_cudagraph_support(None, None)
        assert result == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def test_backend_returns_tq4_builder(self) -> None:
        """TQ4AttentionBackend.get_builder_cls() returns TQ4MetadataBuilder."""
        assert TQ4AttentionBackend.get_builder_cls() is TQ4MetadataBuilder

    def test_init_cg_buffers_shapes(self, tq4_quantizer) -> None:
        """_init_cg_buffers creates buffers of correct shape/dtype (AC 1)."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=4)

        assert not impl._cg_buffers_ready
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)
        assert impl._cg_buffers_ready

        max_tokens = 4 * self.BLOCK_SIZE  # 64
        H = self.NUM_KV_HEADS
        D = self.HEAD_SIZE

        assert impl._cg_decompress_k.shape == (max_tokens, H, D)
        assert impl._cg_decompress_k.dtype == torch.float16
        assert impl._cg_decompress_v.shape == (max_tokens, H, D)
        assert impl._cg_decompress_v.dtype == torch.float16

        assert impl._cg_compress_packed.shape == (1, H, D // 2)
        assert impl._cg_compress_packed.dtype == torch.uint8
        assert impl._cg_compress_norms.shape == (1, H, 1)
        assert impl._cg_compress_norms.dtype == torch.float32

        assert impl._cg_q_rot.shape == (1, self.NUM_HEADS, D)
        assert impl._cg_q_rot.dtype == torch.float32
        assert impl._cg_q_rot_cast.shape == (1, self.NUM_HEADS, D)
        assert impl._cg_q_rot_cast.dtype == torch.float16

        assert impl._cg_compress_row.shape == (1, impl._total_bytes)
        assert impl._cg_compress_row.dtype == torch.uint8

    def test_preallocated_decompress_matches_dynamic(self, tq4_quantizer) -> None:
        """Blocking acceptance test: pre-allocated buffers match dynamic (AC 4).

        Uses float16 (matching pre-allocated buffer dtype and real forward path).
        """
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=4)

        # Write some tokens
        key = torch.randn(3, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(3, self.NUM_KV_HEADS, self.HEAD_SIZE)
        slot_mapping = torch.tensor([0, 5, 33])
        impl._compress_and_store(key, value, kv_cache, slot_mapping)

        # Dynamic allocation (no out= buffers)
        k_dynamic, v_dynamic = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )

        # Pre-allocated buffers (max-size, sliced by decompress)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)
        k_prealloc, v_prealloc = impl._decompress_cache(
            kv_cache,
            torch.float16,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )

        # Must be IDENTICAL (not just close)
        assert torch.equal(k_dynamic, k_prealloc), (
            "Pre-allocated K decompress differs from dynamic"
        )
        assert torch.equal(v_dynamic, v_prealloc), (
            "Pre-allocated V decompress differs from dynamic"
        )

    def test_preallocated_decompress_bfloat16(self, tq4_quantizer) -> None:
        """Decompress buffers use compute_dtype, not hardcoded float16 (F2)."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=4)

        key = torch.randn(3, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(3, self.NUM_KV_HEADS, self.HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0, 5, 33]))

        # Dynamic path with bfloat16
        k_dynamic, v_dynamic = impl._decompress_cache(
            kv_cache, torch.bfloat16, apply_rotation=False
        )
        assert k_dynamic.dtype == torch.bfloat16

        # Pre-allocated path with bfloat16 compute_dtype
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.bfloat16)
        assert impl._cg_decompress_k.dtype == torch.bfloat16

        k_prealloc, v_prealloc = impl._decompress_cache(
            kv_cache,
            torch.bfloat16,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )
        assert k_prealloc.dtype == torch.bfloat16
        assert torch.equal(k_dynamic, k_prealloc), (
            "BF16 pre-allocated K differs from dynamic"
        )
        assert torch.equal(v_dynamic, v_prealloc), (
            "BF16 pre-allocated V differs from dynamic"
        )

    def test_decode_path_uses_preallocated_compress(self, tq4_quantizer) -> None:
        """Decode compress uses pre-allocated buffers (AC 3)."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=2)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        key = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        slot_mapping = torch.tensor([0])

        # Compress with pre-allocated buffers
        impl._compress_and_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            compress_out=(impl._cg_compress_packed, impl._cg_compress_norms),
            row_out=impl._cg_compress_row,
        )

        # Verify data was written to cache
        flat = kv_cache.view(-1, impl._total_bytes)
        assert flat[0].any(), "Slot 0 should have data after pre-allocated compress"

        # Decompress and verify round-trip
        k_cache, v_cache = impl._decompress_cache(kv_cache, torch.float32)
        recon_k = k_cache.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)[0]
        for h in range(self.NUM_KV_HEADS):
            cos_k = torch.nn.functional.cosine_similarity(
                key[0, h].unsqueeze(0), recon_k[h].unsqueeze(0)
            ).item()
            assert cos_k > 0.85, f"Pre-alloc compress K head {h} cosine {cos_k:.4f}"

    def test_prefill_path_unchanged(self, tq4_quantizer) -> None:
        """Prefill (multi-token) path still uses dynamic allocation (AC 5)."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=4)

        N = 5
        key = torch.randn(N, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(N, self.NUM_KV_HEADS, self.HEAD_SIZE)
        slot_mapping = torch.tensor([0, 1, 2, 3, 4])

        # Without pre-allocated buffers (prefill path)
        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        k_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        flat_k = k_cache.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)

        for i in range(N):
            for h in range(self.NUM_KV_HEADS):
                cos = torch.nn.functional.cosine_similarity(
                    key[i, h].unsqueeze(0), flat_k[i, h].unsqueeze(0)
                ).item()
                assert cos > 0.85, f"Prefill token {i} head {h} cosine {cos:.4f}"

    def test_buffer_reuse_consecutive_decode_steps(self, tq4_quantizer) -> None:
        """Same buffers reused on consecutive decode steps (AC 3)."""
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=2)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        compress_out = (impl._cg_compress_packed, impl._cg_compress_norms)
        row_out = impl._cg_compress_row

        # Step 1
        key1 = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        val1 = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        impl._compress_and_store(
            key1,
            val1,
            kv_cache,
            torch.tensor([0]),
            compress_out=compress_out,
            row_out=row_out,
        )

        # Step 2 (same buffers, different slot)
        key2 = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        val2 = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        impl._compress_and_store(
            key2,
            val2,
            kv_cache,
            torch.tensor([1]),
            compress_out=compress_out,
            row_out=row_out,
        )

        # Both slots should have data
        flat = kv_cache.view(-1, impl._total_bytes)
        assert flat[0].any(), "Slot 0 should have data"
        assert flat[1].any(), "Slot 1 should have data"

        # Verify each slot decompresses correctly
        k_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        flat_k = k_cache.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        cos_1 = torch.nn.functional.cosine_similarity(
            key1[0, 0].unsqueeze(0), flat_k[0, 0].unsqueeze(0)
        ).item()
        cos_2 = torch.nn.functional.cosine_similarity(
            key2[0, 0].unsqueeze(0), flat_k[1, 0].unsqueeze(0)
        ).item()
        assert cos_1 > 0.85, f"Step 1 cosine {cos_1:.4f}"
        assert cos_2 > 0.85, f"Step 2 cosine {cos_2:.4f}"

    def test_stale_data_immunity(self, tq4_quantizer) -> None:
        """Stale data in decompress buffers doesn't affect output (AC 4).

        Fill decompress buffers with garbage before calling decompress+
        flash_attn. Validates that tq4_decompress overwrites all positions
        and that stale data in unused positions doesn't leak.
        """
        impl = self._make_impl(tq4_quantizer)
        kv_cache = self._make_cache(num_blocks=4)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Write some data
        key = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([5]))

        # Fill decompress buffers with garbage
        impl._cg_decompress_k.fill_(99.0)
        impl._cg_decompress_v.fill_(99.0)

        # Decompress into garbage-filled buffers
        k_stale, _ = impl._decompress_cache(
            kv_cache,
            torch.float16,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )

        # Clean run (fresh buffers)
        k_clean, _ = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )

        # The written slot (5) must be identical regardless of stale data
        stale_k5 = k_stale.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)[5]
        clean_k5 = k_clean.view(-1, self.NUM_KV_HEADS, self.HEAD_SIZE)[5]
        assert torch.equal(stale_k5, clean_k5), (
            "Stale data in decompress buffer affected written slot output"
        )
