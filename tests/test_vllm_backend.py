"""Tests for TQ4 vLLM attention backend.

Phase 3a tests: plugin wiring, registration, class hierarchy, interface compliance.
Phase 3b tests: TQ4 compress/decompress round-trip.
Phase 3c tests: packed cache shape, TQ4FullAttentionSpec, compress_and_store,
    decompress_cache round-trip.
Requires vLLM to be installed.
"""

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: E402

from turboquant_consumer.vllm.tq4_backend import (  # noqa: E402
    TQ4AttentionBackend,
    TQ4AttentionImpl,
    TQ4FullAttentionSpec,
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


# ---------------------------------------------------------------------------
# Phase 3c: TQ4FullAttentionSpec
# ---------------------------------------------------------------------------


class TestTQ4FullAttentionSpec:
    """TQ4 cache spec page size override."""

    def test_subclasses_full_attention_spec(self):
        assert issubclass(TQ4FullAttentionSpec, FullAttentionSpec)

    def test_page_size_bytes_molmo2_8b(self):
        """Molmo2-8B: 8 KV heads, head_dim=128, block_size=16."""
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        # 16 tokens * 8 heads * 136 bytes/token/head = 17,408
        assert spec.page_size_bytes == 17_408

    def test_page_size_vs_fp16(self):
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

    def test_dtype_is_uint8(self):
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        assert spec.dtype == torch.uint8

    def test_block_size_scales_page_size(self):
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

    def test_frozen_dataclass(self):
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
    HEAD_SIZE = 128
    BLOCK_SIZE = 16

    def _make_impl(self):
        """Create a TQ4AttentionImpl without full vLLM init."""
        # Bypass FlashAttentionImpl.__init__ -- only init TQ4 primitives
        impl = object.__new__(TQ4AttentionImpl)
        impl.head_size = self.HEAD_SIZE
        impl.num_kv_heads = self.NUM_KV_HEADS

        from turboquant_consumer.quantizer import TurboQuantMSE
        from turboquant_consumer.vllm.tq4_backend import (
            TQ4_BITS,
            TQ4_NORM_BYTES,
            TQ4_SEED,
        )

        quantizer = TurboQuantMSE(self.HEAD_SIZE, TQ4_BITS, seed=TQ4_SEED)
        impl._tq4_rotation = quantizer.rotation
        impl._tq4_centroids = quantizer.codebook.centroids
        impl._tq4_boundaries = quantizer.codebook.boundaries
        rot_t = quantizer.rotation.T.contiguous()
        impl._tq4_rot_T_even = rot_t[:, 0::2].contiguous()
        impl._tq4_rot_T_odd = rot_t[:, 1::2].contiguous()
        impl._tq4_on_device = False

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

    def test_compress_store_decompress_single_token(self):
        impl = self._make_impl()
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

    def test_round_trip_cosine_similarity(self):
        """TQ4 round-trip should achieve >0.85 cosine on random data.

        Note: real model KV cache data achieves >0.97 cosine (validated
        in experiment 003/005).  Random Gaussian vectors give lower cosine
        because their distribution differs from actual KV activations.
        """
        impl = self._make_impl()
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

    def test_multi_token_scatter_write(self):
        """Multiple tokens scattered to different slots."""
        impl = self._make_impl()
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

    def test_empty_slots_decompress_to_zero(self):
        """Unwritten slots should decompress to zero."""
        impl = self._make_impl()
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

    def test_overwrite_slot(self):
        """Writing to the same slot twice should overwrite."""
        impl = self._make_impl()
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

    def test_cache_shape_matches_backend(self):
        """Cache shape from backend matches what decompress expects."""
        shape = TQ4AttentionBackend.get_kv_cache_shape(
            num_blocks=10,
            block_size=self.BLOCK_SIZE,
            num_kv_heads=self.NUM_KV_HEADS,
            head_size=self.HEAD_SIZE,
        )
        cache = torch.zeros(*shape, dtype=torch.uint8)
        impl = self._make_impl()

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

    def test_bfloat16_output_dtype(self):
        """Decompress can produce bf16 output."""
        impl = self._make_impl()
        kv_cache = self._make_cache(num_blocks=1)

        key = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        value = torch.randn(1, self.NUM_KV_HEADS, self.HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0]))

        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.bfloat16)
        assert key_cache.dtype == torch.bfloat16
        assert value_cache.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Phase 3c.10: Triton vs PyTorch bit-for-bit validation
# ---------------------------------------------------------------------------


class TestTritonPyTorchEquivalence:
    """Validate Triton kernels produce identical output to the old PyTorch path.

    Phase 3c.10 quality gate: the fused Triton compress/decompress kernels
    must match the original multi-op PyTorch implementation exactly.
    """

    HEAD_DIM = 128
    NUM_KV_HEADS = 8

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up quantizer primitives for both paths."""
        from turboquant_consumer.quantizer import TurboQuantMSE
        from turboquant_consumer.vllm.tq4_backend import TQ4_BITS, TQ4_SEED

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quantizer = TurboQuantMSE(self.HEAD_DIM, TQ4_BITS, seed=TQ4_SEED)
        self.rotation = quantizer.rotation.to(self.device)
        self.rotation_t = self.rotation.T.contiguous()
        self.boundaries = quantizer.codebook.boundaries.to(self.device)
        self.centroids = quantizer.codebook.centroids.to(self.device)
        self.rot_T_even = self.rotation_t[:, 0::2].contiguous()
        self.rot_T_odd = self.rotation_t[:, 1::2].contiguous()

    # --- helpers: old PyTorch path ---

    def _pytorch_compress(self, x):
        """Original PyTorch compress (pre-3c.9)."""
        N, H, D = x.shape
        flat = x.reshape(N * H, D).float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ self.rotation_t
        indices = torch.bucketize(rotated, self.boundaries)
        indices = indices.clamp(0, 15)
        idx_u8 = indices.to(torch.uint8)
        packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]
        return packed.reshape(N, H, D // 2), norms.reshape(N, H, 1)

    def _pytorch_decompress(self, packed, norms, dtype):
        """Original PyTorch decompress (pre-3c.8), with rotation."""
        N, H, half_D = packed.shape
        D = half_D * 2
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
        flat_norms = norms.reshape(N * H, 1)
        reconstructed = self.centroids[indices]
        unrotated = reconstructed @ self.rotation
        result = unrotated * flat_norms
        return result.reshape(N, H, D).to(dtype)

    # --- compress tests ---

    def test_compress_packed_match(self):
        """Triton compress produces identical packed bytes as PyTorch."""
        from turboquant_consumer.triton.tq4_compress import tq4_compress

        x = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        assert torch.equal(pt_packed, tr_packed), "Packed bytes differ"

    def test_compress_norms_match(self):
        """Triton compress produces identical norms as PyTorch."""
        from turboquant_consumer.triton.tq4_compress import tq4_compress

        x = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        torch.testing.assert_close(pt_norms, tr_norms, atol=1e-5, rtol=1e-5)

    def test_compress_single_token(self):
        """Triton compress matches PyTorch for a single token (decode case)."""
        from turboquant_consumer.triton.tq4_compress import tq4_compress

        x = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        assert torch.equal(pt_packed, tr_packed), "Single-token packed bytes differ"
        torch.testing.assert_close(pt_norms, tr_norms, atol=1e-5, rtol=1e-5)

    # --- decompress tests ---

    def test_decompress_with_rotation_matches_pytorch(self):
        """Triton decompress + rotation matches old PyTorch decompress."""
        from turboquant_consumer.triton.tq4_decompress import tq4_decompress

        packed = torch.randint(
            0,
            255,
            (16, self.NUM_KV_HEADS, self.HEAD_DIM // 2),
            device=self.device,
            dtype=torch.uint8,
        )
        norms = (
            torch.randn(
                16,
                self.NUM_KV_HEADS,
                1,
                device=self.device,
                dtype=torch.float32,
            )
            .abs_()
            .clamp_(min=0.1)
        )

        # Old path: decompress with rotation
        pt_out = self._pytorch_decompress(packed, norms, torch.float32)

        # New path: Triton decompress (no rotation) + apply rotation
        tr_out_rot = tq4_decompress(packed, norms, self.centroids, torch.float32)
        tr_out = (tr_out_rot.float() @ self.rotation).to(torch.float32)

        torch.testing.assert_close(pt_out, tr_out, atol=1e-4, rtol=1e-4)

    def test_decompress_no_rotation_stays_in_rotated_space(self):
        """Triton decompress without rotation is NOT close to original data."""
        from turboquant_consumer.triton.tq4_decompress import tq4_decompress

        x = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)

        # Decompress without rotation: should be in rotated space
        tr_out_rot = tq4_decompress(pt_packed, pt_norms, self.centroids, torch.float32)
        # This should NOT match the original (it's rotated)
        cos = torch.nn.functional.cosine_similarity(
            x[0, 0].float().unsqueeze(0).to(self.device),
            tr_out_rot[0, 0].unsqueeze(0),
        ).item()
        # Rotated output should have low cosine with original
        assert cos < 0.5, f"Expected low cosine (rotated space), got {cos:.4f}"

    # --- end-to-end: pre/post-rotation equivalence ---

    def test_pre_post_rotation_attention_equivalence(self):
        """Pre-rotate Q + Triton decompress + post-rotate == old path.

        This is the core 3c.10 validation: the full attention output must
        be equivalent whether we rotate inside decompress (old) or rotate
        Q/output outside (new).
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for Flash Attention")

        from vllm.vllm_flash_attn import flash_attn_varlen_func

        from turboquant_consumer.triton.tq4_decompress import tq4_decompress

        seq_len = 64
        packed = torch.randint(
            0,
            255,
            (seq_len, self.NUM_KV_HEADS, self.HEAD_DIM // 2),
            device=self.device,
            dtype=torch.uint8,
        )
        norms = (
            torch.randn(
                seq_len,
                self.NUM_KV_HEADS,
                1,
                device=self.device,
                dtype=torch.float32,
            )
            .abs_()
            .clamp_(min=0.1)
        )
        query = torch.randn(
            1,
            32,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )

        cu_q = torch.tensor([0, 1], device=self.device, dtype=torch.int32)
        cu_k = torch.tensor([0, seq_len], device=self.device, dtype=torch.int32)

        # Old path: decompress with rotation → standard FA
        k_old = self._pytorch_decompress(packed, norms, torch.float16)
        v_old = self._pytorch_decompress(packed, norms, torch.float16)
        out_old = flash_attn_varlen_func(
            q=query,
            k=k_old,
            v=v_old,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=seq_len,
            cu_seqlens_k=cu_k,
        )

        # New path: pre-rotate Q → Triton decompress (no rotation) → FA → post-rotate
        q_rot = (query.float() @ self.rotation_t).to(torch.float16)
        k_rot = tq4_decompress(packed, norms, self.centroids, torch.float16)
        v_rot = tq4_decompress(packed, norms, self.centroids, torch.float16)
        out_new_rot = flash_attn_varlen_func(
            q=q_rot,
            k=k_rot,
            v=v_rot,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=seq_len,
            cu_seqlens_k=cu_k,
        )
        out_new = (out_new_rot.float() @ self.rotation).to(torch.float16)

        # Should be very close (not exact due to fp16 accumulation order)
        torch.testing.assert_close(out_old, out_new, atol=5e-3, rtol=5e-3)

    # --- round-trip: compress then decompress ---

    def test_full_round_trip_triton_vs_pytorch(self):
        """Full round-trip: Triton compress then decompress matches PyTorch.

        Triton compress → Triton decompress + rotation should produce the
        same reconstructed vectors as PyTorch compress → PyTorch decompress.
        """
        from turboquant_consumer.triton.tq4_compress import tq4_compress
        from turboquant_consumer.triton.tq4_decompress import tq4_decompress

        x = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )

        # PyTorch round-trip
        pt_packed, pt_norms = self._pytorch_compress(x)
        pt_recon = self._pytorch_decompress(pt_packed, pt_norms, torch.float32)

        # Triton round-trip (compress → decompress + rotation)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        tr_recon_rot = tq4_decompress(
            tr_packed,
            tr_norms,
            self.centroids,
            torch.float32,
        )
        tr_recon = (tr_recon_rot.float() @ self.rotation).to(torch.float32)

        torch.testing.assert_close(pt_recon, tr_recon, atol=1e-4, rtol=1e-4)
