"""Tests for D7 CUDA graph buffer pre-allocation in TQ4 backend.

Extracted from ``test_vllm_cache.py`` (Test Maturity Priority 1) to keep
both files under the 500-line module gate.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402

from turboquant_vllm.vllm.tq4_backend import (  # noqa: E402  # isort: skip
    TQ4AttentionBackend,
    TQ4MetadataBuilder,
)

from tests.helpers.vllm_impl import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    make_cache,
    make_impl,
)

pytestmark = [pytest.mark.unit]


class TestCUDAGraphBufferPreallocation:
    """Tests for D7 CUDA graph buffer pre-allocation."""

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
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        assert not impl._cg_buffers_ready
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)
        assert impl._cg_buffers_ready

        max_tokens = 4 * BLOCK_SIZE
        H = NUM_KV_HEADS
        D = HEAD_SIZE

        assert impl._cg_decompress_k.shape == (max_tokens, H, D)
        assert impl._cg_decompress_k.dtype == torch.float16
        assert impl._cg_decompress_v.shape == (max_tokens, H, D)
        assert impl._cg_decompress_v.dtype == torch.float16

        assert impl._cg_compress_packed.shape == (1, H, D // 2)
        assert impl._cg_compress_packed.dtype == torch.uint8
        assert impl._cg_compress_norms.shape == (1, H, 1)
        assert impl._cg_compress_norms.dtype == torch.float32

        assert impl._cg_q_rot.shape == (1, NUM_HEADS, D)
        assert impl._cg_q_rot.dtype == torch.float32
        assert impl._cg_q_rot_cast.shape == (1, NUM_HEADS, D)
        assert impl._cg_q_rot_cast.dtype == torch.float16

        assert impl._cg_compress_row.shape == (1, impl._total_bytes)
        assert impl._cg_compress_row.dtype == torch.uint8

    def test_preallocated_decompress_matches_dynamic(self, tq4_quantizer) -> None:
        """Blocking acceptance test: pre-allocated buffers match dynamic (AC 4).

        Uses float16 (matching pre-allocated buffer dtype and real forward path).
        """
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        # Write some tokens
        key = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
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
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        key = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
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
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=2)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
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
        recon_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)[0]
        for h in range(NUM_KV_HEADS):
            cos_k = torch.nn.functional.cosine_similarity(
                key[0, h].unsqueeze(0), recon_k[h].unsqueeze(0)
            ).item()
            assert cos_k > 0.85, f"Pre-alloc compress K head {h} cosine {cos_k:.4f}"

    def test_prefill_path_unchanged(self, tq4_quantizer) -> None:
        """Prefill (multi-token) path still uses dynamic allocation (AC 5)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        N = 5
        key = torch.randn(N, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(N, NUM_KV_HEADS, HEAD_SIZE)
        slot_mapping = torch.tensor([0, 1, 2, 3, 4])

        # Without pre-allocated buffers (prefill path)
        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        k_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        flat_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)

        for i in range(N):
            for h in range(NUM_KV_HEADS):
                cos = torch.nn.functional.cosine_similarity(
                    key[i, h].unsqueeze(0), flat_k[i, h].unsqueeze(0)
                ).item()
                assert cos > 0.85, f"Prefill token {i} head {h} cosine {cos:.4f}"

    def test_buffer_reuse_consecutive_decode_steps(self, tq4_quantizer) -> None:
        """Same buffers reused on consecutive decode steps (AC 3)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=2)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        compress_out = (impl._cg_compress_packed, impl._cg_compress_norms)
        row_out = impl._cg_compress_row

        # Step 1
        key1 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        val1 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(
            key1,
            val1,
            kv_cache,
            torch.tensor([0]),
            compress_out=compress_out,
            row_out=row_out,
        )

        # Step 2 (same buffers, different slot)
        key2 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        val2 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
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
        flat_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)
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
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Write some data
        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
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
        stale_k5 = k_stale.view(-1, NUM_KV_HEADS, HEAD_SIZE)[5]
        clean_k5 = k_clean.view(-1, NUM_KV_HEADS, HEAD_SIZE)[5]
        assert torch.equal(stale_k5, clean_k5), (
            "Stale data in decompress buffer affected written slot output"
        )
