"""Tests for _decompress_cache_paged method in TQ4 backend.

Extracted from ``test_vllm_cache_cudagraph.py`` (Test Maturity) to keep
both files under the 500-line module gate.
"""

from __future__ import annotations

import logging

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402

from tests.helpers.vllm_impl import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_SIZE,
    NUM_KV_HEADS,
    make_cache,
    make_impl,
)

pytestmark = [pytest.mark.unit]


class TestPagedDecompress:
    """Tests for _decompress_cache_paged method (AC 2-7)."""

    def test_paged_decompress_common_case(self, tq4_quantizer) -> None:
        """Paged decompress of 5 unique blocks from a 100-block cache (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=100)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Write data to specific blocks (blocks 10, 20, 30, 40, 50)
        target_blocks = [10, 20, 30, 40, 50]
        for blk in target_blocks:
            for pos in range(BLOCK_SIZE):
                slot = blk * BLOCK_SIZE + pos
                key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
                value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
                impl._compress_and_store(key, value, kv_cache, torch.tensor([slot]))

        # Build a block_table referencing only those 5 blocks
        block_table = torch.tensor([target_blocks], dtype=torch.int32)
        seq_lens = torch.tensor([5 * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # Should contain exactly 5 decompressed blocks
        assert k_paged.shape == (5, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        assert v_paged.shape == (5, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

        # Verify cosine parity with full decompress reference
        k_full, v_full = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )
        for i, blk in enumerate(target_blocks):
            assert torch.equal(k_paged[i], k_full[blk]), (
                f"Block {blk}: paged K differs from full decompress"
            )
            assert torch.equal(v_paged[i], v_full[blk]), (
                f"Block {blk}: paged V differs from full decompress"
            )

    def test_remapped_block_table_correctness(self, tq4_quantizer) -> None:
        """Remapped block table maps logical blocks to compact indices (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=20)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Sequence using blocks 5, 10, 15 (non-contiguous)
        block_table = torch.tensor([[5, 10, 15]], dtype=torch.int32)
        seq_lens = torch.tensor([3 * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # unique sorted = [5, 10, 15] -> compact [0, 1, 2]
        expected_remap = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        assert torch.equal(remap_bt, expected_remap), (
            f"Expected remapped block table {expected_remap}, got {remap_bt}"
        )

        # Verify Flash Attention would read same data:
        # k_paged[remap_bt[0, j]] should equal full_decompress[block_table[0, j]]
        k_full, _ = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )
        for j in range(3):
            compact_idx = int(remap_bt[0, j])
            original_idx = int(block_table[0, j])
            assert torch.equal(k_paged[compact_idx], k_full[original_idx])

    def test_paged_decompress_fallback(self, tq4_quantizer) -> None:
        """Dynamic fallback when unique blocks exceed buffer capacity (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=200)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Force fallback: reference more blocks than max_prefill_blocks
        num_blocks_needed = impl._max_prefill_blocks + 5
        block_indices = list(range(num_blocks_needed))
        block_table = torch.tensor([block_indices], dtype=torch.int32)
        seq_lens = torch.tensor([num_blocks_needed * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # Should still return correctly shaped output (dynamically allocated)
        assert k_paged.shape == (
            num_blocks_needed,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )
        assert v_paged.shape == k_paged.shape

    def test_decompress_cache_backward_compat(self, tq4_quantizer) -> None:
        """Existing _decompress_cache method unchanged (AC 5)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        key = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0, 5, 33]))

        # Dynamic allocation (no out=)
        k_dyn, v_dyn = impl._decompress_cache(
            kv_cache, torch.float32, apply_rotation=False
        )
        assert k_dyn.shape == (4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

        # With out= buffers
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float32)
        k_pre, v_pre = impl._decompress_cache(
            kv_cache,
            torch.float32,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )
        assert torch.equal(k_dyn, k_pre)
        assert torch.equal(v_dyn, v_pre)

    def test_multi_sequence_batch_dedup(self, tq4_quantizer) -> None:
        """Multi-sequence batch with overlapping physical blocks (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=20)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Two sequences sharing block 5, padding with zeros in unused slots
        # Seq 0: blocks [3, 5], seq_len=2*BS
        # Seq 1: blocks [5, 7], seq_len=2*BS
        block_table = torch.tensor([[3, 5, 0], [5, 7, 0]], dtype=torch.int32)
        seq_lens = torch.tensor([2 * BLOCK_SIZE, 2 * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # Unique blocks: {3, 5, 7} -> 3 compact blocks
        assert k_paged.shape[0] == 3

        # Block 5 appears in both sequences but decompressed only once
        # remap_bt should map: 3->0, 5->1, 7->2
        assert remap_bt[0, 0] == 0  # block 3 -> compact 0
        assert remap_bt[0, 1] == 1  # block 5 -> compact 1
        assert remap_bt[1, 0] == 1  # block 5 -> compact 1 (same)
        assert remap_bt[1, 1] == 2  # block 7 -> compact 2

        # Padding columns (beyond blocks_needed) must be zero (safe sentinel)
        assert remap_bt[0, 2] == 0
        assert remap_bt[1, 2] == 0

    def test_paged_decompress_fallback_logs_warning(
        self, tq4_quantizer, caplog
    ) -> None:
        """Warning logged when fallback to dynamic allocation (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=200)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        num_blocks_needed = impl._max_prefill_blocks + 1
        block_table = torch.tensor([list(range(num_blocks_needed))], dtype=torch.int32)
        seq_lens = torch.tensor([num_blocks_needed * BLOCK_SIZE], dtype=torch.int32)

        with caplog.at_level(
            logging.WARNING, logger="turboquant_vllm.vllm.tq4_backend"
        ):
            impl._decompress_cache_paged(
                kv_cache,
                block_table,
                seq_lens,
                torch.float16,
                out_k=impl._cg_prefill_k,
                out_v=impl._cg_prefill_v,
            )

        assert any("dynamic fallback" in msg for msg in caplog.messages)


class TestDecodePagedDecompress:
    """Tests for decode path using _decompress_cache_paged (AC 2, 3, 5)."""

    def test_tq4_decode_returns_4_tuple(self, tq4_quantizer) -> None:
        """_tq4_decode returns (q_rot, key_cache, value_cache, remapped_bt) (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=10)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Write a token to block 3
        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([3 * BLOCK_SIZE]))

        class FakeMetadata:
            num_actual_tokens = 1
            slot_mapping = torch.tensor([3 * BLOCK_SIZE + 1])
            block_table = torch.tensor([[3]], dtype=torch.int32)
            seq_lens = torch.tensor([BLOCK_SIZE], dtype=torch.int32)

        query = torch.randn(1, impl.num_heads, HEAD_SIZE)
        result = impl._tq4_decode(query, key, value, kv_cache, FakeMetadata())

        assert len(result) == 4
        q_rot, k_cache, v_cache, remap_bt = result
        assert q_rot.shape[0] == 1
        assert remap_bt.dtype == torch.int32

    def test_decode_paged_cosine_parity(self, tq4_quantizer) -> None:
        """Decode paged decompress matches _decompress_cache reference (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=10)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Write data to blocks 2 and 5
        for blk in [2, 5]:
            for pos in range(BLOCK_SIZE):
                slot = blk * BLOCK_SIZE + pos
                key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
                value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
                impl._compress_and_store(key, value, kv_cache, torch.tensor([slot]))

        # Reference: full decompress
        k_full, v_full = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )

        # Paged decompress via decode buffers
        block_table = torch.tensor([[2, 5]], dtype=torch.int32)
        seq_lens = torch.tensor([2 * BLOCK_SIZE], dtype=torch.int32)
        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )

        # Compare block-by-block
        for j, blk in enumerate([2, 5]):
            compact_idx = int(remap_bt[0, j])
            paged_flat = k_paged[compact_idx].reshape(-1)
            full_flat = k_full[blk].reshape(-1)
            cos = torch.nn.functional.cosine_similarity(
                paged_flat.unsqueeze(0).float(),
                full_flat.unsqueeze(0).float(),
            ).item()
            assert cos > 0.999, f"Block {blk}: decode paged K cosine {cos:.6f} < 0.999"

    def test_capacity_check_uses_buffer_shape(self, tq4_quantizer) -> None:
        """_decompress_cache_paged capacity threshold from buffer shape (AC 3)."""
        impl = make_impl(tq4_quantizer)
        # 500 blocks * 16 = 8000 > max_model_len=6144, so decode buffers
        # are bounded to 6144 tokens while prefill stays at 2048 tokens.
        kv_cache = make_cache(num_blocks=500)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Prefill buffers: 2048 tokens / 16 = 128 blocks capacity
        prefill_capacity = impl._cg_prefill_k.shape[0] // BLOCK_SIZE
        assert prefill_capacity == impl._max_prefill_blocks

        # Decode buffers: 6144 tokens / 16 = 384 blocks capacity
        decode_capacity = impl._cg_decompress_k.shape[0] // BLOCK_SIZE
        assert decode_capacity == impl._max_model_len // BLOCK_SIZE

        # Different buffer sizes -> different capacity thresholds
        assert decode_capacity > prefill_capacity

    def test_prefill_paged_still_works(self, tq4_quantizer) -> None:
        """Prefill paged decompress unchanged after buffer-agnostic change (AC 5)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=20)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        block_table = torch.tensor([[3, 7]], dtype=torch.int32)
        seq_lens = torch.tensor([2 * BLOCK_SIZE], dtype=torch.int32)

        # Using prefill buffers (original path)
        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        assert k_paged.shape == (2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        expected_remap = torch.tensor([[0, 1]], dtype=torch.int32)
        assert torch.equal(remap_bt, expected_remap)
