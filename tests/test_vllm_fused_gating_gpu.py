"""GPU integration tests for TQ4 fused paged decode kernel.

Extracted from test_vllm_fused_gating.py (Story 5.5 test maturity):
TestFusedPagedGPUIntegration validates fused vs decompress-all path
output parity on real CUDA hardware.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402

from turboquant_vllm.vllm.tq4_backend import (  # noqa: E402  # isort: skip
    TQ4AttentionImpl,
    TQ4_NORM_BYTES,
    _tq4_bytes_per_token_kv,
)

pytestmark = [pytest.mark.gpu]

# ---------------------------------------------------------------------------
# Constants (Molmo2-8B config)
# ---------------------------------------------------------------------------

NUM_KV_HEADS = 4
NUM_HEADS = 28
HEAD_SIZE = 128
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_impl(quantizer, *, fused_paged_available=False, max_prefill_len=2048):
    """Create a TQ4AttentionImpl without full vLLM init.

    Args:
        quantizer: TurboQuantMSE instance.
        fused_paged_available: Override for ``_fused_paged_available``.
        max_prefill_len: Override for ``_max_prefill_len``.
    """
    impl = object.__new__(TQ4AttentionImpl)
    impl.head_size = HEAD_SIZE
    impl.num_kv_heads = NUM_KV_HEADS
    impl.num_heads = NUM_HEADS
    impl.scale = 1.0 / (HEAD_SIZE**0.5)

    impl._tq4_rotation = quantizer.rotation.clone()
    impl._tq4_centroids = quantizer.codebook.centroids.clone()
    impl._tq4_boundaries = quantizer.codebook.boundaries.clone()
    rot_t = quantizer.rotation.T.contiguous()
    impl._tq4_rot_T_even = rot_t[:, 0::2].contiguous()
    impl._tq4_rot_T_odd = rot_t[:, 1::2].contiguous()
    impl._cg_buffers_ready = False

    half_D = HEAD_SIZE // 2
    impl._half_D = half_D
    impl._k_idx_end = NUM_KV_HEADS * half_D
    impl._k_norm_end = impl._k_idx_end + NUM_KV_HEADS * TQ4_NORM_BYTES
    impl._v_idx_end = impl._k_norm_end + NUM_KV_HEADS * half_D
    impl._total_bytes = impl._v_idx_end + NUM_KV_HEADS * TQ4_NORM_BYTES

    impl._fused_paged_available = fused_paged_available
    impl._int8_prefill_available = False
    impl._max_prefill_len = max_prefill_len
    impl._max_model_len = 6144

    return impl


def _make_cache(num_blocks):
    """Create a zeroed packed TQ4 cache."""
    total_bytes = NUM_KV_HEADS * _tq4_bytes_per_token_kv(HEAD_SIZE)
    return torch.zeros(num_blocks, BLOCK_SIZE, total_bytes, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# GPU integration tests
# ---------------------------------------------------------------------------


class TestFusedPagedGPUIntegration:
    """GPU integration: fused vs decompress-all path output parity.

    Creates a real paged cache via _compress_and_store, runs forward() with
    both paths, and asserts cosine >0.999 between outputs.  Uses Molmo2 config
    (28Q/4KV, D=128).
    """

    def test_fused_vs_decompress_all_cosine(self, tq4_quantizer) -> None:
        """Fused decode output matches decompress-all decode within >0.999 cosine."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        from turboquant_vllm.triton.fused_paged_tq4_attention import (
            fused_paged_tq4_decode,
        )

        num_blocks = 8
        seq_len = 32  # tokens already in cache
        num_seqs = 1

        # Build impl on GPU
        impl = _make_impl(tq4_quantizer, fused_paged_available=False)
        impl._tq4_rotation = impl._tq4_rotation.to(device)
        impl._tq4_centroids = impl._tq4_centroids.to(device)
        impl._tq4_boundaries = impl._tq4_boundaries.to(device)
        impl._tq4_rot_T_even = impl._tq4_rot_T_even.to(device)
        impl._tq4_rot_T_odd = impl._tq4_rot_T_odd.to(device)

        # Create and populate cache
        kv_cache = _make_cache(num_blocks).to(device)
        for t in range(seq_len):
            k = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE, device=device)
            v = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE, device=device)
            slot = torch.tensor([t], device=device)
            impl._compress_and_store(k, v, kv_cache, slot)

        # Query for decode (one token)
        q = torch.randn(
            num_seqs, NUM_HEADS, HEAD_SIZE, device=device, dtype=torch.float16
        )

        # ---- Path A: decompress-all ----
        key_cache, value_cache = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )
        # Q rotation
        q_rot = (q.float() @ impl._tq4_rotation.T).to(torch.float16)
        # Flash attention (simplified: manual dot-product for single-seq decode)
        # Use the fused wrapper's approach: post-rotate after attention
        # For a fair comparison, use the fused kernel for both but with different entry points.
        # Actually, we compare the fused_paged_tq4_decode output directly.

        # ---- Path B: fused kernel ----
        block_table = torch.arange(
            num_blocks, device=device, dtype=torch.int32
        ).unsqueeze(0)
        seq_lens_t = torch.tensor([seq_len], device=device, dtype=torch.int32)
        block_size = BLOCK_SIZE

        out_fused = fused_paged_tq4_decode(
            q,
            kv_cache,
            block_table,
            seq_lens_t,
            impl._tq4_centroids,
            impl._tq4_rotation,
            NUM_KV_HEADS,
            HEAD_SIZE,
            block_size,
            impl.scale,
        )

        # ---- Path A continued: manual attention for reference ----
        # Decompress cache and compute attention manually
        from torch.nn.functional import cosine_similarity

        # Reshape for attention: key_cache is (NB, BS, H, D), flatten to (1, seq_len, H_KV, D)
        k_flat = key_cache.reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[
            :seq_len
        ]  # (seq_len, H_KV, D)
        v_flat = value_cache.reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len]

        # GQA expansion: expand KV heads to match Q heads
        # Each KV head serves gqa_ratio Q heads contiguously
        gqa_ratio = NUM_HEADS // NUM_KV_HEADS
        k_exp = (
            k_flat.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1)
            .reshape(seq_len, NUM_HEADS, HEAD_SIZE)
        )
        v_exp = (
            v_flat.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1)
            .reshape(seq_len, NUM_HEADS, HEAD_SIZE)
        )

        # Attention: (1, H_Q, D) @ (seq_len, H_Q, D).T -> (1, H_Q, seq_len)
        scale = impl.scale
        scores = torch.einsum("qhd,shd->qhs", q_rot.float(), k_exp.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        # Output in rotated space: (1, H_Q, D)
        out_rotated = torch.einsum("qhs,shd->qhd", weights, v_exp.float())
        # Post-rotate
        out_ref = (out_rotated @ impl._tq4_rotation.float()).to(torch.float16)

        cos = cosine_similarity(
            out_fused.flatten().float(),
            out_ref.flatten().float(),
            dim=0,
        ).item()
        assert cos > 0.999, f"Fused vs decompress-all cosine {cos:.6f} < 0.999"
