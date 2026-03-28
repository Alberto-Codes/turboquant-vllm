"""Tests for Phase 3: Fused TQ4 K+V Flash Attention kernel.

Validates that fused TQ4 decompression of both K and V tiles inside
the FA inner loop (plus post-rotation) matches the unfused path.

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.triton.flash_attention_tq4_kv import (
    triton_flash_attention_tq4_kv,
)

from .conftest import (
    assert_tq4_fused_matches_unfused,
    compress_tq4,
    decompress_tq4,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device():
    """CUDA-only device fixture (overrides conftest parametrized fixture)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton Flash Attention")
    return "cuda"


# ---------------------------------------------------------------------------
# Shape matrix: (B, H_Q, H_KV, S_Q, S_KV, D, is_causal, dtype)
# ---------------------------------------------------------------------------

TQ4_KV_SHAPES = [
    pytest.param(1, 4, 4, 32, 32, 128, False, torch.float16, id="mha_basic"),
    pytest.param(1, 32, 8, 64, 64, 128, False, torch.float16, id="gqa_4to1"),
    pytest.param(1, 28, 4, 64, 64, 128, False, torch.float16, id="gqa_7to1"),
    pytest.param(1, 32, 8, 1, 512, 128, False, torch.float16, id="decode"),
    pytest.param(1, 4, 4, 64, 64, 128, True, torch.float16, id="causal"),
    pytest.param(1, 32, 8, 512, 512, 128, False, torch.float16, id="long_seq"),
    pytest.param(4, 32, 8, 64, 64, 128, False, torch.float16, id="batched"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestTQ4KVFlashAttention:
    """Phase 3: fused K+V TQ4 FA vs unfused (decompress both, vanilla FA)."""

    @pytest.mark.parametrize(
        ("B", "H_Q", "H_KV", "S_Q", "S_KV", "D", "is_causal", "dtype"),
        TQ4_KV_SHAPES,
    )
    def test_fused_kv_matches_unfused(
        self,
        device: str,
        tq4_quantizer,
        B: int,
        H_Q: int,
        H_KV: int,
        S_Q: int,
        S_KV: int,
        D: int,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> None:
        """Fused K+V TQ4 FA matches unfused path for given shape configuration."""
        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=dtype)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=dtype)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=dtype)

        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        def _fused(q, k_packed, k_norms, centroids, rotation, is_causal):
            return triton_flash_attention_tq4_kv(
                q,
                k_packed,
                k_norms,
                v_packed,
                v_norms,
                centroids,
                rotation,
                is_causal=is_causal,
            )

        assert_tq4_fused_matches_unfused(
            q=q,
            k=k,
            v_ref=v_dec,
            tq4_quantizer=tq4_quantizer,
            device=device,
            fused_fn=_fused,
            is_causal=is_causal,
        )
