"""Tests for Phase 2: Fused TQ4 Flash Attention kernel.

Validates that fused TQ4 decompression inside the FA inner loop produces
output matching the unfused path (decompress K first, then vanilla FA).

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.triton.flash_attention_tq4 import triton_flash_attention_tq4

from .conftest import assert_tq4_fused_matches_unfused

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> str:
    """CUDA-only device fixture (overrides conftest parametrized fixture)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton Flash Attention")
    return "cuda"


# ---------------------------------------------------------------------------
# Shape matrix: (B, H_Q, H_KV, S_Q, S_KV, D, is_causal, dtype)
# ---------------------------------------------------------------------------

TQ4_SHAPES = [
    pytest.param(1, 4, 4, 32, 32, 128, False, torch.float16, id="mha_basic"),
    pytest.param(1, 32, 8, 64, 64, 128, False, torch.float16, id="gqa_4to1"),
    pytest.param(1, 28, 4, 64, 64, 128, False, torch.float16, id="gqa_7to1"),
    pytest.param(1, 32, 8, 1, 512, 128, False, torch.float16, id="decode"),
    pytest.param(1, 4, 4, 64, 64, 128, True, torch.float16, id="causal"),
    pytest.param(1, 4, 4, 32, 32, 128, False, torch.bfloat16, id="bf16"),
    pytest.param(1, 32, 8, 512, 512, 128, False, torch.float16, id="long_seq"),
    pytest.param(1, 4, 4, 37, 37, 128, False, torch.float16, id="prime_seq"),
    pytest.param(4, 32, 8, 64, 64, 128, False, torch.float16, id="batched"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestTQ4FlashAttention:
    """Phase 2 validation: fused TQ4 FA vs unfused (decompress + vanilla FA)."""

    @pytest.mark.parametrize(
        ("B", "H_Q", "H_KV", "S_Q", "S_KV", "D", "is_causal", "dtype"),
        TQ4_SHAPES,
    )
    def test_fused_matches_unfused(
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
        """Fused TQ4 FA matches unfused path for given shape configuration."""
        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=dtype)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=dtype)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=dtype)

        def _fused(q, k_packed, k_norms, centroids, rotation, is_causal):
            return triton_flash_attention_tq4(
                q,
                k_packed,
                k_norms,
                centroids,
                rotation,
                v,
                is_causal=is_causal,
            )

        assert_tq4_fused_matches_unfused(
            q=q,
            k=k,
            v_ref=v,
            tq4_quantizer=tq4_quantizer,
            device=device,
            fused_fn=_fused,
            is_causal=is_causal,
        )
