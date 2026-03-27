"""Tests for Phase 1: Triton Flash Attention vanilla kernel.

Validates correctness against ``F.scaled_dot_product_attention`` (SDPA)
across MHA, GQA, causal/non-causal, and decode configurations.

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant_consumer.triton.flash_attention import triton_flash_attention

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device():
    """CUDA-only device fixture (overrides conftest parametrized fixture)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton Flash Attention")
    return "cuda"


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SDPA reference with GQA expansion."""
    H_Q, H_KV = q.shape[1], k.shape[1]
    if H_Q != H_KV:
        repeats = H_Q // H_KV
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        is_causal=is_causal,
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat cosine similarity between two tensors."""
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTritonFlashAttention:
    """Phase 1 validation: Triton FA vs SDPA reference."""

    # -- Basic correctness (MHA, no GQA) --

    def test_mha_non_causal(self, device: str) -> None:
        """Standard multi-head attention, non-causal."""
        B, H, S, D = 1, 8, 64, 128
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999

    def test_mha_causal(self, device: str) -> None:
        """Standard multi-head attention with causal masking."""
        B, H, S, D = 1, 8, 64, 128
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v, is_causal=True)
        actual = triton_flash_attention(q, k, v, is_causal=True)

        assert _cosine_similarity(actual, expected) > 0.999

    # -- GQA (Grouped-Query Attention) --

    def test_gqa_7_to_1(self, device: str) -> None:
        """Molmo2's 28Q / 4KV (7:1) GQA ratio."""
        B, H_Q, H_KV, S, D = 1, 28, 4, 64, 128
        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999

    def test_gqa_4_to_1(self, device: str) -> None:
        """Common 4:1 GQA ratio."""
        B, H_Q, H_KV, S, D = 2, 32, 8, 48, 128
        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999

    def test_gqa_causal(self, device: str) -> None:
        """GQA 7:1 with causal masking (prefill mode)."""
        B, H_Q, H_KV, S, D = 1, 28, 4, 128, 128
        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v, is_causal=True)
        actual = triton_flash_attention(q, k, v, is_causal=True)

        assert _cosine_similarity(actual, expected) > 0.999

    # -- Decode mode (seq_q = 1) --

    def test_decode_mha(self, device: str) -> None:
        """Decode: single query attending to long KV cache (MHA)."""
        B, H, D = 1, 8, 128
        S_Q, S_KV = 1, 512
        q = torch.randn(B, H, S_Q, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S_KV, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S_KV, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999

    def test_decode_gqa(self, device: str) -> None:
        """Decode: single query with GQA 7:1 and long KV cache."""
        B, H_Q, H_KV, D = 1, 28, 4, 128
        S_Q, S_KV = 1, 1024
        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999

    def test_decode_is_causal_forced_off(self, device: str) -> None:
        """is_causal=True with seq_q=1 should be forced to False (no hang)."""
        B, H, D = 1, 8, 128
        S_Q, S_KV = 1, 256
        q = torch.randn(B, H, S_Q, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S_KV, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S_KV, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        # Pass is_causal=True -- wrapper should override to False
        actual = triton_flash_attention(q, k, v, is_causal=True)

        assert _cosine_similarity(actual, expected) > 0.999

    # -- Attention mask --

    def test_additive_mask(self, device: str) -> None:
        """Additive attention mask (HF-compatible format)."""
        B, H, S, D = 1, 8, 32, 128
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        # Create a causal-style mask via additive mask
        mask = torch.zeros(B, 1, S, S, device=device, dtype=torch.float16)
        mask.masked_fill_(
            torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )

        expected = _sdpa_reference(q, k, v, attention_mask=mask)
        actual = triton_flash_attention(q, k, v, attention_mask=mask)

        assert _cosine_similarity(actual, expected) > 0.999

    # -- Dtype support --

    def test_bfloat16(self, device: str) -> None:
        """bfloat16 inputs produce correct output."""
        B, H, S, D = 1, 8, 64, 128
        q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999

    # -- Sequence length edge cases --

    def test_seq_not_multiple_of_block(self, device: str) -> None:
        """Non-power-of-2 sequence length (tests masking logic)."""
        B, H, S, D = 1, 8, 37, 128  # 37 is prime
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999

    def test_long_sequence(self, device: str) -> None:
        """Longer sequence to test multi-tile accumulation."""
        B, H_Q, H_KV, S, D = 1, 28, 4, 512, 128
        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.999, f"Long-sequence cosine similarity {cos:.6f} < 0.999"

    # -- Precision validation --

    def test_fp32_accumulation_precision(self, device: str) -> None:
        """Verify fp32 accumulation doesn't drift on realistic shapes.

        Uses Molmo2-4B config: 28Q/4KV heads, head_dim=128, seq=256.
        This is the core validation for Phase 1 exit criteria.
        """
        B, H_Q, H_KV, S, D = 1, 28, 4, 256, 128
        torch.manual_seed(42)
        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.9995, f"Precision cosine similarity {cos:.6f} < 0.9995"

        # Also check max absolute error
        max_err = (actual.float() - expected.float()).abs().max().item()
        assert max_err < 0.05, f"Max absolute error {max_err:.4f} >= 0.05"

    # -- Custom scale --

    def test_custom_sm_scale(self, device: str) -> None:
        """Custom softmax scale factor."""
        B, H, S, D = 1, 8, 64, 128
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        scale = 0.05  # non-default scale
        expected = F.scaled_dot_product_attention(q, k, v, scale=scale)
        actual = triton_flash_attention(q, k, v, sm_scale=scale)

        assert _cosine_similarity(actual, expected) > 0.999

    # -- Batch size > 1 --

    def test_batched(self, device: str) -> None:
        """Multiple sequences in a batch."""
        B, H_Q, H_KV, S, D = 4, 28, 4, 64, 128
        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        expected = _sdpa_reference(q, k, v)
        actual = triton_flash_attention(q, k, v)

        assert _cosine_similarity(actual, expected) > 0.999
