"""Tests for Phase 3: Fused TQ4 K+V Flash Attention kernel.

Validates that fused TQ4 decompression of both K and V tiles inside
the FA inner loop (plus post-rotation) matches the unfused path.

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.triton.flash_attention import triton_flash_attention
from turboquant_vllm.triton.flash_attention_tq4_kv import (
    triton_flash_attention_tq4_kv,
)

from .conftest import compress_tq4, cosine_similarity_flat, decompress_tq4

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
# Tests
# ---------------------------------------------------------------------------


class TestTQ4KVFlashAttention:
    """Phase 3: fused K+V TQ4 FA vs unfused (decompress both, vanilla FA)."""

    def test_basic_mha(self, device: str, tq4_quantizer) -> None:
        """MHA: fused K+V matches unfused path."""
        B, H, S, D = 1, 4, 32, 128
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        k_dec = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"MHA cosine {cos:.6f} < 0.998"

    def test_gqa_4_to_1(self, device: str, tq4_quantizer) -> None:
        """GQA 4:1 (Molmo2-4B: 32Q/8KV)."""
        B, H_Q, H_KV, S, D = 1, 32, 8, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        k_dec = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"GQA 4:1 cosine {cos:.6f} < 0.998"

    def test_gqa_7_to_1(self, device: str, tq4_quantizer) -> None:
        """GQA 7:1 (Molmo2-8B: 28Q/4KV)."""
        B, H_Q, H_KV, S, D = 1, 28, 4, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        k_dec = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"GQA 7:1 cosine {cos:.6f} < 0.998"

    def test_decode(self, device: str, tq4_quantizer) -> None:
        """Decode: seq_q=1, long compressed KV cache."""
        B, H_Q, H_KV, D = 1, 32, 8, 128
        S_Q, S_KV = 1, 512
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        k_dec = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Decode cosine {cos:.6f} < 0.998"

    def test_causal(self, device: str, tq4_quantizer) -> None:
        """Causal masking with both K and V compressed."""
        B, H, S, D = 1, 4, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        k_dec = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec, is_causal=True)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            is_causal=True,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Causal cosine {cos:.6f} < 0.998"

    def test_long_sequence(self, device: str, tq4_quantizer) -> None:
        """Long sequence multi-tile accumulation precision."""
        B, H_Q, H_KV, S, D = 1, 32, 8, 512, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        k_dec = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Long seq cosine {cos:.6f} < 0.998"

    def test_batched(self, device: str, tq4_quantizer) -> None:
        """Multiple sequences in a batch."""
        B, H_Q, H_KV, S, D = 4, 32, 8, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        v_packed, v_norms = compress_tq4(v, tq4_quantizer)
        k_dec = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)
        v_dec = decompress_tq4(v_packed, v_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Batched cosine {cos:.6f} < 0.998"
