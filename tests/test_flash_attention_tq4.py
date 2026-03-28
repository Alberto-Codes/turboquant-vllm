"""Tests for Phase 2: Fused TQ4 Flash Attention kernel.

Validates that fused TQ4 decompression inside the FA inner loop produces
output matching the unfused path (decompress K first, then vanilla FA).

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.triton.flash_attention import triton_flash_attention
from turboquant_vllm.triton.flash_attention_tq4 import triton_flash_attention_tq4

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


class TestTQ4FlashAttention:
    """Phase 2 validation: fused TQ4 FA vs unfused (decompress + vanilla FA)."""

    def test_basic_mha(self, device: str, tq4_quantizer) -> None:
        """MHA (no GQA): fused TQ4 matches unfused path."""
        B, H, S, D = 1, 4, 32, 128
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        # Unfused reference: decompress then vanilla FA
        expected = triton_flash_attention(q, k_decompressed, v)

        # Fused: TQ4 decompression inside kernel
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Fused vs unfused cosine {cos:.6f} < 0.998"

    def test_gqa_4_to_1(self, device: str, tq4_quantizer) -> None:
        """GQA 4:1 (Molmo2-4B config: 32Q/8KV)."""
        B, H_Q, H_KV, S, D = 1, 32, 8, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"GQA 4:1 cosine {cos:.6f} < 0.998"

    def test_gqa_7_to_1(self, device: str, tq4_quantizer) -> None:
        """GQA 7:1 (Molmo2-8B config: 28Q/4KV)."""
        B, H_Q, H_KV, S, D = 1, 28, 4, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"GQA 7:1 cosine {cos:.6f} < 0.998"

    def test_decode_mode(self, device: str, tq4_quantizer) -> None:
        """Decode: seq_q=1, long KV cache."""
        B, H_Q, H_KV, D = 1, 32, 8, 128
        S_Q, S_KV = 1, 512
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Decode cosine {cos:.6f} < 0.998"

    def test_causal(self, device: str, tq4_quantizer) -> None:
        """Causal masking with TQ4 compressed K."""
        B, H, S, D = 1, 4, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v, is_causal=True)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
            is_causal=True,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Causal cosine {cos:.6f} < 0.998"

    def test_bf16(self, device: str, tq4_quantizer) -> None:
        """bfloat16 inputs with TQ4 compressed K."""
        B, H, S, D = 1, 4, 32, 128
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"bf16 cosine {cos:.6f} < 0.998"

    def test_long_sequence_precision(self, device: str, tq4_quantizer) -> None:
        """Long sequence to validate multi-tile accumulation precision."""
        B, H_Q, H_KV, S, D = 1, 32, 8, 512, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Long seq cosine {cos:.6f} < 0.998"

    def test_prime_seq_length(self, device: str, tq4_quantizer) -> None:
        """Non-power-of-2 sequence length (tests masking)."""
        B, H, S, D = 1, 4, 37, 128
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Prime seq cosine {cos:.6f} < 0.998"

    def test_batched(self, device: str, tq4_quantizer) -> None:
        """Multiple sequences in a batch."""
        B, H_Q, H_KV, S, D = 4, 32, 8, 64, 128
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        k_packed, k_norms = compress_tq4(k, tq4_quantizer)
        k_decompressed = decompress_tq4(k_packed, k_norms, tq4_quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            tq4_quantizer.codebook.centroids.to(device),
            tq4_quantizer.rotation.to(device),
            v,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Batched cosine {cos:.6f} < 0.998"
