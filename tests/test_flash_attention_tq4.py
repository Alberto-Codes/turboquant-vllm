"""Tests for Phase 2: Fused TQ4 Flash Attention kernel.

Validates that fused TQ4 decompression inside the FA inner loop produces
output matching the unfused path (decompress K first, then vanilla FA).

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant_vllm.kv_cache import CompressedDynamicCache
from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.flash_attention import triton_flash_attention
from turboquant_vllm.triton.flash_attention_tq4 import triton_flash_attention_tq4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device():
    """CUDA-only device fixture (overrides conftest parametrized fixture)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton Flash Attention")
    return "cuda"


def _compress_keys(
    keys: torch.Tensor, quantizer: TurboQuantMSE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress keys using TurboQuantMSE and nibble-pack.

    Args:
        keys: ``[batch, heads, seq, head_dim]`` fp16/bf16.
        quantizer: Configured TurboQuantMSE instance.

    Returns:
        ``(packed_indices, norms)`` where packed is uint8
        ``[batch, heads, seq, head_dim//2]`` and norms is fp32
        ``[batch, heads, seq]``.
    """
    B, H, S, D = keys.shape
    flat = keys.float().reshape(-1, D)
    indices, norms = quantizer.quantize(flat)
    indices = indices.to(torch.uint8).reshape(B, H, S, D)
    norms = norms.reshape(B, H, S)
    packed = CompressedDynamicCache._nibble_pack(indices)
    return packed, norms


def _decompress_keys(
    packed: torch.Tensor, norms: torch.Tensor, quantizer: TurboQuantMSE
) -> torch.Tensor:
    """Decompress nibble-packed keys back to fp32 for reference comparison.

    Args:
        packed: Nibble-packed indices ``[batch, heads, seq, head_dim//2]`` uint8.
        norms: Key norms ``[batch, heads, seq]`` fp32.
        quantizer: Configured TurboQuantMSE instance.

    Returns:
        Reconstructed keys ``[batch, heads, seq, head_dim]`` fp32.
    """
    B, H, S, HALF_D = packed.shape
    D = HALF_D * 2
    indices = CompressedDynamicCache._nibble_unpack(packed)
    flat_idx = indices.reshape(-1, D)
    flat_norms = norms.reshape(-1, 1)
    reconstructed = quantizer.dequantize(flat_idx, flat_norms)
    return reconstructed.reshape(B, H, S, D)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat cosine similarity between two tensors."""
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTQ4FlashAttention:
    """Phase 2 validation: fused TQ4 FA vs unfused (decompress + vanilla FA)."""

    def test_basic_mha(self, device: str) -> None:
        """MHA (no GQA): fused TQ4 matches unfused path."""
        B, H, S, D = 1, 4, 32, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        # Unfused reference: decompress then vanilla FA
        expected = triton_flash_attention(q, k_decompressed, v)

        # Fused: TQ4 decompression inside kernel
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Fused vs unfused cosine {cos:.6f} < 0.998"

    def test_gqa_4_to_1(self, device: str) -> None:
        """GQA 4:1 (Molmo2-4B config: 32Q/8KV)."""
        B, H_Q, H_KV, S, D = 1, 32, 8, 64, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"GQA 4:1 cosine {cos:.6f} < 0.998"

    def test_gqa_7_to_1(self, device: str) -> None:
        """GQA 7:1 (Molmo2-8B config: 28Q/4KV)."""
        B, H_Q, H_KV, S, D = 1, 28, 4, 64, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"GQA 7:1 cosine {cos:.6f} < 0.998"

    def test_decode_mode(self, device: str) -> None:
        """Decode: seq_q=1, long KV cache."""
        B, H_Q, H_KV, D = 1, 32, 8, 128
        S_Q, S_KV = 1, 512
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Decode cosine {cos:.6f} < 0.998"

    def test_causal(self, device: str) -> None:
        """Causal masking with TQ4 compressed K."""
        B, H, S, D = 1, 4, 64, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v, is_causal=True)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
            is_causal=True,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Causal cosine {cos:.6f} < 0.998"

    def test_bf16(self, device: str) -> None:
        """bfloat16 inputs with TQ4 compressed K."""
        B, H, S, D = 1, 4, 32, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"bf16 cosine {cos:.6f} < 0.998"

    def test_long_sequence_precision(self, device: str) -> None:
        """Long sequence to validate multi-tile accumulation precision."""
        B, H_Q, H_KV, S, D = 1, 32, 8, 512, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Long seq cosine {cos:.6f} < 0.998"

    def test_prime_seq_length(self, device: str) -> None:
        """Non-power-of-2 sequence length (tests masking)."""
        B, H, S, D = 1, 4, 37, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Prime seq cosine {cos:.6f} < 0.998"

    def test_batched(self, device: str) -> None:
        """Multiple sequences in a batch."""
        B, H_Q, H_KV, S, D = 4, 32, 8, 64, 128
        bits = 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress_keys(k, quantizer)
        k_decompressed = _decompress_keys(k_packed, k_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_decompressed, v)
        actual = triton_flash_attention_tq4(
            q,
            k_packed,
            k_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            v,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Batched cosine {cos:.6f} < 0.998"
