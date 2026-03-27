"""Tests for Phase 3: Fused TQ4 K+V Flash Attention kernel.

Validates that fused TQ4 decompression of both K and V tiles inside
the FA inner loop (plus post-rotation) matches the unfused path.

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant_consumer.kv_cache import CompressedDynamicCache
from turboquant_consumer.quantizer import TurboQuantMSE
from turboquant_consumer.triton.flash_attention import triton_flash_attention
from turboquant_consumer.triton.flash_attention_tq4_kv import (
    triton_flash_attention_tq4_kv,
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


def _compress(
    tensor: torch.Tensor, quantizer: TurboQuantMSE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a tensor and nibble-pack.

    Returns:
        ``(packed, norms)`` — uint8 ``[..., D//2]`` and fp32 ``[..., S]``.
    """
    B, H, S, D = tensor.shape
    flat = tensor.float().reshape(-1, D)
    indices, norms = quantizer.quantize(flat)
    indices = indices.to(torch.uint8).reshape(B, H, S, D)
    norms = norms.reshape(B, H, S)
    packed = CompressedDynamicCache._nibble_pack(indices)
    return packed, norms


def _decompress(
    packed: torch.Tensor, norms: torch.Tensor, quantizer: TurboQuantMSE
) -> torch.Tensor:
    """Decompress nibble-packed tensor back to fp32.

    Returns:
        Reconstructed tensor ``[batch, heads, seq, head_dim]`` fp32.
    """
    B, H, S, HALF_D = packed.shape
    D = HALF_D * 2
    indices = CompressedDynamicCache._nibble_unpack(packed)
    flat_idx = indices.reshape(-1, D)
    flat_norms = norms.reshape(-1, 1)
    return quantizer.dequantize(flat_idx, flat_norms).reshape(B, H, S, D)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat cosine similarity."""
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTQ4KVFlashAttention:
    """Phase 3: fused K+V TQ4 FA vs unfused (decompress both, vanilla FA)."""

    def test_basic_mha(self, device: str) -> None:
        """MHA: fused K+V matches unfused path."""
        B, H, S, D, bits = 1, 4, 32, 128, 4
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress(k, quantizer)
        v_packed, v_norms = _compress(v, quantizer)
        k_dec = _decompress(k_packed, k_norms, quantizer).to(q.dtype)
        v_dec = _decompress(v_packed, v_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"MHA cosine {cos:.6f} < 0.998"

    def test_gqa_4_to_1(self, device: str) -> None:
        """GQA 4:1 (Molmo2-4B: 32Q/8KV)."""
        B, H_Q, H_KV, S, D, bits = 1, 32, 8, 64, 128, 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress(k, quantizer)
        v_packed, v_norms = _compress(v, quantizer)
        k_dec = _decompress(k_packed, k_norms, quantizer).to(q.dtype)
        v_dec = _decompress(v_packed, v_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"GQA 4:1 cosine {cos:.6f} < 0.998"

    def test_gqa_7_to_1(self, device: str) -> None:
        """GQA 7:1 (Molmo2-8B: 28Q/4KV)."""
        B, H_Q, H_KV, S, D, bits = 1, 28, 4, 64, 128, 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress(k, quantizer)
        v_packed, v_norms = _compress(v, quantizer)
        k_dec = _decompress(k_packed, k_norms, quantizer).to(q.dtype)
        v_dec = _decompress(v_packed, v_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"GQA 7:1 cosine {cos:.6f} < 0.998"

    def test_decode(self, device: str) -> None:
        """Decode: seq_q=1, long compressed KV cache."""
        B, H_Q, H_KV, D, bits = 1, 32, 8, 128, 4
        S_Q, S_KV = 1, 512
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress(k, quantizer)
        v_packed, v_norms = _compress(v, quantizer)
        k_dec = _decompress(k_packed, k_norms, quantizer).to(q.dtype)
        v_dec = _decompress(v_packed, v_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Decode cosine {cos:.6f} < 0.998"

    def test_causal(self, device: str) -> None:
        """Causal masking with both K and V compressed."""
        B, H, S, D, bits = 1, 4, 64, 128, 4
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress(k, quantizer)
        v_packed, v_norms = _compress(v, quantizer)
        k_dec = _decompress(k_packed, k_norms, quantizer).to(q.dtype)
        v_dec = _decompress(v_packed, v_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec, is_causal=True)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
            is_causal=True,
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Causal cosine {cos:.6f} < 0.998"

    def test_long_sequence(self, device: str) -> None:
        """Long sequence multi-tile accumulation precision."""
        B, H_Q, H_KV, S, D, bits = 1, 32, 8, 512, 128, 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress(k, quantizer)
        v_packed, v_norms = _compress(v, quantizer)
        k_dec = _decompress(k_packed, k_norms, quantizer).to(q.dtype)
        v_dec = _decompress(v_packed, v_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Long seq cosine {cos:.6f} < 0.998"

    def test_batched(self, device: str) -> None:
        """Multiple sequences in a batch."""
        B, H_Q, H_KV, S, D, bits = 4, 32, 8, 64, 128, 4
        torch.manual_seed(42)

        q = torch.randn(B, H_Q, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H_KV, S, D, device=device, dtype=torch.float16)

        quantizer = TurboQuantMSE(D, bits=bits, seed=42)
        quantizer.rotation = quantizer.rotation.to(device)

        k_packed, k_norms = _compress(k, quantizer)
        v_packed, v_norms = _compress(v, quantizer)
        k_dec = _decompress(k_packed, k_norms, quantizer).to(q.dtype)
        v_dec = _decompress(v_packed, v_norms, quantizer).to(q.dtype)

        expected = triton_flash_attention(q, k_dec, v_dec)
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            quantizer.codebook.centroids.to(device),
            quantizer.rotation.to(device),
        )

        cos = _cosine_similarity(actual, expected)
        assert cos > 0.998, f"Batched cosine {cos:.6f} < 0.998"
