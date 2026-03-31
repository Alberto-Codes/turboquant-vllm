"""Tests for Phase 3: Fused TQ4 K+V Flash Attention kernel.

Validates that fused TQ4 decompression of both K and V tiles inside
the FA inner loop (plus post-rotation) matches the unfused path.

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.flash_attention_tq4_kv import (
    _AUTOTUNE_CONFIGS,
    triton_flash_attention_tq4_kv,
)
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

from .conftest import (
    assert_tq4_fused_matches_unfused,
    compress_tq4,
    cosine_similarity_flat,
    decompress_tq4,
)

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


# ---------------------------------------------------------------------------
# Autotune config validation (Story 5.5)
# ---------------------------------------------------------------------------


class TestAutotuneConfigs:
    """Verify autotune search space covers required BLOCK_M values."""

    def test_block_m_32_in_configs(self) -> None:
        """BLOCK_M=32 must exist for head_dim=256 SRAM optimization."""
        block_m_values = {c.kwargs["BLOCK_M"] for c in _AUTOTUNE_CONFIGS}
        assert 32 in block_m_values

    def test_original_block_m_values_preserved(self) -> None:
        """Original BLOCK_M values (16, 64, 128) must remain in configs."""
        block_m_values = {c.kwargs["BLOCK_M"] for c in _AUTOTUNE_CONFIGS}
        assert {16, 64, 128}.issubset(block_m_values)


# ---------------------------------------------------------------------------
# Multi head-dim flash attention (Story 5.2)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[64, 96, 128], ids=["dim64", "dim96", "dim128"], scope="module")
def multi_dim_quantizer(request: pytest.FixtureRequest) -> TurboQuantMSE:
    """Quantizer at various head_dims for FA kernel validation."""
    return TurboQuantMSE(request.param, 4, seed=42)


def _compress_local(
    tensor: torch.Tensor, q: TurboQuantMSE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a 4D tensor using a specific quantizer."""
    B, H, S, D = tensor.shape
    flat = tensor.float().reshape(B * H * S, 1, D)
    rot_t = q.rotation.T.contiguous().to(tensor.device)
    packed, norms = tq4_compress(
        flat,
        rot_t[:, 0::2].contiguous(),
        rot_t[:, 1::2].contiguous(),
        q.codebook.boundaries.clone().to(tensor.device),
    )
    return packed.reshape(B, H, S, D // 2), norms.reshape(B, H, S)


def _decompress_local(
    packed: torch.Tensor,
    norms: torch.Tensor,
    q: TurboQuantMSE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decompress a 4D nibble-packed tensor using a specific quantizer."""
    B, H, S, HALF_D = packed.shape
    D = HALF_D * 2
    flat_packed = packed.reshape(B * H * S, 1, HALF_D)
    flat_norms = norms.reshape(B * H * S, 1, 1)
    rotated = tq4_decompress(
        flat_packed,
        flat_norms,
        q.codebook.centroids.clone().to(packed.device),
        dtype=dtype,
    )
    return rotated.reshape(B, H, S, D)


TQ4_KV_MULTI_DIM_SHAPES = [
    # head_dim=64: MHA and GQA
    pytest.param(1, 4, 4, 32, 32, 64, False, torch.float16, id="mha_dim64"),
    pytest.param(1, 32, 8, 1, 64, 64, False, torch.float16, id="gqa_decode_dim64"),
    # head_dim=96: MHA and GQA
    pytest.param(1, 4, 4, 32, 32, 96, False, torch.float16, id="mha_dim96"),
    pytest.param(1, 32, 8, 1, 64, 96, False, torch.float16, id="gqa_decode_dim96"),
    # head_dim=128 (regression baseline)
    pytest.param(1, 4, 4, 32, 32, 128, False, torch.float16, id="mha_dim128"),
]


@pytest.mark.gpu
class TestMultiDimTQ4KVFlashAttention:
    """Flash attention fused-vs-unfused validation at head_dim 64, 96, 128."""

    @pytest.mark.parametrize(
        ("B", "H_Q", "H_KV", "S_Q", "S_KV", "D", "is_causal", "dtype"),
        TQ4_KV_MULTI_DIM_SHAPES,
    )
    def test_fused_kv_matches_unfused(
        self,
        device: str,
        multi_dim_quantizer: TurboQuantMSE,
        B: int,
        H_Q: int,
        H_KV: int,
        S_Q: int,
        S_KV: int,
        D: int,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> None:
        """Fused K+V TQ4 FA matches unfused path at non-128 head dims."""
        dim = multi_dim_quantizer.rotation.shape[0]
        if dim != D:
            pytest.skip(f"Quantizer dim={dim} != shape D={D}")

        from turboquant_vllm.triton.flash_attention import triton_flash_attention

        q = torch.randn(B, H_Q, S_Q, D, device=device, dtype=dtype)
        k = torch.randn(B, H_KV, S_KV, D, device=device, dtype=dtype)
        v = torch.randn(B, H_KV, S_KV, D, device=device, dtype=dtype)

        centroids = multi_dim_quantizer.codebook.centroids.to(device)
        rotation = multi_dim_quantizer.rotation.to(device)

        # Compress K and V
        k_packed, k_norms = _compress_local(k, multi_dim_quantizer)
        v_packed, v_norms = _compress_local(v, multi_dim_quantizer)

        # Unfused reference: decompress then standard FA
        k_dec = _decompress_local(k_packed, k_norms, multi_dim_quantizer, dtype)
        k_dec = (k_dec.float() @ rotation).to(dtype)
        v_dec = _decompress_local(v_packed, v_norms, multi_dim_quantizer, dtype)
        v_dec = (v_dec.float() @ rotation).to(dtype)
        expected = triton_flash_attention(q, k_dec, v_dec, is_causal=is_causal)

        # Fused path
        actual = triton_flash_attention_tq4_kv(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            centroids,
            rotation,
            is_causal=is_causal,
        )

        cos = cosine_similarity_flat(actual, expected)
        assert cos > 0.998, f"Fused vs unfused cosine {cos:.6f} < 0.998 at D={D}"
