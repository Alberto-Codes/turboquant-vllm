"""Shared pytest fixtures for TurboQuant tests.

Provides deterministic seeding, cached codebooks, and common quantizer
instances to eliminate redundant computation and ensure reproducibility.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant_vllm.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)
from turboquant_vllm.kv_cache import CompressedDynamicCache
from turboquant_vllm.lloyd_max import LloydMaxCodebook, solve_lloyd_max
from turboquant_vllm.quantizer import TurboQuantMSE, TurboQuantProd

# ---------------------------------------------------------------------------
# Constants shared across test modules
# ---------------------------------------------------------------------------
DIM = 128
BITS = 3
BITS_4 = 4
SEED = 42
N_SAMPLES = 500
N_PAIRS = 300


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Fix torch random seed before every test for reproducibility."""
    torch.manual_seed(SEED)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Device fixture for cross-device validation (CPU and GPU).

    GPU tests are skipped when CUDA is not available.
    Run GPU tests only: pytest -m gpu
    Exclude GPU tests: pytest -m "not gpu"
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def codebook_3bit() -> LloydMaxCodebook:
    """Module-scoped Lloyd-Max codebook for dim=128, bits=3.

    Cached across all tests in a module to avoid redundant ~2s scipy solves.
    """
    centroids, boundaries = solve_lloyd_max(DIM, BITS)
    return LloydMaxCodebook(
        centroids=centroids, boundaries=boundaries, bits=BITS, dim=DIM
    )


@pytest.fixture(scope="module")
def mse_quantizer() -> TurboQuantMSE:
    """Module-scoped TurboQuantMSE(dim=128, bits=3)."""
    return TurboQuantMSE(DIM, BITS, seed=SEED)


@pytest.fixture(scope="module")
def prod_quantizer() -> TurboQuantProd:
    """Module-scoped TurboQuantProd(dim=128, bits=3)."""
    return TurboQuantProd(DIM, BITS, seed=SEED)


@pytest.fixture(scope="module")
def key_compressor() -> TurboQuantCompressorV2:
    """Module-scoped key compressor (dim=128, bits=3)."""
    return TurboQuantCompressorV2(DIM, BITS, seed=SEED)


@pytest.fixture(scope="module")
def value_compressor() -> TurboQuantCompressorMSE:
    """Module-scoped value compressor (dim=128, bits=3)."""
    return TurboQuantCompressorMSE(DIM, BITS, seed=SEED)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat cosine similarity between two tensors."""
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def compress_tq4(
    tensor: torch.Tensor, quantizer: TurboQuantMSE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a tensor using TurboQuantMSE and nibble-pack.

    Args:
        tensor: ``[batch, heads, seq, head_dim]`` fp16/bf16.
        quantizer: Configured TurboQuantMSE instance.

    Returns:
        ``(packed_indices, norms)`` where packed is uint8
        ``[batch, heads, seq, head_dim//2]`` and norms is fp32
        ``[batch, heads, seq]``.
    """
    B, H, S, D = tensor.shape
    flat = tensor.float().reshape(-1, D)
    indices, norms = quantizer.quantize(flat)
    indices = indices.to(torch.uint8).reshape(B, H, S, D)
    norms = norms.reshape(B, H, S)
    packed = CompressedDynamicCache._nibble_pack(indices)
    return packed, norms


def decompress_tq4(
    packed: torch.Tensor, norms: torch.Tensor, quantizer: TurboQuantMSE
) -> torch.Tensor:
    """Decompress nibble-packed tensor back to fp32.

    Args:
        packed: Nibble-packed indices ``[batch, heads, seq, head_dim//2]`` uint8.
        norms: Norms ``[batch, heads, seq]`` fp32.
        quantizer: Configured TurboQuantMSE instance.

    Returns:
        Reconstructed tensor ``[batch, heads, seq, head_dim]`` fp32.
    """
    B, H, S, HALF_D = packed.shape
    D = HALF_D * 2
    indices = CompressedDynamicCache._nibble_unpack(packed)
    flat_idx = indices.reshape(-1, D)
    flat_norms = norms.reshape(-1, 1)
    reconstructed = quantizer.dequantize(flat_idx, flat_norms)
    return reconstructed.reshape(B, H, S, D)
