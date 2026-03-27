"""Shared pytest fixtures for TurboQuant tests.

Provides deterministic seeding, cached codebooks, and common quantizer
instances to eliminate redundant computation and ensure reproducibility.
"""

import pytest
import torch

from turboquant_vllm.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)
from turboquant_vllm.lloyd_max import LloydMaxCodebook, solve_lloyd_max
from turboquant_vllm.quantizer import TurboQuantMSE, TurboQuantProd

# ---------------------------------------------------------------------------
# Constants shared across test modules
# ---------------------------------------------------------------------------
DIM = 128
BITS = 3
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
