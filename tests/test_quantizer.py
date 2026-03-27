"""Tests for TurboQuantMSE and TurboQuantProd quantizers."""

import math

import pytest
import torch

from turboquant_vllm.quantizer import TurboQuantMSE, TurboQuantProd

from .conftest import BITS, DIM, N_PAIRS, N_SAMPLES


@pytest.mark.unit
class TestTurboQuantMSE:
    """Validate Stage 1 MSE quantizer."""

    def test_round_trip_mse_within_bounds(
        self, mse_quantizer: TurboQuantMSE, device: torch.device
    ) -> None:
        """MSE distortion should be within theoretical bounds.

        Paper bound: D_mse <= sqrt(3) * pi/2 * (1/4^b) for dimension d.
        """
        x = torch.randn(N_SAMPLES, DIM).to(device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        indices, norms = mse_quantizer.quantize(x)
        reconstructed = mse_quantizer.dequantize(indices, norms)

        mse = ((x - reconstructed) ** 2).mean().item()
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1.0 / 4**BITS)
        assert mse < theoretical_bound * 5, (
            f"MSE {mse:.6f} exceeds 5x theoretical bound {theoretical_bound:.6f}"
        )

    def test_norms_preserved(
        self, mse_quantizer: TurboQuantMSE, device: torch.device
    ) -> None:
        """Stored norms should closely match original vector norms."""
        x = torch.randn(100, DIM).to(device)
        original_norms = torch.norm(x, dim=-1)

        _, stored_norms = mse_quantizer.quantize(x)
        stored_norms = stored_norms.squeeze(-1)

        relative_error = (
            (original_norms - stored_norms).abs() / (original_norms + 1e-10)
        ).mean()
        assert relative_error < 0.01, (
            f"Norm relative error {relative_error:.4f} too high"
        )

    def test_higher_bits_lower_mse(self, device: torch.device) -> None:
        """More bits should yield lower reconstruction error."""
        x = torch.randn(200, DIM).to(device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        mse_values = []
        for bits in (2, 3, 4, 5):
            q = TurboQuantMSE(DIM, bits)
            indices, norms = q.quantize(x)
            reconstructed = q.dequantize(indices, norms)
            mse = ((x - reconstructed) ** 2).mean().item()
            mse_values.append(mse)

        for i in range(len(mse_values) - 1):
            assert mse_values[i] > mse_values[i + 1], (
                f"MSE should decrease with more bits: {mse_values}"
            )

    def test_zero_vector(
        self, mse_quantizer: TurboQuantMSE, device: torch.device
    ) -> None:
        """Zero vectors should not cause errors and should reconstruct near zero."""
        x = torch.zeros(1, DIM, device=device)
        indices, norms = mse_quantizer.quantize(x)
        reconstructed = mse_quantizer.dequantize(indices, norms)

        assert reconstructed.shape == x.shape
        assert torch.allclose(reconstructed, x, atol=1e-6)

    def test_single_vector(
        self, mse_quantizer: TurboQuantMSE, device: torch.device
    ) -> None:
        """Single vector input should work without broadcast issues."""
        x = torch.randn(1, DIM).to(device)
        indices, norms = mse_quantizer.quantize(x)
        reconstructed = mse_quantizer.dequantize(indices, norms)
        assert reconstructed.shape == (1, DIM)

    def test_large_magnitude_vectors(
        self, mse_quantizer: TurboQuantMSE, device: torch.device
    ) -> None:
        """Vectors with large magnitudes should reconstruct proportionally.

        Magnitude should not affect reconstruction quality since norms are
        stored separately. Cosine similarity is magnitude-invariant.
        At 3-bit MSE, empirical cosine sim is ~0.86 for random vectors.
        """
        x = (torch.randn(50, DIM) * 1000.0).to(device)
        indices, norms = mse_quantizer.quantize(x)
        reconstructed = mse_quantizer.dequantize(indices, norms)

        cos_sim = torch.nn.functional.cosine_similarity(x, reconstructed, dim=-1)
        assert cos_sim.mean() > 0.80


@pytest.mark.unit
class TestTurboQuantProd:
    """Validate Stage 2 inner product estimator."""

    def test_inner_product_unbiased(
        self, prod_quantizer: TurboQuantProd, device: torch.device
    ) -> None:
        """QJL-corrected inner product estimates should be approximately unbiased."""
        queries = (torch.randn(N_PAIRS, DIM) * 0.1).to(device)
        keys = (torch.randn(N_PAIRS, DIM) * 0.1).to(device)

        true_ip = (queries * keys).sum(dim=-1)

        indices, norms, qjl_signs, res_norms = prod_quantizer.quantize(keys)
        estimated_ip = prod_quantizer.estimate_inner_product(
            queries, indices, norms, qjl_signs, res_norms
        ).squeeze(-1)

        bias = (estimated_ip - true_ip).mean().item()
        signal = true_ip.abs().mean().item()
        relative_bias = abs(bias) / (signal + 1e-10)

        assert relative_bias < 0.15, (
            f"Relative bias {relative_bias:.4f} too high "
            f"(bias={bias:.6f}, signal={signal:.6f})"
        )

    def test_correlation_with_true_ip(
        self, prod_quantizer: TurboQuantProd, device: torch.device
    ) -> None:
        """Estimated IPs should correlate strongly with true IPs."""
        queries = (torch.randn(N_SAMPLES, DIM) * 0.1).to(device)
        keys = (torch.randn(N_SAMPLES, DIM) * 0.1).to(device)

        true_ip = (queries * keys).sum(dim=-1)
        indices, norms, qjl_signs, res_norms = prod_quantizer.quantize(keys)
        estimated_ip = prod_quantizer.estimate_inner_product(
            queries, indices, norms, qjl_signs, res_norms
        ).squeeze(-1)

        correlation = torch.corrcoef(torch.stack([true_ip, estimated_ip]))[0, 1].item()
        assert correlation > 0.9, f"Correlation {correlation:.4f} too low"

    def test_bits_minimum_is_2(self) -> None:
        """TurboQuantProd requires bits >= 2; bits=1 should raise ValueError."""
        with pytest.raises(ValueError, match="bits >= 2"):
            TurboQuantProd(DIM, bits=1)

    def test_bits_2_works(self, device: torch.device) -> None:
        """bits=2 is the minimum valid configuration (1 MSE + 1 QJL)."""
        q = TurboQuantProd(DIM, bits=2)
        x = torch.randn(10, DIM).to(device)
        indices, norms, qjl_signs, res_norms = q.quantize(x)
        assert indices.shape == (10, DIM)
        assert qjl_signs.shape == (10, DIM)

    def test_zero_vector_inner_product(
        self, prod_quantizer: TurboQuantProd, device: torch.device
    ) -> None:
        """Inner product with zero key should be approximately zero."""
        query = torch.randn(1, DIM).to(device)
        key = torch.zeros(1, DIM, device=device)

        indices, norms, qjl_signs, res_norms = prod_quantizer.quantize(key)
        estimated = prod_quantizer.estimate_inner_product(
            query, indices, norms, qjl_signs, res_norms
        )
        assert abs(estimated.item()) < 0.01
