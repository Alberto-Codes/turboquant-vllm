"""Lloyd-Max optimal scalar quantizer for Beta-distributed coordinates.

After random orthogonal rotation, each coordinate of a unit-norm vector
follows a Beta distribution concentrated near zero. The Lloyd-Max algorithm
finds the optimal set of centroids (reconstruction points) that minimize
mean squared error for this known distribution.

For dimensions d >= 64, the Beta distribution is well-approximated by
a Gaussian N(0, 1/d), which simplifies the codebook computation.

Reference: Section 3.1 of arXiv 2504.19874.

Examples:
    ```python
    from turboquant_vllm.lloyd_max import solve_lloyd_max, LloydMaxCodebook

    centroids, boundaries = solve_lloyd_max(d=128, bits=3)
    codebook = LloydMaxCodebook(centroids, boundaries, bits=3, dim=128)
    ```

See Also:
    :func:`solve_lloyd_max`: Factory that computes centroids and boundaries.
    :class:`LloydMaxCodebook`: Dataclass wrapping a precomputed codebook.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from scipy import integrate
from scipy.stats import beta as beta_dist
from scipy.stats import norm


def _beta_pdf(x: float, d: int) -> float:
    """Evaluate the Beta PDF for a rotated unit-vector coordinate.

    After applying a random orthogonal rotation to a d-dimensional unit
    vector, each coordinate follows Beta((d-1)/2, (d-1)/2) scaled to
    the interval [-1/sqrt(d), 1/sqrt(d)].

    TODO: This exact Beta path (use_exact=True) is untested. The Gaussian
    approximation is used for all d >= 64 cases. Add coverage if we need
    to support very low-dimensional heads (d < 64).

    Args:
        x: Coordinate value.
        d: Vector dimension.

    Returns:
        Probability density at x.
    """
    alpha = (d - 1) / 2.0
    # Scale from [-1/sqrt(d), 1/sqrt(d)] to [0, 1] for scipy Beta
    scale = 1.0 / math.sqrt(d)
    if abs(x) >= scale:
        return 0.0
    t = (x / scale + 1.0) / 2.0  # Map to [0, 1]
    return beta_dist.pdf(t, alpha, alpha) / (2.0 * scale)


def _gaussian_pdf(x: float, d: int) -> float:
    """Evaluate the Gaussian approximation to the Beta PDF.

    For d >= 64, the Beta distribution of rotated coordinates is
    well-approximated by N(0, 1/d).

    Args:
        x: Coordinate value.
        d: Vector dimension.

    Returns:
        Probability density at x under N(0, 1/d).
    """
    sigma = 1.0 / math.sqrt(d)
    return float(norm.pdf(x, loc=0.0, scale=sigma))


def solve_lloyd_max(
    d: int,
    bits: int,
    *,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the Lloyd-Max conditions for optimal scalar quantization.

    Results are cached by (d, bits, use_exact) so that multi-layer models
    (e.g., 32 layers × 2 K/V compressors = 64 calls) pay the scipy
    integration cost only once. Without caching, initialization takes
    2+ minutes for models like Molmo2-8B.

    Args:
        d: Vector dimension (determines the distribution shape).
        bits: Number of quantization bits (produces 2^bits centroids).
        use_exact: If True, use exact Beta PDF. If False, use Gaussian
            approximation (faster, accurate for d >= 64).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        Tuple of (centroids, boundaries) as 1-D tensors. Centroids has
        length 2^bits, boundaries has length 2^bits - 1.
    """
    return _solve_lloyd_max_cached(d, bits, use_exact, max_iter, tol)


@lru_cache(maxsize=32)
def _solve_lloyd_max_cached(
    d: int,
    bits: int,
    use_exact: bool,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached inner implementation of solve_lloyd_max.

    Separated from the public API so that keyword-only arguments can be
    converted to positional arguments for lru_cache hashability.

    Args:
        d: Vector dimension.
        bits: Number of quantization bits.
        use_exact: Whether to use exact Beta PDF.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Tuple of (centroids, boundaries) tensors.
    """
    n_levels = 1 << bits
    pdf = _beta_pdf if use_exact else _gaussian_pdf

    # Initialize centroids uniformly in the support
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.0 * sigma, 3.0 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        # Compute boundaries as midpoints between adjacent centroids
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]

        # Update centroids as conditional expectations E[X | X in partition_i]
        edges = [lo] + boundaries + [hi]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            if b - a < 1e-15:
                new_centroids.append((a + b) / 2.0)
                continue
            # E[X | a <= X <= b] = integral(x * pdf(x)) / integral(pdf(x))
            numer, _ = integrate.quad(lambda x: x * pdf(x, d), a, b)
            denom, _ = integrate.quad(lambda x: pdf(x, d), a, b)
            if denom < 1e-15:
                new_centroids.append((a + b) / 2.0)
            else:
                new_centroids.append(numer / denom)

        # Check convergence
        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries_final = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]

    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries_final, dtype=torch.float32),
    )


@dataclass
class LloydMaxCodebook:
    """Precomputed optimal scalar quantizer for a given dimension and bit-width.

    The codebook stores centroids and boundaries computed by the Lloyd-Max
    algorithm. It maps continuous coordinate values to discrete indices and
    back via nearest-centroid lookup.

    Attributes:
        centroids (torch.Tensor): Reconstruction values, shape ``(2^bits,)``.
        boundaries (torch.Tensor): Partition boundaries, shape ``(2^bits - 1,)``.
        bits (int): Number of quantization bits.
        dim (int): Vector dimension used to compute the codebook.

    Examples:
        Round-trip quantize and dequantize a tensor:

        ```python
        codebook = LloydMaxCodebook(centroids, boundaries, bits=3, dim=128)
        indices = codebook.quantize(x)
        x_hat = codebook.dequantize(indices)
        ```
    """

    centroids: torch.Tensor
    boundaries: torch.Tensor
    bits: int
    dim: int

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous values to nearest centroid indices.

        Uses bucket search on partition boundaries for O(log n) lookup.

        Args:
            x: Input tensor of any shape.

        Returns:
            Integer tensor of same shape with centroid indices in
            [0, 2^bits - 1].
        """
        bounds = self.boundaries.to(x.device)
        # bucketize returns the index of the bucket each value falls into
        return torch.bucketize(x, bounds)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct continuous values from centroid indices.

        Args:
            indices: Integer tensor of centroid indices.

        Returns:
            Float tensor of reconstructed values with same shape as indices.
        """
        cents = self.centroids.to(indices.device)
        return cents[indices]
