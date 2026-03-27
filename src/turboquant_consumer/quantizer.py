"""TurboQuant two-stage vector quantizer.

Implements the core TurboQuant algorithm: random orthogonal rotation
followed by optimal scalar quantization (Stage 1, MSE) and optional
QJL residual correction (Stage 2, unbiased inner products).

Stage 1 (TurboQuantMSE):
    Rotate → quantize each coordinate independently → store indices.
    Minimizes mean squared error. Best for value cache reconstruction.

Stage 2 (TurboQuantProd):
    Allocate (bits-1) to Lloyd-Max + 1 bit to QJL sign correction.
    Produces unbiased inner product estimates. Best for key cache
    where attention scores depend on Q·K^T dot products.

Reference: Sections 3-4 of arXiv 2504.19874.

Examples:
    MSE quantization for value cache reconstruction:

    ```python
    quantizer = TurboQuantMSE(dim=64, bits=4)
    indices, norms = quantizer.quantize(values)
    reconstructed = quantizer.dequantize(indices, norms)
    ```

    Unbiased inner products for key cache attention:

    ```python
    quantizer = TurboQuantProd(dim=64, bits=4)
    indices, norms, signs, res_norms = quantizer.quantize(keys)
    scores = quantizer.estimate_inner_product(query, indices, norms, signs, res_norms)
    ```

See Also:
    :mod:`turboquant_consumer.lloyd_max`: Lloyd-Max codebook solver.
"""

import math

import torch

from turboquant_consumer.lloyd_max import LloydMaxCodebook, solve_lloyd_max


def _generate_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal matrix.

    Uses QR decomposition of a random Gaussian matrix. The resulting
    matrix is uniformly distributed over the orthogonal group O(d).

    Generator and tensor are explicitly placed on CPU in float32 to
    avoid conflicts when ``torch.set_default_device('cuda')`` or a
    non-float32 default dtype is active (e.g., inside vLLM model init).

    Args:
        dim: Matrix dimension (d x d).
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (dim, dim) in float32 on CPU.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=gen, device="cpu", dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    # Ensure uniform distribution by correcting signs
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return q * diag_sign.unsqueeze(0)


class TurboQuantMSE:
    """Stage 1 quantizer: rotation + Lloyd-Max scalar quantization.

    Achieves near-optimal MSE distortion rate for high-dimensional
    vectors by exploiting the concentrated Beta distribution that
    emerges after random rotation.

    Attributes:
        dim (int): Vector dimension.
        bits (int): Quantization bit-width.
        codebook (LloydMaxCodebook): Precomputed Lloyd-Max codebook.
        rotation (torch.Tensor): Orthogonal rotation matrix, shape (dim, dim).

    Examples:
        ```python
        quantizer = TurboQuantMSE(dim=64, bits=4)
        indices, norms = quantizer.quantize(torch.randn(8, 64))
        reconstructed = quantizer.dequantize(indices, norms)
        ```
    """

    def __init__(self, dim: int, bits: int, *, seed: int = 42) -> None:
        """Initialize the MSE quantizer.

        Args:
            dim: Vector dimension (head dimension of the model).
            bits: Quantization bits per coordinate (2-4 typical).
            seed: Random seed for the rotation matrix.
        """
        self.dim = dim
        self.bits = bits
        centroids, boundaries = solve_lloyd_max(dim, bits)
        self.codebook = LloydMaxCodebook(
            centroids=centroids,
            boundaries=boundaries,
            bits=bits,
            dim=dim,
        )
        self.rotation = _generate_rotation_matrix(dim, seed=seed)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize vectors to centroid indices.

        Applies rotation, extracts norms, normalizes to unit sphere,
        then quantizes each coordinate independently.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Tuple of (indices, norms) where indices is a long tensor of
            shape (..., dim) and norms is a float tensor of shape (..., 1).
        """
        # Store original shape and flatten to 2D
        orig_shape = x.shape
        flat = x.reshape(-1, self.dim).float()

        # Extract and store norms
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)

        # Apply rotation: y = x @ Pi^T
        pi = self.rotation.to(flat.device)
        rotated = normalized @ pi.T

        # Quantize each coordinate independently
        indices = self.codebook.quantize(rotated)

        return indices.reshape(orig_shape), norms.reshape(*orig_shape[:-1], 1)

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from centroid indices and norms.

        Looks up centroids, applies inverse rotation, and rescales
        by stored norms.

        Args:
            indices: Long tensor of centroid indices, shape (..., dim).
            norms: Float tensor of vector norms, shape (..., 1).

        Returns:
            Reconstructed float tensor of shape (..., dim).
        """
        orig_shape = indices.shape
        flat_idx = indices.reshape(-1, self.dim)
        flat_norms = norms.reshape(-1, 1)

        # Lookup centroids
        reconstructed = self.codebook.dequantize(flat_idx)

        # Inverse rotation: x = y @ Pi
        pi = self.rotation.to(reconstructed.device)
        unrotated = reconstructed @ pi

        # Rescale by norms
        result = unrotated * flat_norms

        return result.reshape(orig_shape)


class TurboQuantProd:
    """Two-stage quantizer with QJL correction for unbiased inner products.

    Allocates (bits-1) bits to Lloyd-Max MSE quantization and 1 bit
    to Quantized Johnson-Lindenstrauss residual correction. The QJL
    step eliminates bias in dot-product estimation, which is critical
    for attention score computation (Q·K^T).

    The unbiased estimator:
        <q, k> ~ <q, k_mse> + ||r|| * sqrt(pi/2) / m * <S@q, sign(S@r)>

    where r is the quantization residual and S is a random Gaussian
    projection matrix.

    Attributes:
        dim (int): Vector dimension.
        bits (int): Total bit budget (bits-1 for MSE, 1 for QJL).
        mse_quantizer (TurboQuantMSE): Stage 1 quantizer with (bits-1) bits.
        qjl_dim (int): Number of QJL projection dimensions.
        qjl_matrix (torch.Tensor): Random Gaussian projection matrix.

    Examples:
        ```python
        quantizer = TurboQuantProd(dim=64, bits=4)
        indices, norms, signs, res_norms = quantizer.quantize(torch.randn(8, 64))
        scores = quantizer.estimate_inner_product(
            torch.randn(1, 64), indices, norms, signs, res_norms
        )
        ```
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        *,
        qjl_dim: int | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize the two-stage quantizer.

        Args:
            dim: Vector dimension (head dimension of the model).
            bits: Total bit budget per coordinate. Must be >= 2
                (1 bit for MSE + 1 bit for QJL minimum).
            qjl_dim: Number of QJL projection dimensions. Defaults
                to dim (standard JL dimensionality).
            seed: Random seed for rotation and projection matrices.

        Raises:
            ValueError: If bits < 2.
        """
        if bits < 2:
            msg = f"TurboQuantProd requires bits >= 2, got {bits}"
            raise ValueError(msg)

        self.dim = dim
        self.bits = bits
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, seed=seed)

        self.qjl_dim = qjl_dim or dim
        gen = torch.Generator().manual_seed(seed + 1)
        self.qjl_matrix = torch.randn(self.qjl_dim, dim, generator=gen) / math.sqrt(
            self.qjl_dim
        )

    def quantize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize vectors with MSE + QJL correction.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Tuple of (indices, norms, qjl_signs, residual_norms):
                - indices: Lloyd-Max centroid indices, shape (..., dim)
                - norms: Vector norms, shape (..., 1)
                - qjl_signs: Sign bits of projected residuals, shape (..., qjl_dim)
                - residual_norms: Norms of quantization residuals, shape (..., 1)
        """
        # Stage 1: MSE quantization
        indices, norms = self.mse_quantizer.quantize(x)

        # Compute residual: r = x - dequant(quant(x))
        reconstructed = self.mse_quantizer.dequantize(indices, norms)
        residual = x.float() - reconstructed

        # Residual norms for scaling
        residual_norms = torch.norm(residual, dim=-1, keepdim=True)

        # Stage 2: QJL projection → store only signs
        s = self.qjl_matrix.to(x.device)
        projected = residual.reshape(-1, self.dim) @ s.T
        qjl_signs = torch.sign(projected).reshape(*x.shape[:-1], self.qjl_dim)
        # Replace zeros with +1 (ties go positive)
        qjl_signs[qjl_signs == 0] = 1.0

        return indices, norms, qjl_signs, residual_norms

    def dequantize(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct vectors from compressed representation.

        Note: Full reconstruction is approximate. For attention computation,
        use ``estimate_inner_product`` instead — it's more accurate because
        QJL corrects inner-product bias, not reconstruction bias.

        Args:
            indices: Lloyd-Max centroid indices, shape (..., dim).
            norms: Vector norms, shape (..., 1).
            qjl_signs: QJL sign bits, shape (..., qjl_dim).
            residual_norms: Residual norms, shape (..., 1).

        Returns:
            Approximately reconstructed tensor of shape (..., dim).
        """
        return self.mse_quantizer.dequantize(indices, norms)

    def estimate_inner_product(
        self,
        query: torch.Tensor,
        indices: torch.Tensor,
        norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unbiased inner product estimate between query and compressed key.

        Uses the two-stage estimator:
            <q, k> ~ <q, k_mse> + ||r|| * sqrt(pi/2) / m * <S@q, signs>

        Args:
            query: Query vectors, shape (..., dim).
            indices: Compressed key indices, shape (..., dim).
            norms: Key norms, shape (..., 1).
            qjl_signs: QJL sign bits for keys, shape (..., qjl_dim).
            residual_norms: Key residual norms, shape (..., 1).

        Returns:
            Inner product estimates, shape matching broadcast of query and key
            batch dimensions.
        """
        # MSE component: <q, k_mse>
        k_mse = self.mse_quantizer.dequantize(indices, norms)
        mse_term = (query.float() * k_mse).sum(dim=-1, keepdim=True)

        # QJL correction: ||r|| * sqrt(pi/2) / m * <S@q, signs>
        s = self.qjl_matrix.to(query.device)
        q_projected = query.float().reshape(-1, self.dim) @ s.T
        q_projected = q_projected.reshape(*query.shape[:-1], self.qjl_dim)

        qjl_correction = (q_projected * qjl_signs).sum(dim=-1, keepdim=True)
        scale = residual_norms * math.sqrt(math.pi / 2.0) / self.qjl_dim
        qjl_term = scale * qjl_correction

        return mse_term + qjl_term
