"""Production-grade compressors for transformer KV cache tensors.

Wraps the core TurboQuant quantizers to handle real model tensor shapes
([batch, heads, seq_len, head_dim]), dtype conversion, and device placement.

- TurboQuantCompressorV2: For key cache — full two-stage with QJL correction.
  Supports asymmetric_attention_scores() for computing attention directly
  from compressed keys without full dequantization.

- TurboQuantCompressorMSE: For value cache — MSE-only Stage 1 compression.
  Lighter weight, appropriate since values only need reconstruction accuracy
  (not inner-product preservation).

Reference: Section 5 of arXiv 2504.19874.

Examples:
    ```python
    key_comp = TurboQuantCompressorV2(head_dim=128, bits=3)
    val_comp = TurboQuantCompressorMSE(head_dim=128, bits=3)

    compressed_k = key_comp.compress(key_states)
    compressed_v = val_comp.compress(value_states)

    scores = key_comp.asymmetric_attention_scores(query, compressed_k)
    values = val_comp.decompress(compressed_v)
    ```

See Also:
    :mod:`turboquant_vllm.quantizer`: Core TurboQuantProd and TurboQuantMSE algorithms.
"""

from dataclasses import dataclass

import torch

from turboquant_vllm.quantizer import TurboQuantMSE, TurboQuantProd


@dataclass
class CompressedKeys:
    """Compressed key cache representation.

    Stores all components needed to compute attention scores from
    compressed keys without full dequantization.

    Attributes:
        indices (torch.Tensor): Lloyd-Max centroid indices,
            shape (batch, heads, seq, head_dim).
        norms (torch.Tensor): Vector norms, shape (batch, heads, seq, 1).
        qjl_signs (torch.Tensor): QJL sign bits,
            shape (batch, heads, seq, qjl_dim).
        residual_norms (torch.Tensor): Residual norms,
            shape (batch, heads, seq, 1).
        original_dtype (torch.dtype): Original tensor dtype for casting results.

    Examples:
        Typically created via ``TurboQuantCompressorV2.compress()``:

        ```python
        comp = TurboQuantCompressorV2(head_dim=128)
        ck = comp.compress(key_states)
        ck.indices.shape  # (batch, heads, seq, head_dim)
        ```
    """

    indices: torch.Tensor
    norms: torch.Tensor
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor
    original_dtype: torch.dtype = torch.float16


@dataclass
class CompressedValues:
    """Compressed value cache representation.

    Stores components needed to reconstruct value vectors.

    Attributes:
        indices (torch.Tensor): Lloyd-Max centroid indices,
            shape (batch, heads, seq, head_dim).
        norms (torch.Tensor): Vector norms, shape (batch, heads, seq, 1).
        original_dtype (torch.dtype): Original tensor dtype for casting results.

    Examples:
        Typically created via ``TurboQuantCompressorMSE.compress()``:

        ```python
        comp = TurboQuantCompressorMSE(head_dim=128)
        cv = comp.compress(value_states)
        cv.indices.shape  # (batch, heads, seq, head_dim)
        ```
    """

    indices: torch.Tensor
    norms: torch.Tensor
    original_dtype: torch.dtype = torch.float16


class TurboQuantCompressorV2:
    """Key cache compressor with unbiased attention score estimation.

    Uses the full two-stage TurboQuantProd algorithm to compress key
    vectors while preserving accurate inner product estimation for
    attention computation (Q·K^T).

    Attributes:
        quantizer (TurboQuantProd): Two-stage TurboQuantProd instance.
        bits (int): Total bit budget per coordinate.
        head_dim (int): Model head dimension.

    Examples:
        Compress keys and compute attention scores directly:

        ```python
        comp = TurboQuantCompressorV2(head_dim=128, bits=3)
        compressed = comp.compress(key_states)
        scores = comp.asymmetric_attention_scores(query, compressed)
        ```
    """

    def __init__(self, head_dim: int, bits: int = 3, *, seed: int = 42) -> None:
        """Initialize the key compressor.

        Args:
            head_dim: Dimension of each attention head.
            bits: Total bits per coordinate (default 3).
            seed: Random seed for reproducibility.
        """
        self.head_dim = head_dim
        self.bits = bits
        self.quantizer = TurboQuantProd(head_dim, bits, seed=seed)

    def compress(self, keys: torch.Tensor) -> CompressedKeys:
        """Compress key tensors.

        Args:
            keys: Key tensor of shape (batch, heads, seq_len, head_dim).

        Returns:
            CompressedKeys containing all components for attention estimation.
        """
        original_dtype = keys.dtype
        indices, norms, qjl_signs, residual_norms = self.quantizer.quantize(
            keys.float()
        )
        return CompressedKeys(
            indices=indices,
            norms=norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            original_dtype=original_dtype,
        )

    def decompress(self, compressed: CompressedKeys) -> torch.Tensor:
        """Reconstruct key tensors from compressed representation.

        Note: For attention, prefer ``asymmetric_attention_scores()`` which
        uses the QJL-corrected inner product estimator for better accuracy.

        Args:
            compressed: CompressedKeys from compress().

        Returns:
            Reconstructed key tensor in the original dtype.
        """
        result = self.quantizer.dequantize(
            compressed.indices,
            compressed.norms,
            compressed.qjl_signs,
            compressed.residual_norms,
        )
        return result.to(compressed.original_dtype)

    def asymmetric_attention_scores(
        self, query: torch.Tensor, compressed: CompressedKeys
    ) -> torch.Tensor:
        """Compute attention scores directly from compressed keys.

        Uses the unbiased two-stage inner product estimator rather than
        decompressing keys and computing standard dot products. This is
        both more memory-efficient and more accurate.

        .. warning:: MEMORY SCALING

            The current implementation expands tensors to
            (batch, heads, q_len, kv_len, dim) for broadcasting.
            This allocates ~5 intermediate tensors at that shape.
            For real sequence lengths (kv_len=6144, heads=32, dim=128)
            this would use ~500MB+ per call. Suitable for correctness
            testing on short sequences only.

            TODO: Replace with a chunked or fused Triton kernel for
            production use at real sequence lengths.

        Args:
            query: Query tensor, shape (batch, heads, q_len, head_dim).
            compressed: CompressedKeys from compress().

        Returns:
            Attention logits, shape (batch, heads, q_len, kv_len).
        """
        b, h, q_len, d = query.shape
        _, _, kv_len, _ = compressed.indices.shape

        # Expand query for broadcasting: (b, h, q_len, 1, d)
        # NOTE: This expand pattern is O(q_len * kv_len * dim) memory.
        # Fine for benchmarking short sequences, not for production.
        q_exp = query.float().unsqueeze(3).expand(b, h, q_len, kv_len, d)
        # Expand compressed key components: (b, h, 1, kv_len, ...)
        idx_exp = compressed.indices.unsqueeze(2).expand(b, h, q_len, kv_len, d)
        n_exp = compressed.norms.unsqueeze(2).expand(b, h, q_len, kv_len, 1)
        qjl_exp = compressed.qjl_signs.unsqueeze(2).expand(
            b, h, q_len, kv_len, self.quantizer.qjl_dim
        )
        rn_exp = compressed.residual_norms.unsqueeze(2).expand(b, h, q_len, kv_len, 1)

        scores = self.quantizer.estimate_inner_product(
            q_exp, idx_exp, n_exp, qjl_exp, rn_exp
        )
        return scores.squeeze(-1).to(query.dtype)


class TurboQuantCompressorMSE:
    """Value cache compressor with MSE-optimal reconstruction.

    Uses Stage 1 only (TurboQuantMSE) for value vectors. Values appear
    in the ``softmax(scores) @ V`` multiplication where reconstruction
    quality matters but inner-product structure does not.

    Attributes:
        quantizer (TurboQuantMSE): TurboQuantMSE instance.
        bits (int): Bits per coordinate.
        head_dim (int): Model head dimension.

    Examples:
        Compress and reconstruct value tensors:

        ```python
        comp = TurboQuantCompressorMSE(head_dim=128, bits=3)
        compressed = comp.compress(value_states)
        reconstructed = comp.decompress(compressed)
        ```
    """

    def __init__(self, head_dim: int, bits: int = 3, *, seed: int = 42) -> None:
        """Initialize the value compressor.

        Args:
            head_dim: Dimension of each attention head.
            bits: Bits per coordinate (default 3).
            seed: Random seed for reproducibility.
        """
        self.head_dim = head_dim
        self.bits = bits
        self.quantizer = TurboQuantMSE(head_dim, bits, seed=seed)

    def compress(self, values: torch.Tensor) -> CompressedValues:
        """Compress value tensors.

        Args:
            values: Value tensor of shape (batch, heads, seq_len, head_dim).

        Returns:
            CompressedValues containing indices and norms.
        """
        original_dtype = values.dtype
        indices, norms = self.quantizer.quantize(values.float())
        return CompressedValues(
            indices=indices,
            norms=norms,
            original_dtype=original_dtype,
        )

    def decompress(self, compressed: CompressedValues) -> torch.Tensor:
        """Reconstruct value tensors from compressed representation.

        Args:
            compressed: CompressedValues from compress().

        Returns:
            Reconstructed value tensor in the original dtype.
        """
        result = self.quantizer.dequantize(compressed.indices, compressed.norms)
        return result.to(compressed.original_dtype)
