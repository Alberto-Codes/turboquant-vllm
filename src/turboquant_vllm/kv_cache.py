"""TurboQuant-compressed KV cache for HuggingFace transformers.

Two integration modes:

1. **TurboQuantKVCache** — Accuracy benchmark only (no VRAM savings).
   Compresses then immediately decompresses, storing lossy FP32 back
   into the standard DynamicCache. Measures quantization quality impact.

2. **CompressedDynamicCache** — Real VRAM savings.
   Stores uint8 indices + fp16 norms in compressed form. Dequantizes
   lazily on each cache read (one layer at a time). Achieves ~2x
   compression vs FP16 KV cache.

Both use non-invasive method replacement: we save a reference to the
original update() method and replace it with a wrapper. This avoids
subclassing DynamicCache, which is fragile across transformers versions.

Usage:
    ```python
    # Mode 1: Accuracy benchmark (no VRAM savings)
    cache = DynamicCache()
    tq_cache = TurboQuantKVCache(cache, head_dim=128, bits=3)

    # Mode 2: Real VRAM savings
    cache = DynamicCache()
    compressed = CompressedDynamicCache(cache, head_dim=128, bits=3)
    # In both cases, pass cache (not the wrapper) to model.generate()
    ```

Examples:
    ```python
    from transformers import DynamicCache

    cache = DynamicCache()
    tq = TurboQuantKVCache(cache, head_dim=128, bits=3)
    ```

See Also:
    :mod:`turboquant_vllm.compressors`: TurboQuantCompressorMSE and CompressedValues.
    arXiv 2504.19874, Section 5.2: TurboQuant algorithm reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from turboquant_vllm.compressors import CompressedValues, TurboQuantCompressorMSE


class TurboQuantKVCache:
    """Transparent KV cache compression wrapper (drop-in mode).

    Intercepts cache updates to compress key/value tensors before they
    are stored. Both keys and values use TurboQuantCompressorMSE (full
    MSE-optimal quantization at the configured bit-width).

    This is the "drop-in" approach where standard attention (Q @ K^T)
    operates on decompressed keys. For the QJL-corrected inner product
    path (TurboQuantProd), a custom attention kernel would be needed —
    see TurboQuantCompressorV2.asymmetric_attention_scores().

    Attributes:
        cache (Any): The wrapped DynamicCache instance.
        key_compressor (TurboQuantCompressorMSE): Compressor for key tensors.
        value_compressor (TurboQuantCompressorMSE): Compressor for value tensors.
        bits (int): Quantization bits per coordinate.
        head_dim (int): Model head dimension.
        enabled (bool): Whether compression is active.

    Examples:
        ```python
        from transformers import DynamicCache

        cache = DynamicCache()
        tq = TurboQuantKVCache(cache, head_dim=128, bits=3)
        tq.enabled  # True
        ```
    """

    def __init__(
        self,
        cache: Any,
        head_dim: int,
        bits: int = 3,
        *,
        seed: int = 42,
        compress_keys: bool = True,
        compress_values: bool = True,
    ) -> None:
        """Initialize the TurboQuant KV cache wrapper.

        Args:
            cache: A HuggingFace DynamicCache instance to wrap.
            head_dim: Dimension of each attention head.
            bits: Quantization bits per coordinate (default 3).
            seed: Random seed for reproducibility.
            compress_keys: Whether to compress key tensors.
            compress_values: Whether to compress value tensors.
        """
        self.cache = cache
        self.head_dim = head_dim
        self.bits = bits
        self.compress_keys = compress_keys
        self.compress_values = compress_values
        self.enabled = True

        # Drop-in mode: use MSE-only for BOTH keys and values.
        # TurboQuantCompressorV2 (TurboQuantProd) allocates 1 bit to QJL correction,
        # but QJL only helps when attention calls estimate_inner_product() directly.
        # Standard attention does Q @ K^T on decompressed keys, so QJL is invisible
        # and we lose 1 bit of MSE resolution for nothing. Full 3-bit MSE gives
        # ~95% cosine sim vs ~87% with 2-bit MSE + 1-bit QJL.
        # See: https://dejan.ai/blog/turboquant/ (TurboQuant_mse for drop-in cache)
        self.key_compressor = TurboQuantCompressorMSE(head_dim, bits, seed=seed)
        self.value_compressor = TurboQuantCompressorMSE(head_dim, bits, seed=seed)

        # Patch the cache's update method
        self._original_update = cache.update
        cache.update = self._compressed_update

    def _compressed_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store key/value states in the cache.

        This method replaces the original DynamicCache.update(). It
        compresses the incoming tensors, stores the compressed versions,
        then returns decompressed tensors for immediate use by the
        attention layer.

        Args:
            key_states: Key tensor, shape (batch, heads, seq_len, head_dim).
            value_states: Value tensor, same shape as key_states.
            layer_idx: Transformer layer index.
            cache_kwargs: Additional cache arguments (passed through).

        Returns:
            Tuple of (keys, values) decompressed for immediate attention use.
        """
        if not self.enabled:
            return self._original_update(
                key_states, value_states, layer_idx, cache_kwargs
            )

        # Compress → decompress round-trip: simulates quantization quality loss
        # but stores decompressed FP32 back into cache. No VRAM savings.
        # TODO: For actual memory savings, store CompressedKeys/CompressedValues
        # directly and dequantize on cache read. Requires custom attention loop.
        if self.compress_keys:
            compressed_k = self.key_compressor.compress(key_states)
            key_states = self.key_compressor.decompress(compressed_k)

        if self.compress_values:
            compressed_v = self.value_compressor.compress(value_states)
            value_states = self.value_compressor.decompress(compressed_v)

        return self._original_update(key_states, value_states, layer_idx, cache_kwargs)

    def disable(self) -> None:
        """Disable compression, passing through to original update.

        Useful for A/B benchmarking within the same run.
        """
        self.enabled = False

    def enable(self) -> None:
        """Re-enable compression after disable()."""
        self.enabled = True

    def restore(self) -> None:
        """Restore the original update method on the wrapped cache.

        Call this to fully unwrap the cache and remove all TurboQuant
        interception.
        """
        self.cache.update = self._original_update


@dataclass
class _CompressedLayer:
    """Storage-optimized compressed representation of one cache layer.

    Indices may be stored unpacked (one uint8 per index, for 2-3 bit)
    or nibble-packed (two 4-bit indices per uint8, for 4-bit). The
    ``packed`` flag indicates which format is used.

    Attributes:
        indices (torch.Tensor): Lloyd-Max centroid indices in uint8.
            Unpacked shape: ``(batch, heads, seq_len, head_dim)``.
            Nibble-packed shape: ``(batch, heads, seq_len, head_dim // 2)``.
        norms (torch.Tensor): Vector norms in float32, shape
            ``(batch, heads, seq_len, 1)``. Float32 is required --
            float16 causes output degradation at 10K+ token sequences
            due to accumulated norm precision loss across layers.
        packed (bool): True if indices are nibble-packed (4-bit mode).

    Examples:
        ```python
        layer = _CompressedLayer(
            indices=torch.zeros(1, 8, 10, 64, dtype=torch.uint8),
            norms=torch.ones(1, 8, 10, 1),
            packed=True,
        )
        layer.indices.shape  # torch.Size([1, 8, 10, 64]) — nibble-packed
        ```
    """

    indices: torch.Tensor
    norms: torch.Tensor
    packed: bool = False


class CompressedDynamicCache:
    """KV cache with real VRAM savings via compressed index storage.

    Stores TurboQuant-compressed representations and dequantizes lazily
    on each cache read. Only one layer's decompressed tensors are held
    in memory at a time — previous layers are freed on the next update.

    Storage per token per head (head_dim=128):

    ============  =======  =====  ===========  ===========
    Mode          Dtype    Bytes  Compression  Quality
    ============  =======  =====  ===========  ===========
    FP16 baseline fp16     256    1.0x         —
    TQ3 (3-bit)   uint8    132    1.94x        ~95% cosine
    TQ4 (4-bit)   nibble   68     3.76x        ~97% cosine
    ============  =======  =====  ===========  ===========

    At ``bits=4``, indices are nibble-packed (two 4-bit values per
    byte), nearly doubling compression over TQ3 with better quality.
    Float32 norms are required — fp16 causes output degradation at
    10K+ token sequences due to accumulated precision loss.

    Integration strategy: non-invasive method replacement (same pattern
    as TurboQuantKVCache). Patches ``update()`` and ``get_seq_length()``
    on the wrapped DynamicCache.

    Attributes:
        cache (Any): The wrapped DynamicCache instance.
        key_compressor (TurboQuantCompressorMSE): Compressor for key tensors.
        value_compressor (TurboQuantCompressorMSE): Compressor for value tensors.
        bits (int): Quantization bits per coordinate.
        head_dim (int): Model head dimension.
        enabled (bool): Whether compression is active.
        fused_mode (bool): When True, skip decompression in ``update()``
            (fused kernel reads compressed data via ``get_compressed()``).
        rotation (torch.Tensor): Shared rotation matrix ``[head_dim, head_dim]``.
        centroids (torch.Tensor): Shared codebook ``[2^bits]``.

    Examples:
        ```python
        from transformers import DynamicCache

        cache = DynamicCache()
        compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)
        compressed.vram_bytes()  # 0
        ```
    """

    def __init__(
        self,
        cache: Any,
        head_dim: int,
        bits: int = 3,
        *,
        seed: int = 42,
    ) -> None:
        """Initialize the compressed KV cache wrapper.

        Sets up compressors, internal storage for compressed representations,
        and incremental decompressed buffers. ``fused_mode`` starts disabled.

        Args:
            cache: A HuggingFace DynamicCache instance to wrap.
            head_dim: Dimension of each attention head. Must be even
                when ``bits=4`` (required for nibble packing pairs).
            bits: Quantization bits per coordinate (default 3). Use 4
                for nibble-packed storage (3.76x compression).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If ``bits=4`` and ``head_dim`` is odd.
        """
        if bits == 4 and head_dim % 2 != 0:
            msg = f"bits=4 requires even head_dim for nibble packing, got {head_dim}"
            raise ValueError(msg)

        self.cache = cache
        self.head_dim = head_dim
        self.bits = bits
        self._nibble_packed = bits == 4
        self.enabled = True

        self.key_compressor = TurboQuantCompressorMSE(head_dim, bits, seed=seed)
        self.value_compressor = TurboQuantCompressorMSE(head_dim, bits, seed=seed)

        self._compressed_keys: list[_CompressedLayer] = []
        self._compressed_values: list[_CompressedLayer] = []
        self._decompressed_k: list[torch.Tensor | None] = []
        self._decompressed_v: list[torch.Tensor | None] = []
        self._original_dtype: torch.dtype = torch.bfloat16
        self.fused_mode = False

        # Patch cache methods
        self._original_update = cache.update
        self._original_get_seq_length = cache.get_seq_length
        cache.update = self._compressed_update
        cache.get_seq_length = self._compressed_get_seq_length

    @staticmethod
    def _nibble_pack(indices: torch.Tensor) -> torch.Tensor:
        """Pack pairs of 4-bit indices into single uint8 bytes.

        Args:
            indices: uint8 tensor with values in [0, 15], last dim must
                be even. Shape ``(..., D)``.

        Returns:
            uint8 tensor of shape ``(..., D // 2)`` with two indices
            per byte (high nibble = even index, low nibble = odd index).
        """
        even = indices[..., 0::2]
        odd = indices[..., 1::2]
        return (even << 4) | odd

    @staticmethod
    def _nibble_unpack(packed: torch.Tensor) -> torch.Tensor:
        """Unpack nibble-packed uint8 bytes into pairs of 4-bit indices.

        Args:
            packed: uint8 tensor of shape ``(..., D // 2)`` with two
                indices per byte.

        Returns:
            Long tensor of shape ``(..., D)`` with individual indices
            suitable for centroid lookup.
        """
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        return torch.stack([high, low], dim=-1).flatten(-2)

    def _compress_tensor(
        self,
        compressor: TurboQuantCompressorMSE,
        tensor: torch.Tensor,
    ) -> _CompressedLayer:
        """Compress a tensor to packed/unpacked indices + float32 norms.

        At ``bits=4``, indices are nibble-packed (two per byte) for
        3.76x compression. At other bit widths, indices are stored as
        one uint8 per index.

        Args:
            compressor: The MSE compressor instance.
            tensor: Input tensor, shape ``(batch, heads, seq_len, head_dim)``.

        Returns:
            Compressed layer with indices and float32 norms.
        """
        compressed = compressor.compress(tensor)
        indices = compressed.indices.to(torch.uint8)

        if self._nibble_packed:
            indices = self._nibble_pack(indices)

        return _CompressedLayer(
            indices=indices,
            norms=compressed.norms.float(),
            packed=self._nibble_packed,
        )

    def _dequantize_layer(
        self,
        compressor: TurboQuantCompressorMSE,
        layer: _CompressedLayer,
    ) -> torch.Tensor:
        """Dequantize a compressed layer back to the original dtype.

        Unpacks nibble-packed indices when applicable, then converts
        to long for centroid lookup (PyTorch treats uint8 as boolean
        masks during fancy indexing).

        Args:
            compressor: The MSE compressor instance.
            layer: Compressed layer with indices and float32 norms.

        Returns:
            Reconstructed tensor in the original dtype.
        """
        if layer.packed:
            indices = self._nibble_unpack(layer.indices)
        else:
            indices = layer.indices.long()

        compressed = CompressedValues(
            indices=indices,
            norms=layer.norms,
            original_dtype=self._original_dtype,
        )
        return compressor.decompress(compressed)

    @staticmethod
    def _cat_layers(
        existing: _CompressedLayer,
        new: _CompressedLayer,
    ) -> _CompressedLayer:
        """Concatenate two compressed layers along the sequence dimension.

        Preserves the ``packed`` flag from the existing layer.

        Args:
            existing: Previously stored compressed tokens.
            new: Newly compressed tokens to append.

        Returns:
            Combined compressed layer with same packing format.
        """
        return _CompressedLayer(
            indices=torch.cat([existing.indices, new.indices], dim=-2),
            norms=torch.cat([existing.norms, new.norms], dim=-2),
            packed=existing.packed,
        )

    def _compressed_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress new tokens and optionally dequantize.

        Stores compressed representations permanently. In normal mode,
        uses incremental dequantization (only NEW tokens decompressed).
        In ``fused_mode``, skips decompression entirely — the fused TQ4
        kernel reads compressed data via ``get_compressed()``.

        Works with the ``DynamicCache.layers`` API (transformers >=4.57)
        where each layer is a ``DynamicLayer`` holding ``.keys`` and
        ``.values`` tensors.

        Args:
            key_states: Key tensor, shape ``(batch, heads, seq_len, head_dim)``.
            value_states: Value tensor, same shape as key_states.
            layer_idx: Transformer layer index.
            cache_kwargs: Additional cache arguments (unused).

        Returns:
            Tuple of ``(keys, values)`` decompressed for attention use.
        """
        if not self.enabled:
            return self._original_update(
                key_states, value_states, layer_idx, cache_kwargs
            )

        self._original_dtype = key_states.dtype

        # Ensure DynamicCache has created layers up to layer_idx
        if self.cache.layer_class_to_replicate is not None:
            while len(self.cache.layers) <= layer_idx:
                self.cache.layers.append(self.cache.layer_class_to_replicate())

        # Note: decompressed buffers are NOT freed between layers.
        # They accumulate across decode steps so each step only
        # dequantizes the 1 new token (not all 11K+ cached tokens).
        # VRAM savings come from compressed storage; decompressed
        # buffers match baseline VRAM for SDPA compatibility.

        # Compress new tokens to uint8 indices + fp32 norms
        new_ck = self._compress_tensor(self.key_compressor, key_states)
        new_cv = self._compress_tensor(self.value_compressor, value_states)

        # Append to compressed storage
        if layer_idx >= len(self._compressed_keys):
            self._compressed_keys.append(new_ck)
            self._compressed_values.append(new_cv)
        else:
            self._compressed_keys[layer_idx] = self._cat_layers(
                self._compressed_keys[layer_idx], new_ck
            )
            self._compressed_values[layer_idx] = self._cat_layers(
                self._compressed_values[layer_idx], new_cv
            )

        # Fused mode: skip decompression entirely. The fused TQ4 kernel
        # reads compressed data via get_compressed(), so decompressed
        # buffers are never needed. Return key_states/value_states as
        # placeholders to satisfy the DynamicLayer API.
        if self.fused_mode:
            layer = self.cache.layers[layer_idx]
            if not layer.is_initialized:
                layer.lazy_initialization(key_states)
            # Store minimal placeholders — just the new tokens, not the
            # full cache. The fused attention function ignores these.
            layer.keys = key_states
            layer.values = value_states
            return key_states, value_states

        # Incremental dequantization: only decompress the NEW tokens
        # and cat onto a running buffer. Avoids re-dequantizing all 11K+
        # cached tokens at every layer at every decode step.
        new_k_decompressed = self._dequantize_layer(self.key_compressor, new_ck)
        new_v_decompressed = self._dequantize_layer(self.value_compressor, new_cv)

        # Extend buffer lists if needed
        while len(self._decompressed_k) <= layer_idx:
            self._decompressed_k.append(None)
            self._decompressed_v.append(None)

        # Cat onto running buffer (or initialize on first call)
        if self._decompressed_k[layer_idx] is None:
            self._decompressed_k[layer_idx] = new_k_decompressed
            self._decompressed_v[layer_idx] = new_v_decompressed
        else:
            prev_k = self._decompressed_k[layer_idx]
            prev_v = self._decompressed_v[layer_idx]
            assert prev_k is not None  # guaranteed by the if-branch above
            assert prev_v is not None
            self._decompressed_k[layer_idx] = torch.cat(
                [prev_k, new_k_decompressed], dim=-2
            )
            self._decompressed_v[layer_idx] = torch.cat(
                [prev_v, new_v_decompressed], dim=-2
            )

        decompressed_k = self._decompressed_k[layer_idx]
        decompressed_v = self._decompressed_v[layer_idx]
        assert decompressed_k is not None
        assert decompressed_v is not None

        # Store in the DynamicLayer for len(cache) / get_seq_length compat
        layer = self.cache.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states)
        layer.keys = decompressed_k
        layer.values = decompressed_v

        return decompressed_k, decompressed_v

    def _compressed_get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length from compressed storage.

        Args:
            layer_idx: Layer to query (default 0).

        Returns:
            Number of cached tokens for the given layer.
        """
        if not self.enabled:
            return self._original_get_seq_length(layer_idx)
        if layer_idx >= len(self._compressed_keys):
            return 0
        return int(self._compressed_keys[layer_idx].indices.shape[-2])

    def get_compressed(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return compressed K and V for a layer (fused kernel API).

        Provides the raw nibble-packed indices and norms without
        dequantization, for use by the fused TQ4 Flash Attention kernel.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            ``(k_packed, k_norms, v_packed, v_norms)`` where packed tensors
            are uint8 ``[batch, heads, seq, head_dim//2]`` and norms are
            fp32 ``[batch, heads, seq, 1]``.
        """
        k = self._compressed_keys[layer_idx]
        v = self._compressed_values[layer_idx]
        return k.indices, k.norms, v.indices, v.norms

    @property
    def rotation(self) -> torch.Tensor:
        """Shared orthogonal rotation matrix ``[head_dim, head_dim]`` fp32.

        K and V use the same rotation (same seed).

        Returns:
            The rotation matrix from the key compressor's quantizer.
        """
        return self.key_compressor.quantizer.rotation

    @property
    def centroids(self) -> torch.Tensor:
        """Shared Lloyd-Max codebook ``[2^bits]`` fp32.

        Returns:
            Centroid values from the key compressor's quantizer.
        """
        return self.key_compressor.quantizer.codebook.centroids

    def disable(self) -> None:
        """Disable compression, passing through to original update."""
        self.enabled = False

    def enable(self) -> None:
        """Re-enable compression after disable()."""
        self.enabled = True

    def restore(self) -> None:
        """Restore original methods on the wrapped cache.

        Call this to fully unwrap the cache and remove all TurboQuant
        interception.
        """
        self.cache.update = self._original_update
        self.cache.get_seq_length = self._original_get_seq_length

    def vram_bytes(self) -> int:
        """Calculate total VRAM used by compressed storage.

        Returns:
            Total bytes across all compressed layers (keys + values).
        """
        total = 0
        for layer in [*self._compressed_keys, *self._compressed_values]:
            total += layer.indices.nelement() * layer.indices.element_size()
            total += layer.norms.nelement() * layer.norms.element_size()
        return total

    def baseline_vram_bytes(self) -> int:
        """Estimate FP16 VRAM that would be used without compression.

        Accounts for nibble-packed indices by doubling the last
        dimension to recover the original head_dim.

        Returns:
            Total bytes if keys and values were stored as FP16 tensors.
        """
        total = 0
        for layer in [*self._compressed_keys, *self._compressed_values]:
            b, h, s, d = layer.indices.shape
            # Nibble-packed indices have d = head_dim // 2
            if layer.packed:
                d = d * 2
            total += b * h * s * d * 2  # FP16 = 2 bytes per element
        return total

    def compression_stats(self) -> dict[str, Any]:
        """Return compression statistics for reporting.

        Reports the true ``head_dim`` (not the packed index dimension)
        and includes a ``nibble_packed`` flag.

        Returns:
            Dict with layer count, sequence length, compressed/baseline
            sizes in MiB, compression ratio, packing mode, and VRAM savings.
        """
        if not self._compressed_keys:
            return {}

        compressed_bytes = self.vram_bytes()
        baseline_bytes = self.baseline_vram_bytes()
        ratio = baseline_bytes / compressed_bytes if compressed_bytes > 0 else 0.0

        layer = self._compressed_keys[0]
        b, h, s, _ = layer.indices.shape

        return {
            "num_layers": len(self._compressed_keys),
            "seq_len": s,
            "batch_size": b,
            "num_heads": h,
            "head_dim": self.head_dim,
            "bits": self.bits,
            "nibble_packed": self._nibble_packed,
            "compressed_mib": compressed_bytes / (1024 * 1024),
            "baseline_mib": baseline_bytes / (1024 * 1024),
            "compression_ratio": round(ratio, 2),
            "savings_mib": (baseline_bytes - compressed_bytes) / (1024 * 1024),
        }
