"""TQ4 compressed KV cache attention backend for vLLM.

Phase 3c: Packed TQ4 cache layout with real VRAM savings.

The KV cache is stored as uint8 bytes in a packed TQ4 format (68 bytes
per token per head per K/V = 136 bytes total vs 512 bytes FP16 = 3.76x
compression).  Buffer allocation uses a custom ``TQ4FullAttentionSpec``
that overrides ``page_size_bytes`` so the block allocator provisions
3.76x more blocks in the same VRAM budget.  Each ``forward()`` call
decompresses the relevant blocks to FP16 and delegates to Flash Attention.

Implementation phases:
    3a (done): Passthrough skeleton -- validated plugin wiring.
    3b (done): Compress-decompress round-trip in standard FP16 cache.
    3c (this): Packed uint8 cache with real VRAM savings.
    3d: Production benchmark against vLLM baseline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

if TYPE_CHECKING:
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionImplBase,
        AttentionMetadataBuilder,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TQ4 constants
# ---------------------------------------------------------------------------

TQ4_BITS = 4
TQ4_SEED = 42

# Per-token per-head storage: head_dim/2 bytes (nibble-packed) + 4 bytes (fp32 norm)
# For head_dim=128: 64 + 4 = 68 bytes vs 256 bytes FP16 = 3.76x compression
TQ4_NORM_BYTES = 4  # fp32


def _tq4_bytes_per_token(head_dim: int) -> int:
    """Packed byte count for one token, one KV head, one of K or V.

    Returns:
        Byte count: ``head_dim // 2`` (nibble-packed indices) + 4 (fp32 norm).
    """
    return head_dim // 2 + TQ4_NORM_BYTES


def _tq4_bytes_per_token_kv(head_dim: int) -> int:
    """Total packed bytes per token per KV head (K + V combined)."""
    return 2 * _tq4_bytes_per_token(head_dim)


# ---------------------------------------------------------------------------
# Fused paged decode feature gate (Story 6.3)
# ---------------------------------------------------------------------------

# Try importing the fused kernel at module load time.  If Triton is missing
# or the kernel's JIT compilation fails at import, the flag stays False and
# the decompress-all path is used unconditionally.
_fused_paged_kernel_available = False
_fused_paged_tq4_decode_fn = None
try:
    from turboquant_vllm.triton.fused_paged_tq4_attention import (
        fused_paged_tq4_decode as _fused_paged_tq4_decode_fn,
    )

    _fused_paged_kernel_available = True
except (ImportError, RuntimeError) as exc:
    logger.info("Fused paged TQ4 decode kernel unavailable: %s", exc)


def _parse_fused_paged_env() -> bool:
    """Parse ``TQ4_USE_FUSED_PAGED`` environment variable.

    Returns:
        ``True`` when the env var is set to a truthy value
        (``"1"``, ``"true"``, ``"yes"``; case-insensitive).
        ``False`` for everything else including absent.
    """
    return os.environ.get("TQ4_USE_FUSED_PAGED", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# KV cache spec (3c.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TQ4FullAttentionSpec(FullAttentionSpec):
    """KV cache spec with TQ4 packed page size.

    Overrides ``real_page_size_bytes`` so the block allocator provisions
    buffers sized for the packed TQ4 format (3.76x smaller than FP16).
    Follows the same pattern as ``MLAAttentionSpec`` which overrides
    page size for the 656-byte FlashMLA format.
    """

    @property
    def real_page_size_bytes(self) -> int:  # noqa: D102
        return (
            self.block_size
            * self.num_kv_heads
            * _tq4_bytes_per_token_kv(self.head_size)
        )


# ---------------------------------------------------------------------------
# Backend (3c.2 - 3c.3)
# ---------------------------------------------------------------------------


class TQ4MetadataBuilder(FlashAttentionMetadataBuilder):
    """Metadata builder for TQ4 with single-token-decode CUDA graph support.

    TQ4 only supports CUDA graphs for single-token decode (the prefill
    path has dynamic allocations).  Inherits all metadata-building logic
    from Flash Attention; only the CUDA graph support level differs.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: object,
        kv_cache_spec: object,
    ) -> AttentionCGSupport:
        """Report single-token-decode CUDA graph support (D7 mod 3)."""
        from vllm.v1.attention.backend import AttentionCGSupport

        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


class TQ4AttentionBackend(FlashAttentionBackend):
    """TQ4 compressed KV cache attention backend.

    Phase 3c: packed uint8 cache layout with real VRAM savings.
    The cache stores nibble-packed TQ4 indices + fp32 norms as raw bytes.
    ``get_kv_cache_shape()`` returns a 3D ``(NB, BS, bytes_per_token)``
    layout matching the packed format.
    """

    forward_includes_kv_cache_update = True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """Required for VLMs like Molmo2 with bidirectional visual tokens."""
        return True

    @staticmethod
    def get_name() -> str:
        """Must return ``"CUSTOM"`` to match ``AttentionBackendEnum.CUSTOM``."""
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase]:
        """Return :class:`TQ4AttentionImpl`."""
        return TQ4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        """Return :class:`TQ4MetadataBuilder` for CUDA graph support."""
        return TQ4MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Packed TQ4 cache: ``(num_blocks, block_size, total_bytes)``.

        The last dimension packs K and V data for all heads as raw bytes:
        ``[K_indices | K_norms | V_indices | V_norms]``.
        """
        total_bytes = num_kv_heads * _tq4_bytes_per_token_kv(head_size)
        return (num_blocks, block_size, total_bytes)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Raise to trigger identity fallback in reshape.

        The inherited FlashAttentionBackend returns a 5-element stride
        order for the standard ``(2, NB, BS, H, D)`` shape. Our 3D
        packed layout ``(NB, BS, total_bytes)`` needs identity ordering.
        Raising ``NotImplementedError`` triggers the fallback in
        ``_reshape_kv_cache_tensors`` (same pattern as FlashMLA which
        does not implement this method at all).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Attention implementation (3c.4 - 3c.5)
# ---------------------------------------------------------------------------


class TQ4AttentionImpl(FlashAttentionImpl):
    """TQ4 attention: compress -> store -> decompress -> Flash Attention.

    Phase 3c: stores packed TQ4 bytes in a uint8 cache for real VRAM
    savings.  Each ``forward()`` call:

    1. Compresses incoming K/V tokens to TQ4 packed bytes.
    2. Scatter-writes packed bytes to the uint8 cache via ``slot_mapping``.
    3. Decompresses the full cache to FP16 for Flash Attention.
    4. Calls ``flash_attn_varlen_func`` directly with the FP16 data.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize TQ4 attention with compression primitives."""
        super().__init__(*args, **kwargs)

        # Use attributes set by super().__init__()
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads

        # TQ4 compression primitives (deterministic from seed, shared across layers)
        quantizer = TurboQuantMSE(head_size, TQ4_BITS, seed=TQ4_SEED)

        # Eagerly move primitives to the target device (D7 mod 5).
        # FlashAttentionImpl.__init__ doesn't expose device, but
        # vLLM's global config is available during model construction.
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        device = (
            vllm_config.device_config.device
            if vllm_config is not None
            else torch.device("cpu")
        )

        self._tq4_rotation = quantizer.rotation.to(device)  # (D, D) fp32
        self._tq4_centroids = quantizer.codebook.centroids.to(device)  # (16,) fp32
        self._tq4_boundaries = quantizer.codebook.boundaries.to(device)  # (15,) fp32
        # Pre-split rotation.T for fused compress kernel (contiguous loads)
        rot_t = quantizer.rotation.T.contiguous()
        self._tq4_rot_T_even = rot_t[:, 0::2].contiguous().to(device)  # (D, D//2) fp32
        self._tq4_rot_T_odd = rot_t[:, 1::2].contiguous().to(device)  # (D, D//2) fp32

        # Byte layout offsets within the last dimension of the packed cache.
        # Layout: [K_indices(H*D/2) | K_norms(H*4) | V_indices(H*D/2) | V_norms(H*4)]
        half_D = head_size // 2
        self._half_D = half_D
        self._k_idx_end = num_kv_heads * half_D
        self._k_norm_end = self._k_idx_end + num_kv_heads * TQ4_NORM_BYTES
        self._v_idx_end = self._k_norm_end + num_kv_heads * half_D
        self._total_bytes = self._v_idx_end + num_kv_heads * TQ4_NORM_BYTES

        # CUDA graph scratch buffers (D7 mod 2) — lazy-allocated on first
        # forward() from kv_cache.shape, which is stable for engine lifetime.
        # First forward runs during vLLM warmup, before graph capture.
        self._cg_buffers_ready = False

        # Fused paged decode feature gate (Story 6.3, AC 1+6).
        # Explicit opt-in via TQ4_USE_FUSED_PAGED env var AND successful
        # kernel import.  Default is False (decompress-all path).
        self._fused_paged_available = (
            _parse_fused_paged_env() and _fused_paged_kernel_available
        )

        # Buffer downsizing source: scheduler knows its own max prefill length.
        # Fallback 2048 matches vLLM's default max_num_batched_tokens for
        # chunked prefill.
        self._max_prefill_len = (
            vllm_config.scheduler_config.max_num_batched_tokens
            if vllm_config is not None
            else 2048
        )

        logger.info(
            "TQ4AttentionImpl: %d KV heads, head_size=%d, "
            "%d bytes/token (%.2fx compression vs FP16)",
            num_kv_heads,
            head_size,
            self._total_bytes,
            (2 * num_kv_heads * head_size * 2) / self._total_bytes,
        )
        logger.info(
            "Fused paged TQ4 decode: %s",
            "enabled" if self._fused_paged_available else "disabled",
        )

    def _init_cg_buffers(
        self, kv_cache: torch.Tensor, compute_dtype: torch.dtype
    ) -> None:
        """Pre-allocate CUDA graph scratch buffers from kv_cache shape.

        Called once during vLLM warmup (first forward), before CUDA graph
        capture.  Uses max-size + slicing (D7 pattern), NOT per-batch
        allocations.

        Args:
            kv_cache: ``(num_blocks, block_size, total_bytes)`` uint8 cache.
            compute_dtype: Model compute dtype (e.g. ``torch.bfloat16``).
        """
        num_blocks, block_size, _ = kv_cache.shape
        max_tokens = num_blocks * block_size
        device = kv_cache.device
        H = self.num_kv_heads
        D = self.head_size

        # Decompress buffers: when fused decode is available, decode skips
        # decompression entirely — only prefill needs these buffers.
        # Downsize from (max_blocks*block_size, H, D) to (max_prefill_len, H, D).
        if self._fused_paged_available:
            decompress_tokens = self._max_prefill_len
            buffer_source = "max_prefill_len"
        else:
            decompress_tokens = max_tokens
            buffer_source = "full_cache"

        self._cg_decompress_k = torch.empty(
            decompress_tokens, H, D, dtype=compute_dtype, device=device
        )
        self._cg_decompress_v = torch.empty_like(self._cg_decompress_k)

        # Compress output buffers for one decode step (single token)
        half_D = self._half_D
        self._cg_compress_packed = torch.empty(
            1, H, half_D, dtype=torch.uint8, device=device
        )
        self._cg_compress_norms = torch.empty(
            1, H, 1, dtype=torch.float32, device=device
        )

        # Q rotation buffer for decode (single token, fp32 for precision)
        self._cg_q_rot = torch.empty(
            1, self.num_heads, D, dtype=torch.float32, device=device
        )

        # Q rotation cast buffer (compute dtype for Flash Attention input)
        self._cg_q_rot_cast = torch.empty(
            1, self.num_heads, D, dtype=compute_dtype, device=device
        )

        # Compress row assembly buffer for _compress_and_store
        self._cg_compress_row = torch.empty(
            1, self._total_bytes, dtype=torch.uint8, device=device
        )

        self._cg_buffers_ready = True
        dtype_bytes = self._cg_decompress_k.element_size()
        logger.info(
            "TQ4 CUDA graph buffers allocated: decompress=%s "
            "(fused_paged=%s, tokens=%d, source=%s), "
            "decompress=2×%.1f MiB, compress+row+q_rot=%.1f KiB",
            self._cg_decompress_k.shape,
            self._fused_paged_available,
            decompress_tokens,
            buffer_source,
            decompress_tokens * H * D * dtype_bytes / (1024 * 1024),
            (half_D * H + 4 * H + self.num_heads * D * 4 + self._total_bytes) / 1024,
        )

    # ----- packed cache operations (3c.4 - 3c.5) -----

    def _compress_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        compress_out: tuple[torch.Tensor, torch.Tensor] | None = None,
        row_out: torch.Tensor | None = None,
    ) -> None:
        """Compress K/V and scatter-write TQ4 bytes to packed cache.

        Args:
            key: ``(N, H, D)`` new key tokens.
            value: ``(N, H, D)`` new value tokens.
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            slot_mapping: ``(num_actual_tokens,)`` flat slot indices.
            compress_out: Optional pre-allocated ``(packed, norms)`` buffers
                for tq4_compress (D7 CUDA graph decode path).
            row_out: Optional pre-allocated row assembly buffer ``(N, total_bytes)``
                uint8 (D7 CUDA graph decode path).
        """
        N = key.shape[0]
        H = self.num_kv_heads

        # Build packed byte row per token: [K_idx | K_norm | V_idx | V_norm]
        # When compress_out is shared between K and V, we must copy K's
        # result into the row before V overwrites the shared buffer.
        row = (
            row_out[:N]
            if row_out is not None
            else torch.empty(N, self._total_bytes, dtype=torch.uint8, device=key.device)
        )

        k_packed, k_norms = tq4_compress(
            key,
            self._tq4_rot_T_even,
            self._tq4_rot_T_odd,
            self._tq4_boundaries,
            out=compress_out,
        )
        row[:, : self._k_idx_end] = k_packed.reshape(N, -1)
        row[:, self._k_idx_end : self._k_norm_end] = (
            k_norms.reshape(N, H).contiguous().view(torch.uint8)
        )

        v_packed, v_norms = tq4_compress(
            value,
            self._tq4_rot_T_even,
            self._tq4_rot_T_odd,
            self._tq4_boundaries,
            out=compress_out,
        )
        row[:, self._k_norm_end : self._v_idx_end] = v_packed.reshape(N, -1)
        row[:, self._v_idx_end :] = v_norms.reshape(N, H).contiguous().view(torch.uint8)

        # Scatter-write to flat cache using slot_mapping
        num_actual = slot_mapping.shape[0]
        flat_cache = kv_cache.view(-1, self._total_bytes)
        flat_cache[slot_mapping[:num_actual]] = row[:num_actual]

    def _decompress_cache(
        self,
        kv_cache: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        apply_rotation: bool = True,
        out_k: torch.Tensor | None = None,
        out_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress packed uint8 cache -> key_cache, value_cache.

        Uses the fused Triton kernel (Phase 3c.8) for decompress.  When
        ``apply_rotation=False``, output stays in rotated space and the
        caller must pre-rotate Q by ``Pi^T`` and post-rotate the output
        by ``Pi``.  Default ``True`` applies unrotation for backward
        compatibility with tests.

        Args:
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            compute_dtype: Output dtype (e.g., ``torch.bfloat16``).
            apply_rotation: If ``True`` (default), apply unrotation to
                return tensors in original space.  ``False`` returns
                rotated-space tensors for the optimized forward path.
            out_k: Optional pre-allocated ``(max_tokens, H, D)`` buffer for
                decompressed keys (D7 CUDA graph decode path).
            out_v: Optional pre-allocated ``(max_tokens, H, D)`` buffer for
                decompressed values (D7 CUDA graph decode path).

        Returns:
            key_cache: ``(NB, BS, H, D)`` in ``compute_dtype``.
            value_cache: ``(NB, BS, H, D)`` in ``compute_dtype``.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        half_D = self._half_D
        D = self.head_size

        flat = kv_cache.reshape(NB * BS, self._total_bytes)

        # Extract K regions
        k_packed = flat[:, : self._k_idx_end].contiguous().reshape(-1, H, half_D)
        k_norms = (
            flat[:, self._k_idx_end : self._k_norm_end]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )

        # Extract V regions
        v_packed = (
            flat[:, self._k_norm_end : self._v_idx_end]
            .contiguous()
            .reshape(-1, H, half_D)
        )
        v_norms = (
            flat[:, self._v_idx_end :]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )

        # Fused Triton decompress (no rotation applied)
        key_out = tq4_decompress(
            k_packed, k_norms, self._tq4_centroids, compute_dtype, out=out_k
        )
        value_out = tq4_decompress(
            v_packed, v_norms, self._tq4_centroids, compute_dtype, out=out_v
        )

        # Optionally unrotate (backward compat for tests; forward() skips this)
        if apply_rotation:
            key_out = (key_out.float() @ self._tq4_rotation).to(compute_dtype)
            value_out = (value_out.float() @ self._tq4_rotation).to(compute_dtype)

        return key_out.reshape(NB, BS, H, D), value_out.reshape(NB, BS, H, D)

    # ----- TQ4 encode / decode helpers -----

    def _tq4_decode(
        self, query, key, value, kv_cache, attn_metadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode path: compress, rotate Q, decompress using pre-allocated buffers."""
        if key is not None and value is not None:
            self._compress_and_store(
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
                compress_out=(self._cg_compress_packed, self._cg_compress_norms),
                row_out=self._cg_compress_row,
            )

        q_slice = query[: attn_metadata.num_actual_tokens]
        q_rot_buf = self._cg_q_rot[:1]
        torch.matmul(q_slice.float(), self._tq4_rotation.T, out=q_rot_buf)
        self._cg_q_rot_cast[:1].copy_(q_rot_buf)

        key_cache, value_cache = self._decompress_cache(
            kv_cache,
            query.dtype,
            apply_rotation=False,
            out_k=self._cg_decompress_k,
            out_v=self._cg_decompress_v,
        )
        return self._cg_q_rot_cast[:1], key_cache, value_cache

    def _tq4_prefill(
        self, query, key, value, kv_cache, attn_metadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill path: compress, rotate Q, decompress with dynamic allocation."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        if kv_cache is not None and key is not None and value is not None:
            self._compress_and_store(key, value, kv_cache, attn_metadata.slot_mapping)

        q_slice = query[:num_actual_tokens]
        q_rot = (q_slice.float() @ self._tq4_rotation.T).to(q_slice.dtype)

        key_cache, value_cache = self._decompress_cache(
            kv_cache,
            query.dtype,
            apply_rotation=False,
        )
        return q_rot, key_cache, value_cache

    # ----- fused decode path (Story 6.3) -----

    def _fused_decode_path(
        self, query, key, value, kv_cache, attn_metadata, output
    ) -> torch.Tensor:
        """Fused paged decode: compress → fused kernel (in-place attention + rotation).

        Replaces the decompress-all → FlashAttn → post-rotate pipeline
        with a single ``fused_paged_tq4_decode()`` call.  The fused
        wrapper handles Q pre-rotation, in-tile TQ4 decompression,
        attention scoring, and output post-rotation.
        """
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Step 1: Compress and store new tokens (same as decompress-all path)
        if key is not None and value is not None:
            self._compress_and_store(
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
                compress_out=(self._cg_compress_packed, self._cg_compress_norms),
                row_out=self._cg_compress_row,
            )

        # Step 2: Fused kernel — handles Q rotation, paged decompression,
        # attention, and output post-rotation in one call.
        q_slice = query[:num_actual_tokens]
        # Guaranteed non-None: _fused_paged_available requires successful import.
        assert _fused_paged_tq4_decode_fn is not None
        _fused_paged_tq4_decode_fn(
            q_slice,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self._tq4_centroids,
            self._tq4_rotation,
            self.num_kv_heads,
            self.head_size,
            kv_cache.shape[1],  # block_size
            self.scale,
            out=output[:num_actual_tokens],
        )

        return output

    # ----- forward -----

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        """TQ4 attention: compress -> store -> pre-rotate Q -> decompress -> FA -> post-rotate.

        Phase 3c.8: Uses fused Triton decompress (no rotation). The
        rotation is applied to Q before attention and to the output
        after, saving O(cache_len) matmuls per decode step.
        """
        assert output is not None

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported with TQ4 backend"
            )

        # Profiling mode
        if attn_metadata is None:
            output.zero_()
            return output

        # Encoder attention: no TQ4, delegate to parent
        # (VIT uses a separate backend, but guard just in case)
        from vllm.v1.attention.backend import AttentionType

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[: attn_metadata.num_actual_tokens],
                key[: attn_metadata.num_actual_tokens],
                value[: attn_metadata.num_actual_tokens],
                output[: attn_metadata.num_actual_tokens],
                attn_metadata,
                layer,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Lazy-init CUDA graph buffers on first forward (during warmup)
        if not self._cg_buffers_ready and kv_cache is not None:
            self._init_cg_buffers(kv_cache, compute_dtype=query.dtype)

        # Steps 1-3: compress, rotate Q, decompress (decode vs prefill path)
        is_decode = self._cg_buffers_ready and num_actual_tokens == 1

        # Fused paged decode (Story 6.3): single kernel replaces
        # decompress + FlashAttn + post-rotate for decode steps.
        if self._fused_paged_available and is_decode:
            return self._fused_decode_path(
                query, key, value, kv_cache, attn_metadata, output
            )

        if is_decode:
            q_rot, key_cache, value_cache = self._tq4_decode(
                query, key, value, kv_cache, attn_metadata
            )
        else:
            q_rot, key_cache, value_cache = self._tq4_prefill(
                query, key, value, kv_cache, attn_metadata
            )

        # Step 4: Run Flash Attention with rotated Q and rotated KV
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if attn_metadata.use_cascade:
            raise NotImplementedError("TQ4 does not yet support cascade attention")

        descale_shape = (
            attn_metadata.query_start_loc.shape[0] - 1,
            self.num_kv_heads,
        )
        q_descale = layer._q_scale.expand(descale_shape)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        flash_attn_varlen_func(
            q=q_rot,
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window)
            if self.sliding_window is not None
            else None,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
            s_aux=self.sinks,
        )

        # Step 5: Post-rotate output by Pi (undo rotation space)
        out_slice = output[:num_actual_tokens]
        output[:num_actual_tokens] = (out_slice.float() @ self._tq4_rotation).to(
            out_slice.dtype
        )

        return output


# ---------------------------------------------------------------------------
# Registration (3c.1 -- monkey-patch for TQ4 page size)
# ---------------------------------------------------------------------------

_original_get_kv_cache_spec = None


def register_tq4_backend() -> None:
    """Register TQ4 as the CUSTOM attention backend.

    In addition to registering the backend class, this monkey-patches
    ``Attention.get_kv_cache_spec`` so that decoder attention layers
    return :class:`TQ4FullAttentionSpec` (with ``dtype=torch.uint8``
    and TQ4-sized pages) instead of the standard ``FullAttentionSpec``.

    Called automatically by the ``vllm.general_plugins`` entry point,
    or manually before starting vLLM::

        from turboquant_vllm.vllm import register_tq4_backend

        register_tq4_backend()
        # then start vLLM with --attention-backend CUSTOM
    """
    global _original_get_kv_cache_spec  # noqa: PLW0603

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "turboquant_vllm.vllm.tq4_backend.TQ4AttentionBackend",
    )

    # Register TQ4FullAttentionSpec in the KV cache manager mapping.
    # vLLM uses exact type() match, not isinstance(), so subclasses
    # of FullAttentionSpec must be explicitly added.
    from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

    if TQ4FullAttentionSpec not in spec_manager_map:
        spec_manager_map[TQ4FullAttentionSpec] = spec_manager_map[FullAttentionSpec]

    # Monkey-patch Attention.get_kv_cache_spec to return TQ4 spec
    from vllm.model_executor.layers.attention.attention import Attention

    if _original_get_kv_cache_spec is None:
        _original_get_kv_cache_spec = Attention.get_kv_cache_spec

    def _tq4_get_kv_cache_spec(self, vllm_config):
        spec = _original_get_kv_cache_spec(self, vllm_config)
        if isinstance(spec, FullAttentionSpec) and not isinstance(
            spec, TQ4FullAttentionSpec
        ):
            kwargs = {f.name: getattr(spec, f.name) for f in dc_fields(spec)}
            kwargs["dtype"] = torch.uint8
            return TQ4FullAttentionSpec(**kwargs)
        return spec

    Attention.get_kv_cache_spec = _tq4_get_kv_cache_spec
    logger.info("TQ4 attention backend registered as CUSTOM (packed cache)")
