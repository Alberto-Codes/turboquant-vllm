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

from turboquant_consumer.quantizer import TurboQuantMSE
from turboquant_consumer.triton.tq4_compress import tq4_compress
from turboquant_consumer.triton.tq4_decompress import tq4_decompress

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionImplBase, AttentionMetadataBuilder

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
        """Return :class:`FlashAttentionMetadataBuilder` -- reused."""
        return FlashAttentionMetadataBuilder

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
        self._tq4_rotation = quantizer.rotation  # (D, D) fp32
        self._tq4_centroids = quantizer.codebook.centroids  # (16,) fp32
        self._tq4_boundaries = quantizer.codebook.boundaries  # (15,) fp32
        # Pre-split rotation.T for fused compress kernel (contiguous loads)
        rot_t = quantizer.rotation.T.contiguous()
        self._tq4_rot_T_even = rot_t[:, 0::2].contiguous()  # (D, D//2) fp32
        self._tq4_rot_T_odd = rot_t[:, 1::2].contiguous()  # (D, D//2) fp32
        self._tq4_on_device = False

        # Byte layout offsets within the last dimension of the packed cache.
        # Layout: [K_indices(H*D/2) | K_norms(H*4) | V_indices(H*D/2) | V_norms(H*4)]
        half_D = head_size // 2
        self._half_D = half_D
        self._k_idx_end = num_kv_heads * half_D
        self._k_norm_end = self._k_idx_end + num_kv_heads * TQ4_NORM_BYTES
        self._v_idx_end = self._k_norm_end + num_kv_heads * half_D
        self._total_bytes = self._v_idx_end + num_kv_heads * TQ4_NORM_BYTES

        logger.info(
            "TQ4AttentionImpl: %d KV heads, head_size=%d, "
            "%d bytes/token (%.2fx compression vs FP16)",
            num_kv_heads,
            head_size,
            self._total_bytes,
            (2 * num_kv_heads * head_size * 2) / self._total_bytes,
        )

    # ----- device management -----

    def _ensure_device(self, device: torch.device) -> None:
        """Move compression primitives to GPU on first use."""
        if not self._tq4_on_device:
            self._tq4_rotation = self._tq4_rotation.to(device)
            self._tq4_centroids = self._tq4_centroids.to(device)
            self._tq4_boundaries = self._tq4_boundaries.to(device)
            self._tq4_rot_T_even = self._tq4_rot_T_even.to(device)
            self._tq4_rot_T_odd = self._tq4_rot_T_odd.to(device)
            self._tq4_on_device = True

    # ----- compression / decompression -----

    def _compress(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress ``(N, H, D)`` -> nibble-packed indices + fp32 norms.

        Uses the fused Triton kernel (Phase 3c.9) that performs norm +
        normalize + rotation + bucketize + pack in a single launch.

        Returns:
            packed: ``(N, H, D//2)`` uint8 -- two 4-bit centroid indices per byte.
            norms: ``(N, H, 1)`` fp32 -- vector norms.
        """
        return tq4_compress(
            x,
            self._tq4_rot_T_even,
            self._tq4_rot_T_odd,
            self._tq4_boundaries,
        )

    def _decompress(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decompress nibble-packed indices + norms -> ``(N, H, D)``.

        Args:
            packed: ``(N, H, D//2)`` uint8.
            norms: ``(N, H, 1)`` fp32.
            dtype: Output dtype (e.g., ``torch.bfloat16``).

        Returns:
            Reconstructed tensor ``(N, H, D)`` in ``dtype``.
        """
        N, H, half_D = packed.shape
        D = half_D * 2

        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)

        flat_norms = norms.reshape(N * H, 1)
        reconstructed = self._tq4_centroids[indices]
        unrotated = reconstructed @ self._tq4_rotation
        result = unrotated * flat_norms

        return result.reshape(N, H, D).to(dtype)

    # ----- packed cache operations (3c.4 - 3c.5) -----

    def _compress_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Compress K/V and scatter-write TQ4 bytes to packed cache.

        Args:
            key: ``(N, H, D)`` new key tokens.
            value: ``(N, H, D)`` new value tokens.
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            slot_mapping: ``(num_actual_tokens,)`` flat slot indices.
        """
        k_packed, k_norms = self._compress(key)  # (N, H, D//2), (N, H, 1)
        v_packed, v_norms = self._compress(value)

        N = k_packed.shape[0]
        H = self.num_kv_heads
        device = key.device

        # Build packed byte row per token: [K_idx | K_norm | V_idx | V_norm]
        row = torch.empty(N, self._total_bytes, dtype=torch.uint8, device=device)
        row[:, : self._k_idx_end] = k_packed.reshape(N, -1)
        row[:, self._k_idx_end : self._k_norm_end] = (
            k_norms.reshape(N, H).contiguous().view(torch.uint8)
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

        Returns:
            key_cache: ``(NB, BS, H, D)`` in ``compute_dtype``.
            value_cache: ``(NB, BS, H, D)`` in ``compute_dtype``.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        half_D = self._half_D
        D = self.head_size

        self._ensure_device(kv_cache.device)
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
        key_out = tq4_decompress(k_packed, k_norms, self._tq4_centroids, compute_dtype)
        value_out = tq4_decompress(
            v_packed,
            v_norms,
            self._tq4_centroids,
            compute_dtype,
        )

        # Optionally unrotate (backward compat for tests; forward() skips this)
        if apply_rotation:
            key_out = (key_out.float() @ self._tq4_rotation).to(compute_dtype)
            value_out = (value_out.float() @ self._tq4_rotation).to(compute_dtype)

        return key_out.reshape(NB, BS, H, D), value_out.reshape(NB, BS, H, D)

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

        # Step 1: Compress and store new K/V tokens
        if kv_cache is not None and key is not None and value is not None:
            self._ensure_device(query.device)
            self._compress_and_store(key, value, kv_cache, attn_metadata.slot_mapping)

        # Step 2: Pre-rotate Q by Pi^T (O(num_actual_tokens), not O(cache_len))
        self._ensure_device(query.device)
        q_slice = query[:num_actual_tokens]
        q_rot = (q_slice.float() @ self._tq4_rotation.T).to(q_slice.dtype)

        # Step 3: Decompress full cache (Triton fused, skip rotation)
        key_cache, value_cache = self._decompress_cache(
            kv_cache,
            query.dtype,
            apply_rotation=False,
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
            alibi_slopes=self.alibi_slopes,  # ty: ignore[invalid-argument-type]
            window_size=list(self.sliding_window)
            if self.sliding_window is not None
            else None,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,  # ty: ignore[invalid-argument-type]
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

        from turboquant_consumer.vllm import register_tq4_backend

        register_tq4_backend()
        # then start vLLM with --attention-backend CUSTOM
    """
    global _original_get_kv_cache_spec  # noqa: PLW0603

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "turboquant_consumer.vllm.tq4_backend.TQ4AttentionBackend",
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

    Attention.get_kv_cache_spec = _tq4_get_kv_cache_spec  # ty: ignore[invalid-assignment]
    logger.info("TQ4 attention backend registered as CUSTOM (packed cache)")
