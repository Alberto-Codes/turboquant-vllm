"""Fused Triton kernel for TQ4 cache decompression (no rotation).

Phase 3c.8: Replaces the multi-op PyTorch decompress path with a single
fused kernel that performs nibble unpack -> centroid gather -> norm scale
-> dtype cast in one launch.  The rotation is **not** applied here -- the
caller pre-rotates Q by ``Pi^T`` and post-rotates the attention output
by ``Pi``, saving O(cache_len) matmuls per decode step.

Experiment 015 showed that at cache_len=4096 on RTX 4090, decompress
accounts for 68% of decode time.  The rotation matmul (128x128) is the
dominant cost within that 68%.  By moving rotation to Q/output (O(1)
per decode step), the kernel only needs elementwise + gather ops.

Attributes:
    tq4_decompress: Python wrapper that launches the fused kernel.

Examples:
    ```python
    from turboquant_vllm.triton.tq4_decompress import tq4_decompress

    # packed: (N, H, D//2) uint8, norms: (N, H, 1) fp32
    out = tq4_decompress(packed, norms, centroids, dtype=torch.float16)
    # out: (N, H, D) fp16 -- still in rotated space (no Pi applied)
    ```

See Also:
    :mod:`turboquant_vllm.triton.flash_attention_tq4`: Phase 2 fused FA+K kernel.
    :mod:`turboquant_vllm.vllm.tq4_backend`: vLLM backend that calls this kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _tq4_decompress_kernel(
    Packed,
    Norms,
    Centroids,
    Out,
    M,
    HALF_D: tl.constexpr,
):
    """Fused TQ4 decompress: unpack + gather + scale + cast.

    One program per row (one KV head of one token).  Each row reads
    ``HALF_D`` packed uint8 bytes, unpacks two 4-bit indices per byte,
    gathers centroids, multiplies by the fp32 norm, and writes ``D``
    output values in the target dtype.

    Args:
        Packed (tl.pointer_type): ``(M, HALF_D)`` uint8 packed indices.
        Norms (tl.pointer_type): ``(M,)`` fp32 per-vector norms.
        Centroids (tl.pointer_type): ``(16,)`` fp32 centroid table.
        Out (tl.pointer_type): ``(M, D)`` output buffer in target dtype.
        M (int): Total rows (N * H).
        HALF_D (tl.constexpr): ``D // 2`` for compile-time tiling.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    # Load HALF_D packed bytes
    p_offs = tl.arange(0, HALF_D)
    packed = tl.load(Packed + row * HALF_D + p_offs)

    # Nibble unpack -> two index streams
    hi_idx = ((packed >> 4) & 0x0F).to(tl.int32)
    lo_idx = (packed & 0x0F).to(tl.int32)

    # Centroid gather (16-entry table, L1-cached)
    c_hi = tl.load(Centroids + hi_idx)  # (HALF_D,) fp32
    c_lo = tl.load(Centroids + lo_idx)  # (HALF_D,) fp32

    # Load norm and scale
    norm = tl.load(Norms + row)
    c_hi = c_hi * norm
    c_lo = c_lo * norm

    # Interleave and store: even positions = hi, odd = lo
    # This matches the pack convention: packed[i] = (idx[2i] << 4) | idx[2i+1]
    D: tl.constexpr = HALF_D * 2
    out_base = row * D
    tl.store(Out + out_base + p_offs * 2, c_hi.to(Out.dtype.element_ty))
    tl.store(Out + out_base + p_offs * 2 + 1, c_lo.to(Out.dtype.element_ty))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def tq4_decompress(
    packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decompress TQ4 nibble-packed data to full-precision vectors.

    Fused Triton path: unpack + centroid gather + norm scale + cast in a
    single kernel launch.  Does **not** apply the rotation matrix --
    output remains in rotated space.

    Args:
        packed: ``(N, H, D//2)`` uint8 -- nibble-packed centroid indices.
        norms: ``(N, H, 1)`` fp32 -- per-vector norms.
        centroids: ``(C,)`` fp32 -- centroid table (C=16 for TQ4).
        dtype: Output dtype (default: ``torch.float16``).
        out: Optional pre-allocated ``(N, H, D)`` output tensor.  When
            provided, results are written into it and the same tensor is
            returned.  Follows PyTorch ``out`` convention.

    Returns:
        Tensor of shape ``(N, H, D)`` in ``dtype``, still in rotated
        space (caller must apply post-rotation if needed).
    """
    N, H, half_D = packed.shape
    D = half_D * 2
    M = N * H

    # CPU fallback: PyTorch path (Triton requires CUDA)
    if not packed.is_cuda:
        return _tq4_decompress_cpu(packed, norms, centroids, dtype, out)

    # Flatten to 2D for the kernel
    packed_flat = packed.reshape(M, half_D).contiguous()
    norms_flat = norms.reshape(M).contiguous()

    caller_out = out
    if out is None:
        out = torch.empty(M, D, dtype=dtype, device=packed.device)

    grid = (M,)
    _tq4_decompress_kernel[grid](
        packed_flat,
        norms_flat,
        centroids,
        out,
        M,
        HALF_D=half_D,  # ty: ignore[invalid-argument-type]
    )

    if caller_out is not None:
        return caller_out
    return out.reshape(N, H, D)


def _tq4_decompress_cpu(
    packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch fallback for CPU tensors (no rotation).

    Args:
        packed: ``(N, H, D//2)`` uint8.
        norms: ``(N, H, 1)`` fp32.
        centroids: ``(C,)`` fp32.
        dtype: Output dtype.
        out: Optional pre-allocated output tensor.

    Returns:
        Tensor ``(N, H, D)`` in ``dtype``, rotated space.
    """
    N, H, half_D = packed.shape
    D = half_D * 2
    high = (packed >> 4).long()
    low = (packed & 0x0F).long()
    indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
    flat_norms = norms.reshape(N * H, 1)
    reconstructed = centroids[indices]
    result = (reconstructed * flat_norms).reshape(N, H, D).to(dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result
