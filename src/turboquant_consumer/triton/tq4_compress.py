"""Fused Triton kernel for TQ4 compression (norm + rotate + quantize + pack).

Phase 3c.9: Replaces the multi-op PyTorch compress path with a single
fused kernel.  The rotation matrix is pre-split into even/odd column
halves so the kernel writes packed nibble output directly without a
separate interleave step.

Experiment 015 (post-3c.8) showed compress accounts for 53% of decode
time (~0.149ms for K+V at 1 token).  The PyTorch path launches 6+
CUDA kernels (norm, divide, matmul, bucketize, clamp, pack).  Fusing
into one Triton launch eliminates kernel-launch overhead.

Attributes:
    tq4_compress: Python wrapper that launches the fused kernel.

Examples:
    ```python
    from turboquant_consumer.triton.tq4_compress import tq4_compress

    packed, norms = tq4_compress(
        x,
        rotation_T_even,
        rotation_T_odd,
        boundaries,
    )
    # packed: (N, H, D//2) uint8, norms: (N, H, 1) fp32
    ```

See Also:
    :mod:`turboquant_consumer.triton.tq4_decompress`: Phase 3c.8 fused decompress.
    :mod:`turboquant_consumer.vllm.tq4_backend`: vLLM backend that calls this kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _tq4_compress_kernel(
    X,
    Rot_T_even,
    Rot_T_odd,
    Boundaries,
    Packed_out,
    Norms_out,
    M,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    N_BOUND: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused TQ4 compress: norm + normalize + rotate + bucketize + pack.

    One program per row (one KV head of one token).  Computes the L2
    norm, normalizes, tiles the rotation matmul using pre-split even/odd
    column halves, performs a linear-scan bucketize against 15 boundaries,
    and writes one ``HALF_D``-byte packed uint8 row plus one fp32 norm.

    Args:
        X (tl.pointer_type): ``(M, D)`` fp16 input vectors.
        Rot_T_even (tl.pointer_type): ``(D, HALF_D)`` fp32 even cols of
            ``rotation.T``.
        Rot_T_odd (tl.pointer_type): ``(D, HALF_D)`` fp32 odd cols of
            ``rotation.T``.
        Boundaries (tl.pointer_type): ``(N_BOUND,)`` fp32 quantization
            boundaries.
        Packed_out (tl.pointer_type): ``(M, HALF_D)`` uint8 output.
        Norms_out (tl.pointer_type): ``(M,)`` fp32 output norms.
        M (int): Total rows (N * H).
        D (tl.constexpr): Head dimension.
        HALF_D (tl.constexpr): ``D // 2``.
        N_BOUND (tl.constexpr): Number of boundaries (15 for TQ4).
        BLOCK_K (tl.constexpr): Tile size for rotation matmul.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    d_offs = tl.arange(0, D)
    half_offs = tl.arange(0, HALF_D)

    # Step 1: Load input and compute norm
    x = tl.load(X + row * D + d_offs).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x))
    inv_norm = 1.0 / (norm + 1e-10)

    # Step 2: Tiled rotation with even/odd output split
    # result_even[j] = sum_k x_hat[k] * R^T[k, 2j]
    # result_odd[j]  = sum_k x_hat[k] * R^T[k, 2j+1]
    rotated_even = tl.zeros([HALF_D], dtype=tl.float32)
    rotated_odd = tl.zeros([HALF_D], dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Re-load chunk and normalize (avoids register gather)
        x_chunk = tl.load(X + row * D + k_offs).to(tl.float32) * inv_norm

        # Load pre-split rotation tiles (contiguous memory)
        re = tl.load(
            Rot_T_even + k_offs[:, None] * HALF_D + half_offs[None, :],
        )
        ro = tl.load(
            Rot_T_odd + k_offs[:, None] * HALF_D + half_offs[None, :],
        )

        rotated_even += tl.sum(x_chunk[:, None] * re, axis=0)
        rotated_odd += tl.sum(x_chunk[:, None] * ro, axis=0)

    # Step 3: Bucketize (linear scan over sorted boundaries)
    idx_even = tl.zeros([HALF_D], dtype=tl.int32)
    idx_odd = tl.zeros([HALF_D], dtype=tl.int32)
    for b in range(N_BOUND):
        boundary = tl.load(Boundaries + b)
        idx_even += (rotated_even >= boundary).to(tl.int32)
        idx_odd += (rotated_odd >= boundary).to(tl.int32)

    # Step 4: Pack nibbles — high=even, low=odd (matches unpack convention)
    packed = ((idx_even & 0xF) << 4) | (idx_odd & 0xF)

    # Step 5: Store packed indices and norm
    tl.store(Packed_out + row * HALF_D + half_offs, packed.to(tl.uint8))
    tl.store(Norms_out + row, norm)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def tq4_compress(
    x: torch.Tensor,
    rotation_T_even: torch.Tensor,
    rotation_T_odd: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress vectors to TQ4 nibble-packed format.

    Fused Triton path: norm + normalize + tiled rotation + bucketize +
    nibble-pack in a single kernel launch.

    Args:
        x: ``(N, H, D)`` fp16/bf16 input vectors.
        rotation_T_even: ``(D, D//2)`` fp32 -- even columns of
            ``rotation.T``, pre-split for contiguous loads.
        rotation_T_odd: ``(D, D//2)`` fp32 -- odd columns.
        boundaries: ``(N_BOUND,)`` fp32 quantization boundaries.

    Returns:
        Tuple of ``(packed, norms)`` where packed is ``(N, H, D//2)``
        uint8 and norms is ``(N, H, 1)`` fp32.
    """
    N, H, D = x.shape
    HALF_D = D // 2
    M = N * H

    if not x.is_cuda:
        return _tq4_compress_cpu(x, rotation_T_even, rotation_T_odd, boundaries)

    x_flat = x.reshape(M, D).contiguous()
    packed = torch.empty(M, HALF_D, dtype=torch.uint8, device=x.device)
    norms = torch.empty(M, dtype=torch.float32, device=x.device)

    N_BOUND = boundaries.shape[0]
    BLOCK_K = min(32, D)

    grid = (M,)
    _tq4_compress_kernel[grid](
        x_flat,
        rotation_T_even,
        rotation_T_odd,
        boundaries,
        packed,
        norms,
        M,
        D=D,  # ty: ignore[invalid-argument-type]
        HALF_D=HALF_D,  # ty: ignore[invalid-argument-type]
        N_BOUND=N_BOUND,  # ty: ignore[invalid-argument-type]
        BLOCK_K=BLOCK_K,  # ty: ignore[invalid-argument-type]
    )

    return packed.reshape(N, H, HALF_D), norms.reshape(N, H, 1)


def _tq4_compress_cpu(
    x: torch.Tensor,
    rotation_T_even: torch.Tensor,
    rotation_T_odd: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback for CPU tensors.

    Args:
        x: ``(N, H, D)`` input vectors.
        rotation_T_even: ``(D, D//2)`` fp32 even cols of rotation.T.
        rotation_T_odd: ``(D, D//2)`` fp32 odd cols of rotation.T.
        boundaries: ``(N_BOUND,)`` fp32 boundaries.

    Returns:
        Tuple of ``(packed, norms)`` — same shapes as Triton path.
    """
    N, H, D = x.shape
    HALF_D = D // 2
    flat = x.reshape(N * H, D).float()

    norms = torch.norm(flat, dim=-1, keepdim=True)
    normalized = flat / (norms + 1e-10)

    # Reconstruct full rotation.T from even/odd halves
    rotation_T = torch.empty(D, D, dtype=torch.float32, device=x.device)
    rotation_T[:, 0::2] = rotation_T_even
    rotation_T[:, 1::2] = rotation_T_odd
    rotated = normalized @ rotation_T

    indices = torch.bucketize(rotated, boundaries)
    indices = indices.clamp(0, 2**4 - 1)

    idx_u8 = indices.to(torch.uint8)
    packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

    return packed.reshape(N, H, HALF_D), norms.reshape(N, H, 1)
