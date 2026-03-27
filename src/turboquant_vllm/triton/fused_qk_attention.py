"""Fused Triton kernel for TQ4 nibble-packed attention scores.

Computes ``Q @ compressed_K^T`` directly from nibble-packed 4-bit
indices without materializing decompressed key tensors. Reduces
per-layer memory traffic from ~25 MB to ~3 MB during decode.

Key math::

    <q, R^T @ centroids[idx]> = <R @ q, centroids[idx]>

Pre-rotate the query once (``q_rot = q @ Pi_T``), then the kernel
does: ``score[s] = norm[s] * sum_d(q_rot[d] * centroids[idx[s,d]]) * scale``

Based on the Dejan.ai TurboQuant Triton kernel, adapted for:

- **Nibble-packed 4-bit indices** (two per uint8 byte)
- **fp32 norms** (fp16 causes precision loss at 10K+ tokens)
- **Configurable GQA** (tested with 4:1 ratio for Molmo2)

Attributes:
    fused_qk_scores: Python wrapper that launches the Triton kernel.

Examples:
    ```python
    scores = fused_qk_scores(
        q_rotated,  # [B, n_q_heads, q_len, head_dim]
        packed_indices,  # [B, n_kv_heads, kv_len, head_dim // 2] uint8
        norms,  # [B, n_kv_heads, kv_len] fp32
        centroids,  # [16] fp32 (for 4-bit)
        scale=1 / 128**0.5,
        n_q_heads=32,
        n_kv_heads=8,
    )
    ```

See Also:
    :mod:`turboquant_vllm.kv_cache`: CompressedDynamicCache that produces
        the nibble-packed indices and fp32 norms consumed by this kernel.
    `Dejan.ai TurboQuant blog <https://dejan.ai/blog/turboquant/>`_:
        Original Triton kernel reference.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 128}, num_warps=8),
    ],
    key=["kv_len", "head_dim"],
)
@triton.jit
def _fused_qk_nibble_kernel(
    # Pre-rotated query: [BH_q, head_dim]
    Q_ptr,
    # Nibble-packed key indices: [BH_kv, kv_len, head_dim // 2] uint8
    KI_ptr,
    # Key norms: [BH_kv, kv_len] float32
    KN_ptr,
    # Centroid table: [16] float32
    C_ptr,
    # Output scores: [BH_q, kv_len] float32
    Out_ptr,
    # Dimensions
    kv_len,
    head_dim: tl.constexpr,
    n_q_heads,
    n_kv_heads,
    scale,
    # Strides — Q: [BH_q, head_dim]
    stride_q_bh,
    stride_q_d,
    # Strides — KI: [BH_kv, kv_len, head_dim // 2]
    stride_ki_bh,
    stride_ki_s,
    stride_ki_d,
    # Strides — KN: [BH_kv, kv_len]
    stride_kn_bh,
    stride_kn_s,
    # Strides — Out: [BH_q, kv_len]
    stride_o_bh,
    stride_o_s,
    # Block sizes (autotuned)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention scores from pre-rotated queries and nibble-packed keys.

    For each (query_head, kv_block):
        score[s] = norm[s] * sum_d(q_rot[d] * centroids[nibble_unpack(packed[s,d])]) * scale

    The inner loop unpacks two 4-bit indices per byte via bit-shift:
        hi = byte >> 4 (even index), lo = byte & 0x0F (odd index).
    """
    pid_bh = tl.program_id(0)  # batch * query_head
    pid_s = tl.program_id(1)  # kv sequence block

    # GQA: map query head -> KV head
    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    # KV sequence positions for this block
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < kv_len

    # Accumulate dot product over head_dim in blocks of BLOCK_D
    # BLOCK_D operates on UNPACKED indices, so we load BLOCK_D // 2 packed bytes
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    HALF_BLOCK_D: tl.constexpr = BLOCK_D // 2

    for d_start in range(0, head_dim, BLOCK_D):
        # Packed byte offsets: each byte holds 2 indices
        packed_d_offs = d_start // 2 + tl.arange(0, HALF_BLOCK_D)
        packed_d_mask = packed_d_offs < (head_dim // 2)

        # Load packed bytes: KI[kv_bh, s_offs, packed_d_offs] -> [BLOCK_S, HALF_BLOCK_D]
        ki_ptrs = (
            KI_ptr
            + kv_bh * stride_ki_bh
            + s_offs[:, None] * stride_ki_s
            + packed_d_offs[None, :] * stride_ki_d
        )
        combined_mask = s_mask[:, None] & packed_d_mask[None, :]
        packed_bytes = tl.load(ki_ptrs, mask=combined_mask, other=0)

        # Unpack nibbles: hi = even indices, lo = odd indices
        hi_idx = (packed_bytes >> 4).to(tl.int32)
        lo_idx = (packed_bytes & 0x0F).to(tl.int32)

        # Gather centroids for both halves -> [BLOCK_S, HALF_BLOCK_D]
        hi_vals = tl.load(C_ptr + hi_idx, mask=combined_mask, other=0.0).to(tl.float32)
        lo_vals = tl.load(C_ptr + lo_idx, mask=combined_mask, other=0.0).to(tl.float32)

        # Load query values for even and odd positions
        even_d = d_start + tl.arange(0, HALF_BLOCK_D) * 2
        odd_d = even_d + 1
        even_d_mask = even_d < head_dim
        odd_d_mask = odd_d < head_dim

        q_even = tl.load(
            Q_ptr + pid_bh * stride_q_bh + even_d * stride_q_d,
            mask=even_d_mask,
            other=0.0,
        ).to(tl.float32)
        q_odd = tl.load(
            Q_ptr + pid_bh * stride_q_bh + odd_d * stride_q_d,
            mask=odd_d_mask,
            other=0.0,
        ).to(tl.float32)

        # Dot product: sum over D block (even + odd positions)
        acc += tl.sum(hi_vals * q_even[None, :], axis=1)
        acc += tl.sum(lo_vals * q_odd[None, :], axis=1)

    # Load key norms: KN[kv_bh, s_offs]
    kn_ptrs = KN_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s
    norms = tl.load(kn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    # Final score = norm * dot_product * scale
    scores = norms * acc * scale

    # Store
    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


def fused_qk_scores(
    q_rotated: torch.Tensor,
    packed_indices: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    *,
    n_q_heads: int,
    n_kv_heads: int,
) -> torch.Tensor:
    """Compute attention scores from pre-rotated queries and nibble-packed keys.

    The query must be pre-rotated by the TurboQuant rotation matrix
    (``q_rot = q @ Pi_T``) so the kernel avoids the expensive 128x128
    rotation matmul in its inner loop.

    Args:
        q_rotated: Pre-rotated query, shape
            ``(batch, n_q_heads, q_len, head_dim)``.
        packed_indices: Nibble-packed 4-bit key indices, shape
            ``(batch, n_kv_heads, kv_len, head_dim // 2)``, dtype uint8.
        norms: Key vector norms, shape
            ``(batch, n_kv_heads, kv_len)``, dtype float32.
        centroids: Lloyd-Max centroid values, shape ``(n_levels,)``,
            dtype float32.
        scale: Attention scale factor (typically ``1 / sqrt(head_dim)``).
        n_q_heads: Number of query attention heads.
        n_kv_heads: Number of key-value attention heads.

    Returns:
        Attention scores, shape ``(batch, n_q_heads, q_len, kv_len)``.
    """
    batch, _, q_len, head_dim = q_rotated.shape
    _, _, kv_len, _ = packed_indices.shape

    ki_flat = packed_indices.reshape(
        batch * n_kv_heads, kv_len, head_dim // 2
    ).contiguous()
    kn_flat = norms.reshape(batch * n_kv_heads, kv_len).contiguous()
    centroids = centroids.contiguous().float()

    # For q_len > 1 (prefill), process each query position separately
    # to keep the GQA head mapping correct. The kernel maps
    # q_head → kv_head via gqa_ratio = n_q_heads // n_kv_heads.
    # Flattening q_len into heads would break this mapping.
    results = []
    for q_pos in range(q_len):
        q_slice = (
            q_rotated[:, :, q_pos : q_pos + 1, :]
            .reshape(batch * n_q_heads, head_dim)
            .contiguous()
        )

        out = torch.empty(
            batch * n_q_heads,
            kv_len,
            device=q_rotated.device,
            dtype=torch.float32,
        )

        grid = (batch * n_q_heads, triton.cdiv(kv_len, 64))

        _fused_qk_nibble_kernel[grid](
            q_slice,
            ki_flat,
            kn_flat,
            centroids,
            out,
            kv_len,
            head_dim,
            n_q_heads,
            n_kv_heads,
            scale,
            q_slice.stride(0),
            q_slice.stride(1),
            ki_flat.stride(0),
            ki_flat.stride(1),
            ki_flat.stride(2),
            kn_flat.stride(0),
            kn_flat.stride(1),
            out.stride(0),
            out.stride(1),
        )

        results.append(out.reshape(batch, n_q_heads, 1, kv_len))

    return torch.cat(results, dim=2)
