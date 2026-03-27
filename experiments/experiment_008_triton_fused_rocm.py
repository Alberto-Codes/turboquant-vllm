"""Test the fused Q@K^T Triton kernel on ROCm/HIP backend.

Examples:
    ```bash
    ./infra/run-rocm.sh python experiments/experiment_008_triton_fused_rocm.py
    ```

See Also:
    :mod:`turboquant_consumer.triton.fused_qk_attention`: Kernel implementation.
"""

import time

import torch

from turboquant_consumer import solve_lloyd_max
from turboquant_consumer.triton.fused_qk_attention import fused_qk_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
)

# Setup: simulate compressed KV cache scenario
batch, n_q_heads, n_kv_heads, head_dim = 1, 32, 8, 128
kv_len = 1024
bits = 4
n_levels = 2**bits  # 16

# Get real Lloyd-Max centroids
centroids_cpu, _ = solve_lloyd_max(head_dim, bits)
centroids = centroids_cpu.float().to(device)
print(f"Centroids: {centroids.shape} on {centroids.device}")

# Simulate nibble-packed key indices and norms
packed_indices = torch.randint(
    0,
    256,
    (batch, n_kv_heads, kv_len, head_dim // 2),
    dtype=torch.uint8,
    device=device,
)
norms = torch.randn(batch, n_kv_heads, kv_len, device=device).float().abs()

# Pre-rotated query (in practice: q @ Pi_T)
q_rotated = torch.randn(batch, n_q_heads, 1, head_dim, device=device).float()

scale = 1.0 / (head_dim**0.5)

print(
    f"\nInputs: batch={batch}, q_heads={n_q_heads}, kv_heads={n_kv_heads}, "
    f"kv_len={kv_len}, head_dim={head_dim}"
)

# Run fused kernel
print("\nRunning fused Q@K^T kernel...")
try:
    scores = fused_qk_scores(
        q_rotated,
        packed_indices,
        norms,
        centroids,
        scale=scale,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
    )
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"Any NaN: {scores.isnan().any().item()}")
    print(f"Any Inf: {scores.isinf().any().item()}")

    # Correctness check: compute reference via manual unpack + matmul
    print("\nReference check (manual unpack + matmul)...")
    hi = (packed_indices >> 4).long()
    lo = (packed_indices & 0x0F).long()
    unpacked = torch.stack([hi, lo], dim=-1).flatten(
        -2
    )  # [B, kv_heads, kv_len, head_dim]
    decompressed_keys = centroids[unpacked] * norms.unsqueeze(
        -1
    )  # [B, kv_heads, kv_len, head_dim]

    # GQA expand
    gqa_ratio = n_q_heads // n_kv_heads
    keys_expanded = decompressed_keys.repeat_interleave(gqa_ratio, dim=1)
    ref_scores = torch.einsum("bqhd,bkhd->bqhk", q_rotated, keys_expanded) * scale
    # Note: ref_scores shape is [B, q_heads, 1, kv_len] — matches fused output

    # Actually the einsum order should be: q is [B, q_heads, 1, D], k is [B, q_heads, kv_len, D]
    # => scores [B, q_heads, 1, kv_len]
    ref_scores2 = (q_rotated @ keys_expanded.transpose(-2, -1)) * scale

    cosine = torch.nn.functional.cosine_similarity(
        scores.flatten().unsqueeze(0),
        ref_scores2.flatten().unsqueeze(0),
    ).item()
    max_diff = (scores - ref_scores2).abs().max().item()

    print(f"Cosine similarity: {cosine:.6f}")
    print(f"Max absolute diff: {max_diff:.6f}")

    # Benchmark
    print("\nBenchmark (10 iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        _ = fused_qk_scores(
            q_rotated,
            packed_indices,
            norms,
            centroids,
            scale=scale,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Avg time per call: {elapsed / 10 * 1000:.2f} ms")

    print("\nRESULT: PASS" if cosine > 0.999 else "\nRESULT: FAIL")

except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback

    traceback.print_exc()
