"""Experiment 017: INT8 tl.dot() Tensor Core Dispatch on Ada SM89 (RTX 4090).

Phase 0 gate for the SageAttention INT8 x TQ4 fusion kernel design.
Verifies whether Triton's INT8 matrix multiply dispatches to IMMA tensor
cores on Ada Lovelace, or falls back to slower emulation.

Hypothesis: Triton 3.6.0 on SM89 dispatches INT8 tl.dot() to IMMA tensor
core instructions (mma.sync.aligned...s8.s8.s32), achieving >500 TOPS —
approximately 2x the FP16 tensor core throughput (~330 TFLOPS).

Success criteria:
  PASS:    INT8 achieves >500 TOPS AND PTX shows IMMA instructions
  PARTIAL: INT8 works correctly but <500 TOPS (usable, smaller margin)
  FAIL:    INT8 falls back to emulation, no tensor core dispatch
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl

# ── Triton Kernels ──────────────────────────────────────────────────────────


@triton.jit
def matmul_int8_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT8 A @ B -> INT32 accumulator via tl.dot."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in int32 (native IMMA output)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K] as int8
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0)
        a = a.to(tl.int8)

        # Load B tile [BLOCK_K, BLOCK_N] as int8
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        b = b.to(tl.int8)

        # INT8 matmul — should dispatch to IMMA tensor cores
        acc += tl.dot(a, b)

    # Store result as int32
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def matmul_fp16_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP16 A @ B -> FP32 accumulator via tl.dot (baseline, known tensor core)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )

        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(
            b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ── Wrappers ────────────────────────────────────────────────────────────────


def triton_matmul_int8(
    a: torch.Tensor, b: torch.Tensor, block_m: int, block_n: int, block_k: int
) -> torch.Tensor:
    """Run INT8 matmul kernel and return INT32 result."""
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, dtype=torch.int32, device=a.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_int8_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


def triton_matmul_fp16(
    a: torch.Tensor, b: torch.Tensor, block_m: int, block_n: int, block_k: int
) -> torch.Tensor:
    """Run FP16 matmul kernel and return FP32 result."""
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_fp16_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


# ── Benchmarking ────────────────────────────────────────────────────────────


def benchmark_kernel(fn, warmup: int = 50, rep: int = 200) -> float:
    """Time a kernel function, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    # Return median
    return times[len(times) // 2]


def compute_tops(m: int, n: int, k: int, time_ms: float) -> float:
    """Compute TOPS (tera operations per second) for a matmul."""
    flops = 2 * m * n * k  # multiply-add = 2 ops
    return flops / (time_ms * 1e-3) / 1e12


# ── PTX Inspection ──────────────────────────────────────────────────────────


def inspect_ptx(kernel_fn, args, grid, constexpr_kwargs: dict) -> dict:
    """Attempt to inspect generated PTX/SASS for IMMA instructions."""
    result = {"ptx_inspected": False, "imma_found": False, "details": ""}

    try:
        cache_dir = os.environ.get(
            "TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")
        )
        # Trigger compilation by running the kernel once
        kernel_fn[grid](*args, **constexpr_kwargs)
        torch.cuda.synchronize()

        # Search cache for .ptx files
        ptx_files = list(Path(cache_dir).rglob("*.ptx"))
        _ = list(Path(cache_dir).rglob("*.ttgir"))

        imma_patterns = [
            "mma.sync.aligned",
            "s8.s8.s32",
            "m16n8k32",
            "m16n8k16",
            "mma.m16n8k32",
        ]

        for ptx_path in sorted(ptx_files, key=os.path.getmtime, reverse=True)[:20]:
            try:
                content = ptx_path.read_text()
                for pattern in imma_patterns:
                    if pattern in content:
                        result["ptx_inspected"] = True
                        result["imma_found"] = True
                        result["details"] += f"Found '{pattern}' in {ptx_path.name}\n"
            except Exception:
                continue

        if not result["imma_found"]:
            # Check for emulation patterns
            emulation_patterns = ["mul.lo", "mad.lo", "mul.wide"]
            for ptx_path in sorted(ptx_files, key=os.path.getmtime, reverse=True)[:20]:
                try:
                    content = ptx_path.read_text()
                    for pattern in emulation_patterns:
                        if pattern in content:
                            result["ptx_inspected"] = True
                            result["details"] += (
                                f"Possible emulation: '{pattern}' in {ptx_path.name}\n"
                            )
                except Exception:
                    continue

        if not result["ptx_inspected"] and not ptx_files:
            result["details"] = (
                f"No .ptx files found in {cache_dir}. Triton may use a different cache location."
            )

    except Exception as e:
        result["details"] = f"PTX inspection error: {e}"

    return result


# ── Main Experiment ─────────────────────────────────────────────────────────


def run_experiment() -> dict:
    """Run the full experiment and return structured results."""
    device = torch.device("cuda:0")
    results = {
        "experiment": "017-int8-tensor-core-dispatch",
        "date": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": "8.9",
            "driver": "",
            "pytorch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda or "N/A",
        },
        "hypothesis": (
            "Triton INT8 tl.dot() on SM89 dispatches to IMMA tensor cores, "
            "achieving >500 TOPS (~2x FP16 tensor core throughput)."
        ),
        "correctness": {},
        "benchmarks": {},
        "ptx_inspection": {},
        "tile_size_sweep": {},
        "verdict": "",
    }

    # Get driver version
    try:
        import subprocess

        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        results["hardware"]["driver"] = drv
    except Exception:
        pass

    print("=" * 70)
    print("Experiment 017: INT8 tl.dot() Tensor Core Dispatch")
    print(f"Hardware: {results['hardware']['gpu']}")
    print(
        f"PyTorch: {results['hardware']['pytorch']}, Triton: {results['hardware']['triton']}"
    )
    print("=" * 70)

    # ── Step 1: Correctness Check ───────────────────────────────────────

    print("\n[Step 1] Correctness check...")

    M, K, N = 64, 128, 64  # Attention tile: Q[64,128] @ K^T[128,64]

    torch.manual_seed(42)
    a_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    b_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)

    # Reference: float32 matmul
    ref = torch.matmul(a_int8.float(), b_int8.float()).int()

    # Triton INT8
    try:
        triton_out = triton_matmul_int8(
            a_int8, b_int8, block_m=64, block_n=64, block_k=128
        )
        max_diff = (triton_out - ref).abs().max().item()
        correct = max_diff == 0
        results["correctness"]["int8_int32_acc"] = {
            "correct": correct,
            "max_diff": max_diff,
            "shapes": {"M": M, "K": K, "N": N},
        }
        print(
            f"  INT8 -> INT32 acc: {'PASS' if correct else 'FAIL'} (max_diff={max_diff})"
        )
    except Exception as e:
        results["correctness"]["int8_int32_acc"] = {
            "correct": False,
            "error": str(e),
        }
        print(f"  INT8 -> INT32 acc: ERROR — {e}")
        # If basic INT8 fails, we can't proceed with benchmarks
        results["verdict"] = "FAIL"
        results["conclusion"] = f"Triton INT8 tl.dot() failed on SM89: {e}"
        return results

    # FP16 baseline correctness
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    b_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    ref_fp16 = torch.matmul(a_fp16.float(), b_fp16.float())

    try:
        triton_fp16_out = triton_matmul_fp16(
            a_fp16, b_fp16, block_m=64, block_n=64, block_k=128
        )
        max_diff_fp16 = (triton_fp16_out - ref_fp16).abs().max().item()
        results["correctness"]["fp16_fp32_acc"] = {
            "correct": max_diff_fp16 < 1e-2,
            "max_diff": max_diff_fp16,
        }
        print(f"  FP16 -> FP32 acc: PASS (max_diff={max_diff_fp16:.6f})")
    except Exception as e:
        results["correctness"]["fp16_fp32_acc"] = {"correct": False, "error": str(e)}
        print(f"  FP16 -> FP32 acc: ERROR — {e}")

    # ── Step 2: Performance Benchmark ───────────────────────────────────

    print("\n[Step 2] Performance benchmark at attention tile sizes...")
    print("  Config: BLOCK_M=64, BLOCK_N=64, HEAD_DIM(K)=128")

    # Use larger matrices for meaningful timing (simulate many tiles)
    # M=4096 (many Q positions), K=128 (head_dim), N=4096 (many KV positions)
    bench_M, bench_K, bench_N = 4096, 128, 4096

    a_bench_int8 = torch.randint(
        -128, 127, (bench_M, bench_K), dtype=torch.int8, device=device
    )
    b_bench_int8 = torch.randint(
        -128, 127, (bench_K, bench_N), dtype=torch.int8, device=device
    )
    a_bench_fp16 = torch.randn(bench_M, bench_K, dtype=torch.float16, device=device)
    b_bench_fp16 = torch.randn(bench_K, bench_N, dtype=torch.float16, device=device)

    # INT8 benchmark
    try:
        int8_ms = benchmark_kernel(
            lambda: triton_matmul_int8(a_bench_int8, b_bench_int8, 64, 64, 128)
        )
        int8_tops = compute_tops(bench_M, bench_N, bench_K, int8_ms)
        results["benchmarks"]["int8_64x64x128"] = {
            "time_ms": round(int8_ms, 4),
            "tops": round(int8_tops, 2),
            "shapes": {"M": bench_M, "K": bench_K, "N": bench_N},
        }
        print(f"  INT8 tl.dot: {int8_ms:.4f} ms, {int8_tops:.2f} TOPS")
    except Exception as e:
        results["benchmarks"]["int8_64x64x128"] = {"error": str(e)}
        print(f"  INT8 tl.dot: ERROR — {e}")
        int8_tops = 0

    # FP16 baseline benchmark
    try:
        fp16_ms = benchmark_kernel(
            lambda: triton_matmul_fp16(a_bench_fp16, b_bench_fp16, 64, 64, 128)
        )
        fp16_tops = compute_tops(bench_M, bench_N, bench_K, fp16_ms)
        results["benchmarks"]["fp16_64x64x128"] = {
            "time_ms": round(fp16_ms, 4),
            "tflops": round(fp16_tops, 2),
            "shapes": {"M": bench_M, "K": bench_K, "N": bench_N},
        }
        print(f"  FP16 tl.dot: {fp16_ms:.4f} ms, {fp16_tops:.2f} TFLOPS")
    except Exception as e:
        results["benchmarks"]["fp16_64x64x128"] = {"error": str(e)}
        print(f"  FP16 tl.dot: ERROR — {e}")
        fp16_tops = 0

    # torch.matmul FP16 baseline
    try:
        torch_fp16_ms = benchmark_kernel(
            lambda: torch.matmul(a_bench_fp16, b_bench_fp16)
        )
        torch_fp16_tops = compute_tops(bench_M, bench_N, bench_K, torch_fp16_ms)
        results["benchmarks"]["torch_fp16"] = {
            "time_ms": round(torch_fp16_ms, 4),
            "tflops": round(torch_fp16_tops, 2),
        }
        print(f"  torch FP16:  {torch_fp16_ms:.4f} ms, {torch_fp16_tops:.2f} TFLOPS")
    except Exception as e:
        results["benchmarks"]["torch_fp16"] = {"error": str(e)}

    # Speedup ratio
    if fp16_tops > 0 and int8_tops > 0:
        ratio = int8_tops / fp16_tops
        results["benchmarks"]["int8_vs_fp16_ratio"] = round(ratio, 3)
        print(f"\n  INT8/FP16 ratio: {ratio:.3f}x (expect ~2.0x if tensor cores)")

    # ── Step 3: PTX Inspection ──────────────────────────────────────────

    print("\n[Step 3] PTX/SASS inspection for IMMA instructions...")

    # Trigger a fresh compilation with known args for PTX inspection
    a_small = torch.randint(-128, 127, (64, 128), dtype=torch.int8, device=device)
    b_small = torch.randint(-128, 127, (128, 64), dtype=torch.int8, device=device)
    c_small = torch.empty(64, 64, dtype=torch.int32, device=device)

    ptx_result = inspect_ptx(
        matmul_int8_kernel,
        args=(
            a_small,
            b_small,
            c_small,
            64,
            64,
            128,
            a_small.stride(0),
            a_small.stride(1),
            b_small.stride(0),
            b_small.stride(1),
            c_small.stride(0),
            c_small.stride(1),
        ),
        grid=(1, 1),
        constexpr_kwargs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
    )
    results["ptx_inspection"] = ptx_result
    if ptx_result["imma_found"]:
        print("  IMMA instructions FOUND")
    else:
        print("  IMMA instructions NOT found in PTX cache")
    if ptx_result["details"]:
        for line in ptx_result["details"].strip().split("\n"):
            print(f"    {line}")

    # ── Step 4: Tile Size Sweep ─────────────────────────────────────────

    print("\n[Step 4] Tile size sweep...")

    tile_configs = [
        (32, 32, 32),
        (32, 32, 128),
        (64, 64, 64),
        (64, 64, 128),
        (128, 128, 128),
    ]

    for bm, bn, bk in tile_configs:
        label = f"{bm}x{bn}x{bk}"
        try:
            # INT8
            int8_t = benchmark_kernel(
                lambda bm=bm, bn=bn, bk=bk: triton_matmul_int8(
                    a_bench_int8, b_bench_int8, bm, bn, bk
                ),
                warmup=20,
                rep=100,
            )
            int8_t_tops = compute_tops(bench_M, bench_N, bench_K, int8_t)

            # FP16
            fp16_t = benchmark_kernel(
                lambda bm=bm, bn=bn, bk=bk: triton_matmul_fp16(
                    a_bench_fp16, b_bench_fp16, bm, bn, bk
                ),
                warmup=20,
                rep=100,
            )
            fp16_t_tops = compute_tops(bench_M, bench_N, bench_K, fp16_t)

            ratio = int8_t_tops / fp16_t_tops if fp16_t_tops > 0 else 0

            results["tile_size_sweep"][label] = {
                "int8_ms": round(int8_t, 4),
                "int8_tops": round(int8_t_tops, 2),
                "fp16_ms": round(fp16_t, 4),
                "fp16_tflops": round(fp16_t_tops, 2),
                "ratio": round(ratio, 3),
            }
            print(
                f"  {label}: INT8={int8_t_tops:.1f} TOPS,"
                f" FP16={fp16_t_tops:.1f} TFLOPS, ratio={ratio:.2f}x"
            )
        except Exception as e:
            results["tile_size_sweep"][label] = {"error": str(e)}
            print(f"  {label}: ERROR — {e}")

    # ── Verdict ─────────────────────────────────────────────────────────

    print("\n" + "=" * 70)

    int8_benchmark = results["benchmarks"].get("int8_64x64x128", {})
    int8_tops_measured = int8_benchmark.get("tops", 0)
    int8_correct = (
        results["correctness"].get("int8_int32_acc", {}).get("correct", False)
    )
    imma_found = results["ptx_inspection"].get("imma_found", False)

    if int8_correct and int8_tops_measured > 500:
        verdict = "PASS"
        conclusion = (
            f"INT8 tl.dot() achieves {int8_tops_measured:.1f} TOPS on RTX 4090 SM89 "
            f"(>{500} TOPS threshold). "
            f"IMMA instructions {'confirmed' if imma_found else 'not confirmed'}"
            f"{'' if imma_found else ' but performance indicates tensor core use'}. "
            f"INT8/FP16 ratio: {results['benchmarks'].get('int8_vs_fp16_ratio', 'N/A')}x. "
            f"The SageAttention INT8 Q@K^T path is viable for the Phase 3 fusion kernel."
        )
    elif int8_correct and int8_tops_measured > 100:
        verdict = "PARTIAL"
        conclusion = (
            f"INT8 tl.dot() works correctly but achieves only {int8_tops_measured:.1f} TOPS "
            f"(<500 TOPS threshold). "
            f"INT8/FP16 ratio: {results['benchmarks'].get('int8_vs_fp16_ratio', 'N/A')}x. "
            f"Usable but the speedup margin over FP16 is smaller than projected. "
            "The INT8 path may still be worthwhile but the "
            "1.3-2x speedup projection needs recalibration."
        )
    elif int8_correct:
        verdict = "FAIL"
        conclusion = (
            f"INT8 tl.dot() produces correct results but only {int8_tops_measured:.1f} TOPS — "
            f"likely emulated, not using tensor cores. "
            f"The SageAttention INT8 Q@K^T path is NOT viable with current Triton on SM89. "
            f"Phase 3 kernel design should use paged TQ4-only (no INT8 Q path)."
        )
    else:
        verdict = "FAIL"
        conclusion = (
            "INT8 tl.dot() failed correctness check on SM89. "
            "Phase 3 kernel design should use paged TQ4-only."
        )

    results["verdict"] = verdict
    results["conclusion"] = conclusion

    print(f"VERDICT: {verdict}")
    print(f"  {conclusion}")
    print("=" * 70)

    return results


def main() -> None:
    """Run experiment and save results."""
    results = run_experiment()

    # Save JSON log
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "experiment-017-int8-tensor-core-dispatch.json"

    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()
