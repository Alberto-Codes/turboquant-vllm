# Source Tree Analysis

Annotated directory structure for the turboquant-vllm project.

---

## Directory Tree

```
turboquant-vllm/
├── pyproject.toml                          # Project metadata, dependencies, tool config
├── uv.lock                                 # Lockfile (uv package manager)
├── README.md                               # Project overview, quickstart, key findings
│
├── src/
│   └── turboquant_vllm/                    # Main package (src-layout)
│       ├── __init__.py                     # Public API: 8 exports
│       ├── lloyd_max.py                    # Lloyd-Max optimal scalar codebook solver
│       ├── quantizer.py                    # TurboQuantMSE (Stage 1) + TurboQuantProd (Stage 2)
│       ├── compressors.py                  # Production tensor wrappers for KV cache shapes
│       ├── kv_cache.py                     # HuggingFace DynamicCache integration (2 modes)
│       ├── benchmark.py                    # CLI A/B benchmark harness for Molmo2
│       │
│       ├── triton/                         # Fused Triton GPU kernels (WIP)
│       │   ├── __init__.py                 # Re-exports: 12 public symbols
│       │   ├── flash_attention.py          # Phase 1: Vanilla Flash Attention v2
│       │   ├── flash_attention_tq4.py      # Phase 2: Fused TQ4 K decompression in FA
│       │   ├── flash_attention_tq4_kv.py   # Phase 3: Fused TQ4 K+V decompression in FA
│       │   ├── attention_interface.py      # HF AttentionInterface backend registration
│       │   ├── fused_qk_attention.py       # Standalone fused Q@K^T kernel (legacy)
│       │   ├── molmo2_integration.py       # Molmo2-specific attention patching + runner
│       │   ├── tq4_compress.py             # Fused compress: norm+rotate+quantize+pack
│       │   └── tq4_decompress.py           # Fused decompress: unpack+gather+scale
│       │
│       └── vllm/                           # vLLM serving integration
│           ├── __init__.py                 # Plugin entry point (vllm.general_plugins)
│           └── tq4_backend.py              # TQ4AttentionBackend + TQ4AttentionImpl
│
├── tests/                                  # Test suite (~180+ tests)
│   ├── conftest.py                         # Shared fixtures, seeds, module-scoped codebooks
│   ├── test_lloyd_max.py                   # Codebook solver correctness (10 tests)
│   ├── test_quantizer.py                   # MSE + Prod quantizer validation (12 tests)
│   ├── test_compressors.py                 # Compressor wrappers + attention scores (11 tests)
│   ├── test_kv_cache.py                    # DynamicCache integration (31 tests)
│   ├── test_flash_attention.py             # Phase 1 FA correctness (17 tests, GPU)
│   ├── test_flash_attention_tq4.py         # Phase 2 fused TQ4 K (9 tests, GPU)
│   ├── test_flash_attention_tq4_kv.py      # Phase 3 fused TQ4 K+V (8 tests, GPU)
│   └── test_vllm_backend.py               # vLLM backend registration + round-trip (39 tests)
│
├── experiments/                            # GPU experiment scripts
│   ├── experiment_007_e2e_amd_validation.py
│   ├── experiment_008_triton_fused_rocm.py
│   ├── experiment_009_triton_fa_e2e_validation.py
│   ├── experiment_010_fused_tq4_kv_e2e.py
│   ├── experiment_011_molmo2_8b_tq4_validation.py
│   ├── experiment_012_turboquantprod_quality.py
│   ├── experiment_013_fp32_eager_baseline.py
│   ├── experiment_014_full_episode_benchmark.py
│   ├── experiment_015_profile_tq4_cache_bottleneck.py
│   ├── experiment_016_triton_kernel_benchmark.py
│   └── logs/                               # Experiment results (JSON + markdown)
│       ├── experiment-001-initial-validation.md
│       ├── experiment-002-mse-fix-validation.md
│       ├── experiment-003-compressed-vram.md
│       ├── experiment-004-tq4-nibble-vram.md
│       ├── experiment-005-incremental-dequant.md
│       ├── experiment-006-amd-rocm-gfx1150.md
│       ├── experiment-007-e2e-amd-validation.md
│       ├── experiment-008-triton-fused-rocm.md
│       └── *.json                          # Raw benchmark data
│
├── docs/                                   # Project documentation
│   ├── ARCHITECTURE.md                     # Module map, DAGs, data flow, design decisions
│   ├── ROADMAP.md                          # Implementation status, experiments, next steps
│   └── research/                           # Technical research documents
│       ├── technical-google-turboquant-research-2026-03-25.md
│       ├── technical-fused-turboquant-triton-kernel-research-2026-03-25.md
│       ├── technical-flash-attention-fusion-turboquant-kv-cache-research-2026-03-26.md
│       └── technical-triton-flash-attention-tutorial-deep-dive-2026-03-26.md
│
├── infra/                                  # Infrastructure
│   ├── Containerfile.rocm                  # AMD ROCm dev container (gfx1150/RDNA 3.5)
│   └── run-rocm.sh                         # Container launcher with TTY + HF cache handling
│
├── scripts/                                # (empty — utility scripts placeholder)
├── dist/                                   # Built distributions
│   ├── turboquant_vllm-0.1.0-py3-none-any.whl
│   └── turboquant_vllm-0.1.0.tar.gz
│
└── .github/
    └── workflows/
        └── publish.yml                     # PyPI publish via OIDC trusted publishing
```

---

## Critical Folders

| Directory | Purpose |
|-----------|---------|
| `src/turboquant_vllm/` | Core library — all public API surface |
| `src/turboquant_vllm/triton/` | Fused GPU kernels (Flash Attention + TQ4 compress/decompress) |
| `src/turboquant_vllm/vllm/` | vLLM serving backend plugin |
| `tests/` | Full test suite with unit, GPU, and regression tests |
| `experiments/` | GPU validation scripts and result logs |
| `docs/research/` | Technical research informing implementation decisions |
| `infra/` | AMD ROCm container for cross-platform GPU development |

## Entry Points

| Entry Point | Type | Description |
|-------------|------|-------------|
| `src/turboquant_vllm/__init__.py` | Library API | 8 public exports |
| `src/turboquant_vllm/benchmark.py` | CLI (`__main__`) | `python -m turboquant_vllm.benchmark` |
| `src/turboquant_vllm/vllm/__init__.py` | vLLM plugin | Auto-registered via `vllm.general_plugins` entry point |
| `.github/workflows/publish.yml` | CI/CD | Tag-triggered PyPI publish |
