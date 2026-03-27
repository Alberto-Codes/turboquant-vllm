# Project Documentation Index

## Project Overview

- **Type:** Monolith (single Python library + vLLM plugin)
- **Primary Language:** Python 3.12+
- **Architecture:** Layered library with strict DAG dependency flow

## Quick Reference

- **Package:** `turboquant-vllm` (pip-installable, src-layout)
- **Tech Stack:** PyTorch + Triton + HuggingFace transformers + scipy
- **Build System:** uv (uv_build backend)
- **Entry Point:** `src/turboquant_vllm/__init__.py` (8 public exports)
- **CLI:** `python -m turboquant_vllm.benchmark`
- **vLLM Plugin:** Auto-registered via `vllm.general_plugins` entry point
- **vLLM Usage:** `vllm serve <model> --attention-backend CUSTOM`
- **Architecture Pattern:** Layered library (lloyd_max -> quantizer -> compressors -> kv_cache)

## Generated Documentation

- [Project Overview](./project-overview.md) -- Summary, tech stack, architecture, key decisions
- [Architecture](./ARCHITECTURE.md) -- Module map, dependency DAG, data flow diagrams, design decisions
- [Source Tree Analysis](./source-tree-analysis.md) -- Annotated directory structure with critical folders
- [Development Guide](./development-guide.md) -- Prerequisites, install, build, test, lint, CI/CD

## Existing Documentation

- [README](../README.md) -- Install, quickstart, benchmarks, citation
- [Roadmap](./ROADMAP.md) -- Implementation status (completed/in-progress/planned), experiment results

## Research Documents

- [TurboQuant Paper Analysis](./research/technical-google-turboquant-research-2026-03-25.md) -- Core algorithm deep dive (arXiv 2504.19874)
- [Fused Triton Kernel Research](./research/technical-fused-turboquant-triton-kernel-research-2026-03-25.md) -- Q@K^T kernel design
- [Flash Attention Fusion Research](./research/technical-flash-attention-fusion-turboquant-kv-cache-research-2026-03-26.md) -- FA + TQ4 fusion strategy
- [Triton FA Tutorial Deep Dive](./research/technical-triton-flash-attention-tutorial-deep-dive-2026-03-26.md) -- Triton Flash Attention implementation guide

## Experiment Logs

- [Experiment 001](../experiments/logs/experiment-001-initial-validation.md) -- Initial TurboQuantProd validation (failed: QJL invisible in drop-in mode)
- [Experiment 002](../experiments/logs/experiment-002-mse-fix-validation.md) -- MSE-only fix (passed: identical text output)
- [Experiment 003](../experiments/logs/experiment-003-compressed-vram.md) -- CompressedDynamicCache (passed: 1.94x compression, fp16 norms bug found)
- [Experiment 004](../experiments/logs/experiment-004-tq4-nibble-vram.md) -- TQ4 nibble packing (3.76x compression)
- [Experiment 005](../experiments/logs/experiment-005-incremental-dequant.md) -- Incremental dequant (3.76x at 1.78x overhead)
- [Experiment 006](../experiments/logs/experiment-006-amd-rocm-gfx1150.md) -- AMD ROCm gfx1150 validation
- [Experiment 007](../experiments/logs/experiment-007-e2e-amd-validation.md) -- E2E AMD validation
- [Experiment 008](../experiments/logs/experiment-008-triton-fused-rocm.md) -- Triton fused kernel on ROCm

## Getting Started

```bash
# Install from PyPI
pip install turboquant-vllm[vllm]

# Use with vLLM (no code changes)
vllm serve allenai/Molmo2-4B --attention-backend CUSTOM

# Or use with HuggingFace directly
from turboquant_vllm import CompressedDynamicCache
from transformers import DynamicCache

cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)
```
