# Development Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >=3.12, <3.14 | Required by pyproject.toml |
| uv | Latest | Package manager and build tool |
| CUDA GPU | Optional | Required for Triton kernels and GPU tests |
| AMD ROCm | Optional | Supported via container (gfx1150/RDNA 3.5) |

---

## Installation

```bash
# Clone
git clone https://github.com/Alberto-Codes/turboquant-vllm.git
cd turboquant-vllm

# Install all dependencies (including dev group)
uv sync

# Install with vLLM support
uv sync --extra vllm

# Install with bitsandbytes support
uv sync --extra bnb
```

---

## Build

```bash
# Build wheel and sdist
uv build

# Outputs in dist/
#   turboquant_vllm-0.1.0-py3-none-any.whl
#   turboquant_vllm-0.1.0.tar.gz
```

---

## Testing

```bash
# Run all unit tests (CPU, no GPU required)
uv run pytest tests/ -v

# Run only unit tests
uv run pytest tests/ -v -m unit

# Run GPU tests (requires CUDA)
uv run pytest tests/ -v -m gpu

# Run with coverage
uv run pytest tests/ --cov --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_kv_cache.py -v
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Fast, isolated unit tests (CPU) |
| `gpu` | Require CUDA hardware |
| `benchmark` | Benchmark tests (require GPU) |
| `slow` | Slow-running tests |

### Test Configuration

- **Coverage threshold**: 95% (`fail_under = 95`)
- **Coverage source**: `turboquant_vllm` (excludes benchmark, triton, vllm modules)
- **Strict markers**: All markers must be declared
- **Random ordering**: `pytest-randomly` shuffles test order

---

## Linting and Formatting

```bash
# Format code
uv run ruff format .

# Lint (with auto-fix)
uv run ruff check . --fix

# Type checking
uv run ty check

# Docstring validation
uv run docvet check
```

### Ruff Configuration

- **Line length**: 88 (100 max for pycodestyle)
- **Target**: Python 3.12
- **Style**: Google docstring convention
- **Import sorting**: `known-first-party = ["turboquant_vllm"]`
- **Excludes**: `.venv`, `__pycache__`, `*.egg-info`

### Docvet Configuration

- **Fail on**: enrichment, freshness, coverage, griffe, presence
- **Min coverage**: 100% (all public APIs must be documented)
- **Excludes**: `scripts/`, `src/turboquant_vllm/vllm/`, `experiments/`

---

## Security

```bash
# Check for dependency vulnerabilities
uv run uv-secure

# Vulnerability suppressions are in pyproject.toml:
# [tool.uv-secure.vulnerability_criteria]
# allow_unused_ignores = false  (enforces cleanup of stale entries)
```

---

## Benchmark (requires GPU)

```bash
# Accuracy-only mode (no VRAM savings)
uv run python -m turboquant_vllm.benchmark \
    --model allenai/Molmo2-4B --bits 3

# Compressed mode (real VRAM savings, TQ4)
uv run python -m turboquant_vllm.benchmark \
    --model allenai/Molmo2-4B --bits 4 --compressed \
    --video /path/to/video.mp4 --max-new-tokens 256

# Save results to JSON
uv run python -m turboquant_vllm.benchmark \
    --model allenai/Molmo2-4B --bits 4 --compressed \
    --output results.json
```

---

## AMD ROCm Development

For AMD GPU development (tested on Radeon 890M / gfx1150):

```bash
# Build and run the ROCm container
./infra/run-rocm.sh

# Run tests inside container
./infra/run-rocm.sh pytest tests/ -v -m gpu

# The container:
#   - Uses rocm/pytorch:rocm7.1 base image
#   - Spoofs gfx1150 as gfx1100 (HSA_OVERRIDE_GFX_VERSION=11.0.0)
#   - Mounts project source at /workspace
#   - Mounts ~/.cache/huggingface for model caching
```

---

## CI/CD

### PyPI Publishing (`.github/workflows/publish.yml`)

- **Trigger**: Push of `v*` tags
- **Build**: `uv build` on ubuntu-latest
- **Verification**: Tag version must match `pyproject.toml` version
- **Smoke test**: Install wheel, verify metadata (name + version)
- **Publish**: OIDC trusted publishing (no stored tokens)
- **Concurrency**: Never cancels in-flight publishes

---

## Code Style Rules

From `.claude/rules/python.md`:

- Every file starts with `from __future__ import annotations`
- No mutable defaults (`[]`, `{}`) as function arguments
- No bare exceptions -- always catch specific types
- No `assert` for input validation (use `ValueError`/`TypeError`)
- f-strings for formatting, `%`-formatting for logger calls only
- Comments explain WHY, not WHAT
- Module size gate: split modules over 500 lines
