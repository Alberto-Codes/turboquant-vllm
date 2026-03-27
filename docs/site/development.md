# Development

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >=3.12 | Required by pyproject.toml |
| uv | Latest | Package manager and build tool |
| CUDA GPU | Optional | Required for Triton kernels and GPU tests |

## Setup

```bash
git clone https://github.com/Alberto-Codes/turboquant-vllm.git
cd turboquant-vllm
uv sync --extra vllm
```

## Testing

```bash
# All tests (CPU)
uv run pytest tests/ -v

# Unit tests only
uv run pytest -m unit

# GPU tests
uv run pytest -m gpu

# Specific test file
uv run pytest tests/test_per_layer_cosine.py -v
```

## Linting

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run docvet check --all
```

## Building

```bash
uv build
# dist/turboquant_vllm-*.whl
# dist/turboquant_vllm-*.tar.gz
```

## Documentation

```bash
uv sync --extra docs
uv run mkdocs serve    # local preview at http://127.0.0.1:8000
uv run mkdocs build    # build to site/
```
