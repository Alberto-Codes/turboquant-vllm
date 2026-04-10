# Container Deployment

This page covers containerizing the optional vLLM plugin bridge from `turboquant-vllm`. Use it when you specifically need the repo's out-of-tree CUSTOM backend in a containerized environment.

For the primary project workflow, prefer the HuggingFace/reference path. For native vLLM TurboQuant serving, prefer the upstream in-tree path as it matures.

## Build the Image

```bash
podman build -t vllm-turboquant -f infra/Containerfile.vllm .
```

The build installs `turboquant-vllm` from PyPI and verifies the plugin entry point registers correctly.

## Run with TQ4 Compression

```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  --shm-size=8g \
  -v vllm-models:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm-turboquant \
  --model allenai/Molmo2-8B \
  --attention-backend CUSTOM \
  --dtype auto \
  --max-model-len 6144 \
  --max-num-batched-tokens 6144 \
  --enforce-eager \
  --gpu-memory-utilization 0.90
```

!!! tip "No code changes required"
    The `--attention-backend CUSTOM` flag is the only difference from a standard vLLM deployment. The plugin registers automatically via `vllm.general_plugins`.

## Quadlet (systemd Integration)

For persistent deployments on Podman, use a Quadlet container file at `~/.config/containers/systemd/vllm-turboquant.container`:

```ini
[Container]
Image=localhost/vllm-turboquant:latest
ContainerName=vllm-tq
SecurityLabelDisable=true
ShmSize=8g

AddDevice=nvidia.com/gpu=all

Exec=allenai/Molmo2-8B \
    --attention-backend CUSTOM \
    --dtype auto \
    --max-model-len 6144 \
    --max-num-batched-tokens 6144 \
    --enforce-eager \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

Volume=vllm-models.volume:/root/.cache/huggingface
PublishPort=8000:8000

HealthCmd=bash -c 'echo > /dev/tcp/localhost/8000'
HealthInterval=30s
HealthTimeout=10s
HealthRetries=5
HealthStartPeriod=300s

[Service]
Restart=always
TimeoutStartSec=900

[Install]
WantedBy=default.target
```

Then reload and start:

```bash
systemctl --user daemon-reload
systemctl --user start vllm-turboquant
```

## What Gets Compressed

| Data | Compressed | Format |
|------|-----------|--------|
| Key cache vectors | Yes | uint8 nibble-packed indices + fp32 norms |
| Value cache vectors | Yes | uint8 nibble-packed indices + fp32 norms |
| Rotation matrices | No | Generated once per layer from fixed seed |
| Lloyd-Max codebook | No | Computed once, shared across all layers |

## Memory Considerations

The TQ4 backend compresses KV cache pages to **68 bytes/token/head** vs 256 bytes for FP16 (3.76x compression). This is most impactful at long context lengths where KV cache dominates memory.

!!! warning "GPU memory for model weights is unchanged"
    TurboQuant only compresses the KV cache, not model weights. Peak VRAM during prefill is activation-dominated — compression savings are most visible in the permanent KV cache storage during generation.

| GPU | Model | Max Context | Notes |
|-----|-------|-------------|-------|
| RTX 4090 (24 GB) | Molmo2-8B | 6144 | `--gpu-memory-utilization 0.90` |
| RTX 4090 (24 GB) | Molmo2-4B | 11264 | Validated in experiments |

## Verifying the Plugin

Confirm the TQ4 backend is active in the container logs:

```
INFO [cuda.py:257] Using AttentionBackendEnum.CUSTOM backend.
```

Or check from inside the container:

```bash
podman exec <container> python3 -c "
from turboquant_vllm.vllm import TQ4AttentionBackend
import importlib.metadata
v = importlib.metadata.version('turboquant-vllm')
print(f'turboquant-vllm {v} — plugin loaded')
"
```
