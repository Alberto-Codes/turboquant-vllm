# Architecture — `turboquant_vllm`

Implementation of Google's **TurboQuant** algorithm (ICLR 2026, arXiv 2504.19874) for compressing transformer KV caches to 3–4 bits per coordinate on consumer GPUs.

---

## High-Level Module Map

```mermaid
---
title: turboquant_vllm — module overview
---
flowchart TD
    subgraph api ["Public API  __init__.py"]
        direction LR
        API([turboquant_vllm])
    end

    subgraph core ["Core Quantization"]
        direction LR
        LM["`**lloyd_max.py**
        Optimal scalar codebook`"]
        QZ["`**quantizer.py**
        TurboQuantMSE · TurboQuantProd`"]
    end

    subgraph wrap ["Production Wrappers"]
        direction LR
        CP["`**compressors.py**
        CompressorMSE · CompressorV2`"]
        KV["`**kv_cache.py**
        TurboQuantKVCache
        CompressedDynamicCache`"]
    end

    subgraph triton ["Triton Kernels (WIP)"]
        direction LR
        TI["`**triton/__init__.py**
        Package init`"]
        TK["`**triton/fused_qk_attention.py**
        Fused Q@K^T kernel`"]
        TM["`**triton/molmo2_integration.py**
        Molmo2 attention override`"]
    end

    subgraph bench ["Benchmark Harness"]
        BM["`**benchmark.py**
        Molmo2 inference A/B testing`"]
    end

    API -.-> LM & QZ & CP & KV & TK & TM
    QZ --> LM
    CP --> QZ
    KV --> CP
    TM --> QZ
    TM --> TK
    BM --> KV

    classDef layer fill:#1a1a2e,color:#eee,stroke:#333
    classDef apiNode fill:#0f4c75,color:#fff,stroke:#0f4c75
    classDef tritonNode fill:#c0392b,color:#fff,stroke:#c0392b
    class API apiNode
    class LM,QZ,CP,KV,BM layer
    class TI,TK,TM tritonNode
```

---

## Dependency Flow Between Modules

```mermaid
---
title: Import DAG — each arrow means "is imported by"
---
flowchart LR
    LM([lloyd_max]):::math ==>|imported by| QZ([quantizer]):::algo
    QZ ==>|imported by| CP([compressors]):::wrap
    CP ==>|imported by| KV([kv_cache]):::integ
    KV ==>|imported by| BM([benchmark]):::cli
    TK([triton/fused_qk_attention]):::triton
    QZ ==>|imported by| TM([triton/molmo2_integration]):::triton
    TK ==>|imported by| TM

    classDef math  fill:#16a085,color:#fff,stroke:#16a085
    classDef algo  fill:#2980b9,color:#fff,stroke:#2980b9
    classDef wrap  fill:#8e44ad,color:#fff,stroke:#8e44ad
    classDef integ fill:#d35400,color:#fff,stroke:#d35400
    classDef cli   fill:#7f8c8d,color:#fff,stroke:#7f8c8d
    classDef triton fill:#c0392b,color:#fff,stroke:#c0392b
```

No circular dependencies — the graph is a strict DAG from foundational math (`lloyd_max`) up through integration (`kv_cache`) and the CLI harness (`benchmark`). The `triton/` module is a parallel branch: `fused_qk_attention` is standalone (no internal dependencies), and `molmo2_integration` depends on both `quantizer` (for rotation matrices) and `fused_qk_attention` (for the fused kernel).

---

## Public API Surface

Everything exported from `__init__.py`, grouped by purpose:

```mermaid
mindmap
  root((turboquant_vllm))
    **Quantizers**
      TurboQuantMSE
        Stage 1 only
        MSE-optimal
      TurboQuantProd
        Stage 1 + 2
        QJL correction
    **Compressors**
      TurboQuantCompressorMSE
        Value cache
      TurboQuantCompressorV2
        Key cache
    **Cache Integration**
      TurboQuantKVCache
        Accuracy benchmark
        No VRAM savings
      CompressedDynamicCache
        Real VRAM savings
        uint8 storage
    **Codebook**
      LloydMaxCodebook
        Dataclass
      solve_lloyd_max
        Factory function
        lru_cache'd
```

---

## Module Deep Dives

### 1. `lloyd_max.py` — Optimal Scalar Codebook

Solves the Lloyd-Max conditions for scalar quantization of Beta-distributed coordinates (the distribution that emerges after random orthogonal rotation of unit vectors).

```mermaid
flowchart TD
    A[/"Input: dim d, bits b"/] --> B{"d ≥ 64?"}
    B -- Yes --> C["Gaussian PDF  N(0, 1/d)"]
    B -- No --> D["Exact Beta PDF  Beta(d-1/2, d-1/2)"]
    C --> E["Initialize 2^b centroids uniformly in ±3σ"]
    D --> E
    E --> F["Compute boundaries — midpoints of adjacent centroids"]
    F --> G["Update centroids via conditional expectation
    scipy.integrate.quad"]
    G --> H{"max shift < tol?"}
    H -- No --> F
    H -- Yes --> I(["`Return **centroids + boundaries**
    as torch.Tensor`"]):::done

    classDef done fill:#16a085,color:#fff,stroke:#16a085
```

**Key types:**

| Symbol | Type | Role |
|---|---|---|
| `solve_lloyd_max()` | Function | Computes optimal centroids & boundaries for (dim, bits). Results are `@lru_cache`d. |
| `LloydMaxCodebook` | Dataclass | Holds precomputed centroids/boundaries. `quantize()` → `torch.bucketize`; `dequantize()` → centroid lookup. |

---

### 2. `quantizer.py` — Two-Stage Vector Quantizer

The core TurboQuant algorithm: random orthogonal rotation + Lloyd-Max scalar quantization (Stage 1), with optional QJL residual correction (Stage 2).

```mermaid
flowchart TD
    subgraph stage1 ["Stage 1 — TurboQuantMSE"]
        direction TB
        X[/"x ∈ ℝ^d"/] ==> NORM["Extract norm ‖x‖"]
        NORM ==> UNIT["Normalize  x̂ = x / ‖x‖"]
        UNIT ==> ROT["Rotate  y = x̂ · Πᵀ
        Haar-random orthogonal Π"]
        ROT ==> SQ["Scalar quantize each y_i
        via LloydMaxCodebook"]
        SQ ==> IDX["indices ∈ {0..2^b-1}^d"]
    end

    subgraph stage2 ["Stage 2 — QJL Correction  (TurboQuantProd only)"]
        direction TB
        IDX -.-> DEQMSE["Dequantize → k̂_mse"]
        DEQMSE -.-> RES["Residual  r = x - k̂_mse"]
        RES -.-> PROJ["Project S·r  (Gaussian S ∈ ℝ^m×d)"]
        PROJ -.-> SIGN["Store sign(S·r) — 1 bit per dim"]
    end

    IDX --> OUT1(["`**Output:** indices, norms`"]):::mse
    SIGN --> OUT2(["`**Output:** + qjl_signs, residual_norms`"]):::prod

    classDef mse  fill:#2980b9,color:#fff,stroke:#2980b9
    classDef prod fill:#8e44ad,color:#fff,stroke:#8e44ad
```

**Inner-product estimator** (Stage 2):

$$\langle q, k \rangle \;\approx\; \langle q,\, \hat{k}_{\text{mse}} \rangle \;+\; \|r\| \cdot \sqrt{\tfrac{\pi}{2}} \cdot \tfrac{1}{m} \cdot \langle S q,\; \text{sign}(S r) \rangle$$

| Class | Bits Used | Output | Best For |
|---|---|---|---|
| `TurboQuantMSE` | All *b* bits → Lloyd-Max | `(indices, norms)` | **Value cache** — reconstruction quality |
| `TurboQuantProd` | *(b−1)* Lloyd-Max + 1 QJL | `(indices, norms, qjl_signs, residual_norms)` | **Key cache** — unbiased Q·Kᵀ estimation |

**Bit budget allocation** — the key design tradeoff at 3-bit:

```mermaid
---
title: TurboQuantMSE — 3-bit budget
---
pie
    "Lloyd-Max (MSE)" : 3
```

```mermaid
---
title: TurboQuantProd — 3-bit budget
---
pie
    "Lloyd-Max (MSE)" : 2
    "QJL sign bits" : 1
```

---

### 3. `compressors.py` — Production Tensor Wrappers

Adapts the raw quantizers to real model tensor shapes `(batch, heads, seq_len, head_dim)`, handling dtype conversion and device placement.

```mermaid
---
title: compressors.py — class relationships
---
classDiagram
    namespace DataContainers {
        class CompressedKeys {
            +indices : Tensor
            +norms : Tensor
            +qjl_signs : Tensor
            +residual_norms : Tensor
            +original_dtype : dtype
        }
        class CompressedValues {
            +indices : Tensor
            +norms : Tensor
            +original_dtype : dtype
        }
    }

    namespace Compressors {
        class TurboQuantCompressorV2 {
            -quantizer : TurboQuantProd
            +compress(keys) CompressedKeys
            +decompress(compressed) Tensor
            +asymmetric_attention_scores(query, compressed) Tensor
        }
        class TurboQuantCompressorMSE {
            -quantizer : TurboQuantMSE
            +compress(values) CompressedValues
            +decompress(compressed) Tensor
        }
    }

    TurboQuantCompressorV2 --> CompressedKeys : produces
    TurboQuantCompressorMSE --> CompressedValues : produces
    TurboQuantCompressorV2 ..> TurboQuantProd : wraps
    TurboQuantCompressorMSE ..> TurboQuantMSE : wraps
```

| Compressor | Quantizer | Target | Special Method |
|---|---|---|---|
| `TurboQuantCompressorV2` | `TurboQuantProd` | Key cache | `asymmetric_attention_scores()` — computes Q·Kᵀ directly from compressed keys without full decompression |
| `TurboQuantCompressorMSE` | `TurboQuantMSE` | Value cache | — |

---

### 4. `kv_cache.py` — HuggingFace Cache Integration

Two integration modes that non-invasively monkey-patch `DynamicCache.update()`:

```mermaid
flowchart TD
    subgraph mode1 ["Mode 1 — TurboQuantKVCache  (accuracy benchmark)"]
        direction TB
        U1[/"cache.update(K, V)"/] ==> C1["Compress K, V"]
        C1 ==> D1["Immediately decompress"]
        D1 ==> S1[["Store lossy FP32
        in DynamicCache"]]:::warn
        S1 ==> R1([Return decompressed K, V])
    end

    subgraph mode2 ["Mode 2 — CompressedDynamicCache  (real VRAM savings)"]
        direction TB
        U2[/"cache.update(K, V)"/] ==> C2["Compress → uint8 + fp32"]
        C2 ==> ST2[(Append to
        compressed storage)]:::good
        ST2 ==> FREE["Free layer L−1
        decompressed tensors"]
        FREE ==> DQ2["Dequantize NEW tokens only"]
        DQ2 ==> R2([Return decompressed K, V])
    end

    classDef warn fill:#e74c3c,color:#fff,stroke:#c0392b
    classDef good fill:#16a085,color:#fff,stroke:#16a085
```

**Memory model** (CompressedDynamicCache, head_dim=128):

**TQ3 (3-bit, unpacked) — 132 bytes, 1.94x compression:**

```mermaid
---
title: "TQ3: 132 B per token per head"
---
packet-beta
  0-127: "uint8 indices (128 B) — one per head_dim coordinate"
  128-131: "fp32 norm (4 B)"
```

**TQ4 (4-bit, nibble-packed) — 68 bytes, 3.76x compression:**

```mermaid
---
title: "TQ4: 68 B per token per head"
---
packet-beta
  0-63: "nibble-packed uint8 (64 B) — two 4-bit indices per byte"
  64-67: "fp32 norm (4 B)"
```

> **Why TQ4 over TQ3 packing?** TQ4 nibble packing is trivial (`(a << 4) | b`)
> and gives 3.76x compression with ~97% cosine similarity. TQ3 bit-packing (3-bit
> values crossing byte boundaries) is hard — no PyTorch/Triton implementation exists.
> The 30% extra compression isn't worth a custom kernel at this stage.

> **Why fp32 norms?** fp16 norms caused garbled output at 10K+ token sequences.
> The 0.01% per-vector precision loss accumulated across 36 transformer layers,
> flipping low-confidence token predictions. The 2 extra bytes per vector are
> negligible.

```mermaid
sequenceDiagram
    participant Model as Transformer Layer L
    participant Cache as CompressedDynamicCache
    participant Store as Compressed Storage

    activate Model
    Model ->>+ Cache: update(K, V, layer_idx=L)
    Cache ->> Cache: Free layer L−1 decompressed tensors
    Cache ->> Cache: Compress K, V → uint8 + fp32
    Cache ->>+ Store: Append compressed layer L
    Store -->>- Cache: Stored
    Cache ->> Cache: Dequantize NEW tokens only
    Cache -->>- Model: Return decompressed (K, V)
    deactivate Model
```

Both wrappers use the same integration pattern: save the original `update()` method, replace it with a wrapper, and expose `restore()` to undo the patch.

**Cache wrapper lifecycle** — the monkey-patch state machine applies to both `TurboQuantKVCache` and `CompressedDynamicCache`:

```mermaid
---
title: Cache wrapper lifecycle
---
stateDiagram-v2
    [*] --> Initializing : __init__(cache, head_dim, bits)
    Initializing --> Active : Patch cache.update()
    Active --> Disabled : disable()
    Disabled --> Active : enable()
    Active --> Restored : restore()
    Disabled --> Restored : restore()
    Restored --> [*]

    state Active {
        [*] --> Compressing
        Compressing --> Decompressing : quantize K, V
        Decompressing --> Storing : reconstruct for attention
        Storing --> [*] : return to model
    }

    note right of Active
        cache.update() is intercepted
        Original method saved as _original_update
    end note

    note right of Restored
        cache.update = _original_update
        Wrapper fully detached
    end note
```

---

### 5. `benchmark.py` — CLI Benchmark Harness

Orchestrates end-to-end A/B testing on Molmo2 models via HuggingFace `transformers`.

```mermaid
flowchart LR
    subgraph cli ["CLI Arguments"]
        direction TB
        ARGS[/"`--model  --bits  --compressed
        --prompt  --video  --output`"/]
    end

    ARGS ==> LOAD

    subgraph pipeline ["run_benchmark()"]
        direction TB
        LOAD["load_model()
        AutoModelForImageTextToText"]
        ==> CFG["Detect model config
        head_dim · num_layers · num_kv_heads"]
        ==> BASE["Baseline inference
        standard DynamicCache"]
        ==> PATCH["_patch_cache() — monkey-patch
        DynamicCache.__init__"]
        ==> TQ["TurboQuant inference"]
        ==> RESTORE["Restore original __init__"]
        ==> CMP{"Compare outputs
        text match · VRAM Δ · time ratio"}
    end

    CMP ==> JSON(["`JSON results
    stdout or --output file`"]):::out

    classDef out fill:#16a085,color:#fff,stroke:#16a085
```

---

## End-to-End Data Flow

From raw KV tensors to compressed storage and back:

```mermaid
flowchart TD
    KV[/"`**K or V tensor**
    (batch, heads, seq, head_dim)
    bfloat16`"/]:::input
    ==> FLOAT["Cast to float32"]
    ==> NORM["Extract ‖x‖ → norms"]
    ==> UNIT["Normalize to unit sphere"]
    ==> ROT["Random orthogonal rotation Π"]
    ==> BETA["Each coordinate ~ Beta(d/2, d/2)
    ≈ N(0, 1/d) for d ≥ 64"]
    ==> LM["Lloyd-Max quantize
    torch.bucketize on boundaries"]
    ==> IDX["uint8 indices"]
    ==> PACK{"bits=4?"}
    PACK -- Yes --> NIB["Nibble pack (2 per byte)"]
    PACK -- No --> RAW["Store as uint8"]
    NIB ==> STORE[("`**Compressed storage**
    packed/uint8 indices + fp32 norms`")]:::store
    RAW ==> STORE

    STORE ==> LOOKUP["Centroid lookup  centroids#91;indices#93;"]
    ==> INVROT["Inverse rotation Πᵀ"]
    ==> SCALE["Rescale by norms"]
    ==> CAST["Cast to original dtype"]
    ==> OUT(["`**Reconstructed tensor**
    (batch, heads, seq, head_dim)`"]):::output

    classDef input  fill:#2980b9,color:#fff,stroke:#2980b9
    classDef store  fill:#16a085,color:#fff,stroke:#16a085
    classDef output fill:#8e44ad,color:#fff,stroke:#8e44ad
```

---

## Compression Mode Comparison

```mermaid
quadrantChart
    title Compression modes — tradeoff space
    x-axis Low VRAM Savings --> High VRAM Savings
    y-axis Low Fidelity --> High Fidelity
    quadrant-1 Ideal
    quadrant-2 Quality focus
    quadrant-3 Avoid
    quadrant-4 Memory focus
    FP16 Baseline: [0.05, 0.95]
    TurboQuantKVCache accuracy-only: [0.10, 0.78]
    CompressedDynCache TQ3: [0.55, 0.75]
    CompressedDynCache TQ4 nibble: [0.78, 0.82]
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **MSE-only for drop-in mode** | Standard attention does `Q @ K.T` on decompressed keys, so QJL correction bits are wasted. Full 3-bit MSE gives ~95% cosine similarity vs ~87% with 2-bit MSE + 1-bit QJL. |
| **TQ4 nibble packing over TQ3 bit-packing** | 4-bit indices pack trivially (2 per byte via bit-shift). 3-bit indices cross byte boundaries — no PyTorch/Triton implementation exists. TQ4 gives 3.76x compression with ~97% quality vs TQ3's 4.92x at ~95%. The 30% gap isn't worth a custom kernel. |
| **fp32 norms, not fp16** | fp16 norm precision loss compounds across 36 transformer layers, flipping low-confidence logits at 10K+ token sequences. fp32 costs only 2 extra bytes per vector (1.94x → 1.94x for TQ3, negligible for TQ4). |
| **Non-invasive monkey-patching** | Avoids subclassing `DynamicCache`, which is fragile across `transformers` versions. The wrapper saves and restores the original method. |
| **`@lru_cache` on Lloyd-Max** | A 32-layer model creates 64 compressors (K+V). Without caching, `scipy.integrate.quad` would run for 2+ minutes at init. |
| **Incremental dequantization** | `CompressedDynamicCache` dequantizes only new tokens each decode step (not the full cache), maintaining a running decompressed buffer per layer. Previous layer buffers are freed to keep peak VRAM to one decompressed layer at a time. |
| **Haar-random rotation via QR** | QR decomposition of a Gaussian matrix produces a uniformly distributed orthogonal matrix, ensuring coordinates are i.i.d. Beta-distributed. |
