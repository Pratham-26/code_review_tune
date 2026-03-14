# Training Time & Memory Estimation Guide

## Training Time Estimation

### Basic Formula

```
Total Time = Steps × Time per Step

Steps = Dataset Size / (Batch Size × Gradient Accumulation)
```

### Example Calculation

**Given:**
- Dataset: 40,265 samples
- Batch size: 6
- Gradient accumulation: 1
- Time per step: 3 seconds

**Calculation:**
```
Steps = 40,265 / (6 × 1) = 6,711 steps

Total Time = 6,711 × 3s = 20,133s ≈ 5.6 hours
```

### Impact of Changes

| Change | Steps | Time Impact |
|--------|-------|-------------|
| Batch 6 → 8 | 6,711 → 5,033 | ~25% faster |
| Batch 6 → 4 | 6,711 → 10,067 | ~50% slower |
| Grad Accum 1 → 4 | Same steps | ~3-4× slower (4 forward passes per step) |

---

## Memory Estimation

### VRAM Components

```
Total VRAM = Model Weights + Activations + Optimizer States + Gradients
```

### Component Breakdown

| Component | Size Formula | Notes |
|-----------|--------------|-------|
| Model (4-bit) | Params × 0.5 bytes | 0.8B model ≈ 0.5 GB |
| Model (16-bit) | Params × 2 bytes | 0.8B model ≈ 1.6 GB |
| LoRA Adapters | ~50-200 MB | Depends on rank and modules |
| Activations | Batch × Seq × Hidden × Layers | Largest variable |
| Optimizer (8-bit) | 2 × Trainable Params | AdamW needs 2 states |
| Gradients | 1 × Trainable Params | Per trainable param |

### Activation Memory Deep Dive

```
Activation Memory ∝ Batch Size × Sequence Length × Hidden Dimension × Number of Layers
```

**With Gradient Checkpointing:**
- Reduces activation memory by ~60-75%
- Trades compute for memory (recomputation during backward pass)

### Example: Qwen3.5-0.8B with LoRA

**Config:**
- Batch size: 6
- Seq length: 2056
- 4-bit quantization
- LoRA (r=16) on all linear layers
- Gradient checkpointing: enabled

**Estimated Breakdown:**
```
Model (4-bit):           ~0.5 GB
LoRA adapters:           ~0.1 GB
Activations:             ~12-15 GB (with checkpointing)
Optimizer (8-bit):       ~0.2 GB
Gradients:               ~0.1 GB
Overhead/Fragmentation:  ~2-3 GB
─────────────────────────────────
Total:                   ~15-19 GB
```

---

## Quick Reference: Impact of Changes

### Memory Impact

| Change | Memory Impact | Reason |
|--------|---------------|--------|
| Batch ×2 | ×1.5-2 | More activations per step |
| Seq Length ×2 | ~×2 | Linear scaling with activations |
| Grad Accum ×2 | No change | Same batch per forward pass |
| 4-bit → 16-bit | +1-2 GB | Model weights larger |
| Disable checkpointing | ×2-4 | Store all activations |

### Time Impact

| Change | Time Impact | Reason |
|--------|-------------|--------|
| Batch ×2 | ×0.5-0.6 | Fewer steps, slightly more per step |
| Seq Length ×2 | ×1.5-2 | Quadratic attention, more compute |
| Grad Accum ×2 | ×2 | More forward passes per step |
| Enable checkpointing | ×1.1-1.2 | Recomputation overhead |

---

## Coverage Analysis

### What is Coverage?

Coverage = percentage of samples that fit within `max_seq_length` without truncation.

### Calculation

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Load data
ds = load_dataset("your-dataset", split="train")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Calculate token lengths
lengths = []
for sample in ds:
    text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    tokens = tokenizer.encode(text)
    lengths.append(len(tokens))

lengths = np.array(lengths)

# Coverage at different max lengths
for max_len in [2048, 3072, 4096]:
    coverage = (lengths <= max_len).sum() / len(lengths) * 100
    print(f"{max_len}: {coverage:.1f}%")
```

### Example Output

| Max Length | Coverage | Truncated Samples | Use Case |
|------------|----------|-------------------|----------|
| 2048 | 92.8% | 2,908 | Fast training, general tasks |
| 2500 | 95.9% | 1,644 | Balanced |
| 3072 | 98.2% | 709 | High quality |
| 4096 | 99.7% | 118 | Maximum coverage |

### Choosing Max Length

```
Optimal Max Length = 95-99th percentile of your data
```

**Tradeoffs:**
- Shorter → faster training, more truncation, less memory
- Longer → slower training, less truncation, more memory

---

## Practical Examples

### Example 1: OOM Error Recovery

**Scenario:** Training crashes with 24GB GPU at batch=8, seq=4096

**Analysis:**
- Estimated memory: ~22-24GB (too tight)
- Need headroom: ~4-5GB for spikes

**Solutions (pick one):**

| Solution | New Memory | Time Impact | Quality Impact |
|----------|------------|-------------|----------------|
| Batch 8→6 | ~18-19GB | +20% time | None |
| Seq 4096→3072 | ~17-18GB | +25% time | 1.5% more truncated |
| Batch 8→4 + Grad Accum 2 | ~14-15GB | +50% time | None |

### Example 2: Speed Optimization

**Scenario:** Training too slow, want to finish faster

**Current:** Batch=2, Grad Accum=4, Seq=4096
**Estimated time:** ~14 hours

**Optimization Path:**

| Step | Change | New Time | Cumulative |
|------|--------|----------|------------|
| 1 | Seq 4096→2500 | ~10h | 29% faster |
| 2 | Batch 2→4, Grad Accum 4→2 | ~6h | 57% faster |
| 3 | Batch 4→6, Grad Accum 1 | ~5h | 64% faster |

**Final:** Batch=6, Grad Accum=1, Seq=2500 → ~5 hours

### Example 3: Memory Budget Planning

**Scenario:** Have 16GB VRAM, want to train 7B model

**Estimation:**
```
Model (4-bit 7B):     ~3.5 GB
LoRA adapters:        ~0.2 GB
Optimizer (8-bit):    ~0.4 GB
Gradients:            ~0.2 GB
Overhead:             ~2 GB
───────────────────────────────
Fixed overhead:       ~6.3 GB
Available for activations: ~9.7 GB
```

**Safe Config:**
- Batch size: 1-2
- Seq length: 2048-3072
- Gradient accumulation: 4-8 (to maintain effective batch)
- Gradient checkpointing: Required

---

## Rules of Thumb

### Memory Safety

```
Target Memory Usage = 75-80% of Total VRAM

Headroom needed: 20-25% for:
- Memory fragmentation
- Occasional longer sequences (with packing)
- PyTorch overhead
```

### Time Estimation

```
Per-step time (rough):
- 0.5B model: 1-3 seconds
- 7B model: 3-8 seconds
- 13B model: 8-15 seconds

(Depends heavily on batch size, seq length, and hardware)
```

### Coverage vs Quality

```
Coverage ≥ 95%: Usually sufficient for fine-tuning
Coverage ≥ 98%: Recommended for production models
Coverage ≥ 99%: Overkill for most use cases
```

---

## Debugging Memory Issues

### Check Current Usage

```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Reduce Memory (In Order of Impact)

1. **Reduce batch size** - Direct reduction, increase grad accum to compensate
2. **Reduce seq length** - Significant impact, may truncate samples
3. **Enable gradient checkpointing** - 60-75% activation reduction
4. **Use 4-bit quantization** - Already using if using Unsloth/QLoRA
5. **Reduce LoRA rank** - Minor impact, may affect quality

### Handle Fragmentation

```bash
# PyTorch memory allocator settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
