# PRD: Code Review Model Fine-tuning

## Goal
Fine-tune Qwen3.5-0.8B to generate code review comments for Python code using HF Jobs + Unsloth.

## Requirements
| Item | Value |
|------|-------|
| Model | `unsloth/Qwen3.5-0.8B` (text-only) |
| Max sequence length | 6000 tokens |
| Training method | SFT with LoRA |
| Infrastructure | Hugging Face Jobs |
| Output | LoRA adapter pushed to HF Hub |

## Dataset
- **Source**: [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)
- **Filtered for**: Python, quality >= 0.5, tokens <= 5000
- **Train samples**: ~40,000
- **Format**: Conversations with user (code) / assistant (review) format

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOCAL: Prepare & Upload Dataset                              │
│    uv run scripts/prepare_autotrain.py                          │
│    → Uploads to Pratham-26/code-review-python-autotrain         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. HF JOBS: Submit Training Job                                 │
│    hf jobs uv run scripts/train_hf_job.py \                     │
│        --flavor a10g-small --secrets HF_TOKEN --timeout 6h      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. OUTPUT: Model on HF Hub                                      │
│    https://huggingface.co/Pratham-26/code-review-qwen-0.8b      │
└─────────────────────────────────────────────────────────────────┘
```

## Training Configuration
| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Batch size | 2 × 4 (grad accum) = 8 effective |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Optimizer | adamw_8bit |
| Hardware | a10g-small (~$3.50/hr) |
| Timeout | 6 hours |

## Files
| File | Purpose |
|------|---------|
| `scripts/prepare_autotrain.py` | Format & upload dataset to HF Hub |
| `scripts/train_hf_job.py` | Training script with Unsloth + TRL |

## Checklist
- [x] Create PRD
- [ ] Update prepare_autotrain.py (format as conversations, credit source)
- [ ] Update train_hf_job.py (skill best practices, CLI args)
- [ ] Upload dataset to HF Hub
- [ ] Submit training job
- [ ] Verify model on Hub

## Credits
- Dataset: [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)
- Training: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl)
- Infrastructure: [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs)
