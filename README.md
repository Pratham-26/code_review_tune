# Code Review Tune

Fine-tune Qwen3.5-0.8B for Python code review using Unsloth + LoRA.

## Dataset

- **Source**: [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)
- **Filtered**: [PrathamKotian26/code-review-python-autotrain](https://huggingface.co/datasets/PrathamKotian26/code-review-python-autotrain)
- **Samples**: ~40k train, ~800 validation, ~800 test

## Setup

```bash
# Clone the repository
git clone https://github.com/Pratham-26/code_review_tune.git
cd code_review_tune

# Set HF token
export HF_TOKEN=your_token_here

# Install dependencies with uv
uv pip install -r requirements.txt
uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
uv pip install triton xformers
```

## Run Training

```bash
uv run python scripts/finetune_qwen.py
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | unsloth/Qwen3.5-0.8B |
| Max sequence length | 6000 |
| LoRA rank | 16 |
| Batch size | 2 × 4 (effective: 8) |
| Learning rate | 2e-4 |
| Epochs | 1 |

## Output

Model saved to `models/code_review_model/lora/`

## Credits

- Dataset: [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)
- Training: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl)

## License

MIT
