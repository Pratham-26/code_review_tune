---
license: apache-2.0
base_model: unsloth/Qwen3.5-0.8B
tags:
- code-review
- python
- qwen
- unsloth
- lora
- fine-tuned
language:
- en
pipeline_tag: text-generation
---

# Code Review Qwen 0.8B

A fine-tuned version of [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B) specialized for Python code review. The model analyzes Python code and provides constructive feedback with suggested improvements.

## Model Description

- **Base Model:** Qwen3.5-0.8B (4-bit quantized)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Task:** Python code review and improvement suggestions
- **Language:** English

## Training Details

| Parameter | Value |
|-----------|-------|
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| Learning Rate | 2e-4 |
| Batch Size | 6 |
| Sequence Length | 2056 |
| Epochs | 1 |
| Optimizer | AdamW 8-bit |
| Training Time | ~5.2 hours |
| Final Loss | 1.56 |
| Hardware | NVIDIA RTX 4090 (24GB) |

## Training Data

Trained on [PrathamKotian26/code-review-python-autotrain](https://huggingface.co/datasets/PrathamKotian26/code-review-python-autotrain) containing Python code with review feedback.

- **Training samples:** 40,265
- **Validation samples:** 869
- **Test samples:** 791

## Usage

### Installation

```bash
pip install unsloth transformers datasets torch
```

### Basic Usage

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="PrathamKotian26/code-review-qwen-0.8b",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Prepare prompt
code = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total
"""

prompt = f"<|im_start|>user\nReview this Python code and suggest improvements:\n\n{code}<|im_end|>\n<|im_start|>assistant\n"

# Tokenize
text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
tokenized = text_tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = tokenized["input_ids"].to("cuda")
attention_mask = tokenized["attention_mask"].to("cuda")

# Generate
streamer = TextStreamer(text_tokenizer, skip_prompt=True)
model.generate(
    input_ids,
    attention_mask=attention_mask,
    streamer=streamer,
    max_new_tokens=512,
    temperature=0.7,
    min_p=0.1,
)
```

### Using the Test Script

```bash
# Clone the repo
git clone https://github.com/Pratham-26/code_review_tune.git
cd code_review_tune

# Install dependencies
pip install -r requirements.txt

# Run inference
python scripts/test_from_hub.py --model PrathamKotian26/code-review-qwen-0.8b
```

## Example Output

**Input:**
```python
def process_data(data):
    result = []
    for item in data:
        if item != None:
            result.append(item.strip())
    return result
```

**Output:**
```suggestion
    return [item.strip() for item in data]
```

## Capabilities

The model can:
- Identify code style issues
- Suggest Pythonic improvements
- Recommend best practices
- Simplify complex code patterns
- Provide constructive feedback

## Limitations

- **Model Size:** 0.8B parameters - limited reasoning compared to larger models
- **Context:** Trained on sequences up to 2056 tokens (can extrapolate to ~4k at inference)
- **Language:** Only English
- **Domain:** Python code only
- **Complexity:** May struggle with subtle bugs or complex architectural issues

## Intended Use

- First-pass automated code review
- PR hygiene checks
- Learning Python best practices
- Code improvement suggestions

**Not intended for:**
- Replacing human code review
- Security auditing
- Complex architectural decisions

## Model Architecture

```
Base: Qwen3.5-0.8B (4-bit quantized)
+ LoRA adapters (r=16) on:
  - q_proj, k_proj, v_proj, o_proj
  - gate_proj, up_proj, down_proj
```

## Repository

- **Code:** [github.com/Pratham-26/code_review_tune](https://github.com/Pratham-26/code_review_tune)
- **Dataset:** [PrathamKotian26/code-review-python-autotrain](https://huggingface.co/datasets/PrathamKotian26/code-review-python-autotrain)

## License

Apache 2.0

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- [TRL](https://github.com/huggingface/trl) for training utilities
