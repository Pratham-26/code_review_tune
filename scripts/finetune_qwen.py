# -*- coding: utf-8 -*-
"""
Fine-tune Qwen3.5-0.8B for Python Code Review

Based on Unsloth reference scripts. Uses LoRA for efficient fine-tuning.

## Setup

```bash
uv sync
uv sync --extra train
uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
uv pip install triton xformers
```

## Run training

```bash
uv run python scripts/finetune_qwen.py
```
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import polars as pl
from config import HF_TOKEN, DATA_DIR, OUTPUT_DIR

MODEL_NAME = "unsloth/Qwen3.5-0.8B"

MAX_SEQ_LENGTH = 6000
DTYPE = None
LOAD_IN_4BIT = True

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

INSTRUCTION = """Review the following Python code and provide constructive feedback. If you see issues, suggest fixes.

Code to review:
```python
{before_code}
```
{diff_context}"""


def load_data(split: str) -> Dataset:
    file_path = DATA_DIR / f"python_reviews_{split}.parquet"
    df = pl.read_parquet(file_path)

    def format_sample(row):
        diff_ctx = (
            f"\nDiff context:\n```\n{row['diff_context']}\n```"
            if row["diff_context"]
            else ""
        )
        prompt = INSTRUCTION.format(
            before_code=row["before_code"], diff_context=diff_ctx
        )
        return {
            "instruction": prompt,
            "output": row["reviewer_comment"],
        }

    samples = [format_sample(row) for row in df.to_dicts()]
    return Dataset.from_list(samples)


def convert_to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    }


def main():
    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
    )

    print("\n" + "=" * 60)
    print("Adding LoRA adapters...")
    print("=" * 60)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("\n" + "=" * 60)
    print("Loading and formatting dataset...")
    print("=" * 60)

    train_dataset = load_data("train")
    print(f"Train samples: {len(train_dataset):,}")

    val_dataset = load_data("validation")
    print(f"Validation samples: {len(val_dataset):,}")

    train_converted = [convert_to_conversation(s) for s in train_dataset]
    val_converted = [convert_to_conversation(s) for s in val_dataset]

    print("\nSample formatted conversation:")
    print(train_converted[0])

    print("\n" + "=" * 60)
    print("Testing inference before training...")
    print("=" * 60)

    FastLanguageModel.for_inference(model)

    test_sample = train_dataset[0]
    messages = [{"role": "user", "content": test_sample["instruction"]}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\nGenerating review (before training)...")
    _ = model.generate(
        inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        min_p=0.1,
    )

    print("\n" + "=" * 60)
    print("Setting up trainer...")
    print("=" * 60)

    from trl import SFTTrainer, SFTConfig

    FastLanguageModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_converted,
        eval_dataset=val_converted,
        args=SFTConfig(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=500,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(OUTPUT_DIR),
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            packing=False,
        ),
    )

    print("\n" + "=" * 60)
    print("GPU Memory Stats")
    print("=" * 60)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer_stats = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete - Final Stats")
    print("=" * 60)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']:.1f} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    print("\n" + "=" * 60)
    print("Testing inference after training...")
    print("=" * 60)

    FastLanguageModel.for_inference(model)

    test_dataset = load_data("test")
    test_sample = test_dataset[0]
    messages = [{"role": "user", "content": test_sample["instruction"]}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    print("\nExpected output:")
    print(test_sample["output"])
    print("\nGenerated review:")

    _ = model.generate(
        inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        min_p=0.1,
    )

    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR / "lora"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora"))
    print(f"LoRA adapters saved to: {OUTPUT_DIR / 'lora'}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nModel saved locally to: {OUTPUT_DIR / 'lora'}")
    print("To push to Hugging Face Hub, run:")
    print("  uv run python scripts/push_to_hub.py")


if __name__ == "__main__":
    main()
