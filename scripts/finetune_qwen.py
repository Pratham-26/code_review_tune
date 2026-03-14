# -*- coding: utf-8 -*-
"""
Fine-tune Qwen3.5-0.8B for Python Code Review

Based on Unsloth reference scripts. Uses LoRA for efficient fine-tuning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run training

```bash
export HF_TOKEN=your_token
python scripts/finetune_qwen.py 2>&1 | tee training.log
```
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
import os
import torch
from datasets import load_dataset
from config import HF_TOKEN, OUTPUT_DIR

os.environ["TQDM_MININTERVAL"] = "5"

MODEL_NAME = "unsloth/Qwen3.5-0.8B"
DATASET_NAME = "PrathamKotian26/code-review-python-autotrain"

MAX_SEQ_LENGTH = 2500

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


def main():
    print("=" * 60)
    print("Loading model...")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
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
    print("Loading dataset...")
    print("=" * 60)

    train_dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)
    val_dataset = load_dataset(DATASET_NAME, split="validation", token=HF_TOKEN)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("messages", "conversations")
        val_dataset = val_dataset.rename_column("messages", "conversations")

    train_dataset = standardize_data_formats(train_dataset)
    val_dataset = standardize_data_formats(val_dataset)

    def formatting_prompts_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        bos = tokenizer.bos_token or ""
        return {"text": [x.removeprefix(bos) for x in texts]}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    print("\nSample formatted text:")
    print(train_dataset[0]["text"][:500] + "...")

    print("\n" + "=" * 60)
    print("Setting up trainer...")
    print("=" * 60)

    from trl import SFTTrainer, SFTConfig

    FastLanguageModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            gradient_accumulation_steps=1,
            warmup_steps=20,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(OUTPUT_DIR),
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            packing=True,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
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

    test_dataset = load_dataset(DATASET_NAME, split="test", token=HF_TOKEN)
    test_sample = test_dataset[0]
    messages = test_sample["messages"][:-1]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\nExpected output:")
    print(test_sample["messages"][-1]["content"])
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


if __name__ == "__main__":
    main()
