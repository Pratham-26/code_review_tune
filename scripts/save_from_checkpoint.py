# -*- coding: utf-8 -*-
"""Save the model from the latest checkpoint."""

from unsloth import FastLanguageModel
from pathlib import Path
import shutil

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "code_review_model"
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 4096


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return checkpoints[-1]


def main():
    checkpoint_path = find_latest_checkpoint(OUTPUT_DIR)

    if not checkpoint_path:
        print("No checkpoint found!")
        return

    print("=" * 60)
    print(f"Found checkpoint: {checkpoint_path}")
    print("=" * 60)

    print("\nLoading model from checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    lora_path = OUTPUT_DIR / "lora"

    print(f"\nSaving model to: {lora_path}")
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))

    print("\n" + "=" * 60)
    print("Model saved successfully!")
    print("=" * 60)
    print(f"Location: {lora_path}")


if __name__ == "__main__":
    main()
