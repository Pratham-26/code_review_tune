"""
Push fine-tuned model to Hugging Face Hub.

Usage:
    uv run python scripts/push_to_hub.py
    uv run python scripts/push_to_hub.py --merged  # Push merged 16-bit model
    uv run python scripts/push_to_hub.py --gguf q4_k_m  # Push GGUF format
"""

import argparse
from pathlib import Path
from config import HF_TOKEN, OUTPUT_DIR, REPO_ID


def push_lora():
    """Push LoRA adapters to Hub"""
    from unsloth import FastLanguageModel

    print("=" * 60)
    print("Pushing LoRA adapters to Hugging Face Hub...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(OUTPUT_DIR / "lora"),
        max_seq_length=6000,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    model.push_to_hub(REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)

    print(f"\nLoRA adapters pushed to: https://huggingface.co/{REPO_ID}")


def push_merged():
    """Push merged 16-bit model to Hub"""
    from unsloth import FastLanguageModel

    print("=" * 60)
    print("Pushing merged 16-bit model to Hugging Face Hub...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(OUTPUT_DIR / "lora"),
        max_seq_length=6000,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    model.push_to_hub_merged(REPO_ID, tokenizer, token=HF_TOKEN)

    print(f"\nMerged model pushed to: https://huggingface.co/{REPO_ID}")


def push_gguf(quantization_method: str = "q4_k_m"):
    """Push GGUF model to Hub"""
    from unsloth import FastLanguageModel

    print("=" * 60)
    print(f"Pushing GGUF model ({quantization_method}) to Hugging Face Hub...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(OUTPUT_DIR / "lora"),
        max_seq_length=6000,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    model.push_to_hub_gguf(
        REPO_ID,
        tokenizer,
        quantization_method=quantization_method,
        token=HF_TOKEN,
    )

    print(f"\nGGUF model pushed to: https://huggingface.co/{REPO_ID}")


def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Push merged 16-bit model instead of LoRA adapters",
    )
    parser.add_argument(
        "--gguf",
        type=str,
        nargs="?",
        const="q4_k_m",
        default=None,
        help="Push GGUF format. Options: q4_k_m, q5_k_m, q8_0, f16",
    )
    args = parser.parse_args()

    lora_path = OUTPUT_DIR / "lora"
    if not lora_path.exists():
        print(f"Error: LoRA adapters not found at {lora_path}")
        print("Please run finetune_qwen.py first to train and save the model.")
        return

    print(f"Repository: {REPO_ID}")
    print(f"Local model: {lora_path}")
    print()

    if args.gguf:
        push_gguf(args.gguf)
    elif args.merged:
        push_merged()
    else:
        push_lora()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
