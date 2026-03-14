# -*- coding: utf-8 -*-
"""
Test the fine-tuned code review model.

Usage:
    export HF_TOKEN=your_token
    python scripts/test_model.py --model-path ./models/code_review_model/lora
    # Or from HF Hub:
    python scripts/test_model.py --hub-repo PrathamKotian26/code-review-qwen-0.8b
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test the code review model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path to LoRA adapter",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="HF Hub repo ID for the adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Qwen3.5-0.8B",
        help="Base model name",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Max sequence length for inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    args = parser.parse_args()

    if not args.model_path and not args.hub_repo:
        args.hub_repo = "PrathamKotian26/code-review-qwen-0.8b"
        print(f"No model specified, using default: {args.hub_repo}")

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    from unsloth import FastLanguageModel

    hf_token = os.environ.get("HF_TOKEN", "")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )

    if args.hub_repo:
        print(f"Loading adapter from HF Hub: {args.hub_repo}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.hub_repo, token=hf_token)
    else:
        print(f"Loading adapter from local path: {args.model_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.model_path)

    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!\n")

    print("=" * 60)
    print("Code Review Model - Interactive Testing")
    print("=" * 60)
    print("Enter Python code to review. Type 'quit' to exit.\n")

    while True:
        print("-" * 40)
        code = input("Paste your Python code:\n")

        if code.strip().lower() == "quit":
            print("Goodbye!")
            break

        if not code.strip():
            continue

        prompt = f"""Review the following Python code and provide constructive feedback. If you see issues, suggest fixes.

Code to review:
```python
{code}
```"""

        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        print("\n" + "-" * 40)
        print("Generated Review:")
        print("-" * 40)

        from transformers import TextStreamer

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)

        _ = model.generate(
            inputs,
            streamer=text_streamer,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            temperature=args.temperature,
            min_p=0.1,
        )
        print()


if __name__ == "__main__":
    main()
