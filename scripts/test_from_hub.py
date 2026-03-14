# -*- coding: utf-8 -*-
"""Test the fine-tuned code review model from Hugging Face Hub."""

import argparse
from unsloth import FastLanguageModel
from transformers import TextStreamer
import os

DEFAULT_REPO = "PrathamKotian26/code-review-qwen-0.8b"


def main():
    parser = argparse.ArgumentParser(description="Test code review model from Hub")
    parser.add_argument(
        "--model",
        default=DEFAULT_REPO,
        help=f"HF Hub model ID or local path (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Code to review (if not provided, uses sample)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)
    print(f"Model: {args.model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN", ""),
    )

    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!")

    if args.code:
        code = args.code
    else:
        code = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total

def find_max(lst):
    max_val = lst[0]
    for i in lst:
        if i > max_val:
            max_val = i
    return max_val

def process_data(data):
    result = []
    for item in data:
        if item != None:
            result.append(item.strip())
    return result
"""

    messages = [
        {
            "role": "user",
            "content": f"Review this Python code and suggest improvements:\n\n{code}",
        }
    ]

    prompt = "<|im_start|>user\n"
    for msg in messages:
        if msg["role"] == "user":
            prompt += msg["content"] + "<|im_end|>\n<|im_start|>assistant\n"

    text_tokenizer = (
        tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    )
    tokenized = text_tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].to("cuda")

    streamer = TextStreamer(text_tokenizer, skip_prompt=True)

    print("\n" + "=" * 60)
    print("Code to review:")
    print("=" * 60)
    print(code.strip())

    print("\n" + "=" * 60)
    print("Generated review:")
    print("=" * 60)

    _ = model.generate(
        input_ids,
        streamer=streamer,
        max_new_tokens=args.max_tokens,
        use_cache=True,
        temperature=0.7,
        min_p=0.1,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
