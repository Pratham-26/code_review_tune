# -*- coding: utf-8 -*-
"""Test the fine-tuned code review model."""

from unsloth import FastLanguageModel
from datasets import load_dataset
from config import HF_TOKEN, OUTPUT_DIR, DATASET_NAME

MODEL_PATH = OUTPUT_DIR / "lora"


def main():
    print("=" * 60)
    print("Loading fine-tuned model...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    print("\n" + "=" * 60)
    print("Loading test sample...")
    print("=" * 60)

    test_dataset = load_dataset(DATASET_NAME, split="test", token=HF_TOKEN)
    test_sample = test_dataset[0]

    col = "messages" if "messages" in test_dataset.column_names else "conversations"
    messages = test_sample[col][:-1]

    formatted_messages = []
    for msg in messages:
        content = msg.get("content", msg.get("value", ""))
        role = msg.get("role", msg.get("from", "user"))
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    text_parts.append(c["text"])
                elif isinstance(c, str):
                    text_parts.append(c)
            content = " ".join(text_parts)
        formatted_messages.append({"role": role, "content": content})

    prompt = "<|im_start|>user\n"
    for msg in formatted_messages:
        if msg["role"] == "user":
            prompt += msg["content"] + "<|im_end|>\n<|im_start|>assistant\n"
        elif msg["role"] == "assistant":
            prompt += msg["content"] + "<|im_end|>\n"

    text_tokenizer = (
        tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    )
    tokenized = text_tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].to("cuda")

    from transformers import TextStreamer

    text_streamer = TextStreamer(text_tokenizer, skip_prompt=True)

    print("\n" + "=" * 60)
    print("Input code:")
    print("=" * 60)
    for msg in formatted_messages:
        if msg["role"] == "user":
            print(
                msg["content"][:500] + "..."
                if len(msg["content"]) > 500
                else msg["content"]
            )

    print("\n" + "=" * 60)
    print("Expected review:")
    print("=" * 60)
    last_msg = test_sample[col][-1]
    expected_content = last_msg.get("content", last_msg.get("value", ""))
    if isinstance(expected_content, list):
        text_parts = []
        for c in expected_content:
            if isinstance(c, dict) and "text" in c:
                text_parts.append(c["text"])
            elif isinstance(c, str):
                text_parts.append(c)
        expected_content = " ".join(text_parts)
    print(expected_content)

    print("\n" + "=" * 60)
    print("Generated review:")
    print("=" * 60)

    _ = model.generate(
        input_ids,
        streamer=text_streamer,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.7,
        min_p=0.1,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
