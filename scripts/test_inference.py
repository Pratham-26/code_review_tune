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
        content = msg["content"]
        if isinstance(content, str):
            formatted_messages.append({"role": msg["role"], "content": content})
        else:
            formatted_messages.append(msg)

    inputs = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

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
    expected_content = test_sample[col][-1]["content"]
    if isinstance(expected_content, list):
        expected_content = " ".join([c.get("text", str(c)) for c in expected_content])
    print(expected_content)

    print("\n" + "=" * 60)
    print("Generated review:")
    print("=" * 60)

    _ = model.generate(
        inputs,
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
