# -*- coding: utf-8 -*-
"""Analyze token lengths in the dataset to determine optimal max_seq_length."""

from datasets import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable must be set")

DATASET_NAME = "PrathamKotian26/code-review-python-autotrain"
MODEL_NAME = "unsloth/Qwen3.5-0.8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

print("Loading dataset...")
ds = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)

col = "messages" if "messages" in ds.column_names else "conversations"
print(f"Analyzing {len(ds):,} samples...")

lengths = []
for i, sample in enumerate(ds):
    msgs = sample[col]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    lengths.append(len(tokens))
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1:,}...")

lengths = np.array(lengths)

print()
print("=" * 50)
print("TOKEN LENGTH ANALYSIS")
print("=" * 50)
print(f"Samples: {len(lengths):,}")
print(f"Min: {lengths.min():,}")
print(f"Max: {lengths.max():,}")
print(f"Mean: {lengths.mean():.0f}")
print(f"Median: {np.median(lengths):.0f}")
print(f"Std: {lengths.std():.0f}")
print()
print("Percentiles:")
for p in [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]:
    print(f"  {p}%: {np.percentile(lengths, p):.0f} tokens")
print()
print("Coverage at max_length:")
for m in [2048, 2500, 3072, 4096, 5120, 6144, 8192]:
    covered = (lengths <= m).sum() / len(lengths) * 100
    count = (lengths <= m).sum()
    print(f"  {m}: {covered:.1f}% ({count:,} samples)")
