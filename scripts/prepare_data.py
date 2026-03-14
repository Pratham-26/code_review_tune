"""
Prepare Python code review dataset for fine-tuning.

Downloads the GitHub code review dataset and filters it for high-quality Python reviews.

Usage:
    uv run python scripts/prepare_data.py
"""

import polars as pl
from datasets import load_dataset
from pathlib import Path
from config import HF_TOKEN, DATA_DIR

OUTPUT_DIR = DATA_DIR

MIN_QUALITY_SCORE = 0.5
MAX_TOKENS = 5000
CHARS_PER_TOKEN = 3
INSTRUCTION_OVERHEAD_CHARS = 150

EXCLUDE_COMMENT_TYPES = {"none"}
EXCLUDE_REVIEWERS = {"Copilot"}

SPLITS = ["train", "validation", "test"]


def estimate_tokens(before_code: str, reviewer_comment: str, diff_context: str) -> int:
    input_chars = (
        len(before_code) + len(diff_context or "") + INSTRUCTION_OVERHEAD_CHARS
    )
    output_chars = len(reviewer_comment)
    return (input_chars + output_chars) // CHARS_PER_TOKEN


def download_dataset():
    print("=" * 60)
    print("Downloading dataset from Hugging Face...")
    print("=" * 60)

    dataset = load_dataset(
        "ronantakizawa/github-codereview",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    for split in SPLITS:
        print(f"  {split}: {len(dataset[split]):,} rows")

    return dataset


def filter_split(dataset, split: str) -> pl.DataFrame:
    print(f"\nProcessing {split}...")

    df = pl.from_arrow(dataset[split].data.table)
    original_count = len(df)
    print(f"  Original rows: {original_count:,}")

    filtered = df.filter(
        (pl.col("language") == "Python")
        & (pl.col("quality_score") >= MIN_QUALITY_SCORE)
        & (~pl.col("comment_type").is_in(list(EXCLUDE_COMMENT_TYPES)))
        & (~pl.col("reviewer_username").is_in(list(EXCLUDE_REVIEWERS)))
    )
    print(f"  After basic filters: {len(filtered):,}")

    token_estimates = [
        estimate_tokens(
            row["before_code"], row["reviewer_comment"], row["diff_context"]
        )
        for row in filtered.to_dicts()
    ]
    filtered = filtered.with_columns(pl.Series("estimated_tokens", token_estimates))
    filtered = filtered.filter(pl.col("estimated_tokens") <= MAX_TOKENS)
    filtered = filtered.drop("estimated_tokens")

    print(f"  After token filter (<= {MAX_TOKENS}): {len(filtered):,}")

    comment_types = filtered.group_by("comment_type").len().sort("len", descending=True)
    print(f"  Comment types: {comment_types.to_dicts()}")

    return filtered.select(
        [
            "before_code",
            "reviewer_comment",
            "diff_context",
            "comment_type",
            "quality_score",
        ]
    )


def main():
    dataset = download_dataset()

    print("\n" + "=" * 60)
    print("Filtering dataset...")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for split in SPLITS:
        df = filter_split(dataset, split)
        output_file = OUTPUT_DIR / f"python_reviews_{split}.parquet"
        df.write_parquet(output_file)
        total_rows += len(df)
        print(f"  Saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total rows: {total_rows:,}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nFilters applied:")
    print(f"  - language = 'Python'")
    print(f"  - quality_score >= {MIN_QUALITY_SCORE}")
    print(f"  - comment_type not in {EXCLUDE_COMMENT_TYPES}")
    print(f"  - reviewer_username not in {EXCLUDE_REVIEWERS}")
    print(f"  - estimated_tokens <= {MAX_TOKENS}")


if __name__ == "__main__":
    main()
