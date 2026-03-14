"""
Configuration for code-review-tune project.

Set HF_TOKEN environment variable before running scripts:
    export HF_TOKEN=your_token_here
"""

import os
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable must be set.\n"
        "Please set: export HF_TOKEN=your_token_here"
    )

HF_USERNAME = "pratham"
MODEL_NAME = "code-review-qwen-0.8b"

REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

DATA_DIR = Path(__file__).parent.parent / "dataset" / "python_subset"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "code_review_model"
