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

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "code_review_model"
DATASET_NAME = "PrathamKotian26/code-review-python-autotrain"
