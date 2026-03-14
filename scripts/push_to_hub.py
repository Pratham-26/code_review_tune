"""
Push trained LoRA adapter to Hugging Face Hub.

Usage:
    export HF_TOKEN=your_token
    python scripts/push_to_hub.py [--repo-id username/model-name]
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "code_review_model" / "lora"
DEFAULT_REPO = "PrathamKotian26/code-review-qwen-0.8b"


def main():
    parser = argparse.ArgumentParser(
        description="Push trained model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help=f"HF Hub repo ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--local-path",
        default=str(OUTPUT_DIR),
        help=f"Local model path (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN environment variable must be set.")

    local_path = Path(args.local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Model not found at {local_path}")

    print("=" * 60)
    print("Pushing model to Hugging Face Hub")
    print("=" * 60)
    print(f"Local path: {local_path}")
    print(f"Repo ID: {args.repo_id}")

    api = HfApi(token=token)

    try:
        api.repo_info(args.repo_id, repo_type="model")
        print(f"Repository exists: https://huggingface.co/{args.repo_id}")
    except RepositoryNotFoundError:
        print(f"Creating repository: {args.repo_id}")
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private)

    print("\nUploading files...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload trained LoRA adapter",
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
