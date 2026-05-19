import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training entrypoint for the public research repository."
    )
    parser.add_argument(
        "--experiment",
        choices=("ce", "entropy", "similarity"),
        default="similarity",
        help="Which experiment pipeline to run.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to the underlying training script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_map = {
        "ce": ["main.py", "--mode", "train_eval"],
        "entropy": ["train_adaptive_loss.py"],
        "similarity": ["train_similarity_aware_loss.py"],
    }
    command = [sys.executable, *script_map[args.experiment], *args.extra]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
