import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Convenience runner for the LSTM training script."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Code/Data/complete_dataset.csv"),
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/lstm_seed_study"),
        help="Directory where run artifacts will be stored.",
    )
    parser.add_argument(
        "--run-tag",
        default="local",
        help="Suffix used in saved artifact filenames.",
    )
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def main() -> None:
    args, extra_args = parse_args()
    repo_root = Path(__file__).resolve().parent
    model_script = repo_root / "Code" / "lstm_google_cloud.py"

    if not model_script.exists():
        raise FileNotFoundError(f"Could not find model script: {model_script}")

    command = [
        sys.executable,
        str(model_script),
        "--data-path",
        str((repo_root / args.data_path).resolve()),
        "--artifact-dir",
        str((repo_root / args.artifact_dir).resolve()),
        "--run-tag",
        args.run_tag,
        *extra_args,
    ]

    subprocess.run(command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
