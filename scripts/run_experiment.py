#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "pyyaml",
#   "tqdm",
# ]
# ///
"""Run a complete experiment with manifest generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.config import load_config


def get_git_hash() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def get_jj_hash() -> str | None:
    """Get current jj commit hash."""
    try:
        result = subprocess.run(
            ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def generate_manifest(
    config_path: Path,
    name: str,
    command: list[str],
) -> dict:
    """Generate run manifest."""
    config = load_config(config_path)

    # Read VERSION file
    version_file = Path(__file__).parent.parent / "VERSION"
    version = version_file.read_text().strip() if version_file.exists() else "0.1.0"

    manifest = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "git_hash": get_git_hash(),
        "jj_hash": get_jj_hash(),
        "config_path": str(config_path),
        "config": config.to_dict(),
        "command": command,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "device": {
            "type": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "seeds": {
            "train_seed": config.train.seed,
            "eval_seed": config.eval.seed,
        },
    }

    return manifest


def save_manifest(manifest: dict, output_dir: Path) -> Path:
    """Save manifest to file."""
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = manifest.get("name", "experiment")
    filename = f"{name}_{timestamp}.json"

    manifest_path = manifest_dir / filename
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def run_experiment(
    config_path: Path,
    name: str,
    output_dir: Path = Path("eval/results"),
    skip_train: bool = False,
    skip_eval: bool = False,
    data_path: Path | None = None,
    eval_path: Path | None = None,
) -> None:
    """Run complete experiment.

    Args:
        config_path: Path to config
        name: Experiment name
        output_dir: Output directory
        skip_train: Skip training
        skip_eval: Skip evaluation
        data_path: Path to training data directory
        eval_path: Path to evaluation data
    """
    scripts_dir = Path(__file__).parent

    # Build command
    command = [sys.executable, str(Path(__file__))]
    command.extend(["--config", str(config_path), "--name", name])

    # Generate manifest
    print("Generating run manifest...")
    manifest = generate_manifest(config_path, name, command)
    manifest_path = save_manifest(manifest, output_dir)
    print(f"Manifest saved to {manifest_path}")

    # Set seeds
    config = load_config(config_path)
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # Run training
    if not skip_train:
        print("\n" + "=" * 50)
        print("TRAINING")
        print("=" * 50)
        train_script = scripts_dir / "train.py"
        result = subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--config",
                str(config_path),
                "--name",
                name,
                "--output",
                str(output_dir / "checkpoints"),
            ]
        )
        if result.returncode != 0:
            print("Training failed!")
            return

    # Run evaluation
    if not skip_eval:
        print("\n" + "=" * 50)
        print("EVALUATION")
        print("=" * 50)
        eval_script = scripts_dir / "eval.py"
        model_path = output_dir / "checkpoints" / name / "final.pt"

        eval_args = [
            sys.executable,
            str(eval_script),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ]

        if model_path.exists():
            eval_args.extend(["--model", str(model_path)])

        if eval_path:
            eval_args.extend(["--data", str(eval_path)])

        result = subprocess.run(eval_args)
        if result.returncode != 0:
            print("Evaluation failed!")
            return

    # Update manifest with results
    if (output_dir / "summary.json").exists():
        with open(output_dir / "summary.json") as f:
            summary = json.load(f)
        manifest["results"] = summary.get("results", {})

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Config file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results"),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Training data directory",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Evaluation data path",
    )

    args = parser.parse_args()
    run_experiment(
        args.config,
        args.name,
        args.output,
        args.skip_train,
        args.skip_eval,
        args.data,
        getattr(args, "eval"),
    )


if __name__ == "__main__":
    main()
