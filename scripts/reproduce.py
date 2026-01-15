#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "pyyaml",
#   "tqdm",
#   "tomli-w",
# ]
# ///
"""Reproduce an experiment from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def reproduce(manifest_path: Path, output_dir: Path | None = None) -> None:
    """Reproduce experiment from manifest.

    Args:
        manifest_path: Path to run manifest
        output_dir: Override output directory
    """
    print(f"Loading manifest from {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Experiment: {manifest.get('name', 'unknown')}")
    print(f"Original timestamp: {manifest.get('timestamp', 'unknown')}")
    print(f"Version: {manifest.get('version', 'unknown')}")

    # Extract config
    config = manifest.get("config", {})
    if not config:
        print("ERROR: No config in manifest")
        return

    # Write temporary config file
    import tomllib
    import tomli_w

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".toml", delete=False
    ) as f:
        # Convert nested dicts for TOML
        tomli_w.dump(config, f)
        temp_config = Path(f.name)

    print(f"Recreated config at {temp_config}")

    # Determine output directory
    if output_dir is None:
        output_dir = Path("eval/results/reproduced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    from scripts.run_experiment import run_experiment

    name = f"reproduce_{manifest.get('name', 'exp')}"
    run_experiment(temp_config, name, output_dir)

    # Cleanup
    temp_config.unlink()

    print(f"\nReproduction complete. Results in {output_dir}")


def compare_results(original_manifest: Path, reproduced_dir: Path) -> None:
    """Compare original and reproduced results."""
    print("\nComparing results...")

    with open(original_manifest) as f:
        original = json.load(f)

    reproduced_summary = reproduced_dir / "summary.json"
    if not reproduced_summary.exists():
        print("No reproduced summary found")
        return

    with open(reproduced_summary) as f:
        reproduced = json.load(f)

    # Compare key metrics
    orig_results = original.get("results", {})
    repr_results = reproduced.get("results", {})

    print("\nMetric comparison (original -> reproduced):")
    for tokenizer in ["universal", "raw_bytes", "byte_bpe"]:
        if tokenizer in orig_results and tokenizer in repr_results:
            orig = orig_results[tokenizer]
            repr_ = repr_results[tokenizer]

            print(f"\n{tokenizer}:")
            for key in ["mean_compression_ratio", "mean_bits_per_byte", "lossless_rate"]:
                if key in orig and key in repr_:
                    diff = abs(orig[key] - repr_[key])
                    status = "OK" if diff < 0.01 else "DIFF"
                    print(f"  {key}: {orig[key]:.4f} -> {repr_[key]:.4f} [{status}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce experiment from manifest")
    parser.add_argument("manifest", type=Path, help="Path to manifest JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results after reproduction",
    )

    args = parser.parse_args()

    # Need tomli_w for writing TOML
    try:
        import tomli_w
    except ImportError:
        print("Installing tomli_w...")
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "tomli_w"])
        import tomli_w

    reproduce(args.manifest, args.output)

    if args.compare:
        output_dir = args.output or Path("eval/results/reproduced")
        compare_results(args.manifest, output_dir)


if __name__ == "__main__":
    main()
