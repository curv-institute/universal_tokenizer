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
"""Reproduce an experiment from a manifest.

Supports:
- Single manifest path
- Glob patterns (e.g., eval/results/*/manifests/*.json)
- Strict metric comparison with configurable tolerance
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Fields that are allowed to differ between runs (timestamps, paths, etc.)
ALLOWED_DIFF_FIELDS = {
    "timestamp",
    "git_hash",
    "jj_hash",
    "config_path",
    "command",
    "environment",
    "device",
}

# Metric tolerance for floating point comparison
METRIC_TOLERANCE = 1e-6


def reproduce(manifest_path: Path, output_dir: Path | None = None) -> Path:
    """Reproduce experiment from manifest.

    Args:
        manifest_path: Path to run manifest
        output_dir: Override output directory (default: same parent as manifest)

    Returns:
        Path to reproduced run directory
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
        raise ValueError("No config in manifest")

    # Write temporary config file
    import tomli_w

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".toml", delete=False
    ) as f:
        tomli_w.dump(config, f)
        temp_config = Path(f.name)

    print(f"Recreated config at {temp_config}")

    # Determine output directory
    # Default: reproduce into sibling directory of original run
    if output_dir is None:
        original_run_dir = manifest_path.parent.parent
        output_dir = original_run_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    from scripts.run_experiment import run_experiment

    name = f"reproduce_{manifest.get('name', 'exp')}"
    run_dir = run_experiment(temp_config, name, output_dir)

    # Cleanup
    temp_config.unlink()

    print(f"\nReproduction complete. Results in {run_dir}")
    return run_dir


def compare_metrics(original: dict, reproduced: dict, path: str = "") -> list[str]:
    """Recursively compare metrics, returning list of differences."""
    diffs = []

    if isinstance(original, dict) and isinstance(reproduced, dict):
        all_keys = set(original.keys()) | set(reproduced.keys())
        for key in all_keys:
            key_path = f"{path}.{key}" if path else key
            if key in ALLOWED_DIFF_FIELDS:
                continue
            if key not in original:
                diffs.append(f"NEW: {key_path} = {reproduced[key]}")
            elif key not in reproduced:
                diffs.append(f"MISSING: {key_path} (was {original[key]})")
            else:
                diffs.extend(compare_metrics(original[key], reproduced[key], key_path))
    elif isinstance(original, (int, float)) and isinstance(reproduced, (int, float)):
        if abs(original - reproduced) > METRIC_TOLERANCE:
            diffs.append(f"DIFF: {path}: {original} -> {reproduced} (delta={reproduced - original})")
    elif isinstance(original, list) and isinstance(reproduced, list):
        if len(original) != len(reproduced):
            diffs.append(f"LENGTH: {path}: {len(original)} -> {len(reproduced)}")
        else:
            for i, (o, r) in enumerate(zip(original, reproduced)):
                diffs.extend(compare_metrics(o, r, f"{path}[{i}]"))
    elif original != reproduced:
        diffs.append(f"DIFF: {path}: {original!r} -> {reproduced!r}")

    return diffs


def compare_results(original_manifest: Path, reproduced_dir: Path) -> bool:
    """Compare original and reproduced results.

    Args:
        original_manifest: Path to original manifest
        reproduced_dir: Path to reproduced run directory

    Returns:
        True if results match within tolerance, False otherwise
    """
    print("\n" + "=" * 50)
    print("COMPARING RESULTS")
    print("=" * 50)

    with open(original_manifest) as f:
        original = json.load(f)

    reproduced_summary = reproduced_dir / "summary.json"
    if not reproduced_summary.exists():
        print("ERROR: No reproduced summary found")
        return False

    with open(reproduced_summary) as f:
        reproduced = json.load(f)

    # Compare results section
    orig_results = original.get("results", {})
    repr_results = reproduced.get("results", {})

    if not orig_results:
        print("WARNING: Original manifest has no results section")
        # Try to load from original run's summary.json
        original_run_dir = original_manifest.parent.parent
        original_summary = original_run_dir / "summary.json"
        if original_summary.exists():
            with open(original_summary) as f:
                orig_data = json.load(f)
                orig_results = orig_data.get("results", {})

    diffs = compare_metrics(orig_results, repr_results)

    if diffs:
        print("\nDifferences found:")
        for diff in diffs:
            print(f"  {diff}")
        print(f"\nTotal: {len(diffs)} difference(s)")
        return False
    else:
        print("\nAll metrics match within tolerance!")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce experiment from manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce single run
  uv run scripts/reproduce.py eval/results/paper_v0_1/manifests/paper_v0_1_*.json

  # Reproduce and compare
  uv run scripts/reproduce.py --compare eval/results/paper_v0_1/manifests/*.json

  # Reproduce multiple runs (glob)
  uv run scripts/reproduce.py --manifest "eval/results/*/manifests/*.json" --compare
""",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path or glob pattern to manifest JSON(s)",
    )
    parser.add_argument(
        "manifest_positional",
        type=str,
        nargs="?",
        default=None,
        help="Path to manifest JSON (positional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: sibling of original)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results after reproduction",
    )

    args = parser.parse_args()

    # Resolve manifest path (positional or --manifest)
    manifest_pattern = args.manifest or args.manifest_positional
    if not manifest_pattern:
        parser.error("Must provide manifest path")

    # Expand glob
    manifest_paths = glob.glob(manifest_pattern)
    if not manifest_paths:
        # Try as literal path
        if Path(manifest_pattern).exists():
            manifest_paths = [manifest_pattern]
        else:
            parser.error(f"No manifests found matching: {manifest_pattern}")

    print(f"Found {len(manifest_paths)} manifest(s)")

    all_passed = True
    for manifest_str in manifest_paths:
        manifest_path = Path(manifest_str)
        print(f"\n{'=' * 60}")
        print(f"Processing: {manifest_path}")
        print("=" * 60)

        try:
            run_dir = reproduce(manifest_path, args.output)

            if args.compare:
                passed = compare_results(manifest_path, run_dir)
                if not passed:
                    all_passed = False
        except Exception as e:
            print(f"ERROR: {e}")
            all_passed = False

    if args.compare:
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL REPRODUCTIONS PASSED")
        else:
            print("SOME REPRODUCTIONS FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
