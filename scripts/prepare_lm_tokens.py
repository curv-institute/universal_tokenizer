#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
# ]
# ///
"""Prepare tokenized sequences for LM proxy evaluation.

Tokenizes the shift dataset using both baseline (HHC disabled) and HHC
configurations, saving token sequences as numpy arrays for downstream
language model evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent to path for local tokenizer import
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.config import load_config
from tokenizer.bundle import UniversalTokenizer


def tokenize_data(
    data: bytes,
    config_path: Path,
    name: str,
) -> tuple[list[int], dict]:
    """Tokenize data using the specified configuration.

    Args:
        data: Input byte data
        config_path: Path to config file
        name: Name for progress bar

    Returns:
        Tuple of (token_ids, info_dict)
    """
    config = load_config(config_path)
    tokenizer = UniversalTokenizer(config)

    # Tokenize
    print(f"\nTokenizing with {name} config: {config_path}")
    print(f"  HHC enabled: {config.hhc.enabled}")

    result = tokenizer.encode(data)

    info = {
        "vocab_size": config.codebook.num_codes,
        "num_tokens": len(result.ids),
        "num_bytes": len(data),
        "config_path": str(config_path),
        "hhc_enabled": config.hhc.enabled,
        "hhc_window_tokens": config.hhc.window_tokens if config.hhc.enabled else None,
        "compression_ratio": result.compression_ratio,
        "bytes_per_token": len(data) / len(result.ids) if result.ids else 0,
    }

    return result.ids, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare tokenized sequences for LM proxy evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default paths
  %(prog)s

  # Custom paths
  %(prog)s --data data/custom.bin --output eval/custom_tokens
        """,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/eval/heldout_shift_1mb.bin"),
        help="Path to input data (default: data/eval/heldout_shift_1mb.bin)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/lm_proxy/tokens"),
        help="Output directory (default: eval/results/lm_proxy/tokens)",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/cpu_small.toml"),
        help="Baseline config path (default: configs/cpu_small.toml)",
    )
    parser.add_argument(
        "--hhc-config",
        type=Path,
        default=Path("configs/cpu_small_hhc.toml"),
        help="HHC config path (default: configs/cpu_small_hhc.toml)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    if not args.baseline_config.exists():
        print(f"Error: Baseline config not found: {args.baseline_config}")
        sys.exit(1)

    if not args.hhc_config.exists():
        print(f"Error: HHC config not found: {args.hhc_config}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}")
    with open(args.data, "rb") as f:
        data = f.read()
    print(f"  Loaded {len(data):,} bytes")

    # Tokenize with baseline config
    baseline_ids, baseline_info = tokenize_data(
        data, args.baseline_config, "baseline"
    )

    # Tokenize with HHC config
    hhc_ids, hhc_info = tokenize_data(data, args.hhc_config, "HHC")

    # Save token sequences
    baseline_tokens_path = args.output / "baseline_tokens.npy"
    hhc_tokens_path = args.output / "hhc_tokens.npy"

    np.save(baseline_tokens_path, np.array(baseline_ids, dtype=np.int32))
    np.save(hhc_tokens_path, np.array(hhc_ids, dtype=np.int32))

    print(f"\nSaved baseline tokens to {baseline_tokens_path}")
    print(f"Saved HHC tokens to {hhc_tokens_path}")

    # Save info files
    baseline_info_path = args.output / "baseline_info.json"
    hhc_info_path = args.output / "hhc_info.json"

    with open(baseline_info_path, "w") as f:
        json.dump(baseline_info, f, indent=2)

    with open(hhc_info_path, "w") as f:
        json.dump(hhc_info, f, indent=2)

    print(f"Saved baseline info to {baseline_info_path}")
    print(f"Saved HHC info to {hhc_info_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nInput data: {len(data):,} bytes")

    print(f"\nBaseline (HHC disabled):")
    print(f"  Tokens: {baseline_info['num_tokens']:,}")
    print(f"  Vocab size: {baseline_info['vocab_size']:,}")
    print(f"  Bytes per token: {baseline_info['bytes_per_token']:.2f}")
    print(f"  Compression ratio: {baseline_info['compression_ratio']:.2f}")

    print(f"\nHHC (HHC enabled):")
    print(f"  Tokens: {hhc_info['num_tokens']:,}")
    print(f"  Vocab size: {hhc_info['vocab_size']:,}")
    print(f"  Bytes per token: {hhc_info['bytes_per_token']:.2f}")
    print(f"  Compression ratio: {hhc_info['compression_ratio']:.2f}")

    # Token sequence length ratio
    if baseline_info["num_tokens"] > 0:
        ratio = hhc_info["num_tokens"] / baseline_info["num_tokens"]
        print(f"\nToken sequence length ratio (HHC / baseline): {ratio:.4f}")
        if ratio < 1.0:
            print(f"  HHC produces {(1 - ratio) * 100:.1f}% fewer tokens")
        elif ratio > 1.0:
            print(f"  HHC produces {(ratio - 1) * 100:.1f}% more tokens")
        else:
            print("  HHC produces same number of tokens")

    # Verify both tokenize the same input
    print(f"\nVerification:")
    print(f"  Baseline input bytes: {baseline_info['num_bytes']:,}")
    print(f"  HHC input bytes: {hhc_info['num_bytes']:,}")
    if baseline_info["num_bytes"] == hhc_info["num_bytes"] == len(data):
        print("  Both configurations tokenized the same input bytes")
    else:
        print("  WARNING: Input byte counts do not match!")

    print(f"\nOutput directory: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
