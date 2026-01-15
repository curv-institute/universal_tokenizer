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
"""Encode data using Universal Lossless Tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bundle import UniversalTokenizer
from tokenizer.config import load_config
from tokenizer.metrics import compute_metrics, verify_lossless


def tokenize(
    input_path: Path,
    output_path: Path | None = None,
    model_path: Path | None = None,
    config_path: Path | None = None,
    verify: bool = True,
) -> None:
    """Tokenize a file.

    Args:
        input_path: Input file path
        output_path: Output path for tokens (JSON)
        model_path: Path to trained model
        config_path: Path to config (if no model)
        verify: Verify lossless reconstruction
    """
    # Load input
    print(f"Loading input from {input_path}")
    with open(input_path, "rb") as f:
        data = f.read()
    print(f"Input size: {len(data)} bytes")

    # Load tokenizer
    if model_path and model_path.exists():
        print(f"Loading model from {model_path}")
        tokenizer = UniversalTokenizer.load(model_path)
    elif config_path:
        print(f"Initializing from config {config_path}")
        config = load_config(config_path)
        tokenizer = UniversalTokenizer(config)
    else:
        print("Using default config")
        config = load_config(Path("configs/default.toml"))
        tokenizer = UniversalTokenizer(config)

    # Encode
    print("Encoding...")
    result = tokenizer.encode(data)
    print(f"Generated {result.num_tokens} tokens")

    # Compute metrics
    metrics = compute_metrics(result, data)
    print(f"Compression ratio: {metrics.compression_ratio:.2f}")
    print(f"Bits per byte: {metrics.bits_per_byte:.2f}")
    print(f"Avg token length: {metrics.avg_token_length:.2f}")

    # Verify lossless
    if verify:
        print("Verifying lossless reconstruction...")
        decoded = tokenizer.decode(result)
        if verify_lossless(data, decoded.data):
            print("PASS: Lossless reconstruction verified")
        else:
            print("FAIL: Reconstruction mismatch!")
            print(f"Original length: {len(data)}")
            print(f"Decoded length: {len(decoded.data)}")

    # Save output
    if output_path:
        output = {
            "input_path": str(input_path),
            "input_size": len(data),
            "num_tokens": result.num_tokens,
            "token_ids": result.ids,
            "metrics": metrics.to_dict(),
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved output to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize with Universal Tokenizer")
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output JSON file"
    )
    parser.add_argument(
        "--model", "-m", type=Path, default=None, help="Trained model path"
    )
    parser.add_argument(
        "--config", "-c", type=Path, default=None, help="Config file path"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip lossless verification"
    )

    args = parser.parse_args()
    tokenize(args.input, args.output, args.model, args.config, not args.no_verify)


if __name__ == "__main__":
    main()
