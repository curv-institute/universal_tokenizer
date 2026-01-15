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
"""Evaluate Universal Lossless Tokenizer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.config import load_config
from tokenizer.bundle import UniversalTokenizer
from tokenizer.baselines import RawByteTokenizer, ByteBPETokenizer
from tokenizer.data import generate_text_like, generate_random_bytes
from tokenizer.metrics import (
    compute_metrics,
    aggregate_metrics,
    verify_lossless,
    compute_entropy,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    config_path: Path,
    model_path: Path | None = None,
    data_path: Path | None = None,
    output_dir: Path = Path("eval/results"),
) -> None:
    """Run evaluation.

    Args:
        config_path: Path to config
        model_path: Path to trained model (optional)
        data_path: Path to eval data (generates if None)
        output_dir: Output directory
    """
    config = load_config(config_path)
    set_seed(config.eval.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if data_path and data_path.exists():
        if data_path.is_dir():
            print(f"Loading eval data from directory {data_path}")
            data = b""
            for file in sorted(data_path.iterdir()):
                if file.is_file():
                    print(f"  Loading {file.name}")
                    with open(file, "rb") as f:
                        data += f.read()
        else:
            print(f"Loading eval data from {data_path}")
            with open(data_path, "rb") as f:
                data = f.read()
    else:
        print("Generating eval data...")
        data = generate_text_like(10_000, seed=config.eval.seed)

    # Chunk data
    chunk_size = 256
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    chunks = chunks[: config.eval.max_samples]
    print(f"Evaluating on {len(chunks)} chunks")

    # Initialize tokenizers
    if model_path and model_path.exists():
        print(f"Loading model from {model_path}")
        tokenizer = UniversalTokenizer.load(model_path)
    else:
        print("Using untrained tokenizer")
        tokenizer = UniversalTokenizer(config)

    baselines = [
        RawByteTokenizer(),
        ByteBPETokenizer(vocab_size=1024),
    ]

    # Train BPE baseline on subset
    print("Training BPE baseline...")
    train_data = b"".join(chunks[:100])
    baselines[1].train(train_data)

    results = {"universal": [], "raw_bytes": [], "byte_bpe": []}
    all_tokenizers = [("universal", tokenizer)] + [(b.name, b) for b in baselines]

    # Evaluate each tokenizer
    for name, tok in all_tokenizers:
        print(f"\nEvaluating {name}...")
        metrics_list = []
        lossless_count = 0

        for chunk in tqdm(chunks, desc=name):
            result = tok.encode(chunk)
            decoded = tok.decode(result)

            metrics = compute_metrics(result, chunk)
            metrics_list.append(metrics)

            if verify_lossless(chunk, decoded.data):
                lossless_count += 1

        agg = aggregate_metrics(metrics_list)
        agg.lossless_rate = lossless_count / len(chunks)

        results[name] = agg.to_dict()

        print(f"  Compression ratio: {agg.mean_compression_ratio:.2f}")
        print(f"  Bits per byte: {agg.mean_bits_per_byte:.2f}")
        print(f"  Lossless rate: {agg.lossless_rate:.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Detailed metrics
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for name, res in results.items():
            for m in res.get("sample_metrics", []):
                entry = {"tokenizer": name, **m}
                f.write(json.dumps(entry) + "\n")

    # Summary
    summary = {
        "timestamp": timestamp,
        "config": str(config_path),
        "model": str(model_path) if model_path else None,
        "num_samples": len(chunks),
        "data_entropy": compute_entropy(data),
        "results": {
            name: {k: v for k, v in res.items() if k != "sample_metrics"}
            for name, res in results.items()
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Universal Tokenizer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Config file",
    )
    parser.add_argument("--model", type=Path, default=None, help="Model path")
    parser.add_argument("--data", type=Path, default=None, help="Eval data path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results"),
        help="Output directory",
    )

    args = parser.parse_args()
    evaluate(args.config, args.model, args.data, args.output)


if __name__ == "__main__":
    main()
