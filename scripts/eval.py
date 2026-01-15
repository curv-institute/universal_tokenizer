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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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


@dataclass
class HHCDiagnostics:
    """HHC diagnostics collected during evaluation."""

    hhc_active_fraction: float = 0.0
    hhc_mean_num_neighbors: float = 0.0
    hhc_mean_neighbor_dist: float = 0.0
    hhc_var_neighbor_dist: float = 0.0
    hhc_mean_curvature_delta: float = 0.0
    hhc_nonzero_delta_fraction: float = 0.0

    # Internal accumulators
    _total_samples: int = field(default=0, repr=False)
    _active_samples: int = field(default=0, repr=False)
    _neighbor_counts: list[int] = field(default_factory=list, repr=False)
    _neighbor_dists: list[float] = field(default_factory=list, repr=False)
    _neighbor_vars: list[float] = field(default_factory=list, repr=False)
    _curvature_deltas: list[float] = field(default_factory=list, repr=False)

    def record(
        self,
        num_neighbors: int,
        mean_neighbor_dist: float,
        var_neighbor_dist: float,
        curvature_delta: float,
    ) -> None:
        """Record a single HHC observation."""
        self._total_samples += 1
        if num_neighbors > 0:
            self._active_samples += 1
            self._neighbor_counts.append(num_neighbors)
            self._neighbor_dists.append(mean_neighbor_dist)
            self._neighbor_vars.append(var_neighbor_dist)
        if curvature_delta > 1e-8:
            self._curvature_deltas.append(curvature_delta)

    def finalize(self) -> None:
        """Compute final aggregate metrics."""
        if self._total_samples > 0:
            self.hhc_active_fraction = self._active_samples / self._total_samples
        if self._neighbor_counts:
            self.hhc_mean_num_neighbors = sum(self._neighbor_counts) / len(
                self._neighbor_counts
            )
        if self._neighbor_dists:
            self.hhc_mean_neighbor_dist = sum(self._neighbor_dists) / len(
                self._neighbor_dists
            )
        if self._neighbor_vars:
            self.hhc_var_neighbor_dist = sum(self._neighbor_vars) / len(
                self._neighbor_vars
            )
        if self._curvature_deltas:
            self.hhc_mean_curvature_delta = sum(self._curvature_deltas) / len(
                self._curvature_deltas
            )
        if self._total_samples > 0:
            self.hhc_nonzero_delta_fraction = len(self._curvature_deltas) / self._total_samples

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hhc_active_fraction": self.hhc_active_fraction,
            "hhc_mean_num_neighbors": self.hhc_mean_num_neighbors,
            "hhc_mean_neighbor_dist": self.hhc_mean_neighbor_dist,
            "hhc_var_neighbor_dist": self.hhc_var_neighbor_dist,
            "hhc_mean_curvature_delta": self.hhc_mean_curvature_delta,
            "hhc_nonzero_delta_fraction": self.hhc_nonzero_delta_fraction,
        }


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

    # Check if HHC is enabled
    hhc_enabled = config.hhc.enabled
    hhc_diagnostics: HHCDiagnostics | None = None

    # Evaluate each tokenizer
    for name, tok in all_tokenizers:
        print(f"\nEvaluating {name}...")
        metrics_list = []
        lossless_count = 0

        # Get vocab_size for correct BPB calculation
        vocab_size = getattr(tok, "vocab_size", config.codebook.num_codes)

        # Initialize HHC diagnostics for universal tokenizer with HHC enabled
        collect_hhc = name == "universal" and hhc_enabled
        if collect_hhc:
            hhc_diagnostics = HHCDiagnostics()

        for chunk in tqdm(chunks, desc=name):
            result = tok.encode(chunk)
            decoded = tok.decode(result)

            metrics = compute_metrics(result, chunk, vocab_size=vocab_size)
            metrics_list.append(metrics)

            if verify_lossless(chunk, decoded.data):
                lossless_count += 1

            # Collect HHC diagnostics from the tokenizer
            if collect_hhc and hasattr(tok, "equilibrium"):
                eq_stats = tok.equilibrium.get_hhc_stats()
                if eq_stats is not None:
                    hhc_diagnostics.record(
                        num_neighbors=eq_stats.num_neighbors,
                        mean_neighbor_dist=eq_stats.mean_neighbor_distance,
                        var_neighbor_dist=eq_stats.neighbor_distance_variance,
                        curvature_delta=eq_stats.delta_applied,
                    )
                else:
                    # HHC not applied for this sample
                    hhc_diagnostics.record(
                        num_neighbors=0,
                        mean_neighbor_dist=0.0,
                        var_neighbor_dist=0.0,
                        curvature_delta=0.0,
                    )

        agg = aggregate_metrics(metrics_list)
        agg.lossless_rate = lossless_count / len(chunks)

        results[name] = agg.to_dict()

        print(f"  Compression ratio: {agg.mean_compression_ratio:.2f}")
        print(f"  End-to-end BPB: {agg.mean_end_to_end_bpb:.2f} (lossless)")
        print(f"  Structural BPB: {agg.mean_structural_bpb:.2f} (representational)")
        print(f"  Lossless rate: {agg.lossless_rate:.1%}")

    # Finalize and display HHC diagnostics
    if hhc_diagnostics is not None:
        hhc_diagnostics.finalize()
        print("\n=== HHC Diagnostics ===")
        print(f"  HHC Active: {'Yes' if hhc_enabled else 'No'}")
        print(f"  Active fraction: {hhc_diagnostics.hhc_active_fraction:.1%}")
        print(f"  Mean neighbors: {hhc_diagnostics.hhc_mean_num_neighbors:.1f}")
        print(f"  Mean neighbor distance: {hhc_diagnostics.hhc_mean_neighbor_dist:.3f}")
        print(f"  Neighbor distance variance: {hhc_diagnostics.hhc_var_neighbor_dist:.3f}")
        print(f"  Mean curvature delta: {hhc_diagnostics.hhc_mean_curvature_delta:.3f}")
        print(f"  Nonzero delta fraction: {hhc_diagnostics.hhc_nonzero_delta_fraction:.1%}")

        # Acceptance criteria warnings
        if hhc_diagnostics.hhc_active_fraction < 0.95:
            print(
                f"\n  WARNING: HHC active fraction ({hhc_diagnostics.hhc_active_fraction:.1%}) "
                "is below 95% threshold"
            )
        if hhc_diagnostics.hhc_nonzero_delta_fraction < 0.20:
            print(
                f"\n  WARNING: Nonzero delta fraction ({hhc_diagnostics.hhc_nonzero_delta_fraction:.1%}) "
                "is below 20% threshold - HHC may be producing degenerate results"
            )

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

    # Add HHC diagnostics to summary if collected
    if hhc_diagnostics is not None:
        summary["hhc_enabled"] = hhc_enabled
        summary["hhc_diagnostics"] = hhc_diagnostics.to_dict()

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
