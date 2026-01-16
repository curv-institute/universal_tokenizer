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
    compute_boundary_aware_metrics,
    load_manifest,
    BoundaryAwareMetrics,
)


@dataclass
class HHCDiagnostics:
    """HHC diagnostics collected during evaluation.

    Tracks both HHC neighbor coupling statistics and Harmonizer
    intervention behavior, including correlation with domain boundaries.
    """

    # HHC neighbor coupling metrics
    hhc_active_fraction: float = 0.0
    hhc_mean_num_neighbors: float = 0.0
    hhc_mean_neighbor_dist: float = 0.0
    hhc_var_neighbor_dist: float = 0.0
    hhc_mean_curvature_delta: float = 0.0
    hhc_nonzero_delta_fraction: float = 0.0

    # Harmonizer intervention tracking
    harmonizer_intervention_count: int = 0
    interventions_at_boundary: int = 0
    interventions_away_from_boundary: int = 0
    intervention_boundary_correlation: float = 0.0  # higher = more interventions at boundaries

    # Internal accumulators
    _total_samples: int = field(default=0, repr=False)
    _active_samples: int = field(default=0, repr=False)
    _neighbor_counts: list[int] = field(default_factory=list, repr=False)
    _neighbor_dists: list[float] = field(default_factory=list, repr=False)
    _neighbor_vars: list[float] = field(default_factory=list, repr=False)
    _curvature_deltas: list[float] = field(default_factory=list, repr=False)

    # Intervention tracking accumulators
    _intervention_offsets: list[int] = field(default_factory=list, repr=False)
    _intervention_params_changed: list[dict] = field(default_factory=list, repr=False)

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

    def record_intervention(
        self,
        offset: int,
        params_before: dict[str, float],
        params_after: dict[str, float],
    ) -> None:
        """Record a Harmonizer intervention (parameter adjustment).

        Args:
            offset: Byte offset where intervention occurred
            params_before: Controller parameters before step
            params_after: Controller parameters after step
        """
        # Check if any parameter actually changed
        changed = False
        for key in params_before:
            if key in params_after:
                if abs(params_before[key] - params_after[key]) > 1e-8:
                    changed = True
                    break

        if changed:
            self.harmonizer_intervention_count += 1
            self._intervention_offsets.append(offset)
            self._intervention_params_changed.append({
                "offset": offset,
                "before": params_before.copy(),
                "after": params_after.copy(),
            })

    def finalize(self, domain_boundaries: list[int] | None = None, boundary_window: int = 64) -> None:
        """Compute final aggregate metrics.

        Args:
            domain_boundaries: List of byte offsets for domain boundaries (from manifest)
            boundary_window: Window size (bytes) to consider "at boundary"
        """
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

        # Compute intervention-boundary correlation
        if domain_boundaries and self._intervention_offsets:
            for offset in self._intervention_offsets:
                is_near_boundary = any(
                    abs(offset - db) <= boundary_window for db in domain_boundaries
                )
                if is_near_boundary:
                    self.interventions_at_boundary += 1
                else:
                    self.interventions_away_from_boundary += 1

            # Correlation: ratio of interventions at boundary vs expected if random
            # Expected = total_interventions * (boundary_window_coverage / total_bytes)
            # Simplified: compare fraction at boundary vs fraction of bytes near boundary
            total_interventions = len(self._intervention_offsets)
            if total_interventions > 0:
                frac_at_boundary = self.interventions_at_boundary / total_interventions
                # Higher correlation = more interventions happen at boundaries than expected
                self.intervention_boundary_correlation = frac_at_boundary
        elif self._intervention_offsets:
            # No boundaries provided, all interventions are "away from boundary"
            self.interventions_away_from_boundary = len(self._intervention_offsets)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hhc_active_fraction": self.hhc_active_fraction,
            "hhc_mean_num_neighbors": self.hhc_mean_num_neighbors,
            "hhc_mean_neighbor_dist": self.hhc_mean_neighbor_dist,
            "hhc_var_neighbor_dist": self.hhc_var_neighbor_dist,
            "hhc_mean_curvature_delta": self.hhc_mean_curvature_delta,
            "hhc_nonzero_delta_fraction": self.hhc_nonzero_delta_fraction,
            "harmonizer_intervention_count": self.harmonizer_intervention_count,
            "interventions_at_boundary": self.interventions_at_boundary,
            "interventions_away_from_boundary": self.interventions_away_from_boundary,
            "intervention_boundary_correlation": self.intervention_boundary_correlation,
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
    manifest_path: Path | None = None,
) -> None:
    """Run evaluation.

    Supports two modes:
    1. Standard mode (no manifest): Evaluates on chunked data
    2. Boundary-aware mode (with manifest): Evaluates on full data with
       boundary-aware metrics computed from manifest block information

    Args:
        config_path: Path to config
        model_path: Path to trained model (optional)
        data_path: Path to eval data (generates if None)
        output_dir: Output directory
        manifest_path: Path to manifest file for boundary-aware evaluation
    """
    config = load_config(config_path)
    set_seed(config.eval.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest if provided (enables boundary-aware mode)
    manifest: dict = {}
    domain_boundaries: list[int] = []
    boundary_aware_mode = False

    if manifest_path and manifest_path.exists():
        print(f"Loading manifest from {manifest_path}")
        manifest = load_manifest(manifest_path)
        if manifest:
            boundary_aware_mode = True
            # Extract domain boundary offsets from manifest blocks
            for block in manifest.get("blocks", []):
                start = block.get("start", 0)
                if start > 0:
                    domain_boundaries.append(start)
            print(f"  Boundary-aware mode enabled with {len(domain_boundaries)} domain boundaries")
            print(f"  Block types: {list(manifest.get('block_summary', {}).keys())}")

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

    # Chunk data (for standard mode) or use full data (boundary-aware mode)
    if boundary_aware_mode:
        # In boundary-aware mode, process full data to preserve boundary positions
        chunks = [data]
        print(f"Evaluating full data ({len(data):,} bytes) in boundary-aware mode")
    else:
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

    # Storage for boundary-aware metrics (computed for universal tokenizer)
    boundary_metrics: BoundaryAwareMetrics | None = None
    all_tokens_universal: list = []  # Collect all tokens for boundary analysis

    # Evaluate each tokenizer
    for name, tok in all_tokenizers:
        print(f"\nEvaluating {name}...")
        metrics_list = []
        lossless_count = 0
        current_offset = 0  # Track byte offset for intervention correlation

        # Get vocab_size for correct BPB calculation
        vocab_size = getattr(tok, "vocab_size", config.codebook.num_codes)

        # Initialize HHC diagnostics for universal tokenizer with HHC enabled
        collect_hhc = name == "universal" and hhc_enabled
        if collect_hhc:
            hhc_diagnostics = HHCDiagnostics()

        for chunk in tqdm(chunks, desc=name):
            # Get controller parameters before encoding (for intervention tracking)
            params_before = None
            if collect_hhc and hasattr(tok, "controller"):
                params_before = tok.controller.get_parameters().copy()

            result = tok.encode(chunk)
            decoded = tok.decode(result)

            # Get controller parameters after encoding
            params_after = None
            if collect_hhc and hasattr(tok, "controller"):
                params_after = tok.controller.get_parameters().copy()

            # Track intervention if parameters changed
            if collect_hhc and params_before is not None and params_after is not None:
                hhc_diagnostics.record_intervention(
                    offset=current_offset,
                    params_before=params_before,
                    params_after=params_after,
                )

            metrics = compute_metrics(result, chunk, vocab_size=vocab_size)
            metrics_list.append(metrics)

            if verify_lossless(chunk, decoded.data):
                lossless_count += 1

            # Collect tokens for boundary-aware analysis (universal tokenizer only)
            if name == "universal" and boundary_aware_mode:
                all_tokens_universal.extend(result.tokens)

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

            current_offset += len(chunk)

        agg = aggregate_metrics(metrics_list)
        agg.lossless_rate = lossless_count / len(chunks)

        # Compute boundary-aware metrics for universal tokenizer
        if name == "universal" and boundary_aware_mode and all_tokens_universal:
            boundary_metrics = compute_boundary_aware_metrics(
                tokens=all_tokens_universal,
                manifest=manifest,
                boundary_threshold=8,
                boundary_window=64,
                vocab_size=vocab_size,
            )
            agg.boundary_metrics = boundary_metrics

        results[name] = agg.to_dict()

        print(f"  Compression ratio: {agg.mean_compression_ratio:.2f}")
        print(f"  End-to-end BPB: {agg.mean_end_to_end_bpb:.2f} (lossless)")
        print(f"  Structural BPB: {agg.mean_structural_bpb:.2f} (representational)")
        print(f"  Lossless rate: {agg.lossless_rate:.1%}")

        # Print boundary-aware metrics if available
        if name == "universal" and boundary_metrics is not None:
            print("\n  === Boundary-Aware Metrics ===")
            print(f"  Domain boundaries: {boundary_metrics.total_domain_boundaries}")
            print(f"  Boundary alignment rate: {boundary_metrics.boundary_alignment_rate:.1%}")
            print(f"  Boundary curvature delta: {boundary_metrics.boundary_curvature_delta:.4f}")
            if boundary_metrics.per_domain_bpb:
                print("  Per-domain BPB:")
                for domain, bpb in sorted(boundary_metrics.per_domain_bpb.items()):
                    print(f"    {domain}: {bpb:.2f}")

    # Finalize and display HHC diagnostics
    if hhc_diagnostics is not None:
        # Pass domain boundaries for intervention correlation analysis
        hhc_diagnostics.finalize(
            domain_boundaries=domain_boundaries if boundary_aware_mode else None,
            boundary_window=64,
        )
        print("\n=== HHC Diagnostics ===")
        print(f"  HHC Active: {'Yes' if hhc_enabled else 'No'}")
        print(f"  Active fraction: {hhc_diagnostics.hhc_active_fraction:.1%}")
        print(f"  Mean neighbors: {hhc_diagnostics.hhc_mean_num_neighbors:.1f}")
        print(f"  Mean neighbor distance: {hhc_diagnostics.hhc_mean_neighbor_dist:.3f}")
        print(f"  Neighbor distance variance: {hhc_diagnostics.hhc_var_neighbor_dist:.3f}")
        print(f"  Mean curvature delta: {hhc_diagnostics.hhc_mean_curvature_delta:.3f}")
        print(f"  Nonzero delta fraction: {hhc_diagnostics.hhc_nonzero_delta_fraction:.1%}")

        # Harmonizer intervention statistics
        print("\n  === Harmonizer Interventions ===")
        print(f"  Total interventions: {hhc_diagnostics.harmonizer_intervention_count}")
        if boundary_aware_mode:
            print(f"  Interventions at boundary: {hhc_diagnostics.interventions_at_boundary}")
            print(f"  Interventions away from boundary: {hhc_diagnostics.interventions_away_from_boundary}")
            print(f"  Boundary correlation: {hhc_diagnostics.intervention_boundary_correlation:.2%}")

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

    # Summary with comprehensive structure
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "config": str(config_path),
        "model": str(model_path) if model_path else None,
        "num_samples": len(chunks),
        "data_entropy": compute_entropy(data),
        "evaluation_mode": "boundary_aware" if boundary_aware_mode else "standard",
        "results": {
            name: {k: v for k, v in res.items() if k != "sample_metrics"}
            for name, res in results.items()
        },
    }

    # Add manifest info if in boundary-aware mode
    if boundary_aware_mode:
        summary["manifest"] = {
            "path": str(manifest_path),
            "num_domain_boundaries": len(domain_boundaries),
            "block_types": list(manifest.get("block_summary", {}).keys()),
        }

    # Add HHC diagnostics to summary if collected
    if hhc_diagnostics is not None:
        summary["hhc_enabled"] = hhc_enabled
        summary["hhc_diagnostics"] = hhc_diagnostics.to_dict()

    # Add boundary-aware metrics separately for easy access
    if boundary_metrics is not None:
        summary["boundary_aware_metrics"] = boundary_metrics.to_dict()

    # Add comparison structure for baseline vs HHC analysis
    if "universal" in results:
        universal_results = {k: v for k, v in results["universal"].items() if k != "sample_metrics"}
        baseline_results = {}
        for name in ["raw_bytes", "byte_bpe"]:
            if name in results:
                baseline_results[name] = {
                    k: v for k, v in results[name].items() if k != "sample_metrics"
                }

        summary["comparison"] = {
            "universal_tokenizer": universal_results,
            "baselines": baseline_results,
        }

        # Add HHC-specific comparison if HHC is enabled
        if hhc_enabled and hhc_diagnostics is not None:
            summary["comparison"]["hhc_impact"] = {
                "hhc_enabled": True,
                "hhc_active_fraction": hhc_diagnostics.hhc_active_fraction,
                "intervention_count": hhc_diagnostics.harmonizer_intervention_count,
                "mean_curvature_delta": hhc_diagnostics.hhc_mean_curvature_delta,
            }
            if boundary_aware_mode:
                summary["comparison"]["hhc_impact"]["boundary_correlation"] = (
                    hhc_diagnostics.intervention_boundary_correlation
                )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Universal Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation Modes:
  Standard mode (default): Evaluates on chunked data samples.
  Boundary-aware mode (--manifest): Evaluates on full data with
    boundary-aware metrics from manifest block information.

Examples:
  # Standard evaluation
  %(prog)s --config configs/default.toml --data data/eval.bin

  # Boundary-aware evaluation with manifest
  %(prog)s --config configs/default.toml --data data/eval.bin \\
           --manifest data/eval/manifests/manifest_eval_seed42.json
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Config file (default: configs/default.toml)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model (optional, uses untrained if not specified)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to eval data file or directory (generates synthetic data if not specified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results"),
        help="Output directory for results (default: eval/results)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON file for boundary-aware evaluation. "
        "When provided, enables boundary-aware mode with domain transition metrics.",
    )

    args = parser.parse_args()
    evaluate(
        config_path=args.config,
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
