#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "matplotlib",
# ]
# ///
"""Generate reports and visualizations for LM proxy experiments.

Aggregates evaluation results from lm_proxy_eval.py and generates:
- summary.json with all metrics
- loss_curves.pdf comparing baseline vs HHC
- metrics.jsonl with per-run detailed metrics

Usage:
    # Generate full report from evaluation results
    uv run scripts/lm_proxy_report.py --input eval/results/lm_proxy --output eval/results/lm_proxy

    # Generate report with custom title
    uv run scripts/lm_proxy_report.py --input eval/results/lm_proxy --title "LM Proxy Experiment v1"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_evaluation_results(input_dir: Path) -> dict[str, Any]:
    """Load all evaluation results from the input directory.

    Args:
        input_dir: Directory containing evaluation outputs

    Returns:
        Dictionary with all loaded results
    """
    results: dict[str, Any] = {
        "convergence": {},
        "extension": {},
        "variance": {},
        "trajectories": {},
    }

    # Load combined evaluation results if available
    combined_path = input_dir / "evaluation_results.json"
    if combined_path.exists():
        with open(combined_path) as f:
            data = json.load(f)
            results["convergence"] = data.get("convergence", {})
            results["extension"] = data.get("extension", {})
            results["variance"] = data.get("variance", {})

    # Load individual convergence metrics
    for config_type in ["baseline", "hhc"]:
        config_dir = input_dir / config_type
        if not config_dir.exists():
            continue

        results["trajectories"][config_type] = {}

        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir():
                continue

            # Load convergence metrics
            conv_path = seed_dir / "convergence_metrics.json"
            if conv_path.exists():
                with open(conv_path) as f:
                    if config_type not in results["convergence"]:
                        results["convergence"][config_type] = {}
                    results["convergence"][config_type][seed_dir.name] = json.load(f)

            # Load extension metrics
            ext_path = seed_dir / "extension_metrics.json"
            if ext_path.exists():
                with open(ext_path) as f:
                    if config_type not in results["extension"]:
                        results["extension"][config_type] = {}
                    results["extension"][config_type][seed_dir.name] = json.load(f)

            # Load loss trajectory
            traj_path = seed_dir / "loss_trajectory.jsonl"
            if traj_path.exists():
                trajectory = []
                with open(traj_path) as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        trajectory.append((entry["step"], entry["loss"]))
                results["trajectories"][config_type][seed_dir.name] = trajectory

    # Load variance metrics
    for config_type in ["baseline", "hhc"]:
        var_path = input_dir / f"variance_metrics_{config_type}.json"
        if var_path.exists():
            with open(var_path) as f:
                results["variance"][config_type] = json.load(f)

    return results


def compute_aggregated_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregated summary statistics.

    Args:
        results: Loaded evaluation results

    Returns:
        Summary dictionary matching required format
    """
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {},
        "hhc": {},
        "comparison": {},
    }

    # Process each config type
    for config_type in ["baseline", "hhc"]:
        config_summary: dict[str, Any] = {}

        # Get variance metrics if available
        var_metrics = results.get("variance", {}).get(config_type, {})
        if var_metrics:
            config_summary["final_loss_mean"] = var_metrics.get("final_loss_mean", 0.0)
            config_summary["final_loss_std"] = var_metrics.get("final_loss_std", 0.0)
            config_summary["final_loss_min"] = var_metrics.get("final_loss_min", 0.0)
            config_summary["final_loss_max"] = var_metrics.get("final_loss_max", 0.0)
            config_summary["steps_to_3.5"] = var_metrics.get("steps_to_3_5_mean")
            config_summary["steps_to_3.5_std"] = var_metrics.get("steps_to_3_5_std")
            config_summary["convergence_success_rate"] = var_metrics.get("convergence_success_rate", 0.0)
            config_summary["late_loss_std_mean"] = var_metrics.get("late_loss_std_mean", 0.0)
            config_summary["num_seeds"] = var_metrics.get("num_seeds", 0)
        else:
            # Compute from individual convergence results
            conv_results = results.get("convergence", {}).get(config_type, {})
            if conv_results:
                final_losses = [r.get("final_loss", 0.0) for r in conv_results.values()]
                steps_list = [r.get("steps_to_3_5") for r in conv_results.values() if r.get("steps_to_3_5")]

                config_summary["final_loss_mean"] = float(np.mean(final_losses)) if final_losses else 0.0
                config_summary["final_loss_std"] = float(np.std(final_losses)) if len(final_losses) > 1 else 0.0
                config_summary["final_loss_min"] = float(np.min(final_losses)) if final_losses else 0.0
                config_summary["final_loss_max"] = float(np.max(final_losses)) if final_losses else 0.0
                config_summary["steps_to_3.5"] = float(np.mean(steps_list)) if steps_list else None
                config_summary["convergence_success_rate"] = len(steps_list) / len(conv_results) if conv_results else 0.0
                config_summary["num_seeds"] = len(conv_results)

        # Get extension metrics
        ext_results = results.get("extension", {}).get(config_type, {})
        if ext_results:
            degradations = [r.get("total_degradation", 0.0) for r in ext_results.values()]
            config_summary["prompt_extension_degradation"] = float(np.mean(degradations)) if degradations else 0.0
            config_summary["prompt_extension_degradation_std"] = float(np.std(degradations)) if len(degradations) > 1 else 0.0

        summary[config_type] = config_summary

    # Compute comparison metrics
    baseline = summary.get("baseline", {})
    hhc = summary.get("hhc", {})

    if baseline and hhc:
        baseline_loss = baseline.get("final_loss_mean", 0.0)
        hhc_loss = hhc.get("final_loss_mean", 0.0)

        baseline_std = baseline.get("final_loss_std", 0.0)
        hhc_std = hhc.get("final_loss_std", 0.0)

        baseline_deg = baseline.get("prompt_extension_degradation", 0.0)
        hhc_deg = hhc.get("prompt_extension_degradation", 0.0)

        # Loss improvement (negative means HHC is better)
        loss_improvement = baseline_loss - hhc_loss

        # Stability improvement (lower std is better)
        stability_improvement = baseline_std - hhc_std

        # Variance reduction (as percentage)
        variance_reduction = 0.0
        if baseline_std > 0:
            variance_reduction = (baseline_std - hhc_std) / baseline_std

        # Extension stability improvement
        extension_stability_improvement = baseline_deg - hhc_deg

        summary["comparison"] = {
            "loss_improvement": loss_improvement,
            "loss_improvement_pct": (loss_improvement / baseline_loss * 100) if baseline_loss > 0 else 0.0,
            "stability_improvement": stability_improvement,
            "variance_reduction": variance_reduction,
            "variance_reduction_pct": variance_reduction * 100,
            "extension_stability_improvement": extension_stability_improvement,
        }

    return summary


def generate_metrics_jsonl(results: dict[str, Any], output_path: Path) -> None:
    """Generate detailed metrics.jsonl file.

    Args:
        results: Loaded evaluation results
        output_path: Path to output file
    """
    with open(output_path, "w") as f:
        # Write convergence metrics
        for config_type, seeds in results.get("convergence", {}).items():
            for seed_name, metrics in seeds.items():
                entry = {
                    "type": "convergence",
                    "config": config_type,
                    "seed": seed_name,
                    **metrics,
                }
                f.write(json.dumps(entry) + "\n")

        # Write extension metrics
        for config_type, seeds in results.get("extension", {}).items():
            for seed_name, metrics in seeds.items():
                entry = {
                    "type": "extension",
                    "config": config_type,
                    "seed": seed_name,
                    **metrics,
                }
                f.write(json.dumps(entry) + "\n")

        # Write variance metrics
        for config_type, metrics in results.get("variance", {}).items():
            entry = {
                "type": "variance",
                "config": config_type,
                **metrics,
            }
            f.write(json.dumps(entry) + "\n")


def generate_loss_curves_plot(
    results: dict[str, Any],
    output_path: Path,
    title: str = "LM Proxy Training: Loss Curves",
) -> None:
    """Generate loss curves comparison plot.

    Creates a PDF plot with:
    - X-axis: training steps
    - Y-axis: loss
    - Baseline (dashed) and HHC (solid) lines
    - Mean with shaded std region for multiple seeds

    Args:
        results: Loaded evaluation results with trajectories
        output_path: Path to output PDF
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "baseline": "#1f77b4",  # Blue
        "hhc": "#ff7f0e",  # Orange
    }
    styles = {
        "baseline": "--",  # Dashed
        "hhc": "-",  # Solid
    }
    labels = {
        "baseline": "Baseline",
        "hhc": "HHC",
    }

    trajectories = results.get("trajectories", {})

    for config_type in ["baseline", "hhc"]:
        config_trajectories = trajectories.get(config_type, {})
        if not config_trajectories:
            continue

        # Collect all trajectories and align by step
        all_steps = set()
        for traj in config_trajectories.values():
            all_steps.update(step for step, _ in traj)

        all_steps = sorted(all_steps)

        if not all_steps:
            continue

        # Build matrix of losses (seeds x steps)
        loss_matrix = []
        for traj in config_trajectories.values():
            traj_dict = dict(traj)
            losses = [traj_dict.get(step, np.nan) for step in all_steps]
            loss_matrix.append(losses)

        loss_matrix = np.array(loss_matrix)

        # Compute mean and std across seeds
        with np.errstate(all="ignore"):
            mean_loss = np.nanmean(loss_matrix, axis=0)
            std_loss = np.nanstd(loss_matrix, axis=0)

        # Replace NaN with interpolated values for plotting
        valid_mask = ~np.isnan(mean_loss)
        if not np.any(valid_mask):
            continue

        steps_arr = np.array(all_steps)
        valid_steps = steps_arr[valid_mask]
        valid_mean = mean_loss[valid_mask]
        valid_std = std_loss[valid_mask]

        # Plot mean line
        ax.plot(
            valid_steps,
            valid_mean,
            linestyle=styles[config_type],
            color=colors[config_type],
            label=labels[config_type],
            linewidth=2,
        )

        # Plot shaded std region
        ax.fill_between(
            valid_steps,
            valid_mean - valid_std,
            valid_mean + valid_std,
            color=colors[config_type],
            alpha=0.2,
        )

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add horizontal threshold lines
    for threshold, ls in [(4.0, ":"), (3.5, "-."), (3.0, ":")]:
        ax.axhline(y=threshold, color="gray", linestyle=ls, alpha=0.5, linewidth=0.8)
        ax.annotate(
            f"loss={threshold}",
            xy=(ax.get_xlim()[1], threshold),
            xytext=(-5, 2),
            textcoords="offset points",
            fontsize=8,
            color="gray",
            ha="right",
        )

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Loss curves plot saved to {output_path}")


def generate_report(
    input_dir: Path,
    output_dir: Path,
    title: str = "LM Proxy Experiment",
) -> None:
    """Generate full report from evaluation results.

    Args:
        input_dir: Directory containing evaluation outputs
        output_dir: Output directory for report files
        title: Report title
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evaluation results from {input_dir}...")
    results = load_evaluation_results(input_dir)

    # Check if we have any results
    has_convergence = any(results.get("convergence", {}).values())
    has_variance = any(results.get("variance", {}).values())
    has_trajectories = any(results.get("trajectories", {}).values())

    if not (has_convergence or has_variance):
        print("Warning: No evaluation results found. Creating placeholder report.")
        # Create placeholder summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "status": "no_results",
            "baseline": {},
            "hhc": {},
            "comparison": {},
        }
    else:
        print("Computing aggregated summary...")
        summary = compute_aggregated_summary(results)
        summary["title"] = title

    # Save summary.json
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Generate metrics.jsonl
    if has_convergence or has_variance:
        metrics_path = output_dir / "metrics.jsonl"
        generate_metrics_jsonl(results, metrics_path)
        print(f"Metrics saved to {metrics_path}")

    # Generate loss curves plot
    if has_trajectories:
        curves_path = output_dir / "loss_curves.pdf"
        generate_loss_curves_plot(results, curves_path, f"{title}: Loss Curves")
    else:
        print("No trajectory data available for loss curves plot")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Report: {title}")
    print("=" * 60)

    for config_type in ["baseline", "hhc"]:
        config_data = summary.get(config_type, {})
        if config_data:
            print(f"\n{config_type.upper()}:")
            print(f"  Final loss: {config_data.get('final_loss_mean', 'N/A'):.4f} +/- {config_data.get('final_loss_std', 0):.4f}")
            steps = config_data.get("steps_to_3.5")
            if steps is not None:
                print(f"  Steps to 3.5: {steps:.0f}")
            deg = config_data.get("prompt_extension_degradation")
            if deg is not None:
                print(f"  Extension degradation: {deg:.4f}")
            print(f"  Seeds evaluated: {config_data.get('num_seeds', 0)}")

    comparison = summary.get("comparison", {})
    if comparison:
        print("\nCOMPARISON:")
        print(f"  Loss improvement: {comparison.get('loss_improvement', 0):.4f} ({comparison.get('loss_improvement_pct', 0):.1f}%)")
        print(f"  Variance reduction: {comparison.get('variance_reduction_pct', 0):.1f}%")
        print(f"  Stability improvement: {comparison.get('stability_improvement', 0):.4f}")

    print("\n" + "=" * 60)


def setup_experiment_structure(output_dir: Path) -> None:
    """Create the expected experiment directory structure.

    Creates:
        output_dir/
            baseline/
                seed_42/
                seed_43/
                seed_44/
            hhc/
                seed_42/
                seed_43/
                seed_44/

    Args:
        output_dir: Root output directory
    """
    for config_type in ["baseline", "hhc"]:
        for seed in [42, 43, 44]:
            seed_dir = output_dir / config_type / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created experiment structure in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reports for LM proxy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/results/lm_proxy"),
        help="Input directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (defaults to input directory)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="LM Proxy Experiment",
        help="Report title",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create experiment directory structure only",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    if args.setup:
        setup_experiment_structure(args.output)
    else:
        generate_report(args.input, args.output, args.title)


if __name__ == "__main__":
    main()
