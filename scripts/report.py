#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "pyyaml",
# ]
# ///
"""Generate evaluation reports with multi-run aggregation support."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_report(results_dir: Path, output_dir: Path | None = None) -> None:
    """Generate markdown report from evaluation results.

    Args:
        results_dir: Directory with evaluation results
        output_dir: Output directory (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir

    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json found in {results_dir}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Generate report
    report = generate_markdown_report(summary)

    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Generate paper artifact
    artifact = generate_paper_artifact(summary)
    artifact_path = output_dir / "paper_artifact.md"
    with open(artifact_path, "w") as f:
        f.write(artifact)
    print(f"Paper artifact saved to {artifact_path}")


def generate_markdown_report(summary: dict) -> str:
    """Generate markdown report."""
    lines = [
        "# Universal Lossless Tokenizer - Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        "",
        f"- Config: `{summary.get('config', 'N/A')}`",
        f"- Model: `{summary.get('model', 'N/A')}`",
        f"- Samples: {summary.get('num_samples', 'N/A')}",
        f"- Data entropy: {summary.get('data_entropy', 0):.3f} bits/byte",
        "",
        "## Results",
        "",
        "| Tokenizer | Lossless BPB* | Struct BPB | Compression | Lossless |",
        "|-----------|---------------|------------|-------------|----------|",
    ]

    results = summary.get("results", {})
    for name, metrics in results.items():
        # Primary metric: end-to-end BPB (lossless compression)
        e2e_bpb = metrics.get("mean_end_to_end_bpb", metrics.get("mean_bits_per_byte", 0))
        # Secondary metric: structural BPB (token representation only)
        struct_bpb = metrics.get("mean_structural_bpb", metrics.get("mean_bits_per_byte", 0))
        comp = metrics.get("mean_compression_ratio", 0)
        lossless = metrics.get("lossless_rate", 0)
        lines.append(f"| {name} | {e2e_bpb:.2f} | {struct_bpb:.2f} | {comp:.2f} | {lossless:.1%} |")

    lines.extend([
        "",
        "*\\*Primary compression metric*",
        "",
        "**Metric Definitions:** End-to-end BPB (Lossless BPB) measures the true lossless "
        "encoding cost including token IDs and residual bytes. Structural BPB measures "
        "representational efficiency prior to residual correction and should not be "
        "interpreted as a standalone compression ratio. All compression claims in this "
        "work are based on end-to-end lossless BPB.",
        "",
        "## Detailed Metrics",
        "",
    ])

    for name, metrics in results.items():
        # Get both BPB metrics with fallback for backward compatibility
        e2e_bpb = metrics.get("mean_end_to_end_bpb", metrics.get("mean_bits_per_byte", 0))
        struct_bpb = metrics.get("mean_structural_bpb", metrics.get("mean_bits_per_byte", 0))
        lines.extend([
            f"### {name}",
            "",
            f"- Total tokens: {metrics.get('total_tokens', 'N/A')}",
            f"- Total bytes: {metrics.get('total_bytes', 'N/A')}",
            f"- **Lossless BPB (E2E):** {e2e_bpb:.4f}",
            f"- Structural BPB: {struct_bpb:.4f}",
            f"- Total residual bytes: {metrics.get('total_residual_bytes', 'N/A')}",
            f"- Mean compression ratio: {metrics.get('mean_compression_ratio', 0):.4f}",
            f"- Mean avg token length: {metrics.get('mean_avg_token_length', 0):.4f}",
            f"- Mean curvature: {metrics.get('mean_curvature', 0):.4f}",
            f"- Curvature P90: {metrics.get('curvature_p90', 0):.4f}",
            f"- Mean stability: {metrics.get('mean_stability', 0):.4f}",
            f"- Stability P10: {metrics.get('stability_p10', 0):.4f}",
            f"- Lossless rate: {metrics.get('lossless_rate', 0):.2%}",
            "",
        ])

    # Add HHC diagnostics section if present
    hhc_diag = summary.get("hhc_diagnostics")
    if hhc_diag:
        hhc_enabled = summary.get("hhc_enabled", False)
        lines.extend([
            "## HHC Diagnostics",
            "",
            f"- **HHC Active:** {'Yes' if hhc_enabled else 'No'}",
            f"- Active fraction: {hhc_diag.get('hhc_active_fraction', 0) * 100:.1f}%",
            f"- Mean neighbors: {hhc_diag.get('hhc_mean_num_neighbors', 0):.1f}",
            f"- Mean neighbor distance: {hhc_diag.get('hhc_mean_neighbor_dist', 0):.3f}",
            f"- Neighbor distance variance: {hhc_diag.get('hhc_var_neighbor_dist', 0):.3f}",
            f"- Mean curvature delta: {hhc_diag.get('hhc_mean_curvature_delta', 0):.3f}",
            f"- Nonzero delta fraction: {hhc_diag.get('hhc_nonzero_delta_fraction', 0) * 100:.1f}%",
            "",
        ])

        # Add warnings if thresholds not met
        active_frac = hhc_diag.get("hhc_active_fraction", 0)
        nonzero_frac = hhc_diag.get("hhc_nonzero_delta_fraction", 0)
        if hhc_enabled and active_frac < 0.95:
            lines.extend([
                f"> **Warning:** HHC active fraction ({active_frac * 100:.1f}%) is below 95% threshold",
                "",
            ])
        if hhc_enabled and nonzero_frac < 0.20:
            lines.extend([
                f"> **Warning:** Nonzero delta fraction ({nonzero_frac * 100:.1f}%) is below 20% threshold - HHC may be producing degenerate results",
                "",
            ])

    return "\n".join(lines)


def infer_config_type(config_path: str) -> str:
    """Infer configuration type (baseline or hhc) from config path.

    Args:
        config_path: Path to the config file

    Returns:
        'hhc' if config is HHC-enabled, 'baseline' otherwise
    """
    config_lower = config_path.lower()
    if "_hhc" in config_lower or "/hhc" in config_lower:
        return "hhc"
    return "baseline"


def infer_dataset_tier(results_dir: Path) -> str:
    """Infer dataset tier from results directory path.

    Attempts to extract tier information from directory name.
    Common patterns: tier1, tier2, small, medium, large, etc.

    Args:
        results_dir: Path to results directory

    Returns:
        Inferred tier name or directory name as fallback
    """
    dir_name = results_dir.name.lower()

    # Check for explicit tier naming
    tier_match = re.search(r"tier[_-]?(\d+)", dir_name)
    if tier_match:
        return f"tier{tier_match.group(1)}"

    # Check for size-based naming
    for size in ["small", "medium", "large", "tiny", "huge"]:
        if size in dir_name:
            return size

    # Check for dataset names
    for dataset in ["text", "code", "binary", "mixed", "random"]:
        if dataset in dir_name:
            return dataset

    # Fallback to directory name (strip common prefixes)
    clean_name = dir_name
    for prefix in ["eval_", "results_", "run_"]:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix) :]
    return clean_name or "default"


def aggregate_matrix_results(results_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate metrics from multiple result directories for matrix comparison.

    Reads summary.json from each directory and groups results by:
    - Configuration type (baseline vs HHC)
    - Dataset tier (inferred from directory name)

    Args:
        results_dirs: List of paths to result directories containing summary.json

    Returns:
        Dictionary with structure:
        {
            'tiers': ['tier1', 'tier2', ...],
            'configs': ['baseline', 'hhc'],
            'data': {
                'tier1': {
                    'baseline': {'metrics': {...}, 'runs': [...]},
                    'hhc': {'metrics': {...}, 'runs': [...]}
                },
                ...
            },
            'metadata': {
                'total_runs': N,
                'aggregation_timestamp': '...'
            }
        }
    """
    aggregated: dict[str, Any] = {
        "tiers": set(),
        "configs": set(),
        "data": defaultdict(lambda: defaultdict(lambda: {"runs": [], "metrics": {}})),
        "metadata": {"total_runs": 0, "aggregation_timestamp": datetime.now().isoformat()},
    }

    for results_dir in results_dirs:
        summary_path = results_dir / "summary.json"
        if not summary_path.exists():
            print(f"Warning: No summary.json found in {results_dir}, skipping")
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        config_type = infer_config_type(summary.get("config", ""))
        tier = infer_dataset_tier(results_dir)

        aggregated["tiers"].add(tier)
        aggregated["configs"].add(config_type)
        aggregated["data"][tier][config_type]["runs"].append(summary)
        aggregated["metadata"]["total_runs"] += 1

    # Convert sets to sorted lists
    aggregated["tiers"] = sorted(aggregated["tiers"])
    aggregated["configs"] = sorted(aggregated["configs"])

    # Aggregate metrics across runs for each tier/config combination
    for tier in aggregated["tiers"]:
        for config_type in aggregated["configs"]:
            runs = aggregated["data"][tier][config_type]["runs"]
            if not runs:
                continue

            # Aggregate metrics for the 'universal' tokenizer (primary)
            metrics_by_key: dict[str, list[float]] = defaultdict(list)
            for run in runs:
                universal_results = run.get("results", {}).get("universal", {})
                for key, value in universal_results.items():
                    if isinstance(value, (int, float)):
                        metrics_by_key[key].append(float(value))

                # Handle backward compatibility: map mean_bits_per_byte to new metric names
                if "mean_bits_per_byte" in universal_results:
                    old_bpb = float(universal_results["mean_bits_per_byte"])
                    if "mean_end_to_end_bpb" not in universal_results:
                        metrics_by_key["mean_end_to_end_bpb"].append(old_bpb)
                    if "mean_structural_bpb" not in universal_results:
                        metrics_by_key["mean_structural_bpb"].append(old_bpb)

            # Compute aggregated statistics
            agg_metrics: dict[str, Any] = {}
            for key, values in metrics_by_key.items():
                if len(values) == 1:
                    agg_metrics[key] = {"mean": values[0], "std": 0.0, "n": 1}
                else:
                    agg_metrics[key] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values),
                        "n": len(values),
                    }

            aggregated["data"][tier][config_type]["metrics"] = agg_metrics

    # Convert defaultdicts to regular dicts for JSON serialization
    aggregated["data"] = {
        tier: {config: dict(data) for config, data in configs.items()}
        for tier, configs in aggregated["data"].items()
    }

    return aggregated


def generate_matrix_report(aggregated: dict[str, Any]) -> tuple[str, str]:
    """Generate comparison tables from aggregated matrix results.

    Creates both markdown and LaTeX tables showing:
    - Rows: dataset tiers
    - Columns: baseline vs HHC metrics (E2E BPB, Structural BPB, curvature P90, lossless rate, throughput)

    Args:
        aggregated: Aggregated results from aggregate_matrix_results()

    Returns:
        Tuple of (markdown_report, latex_report)
    """
    tiers = aggregated.get("tiers", [])
    configs = aggregated.get("configs", [])
    data = aggregated.get("data", {})
    metadata = aggregated.get("metadata", {})

    # Key metrics to display
    metric_keys = [
        ("mean_end_to_end_bpb", "E2E BPB", "%.3f"),
        ("mean_structural_bpb", "Struct BPB", "%.3f"),
        ("curvature_p90", "Curv P90", "%.3f"),
        ("lossless_rate", "Lossless", "%.1%%"),
        ("mean_compression_ratio", "Compress", "%.2f"),
    ]

    # Generate Markdown report
    md_lines = [
        "# Evaluation Matrix Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Total runs aggregated: {metadata.get('total_runs', 0)}",
        "",
        "## Comparison Table",
        "",
    ]

    # Build header
    header_parts = ["| Tier |"]
    separator_parts = ["|------|"]
    for config in configs:
        for _, metric_name, _ in metric_keys:
            header_parts.append(f" {config.upper()} {metric_name} |")
            separator_parts.append("---:|")

    md_lines.append("".join(header_parts))
    md_lines.append("".join(separator_parts))

    # Build rows
    for tier in tiers:
        row_parts = [f"| {tier} |"]
        for config in configs:
            tier_config_data = data.get(tier, {}).get(config, {})
            metrics = tier_config_data.get("metrics", {})

            for key, _, fmt in metric_keys:
                metric_data = metrics.get(key, {})
                mean_val = metric_data.get("mean", 0)
                std_val = metric_data.get("std", 0)

                if fmt.endswith("%%"):
                    # Percentage format - multiply by 100 and add % symbol
                    numeric_fmt = fmt.replace("%%", "f")
                    formatted = (numeric_fmt % (mean_val * 100)) + "%"
                else:
                    formatted = fmt % mean_val

                if std_val > 0.001:
                    row_parts.append(f" {formatted} (+-{std_val:.3f}) |")
                else:
                    row_parts.append(f" {formatted} |")

        md_lines.append("".join(row_parts))

    md_lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- **E2E BPB**: End-to-end bits per byte (lossless compression, primary metric)",
            "- **Struct BPB**: Structural bits per byte (token representation only)",
            "- **Curv P90**: 90th percentile curvature (boundary quality)",
            "- **Lossless**: Exact reconstruction rate",
            "- **Compress**: Compression ratio (bytes/token)",
            "",
        ]
    )

    # Detailed per-tier breakdown
    md_lines.extend(["## Detailed Results", ""])
    for tier in tiers:
        md_lines.extend([f"### {tier.title()}", ""])
        for config in configs:
            tier_config_data = data.get(tier, {}).get(config, {})
            metrics = tier_config_data.get("metrics", {})
            runs = tier_config_data.get("runs", [])

            md_lines.append(f"**{config.upper()}** ({len(runs)} runs)")
            md_lines.append("")

            for key, metric_name, fmt in metric_keys:
                metric_data = metrics.get(key, {})
                mean_val = metric_data.get("mean", 0)
                std_val = metric_data.get("std", 0)
                n = metric_data.get("n", 0)

                if fmt.endswith("%%"):
                    numeric_fmt = fmt.replace("%%", "f")
                    formatted = (numeric_fmt % (mean_val * 100)) + "%"
                else:
                    formatted = fmt % mean_val

                md_lines.append(f"- {metric_name}: {formatted} (std: {std_val:.4f}, n={n})")

            md_lines.append("")

    markdown_report = "\n".join(md_lines)

    # Generate LaTeX report
    latex_lines = [
        "% Evaluation Matrix - Paper-ready LaTeX table",
        f"% Generated: {datetime.now().isoformat()}",
        f"% Total runs aggregated: {metadata.get('total_runs', 0)}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Universal Tokenizer Evaluation Results}",
        "\\label{tab:eval-matrix}",
    ]

    # Calculate column spec
    num_metric_cols = len(metric_keys) * len(configs)
    col_spec = "l" + "r" * num_metric_cols
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    # Header row 1: config names spanning metrics
    header1_parts = [""]
    for config in configs:
        header1_parts.append(f"\\multicolumn{{{len(metric_keys)}}}{{c}}{{{config.upper()}}}")
    latex_lines.append(" & ".join(header1_parts) + " \\\\")

    # Add cmidrule for each config group
    cmidrule_parts = []
    col_start = 2
    for _config in configs:
        col_end = col_start + len(metric_keys) - 1
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
    latex_lines.append(" ".join(cmidrule_parts))

    # Header row 2: metric names
    header2_parts = ["Tier"]
    for _config in configs:
        for _, metric_name, _ in metric_keys:
            header2_parts.append(metric_name)
    latex_lines.append(" & ".join(header2_parts) + " \\\\")
    latex_lines.append("\\midrule")

    # Data rows
    for tier in tiers:
        row_parts = [tier.replace("_", "\\_")]
        for config in configs:
            tier_config_data = data.get(tier, {}).get(config, {})
            metrics = tier_config_data.get("metrics", {})

            for key, _, fmt in metric_keys:
                metric_data = metrics.get(key, {})
                mean_val = metric_data.get("mean", 0)

                if fmt.endswith("%%"):
                    numeric_fmt = fmt.replace("%%", "f")
                    formatted = (numeric_fmt % (mean_val * 100)) + "\\%"
                else:
                    formatted = fmt % mean_val

                row_parts.append(formatted)

        latex_lines.append(" & ".join(row_parts) + " \\\\")

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{1em}",
            "\\begin{minipage}{0.9\\textwidth}",
            "\\footnotesize",
            "\\textbf{Metrics:} E2E BPB = end-to-end bits per byte (primary); "
            "Struct BPB = structural bits per byte; "
            "Curv P90 = 90th percentile curvature; "
            "Lossless = exact reconstruction rate; "
            "Compress = compression ratio (bytes/token).",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )

    latex_report = "\n".join(latex_lines)

    return markdown_report, latex_report


def generate_paper_artifact(summary: dict) -> str:
    """Generate paper-ready artifact snippet."""
    results = summary.get("results", {})
    universal = results.get("universal", {})

    # Primary metric: end-to-end BPB (with fallback for backward compatibility)
    e2e_bpb = universal.get("mean_end_to_end_bpb", universal.get("mean_bits_per_byte", 0))
    struct_bpb = universal.get("mean_structural_bpb", universal.get("mean_bits_per_byte", 0))

    lines = [
        "# Paper Artifact - Universal Lossless Tokenizer",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@software{miller2025universal,",
        "  author = {Miller, J. W.},",
        "  title = {Universal Lossless Tokenizer},",
        "  year = {2025},",
        "  url = {https://github.com/curv-institute/universal_tokenizer}",
        "}",
        "```",
        "",
        "## Key Results",
        "",
        "| Metric | Value | Description |",
        "|--------|-------|-------------|",
        f"| **Lossless BPB** | {e2e_bpb:.2f} | End-to-end compression (primary) |",
        f"| Structural BPB | {struct_bpb:.2f} | Token representation only |",
        f"| Compression Ratio | {universal.get('mean_compression_ratio', 0):.2f} | Bytes per token |",
        f"| Lossless Rate | {universal.get('lossless_rate', 0):.1%} | Exact reconstruction |",
        f"| Avg Token Length | {universal.get('mean_avg_token_length', 0):.2f} | Mean bytes per token |",
        "",
        "**Note:** Lossless BPB is the primary compression metric and includes both token IDs "
        "and residual bytes required for exact reconstruction. Structural BPB measures token "
        "representation efficiency only and should not be cited as a compression ratio.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "git clone https://github.com/curv-institute/universal_tokenizer",
        "cd universal_tokenizer",
        "uv venv .venv && source .venv/bin/activate",
        "uv run scripts/run_experiment.py --config configs/default.toml --name paper_v0_1",
        "```",
        "",
        "## Environment",
        "",
        f"- Timestamp: {summary.get('timestamp', 'N/A')}",
        f"- Data entropy: {summary.get('data_entropy', 0):.3f} bits/byte",
        f"- Samples evaluated: {summary.get('num_samples', 'N/A')}",
        "",
    ]

    return "\n".join(lines)


def generate_matrix_artifacts(
    results_dirs: list[Path], output_dir: Path | None = None
) -> None:
    """Generate matrix comparison reports from multiple result directories.

    Args:
        results_dirs: List of directories containing summary.json files
        output_dir: Output directory (defaults to first results_dir parent)
    """
    if not results_dirs:
        print("Error: No results directories provided")
        return

    if output_dir is None:
        output_dir = results_dirs[0].parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating results from {len(results_dirs)} directories...")
    for d in results_dirs:
        print(f"  - {d}")

    aggregated = aggregate_matrix_results(results_dirs)

    if aggregated["metadata"]["total_runs"] == 0:
        print("Error: No valid summary.json files found")
        return

    print(f"Found {aggregated['metadata']['total_runs']} runs across:")
    print(f"  Tiers: {', '.join(aggregated['tiers'])}")
    print(f"  Configs: {', '.join(aggregated['configs'])}")

    # Generate reports
    markdown_report, latex_report = generate_matrix_report(aggregated)

    # Save markdown report
    md_path = output_dir / "matrix_report.md"
    with open(md_path, "w") as f:
        f.write(markdown_report)
    print(f"Markdown report saved to {md_path}")

    # Save LaTeX report
    tex_path = output_dir / "matrix_report.tex"
    with open(tex_path, "w") as f:
        f.write(latex_report)
    print(f"LaTeX report saved to {tex_path}")

    # Save raw aggregated data
    json_path = output_dir / "matrix_aggregated.json"
    with open(json_path, "w") as f:
        # Remove 'runs' from output to keep file size reasonable
        output_data = {
            "tiers": aggregated["tiers"],
            "configs": aggregated["configs"],
            "data": {
                tier: {
                    config: {"metrics": data["metrics"], "num_runs": len(data["runs"])}
                    for config, data in configs.items()
                }
                for tier, configs in aggregated["data"].items()
            },
            "metadata": aggregated["metadata"],
        }
        json.dump(output_data, f, indent=2)
    print(f"Aggregated data saved to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory mode (default)
  uv run scripts/report.py --results eval/results/run1

  # Matrix aggregation mode
  uv run scripts/report.py --matrix --results-dirs eval/results/baseline eval/results/hhc

  # Matrix mode with custom output directory
  uv run scripts/report.py --matrix --results-dirs eval/results/* --output reports/
        """,
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval/results"),
        help="Results directory (single-directory mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Enable matrix aggregation mode for comparing multiple runs",
    )
    parser.add_argument(
        "--results-dirs",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="Result directories to aggregate (matrix mode)",
    )

    args = parser.parse_args()

    if args.matrix:
        # Matrix aggregation mode
        if not args.results_dirs:
            parser.error("--matrix requires --results-dirs with at least one directory")
        generate_matrix_artifacts(args.results_dirs, args.output)
    else:
        # Single directory mode (backward compatible)
        generate_report(args.results, args.output)


if __name__ == "__main__":
    main()
