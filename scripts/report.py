#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "pyyaml",
# ]
# ///
"""Generate evaluation reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


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

    return "\n".join(lines)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval/results"),
        help="Results directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()
    generate_report(args.results, args.output)


if __name__ == "__main__":
    main()
