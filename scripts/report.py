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
        "| Tokenizer | Compression | BPB | Lossless |",
        "|-----------|-------------|-----|----------|",
    ]

    results = summary.get("results", {})
    for name, metrics in results.items():
        comp = metrics.get("mean_compression_ratio", 0)
        bpb = metrics.get("mean_bits_per_byte", 0)
        lossless = metrics.get("lossless_rate", 0)
        lines.append(f"| {name} | {comp:.2f} | {bpb:.2f} | {lossless:.1%} |")

    lines.extend([
        "",
        "## Detailed Metrics",
        "",
    ])

    for name, metrics in results.items():
        lines.extend([
            f"### {name}",
            "",
            f"- Total tokens: {metrics.get('total_tokens', 'N/A')}",
            f"- Total bytes: {metrics.get('total_bytes', 'N/A')}",
            f"- Mean compression ratio: {metrics.get('mean_compression_ratio', 0):.4f}",
            f"- Mean bits per byte: {metrics.get('mean_bits_per_byte', 0):.4f}",
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
        "| Metric | Value |",
        "|--------|-------|",
        f"| Compression Ratio | {universal.get('mean_compression_ratio', 0):.2f} |",
        f"| Bits per Byte | {universal.get('mean_bits_per_byte', 0):.2f} |",
        f"| Lossless Rate | {universal.get('lossless_rate', 0):.1%} |",
        f"| Avg Token Length | {universal.get('mean_avg_token_length', 0):.2f} |",
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
