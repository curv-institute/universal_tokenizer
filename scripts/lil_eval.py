#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
# ]
# ///
"""LIL evaluation script - Evaluate churn, stability, curvature.

This script runs the mandatory LIL experiments:
1. Prompt extension stability - core instruction stability under context changes
2. Role boundary robustness - stability at system/user/assistant transitions
3. Interface efficiency - BPB and token count comparisons

Metrics focus on RIFT principles, not LM loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.lil import (
    BoundaryMetrics,
    ChurnMetrics,
    LanguageInterfaceLayer,
    LILEvalMetrics,
    PromptSegment,
    Role,
    SegmentMetrics,
    StructuredPrompt,
    compute_instruction_stability,
    compute_token_churn,
)
from tokenizer.config import load_config
from tokenizer.bundle import UniversalTokenizer
from tokenizer.metrics import compute_metrics


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    name: str
    baseline_metrics: dict
    lil_metrics: dict
    lil_hhc_metrics: dict | None = None


def create_test_prompts() -> list[tuple[str, StructuredPrompt]]:
    """Create a diverse set of test prompts for evaluation."""
    prompts = []

    # Simple single-turn
    prompts.append(("simple_user", StructuredPrompt(segments=[
        PromptSegment(role=Role.USER, content=b"What is the capital of France?"),
    ])))

    # System + User
    prompts.append(("system_user", StructuredPrompt(segments=[
        PromptSegment(role=Role.SYSTEM, content=b"You are a helpful assistant."),
        PromptSegment(role=Role.USER, content=b"Explain quantum computing."),
    ])))

    # Full conversation
    prompts.append(("full_conversation", StructuredPrompt(segments=[
        PromptSegment(role=Role.SYSTEM, content=b"You are a helpful, harmless, and honest assistant."),
        PromptSegment(role=Role.USER, content=b"What is machine learning?"),
        PromptSegment(role=Role.ASSISTANT, content=b"Machine learning is a subset of AI..."),
        PromptSegment(role=Role.USER, content=b"Can you give me an example?"),
    ])))

    # Code-heavy prompt
    prompts.append(("code_prompt", StructuredPrompt(segments=[
        PromptSegment(role=Role.SYSTEM, content=b"You are a Python expert."),
        PromptSegment(role=Role.USER, content=b"```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```\nExplain this code."),
    ])))

    # JSON schema prompt
    prompts.append(("json_schema", StructuredPrompt(segments=[
        PromptSegment(role=Role.SYSTEM, content=b'Please respond with valid JSON matching this schema: {"type": "object", "properties": {"name": {"type": "string"}}}'),
        PromptSegment(role=Role.USER, content=b"Give me info about Paris."),
    ])))

    return prompts


def run_prompt_extension_experiment(
    lil: LanguageInterfaceLayer,
    tokenizer: UniversalTokenizer,
    tokenizer_hhc: UniversalTokenizer | None = None,
) -> ExperimentResult:
    """Experiment 1: Prompt extension stability.

    Measures how stable core instructions are when extended with context.
    """
    print("\n=== Experiment 1: Prompt Extension Stability ===")

    core_instruction = b"Explain the concept of representational stability in machine learning."

    # Extensions of varying sizes
    extensions = [
        b"",  # No extension (baseline)
        b" Please be concise.",
        b" Consider the following context: Neural networks learn representations...",
        b" " + b"Additional context. " * 20,  # ~400 bytes of noise
        b" " + b"More context padding. " * 50,  # ~1000 bytes of noise
    ]

    results_baseline = []
    results_lil = []
    results_lil_hhc = []

    for i, ext in enumerate(tqdm(extensions, desc="Extensions")):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=core_instruction + ext)
        ])

        # Baseline (raw bytes, no LIL)
        raw_bytes = prompt.to_text().encode()
        enc_baseline = tokenizer.encode(raw_bytes)
        core_tokens_baseline = tokenizer.encode(core_instruction).tokens

        # LIL
        lil_result = lil.pack(prompt)
        enc_lil = tokenizer.encode(lil_result.data)

        # LIL + HHC
        if tokenizer_hhc:
            enc_lil_hhc = tokenizer_hhc.encode(lil_result.data)
        else:
            enc_lil_hhc = None

        results_baseline.append({
            "extension_bytes": len(ext),
            "total_bytes": len(raw_bytes),
            "tokens": len(enc_baseline.tokens),
            "core_tokens": len(core_tokens_baseline),
        })

        results_lil.append({
            "extension_bytes": len(ext),
            "total_bytes": len(lil_result.data),
            "tokens": len(enc_lil.tokens),
            "lil_ratio": lil_result.size_ratio,
        })

        if enc_lil_hhc:
            results_lil_hhc.append({
                "extension_bytes": len(ext),
                "tokens": len(enc_lil_hhc.tokens),
            })

    # Compute churn metrics
    baseline_core_tokens = results_baseline[0]["core_tokens"]
    lil_base_tokens = results_lil[0]["tokens"]

    baseline_churn = []
    lil_churn = []

    for i in range(1, len(extensions)):
        # For baseline, we'd need to re-encode core to compare
        # Simplified: compare total token count change
        baseline_churn.append(
            abs(results_baseline[i]["tokens"] - results_baseline[0]["tokens"]) /
            max(results_baseline[0]["tokens"], 1)
        )
        lil_churn.append(
            abs(results_lil[i]["tokens"] - results_lil[0]["tokens"]) /
            max(results_lil[0]["tokens"], 1)
        )

    return ExperimentResult(
        name="prompt_extension",
        baseline_metrics={
            "results": results_baseline,
            "avg_token_churn": float(np.mean(baseline_churn)) if baseline_churn else 0,
            "max_token_churn": float(np.max(baseline_churn)) if baseline_churn else 0,
        },
        lil_metrics={
            "results": results_lil,
            "avg_token_churn": float(np.mean(lil_churn)) if lil_churn else 0,
            "max_token_churn": float(np.max(lil_churn)) if lil_churn else 0,
            "avg_lil_ratio": float(np.mean([r["lil_ratio"] for r in results_lil])),
        },
        lil_hhc_metrics={
            "results": results_lil_hhc,
        } if results_lil_hhc else None,
    )


def run_role_boundary_experiment(
    lil: LanguageInterfaceLayer,
    tokenizer: UniversalTokenizer,
    tokenizer_hhc: UniversalTokenizer | None = None,
) -> ExperimentResult:
    """Experiment 2: Role boundary robustness.

    Measures stability at system/user/assistant transitions.
    """
    print("\n=== Experiment 2: Role Boundary Robustness ===")

    # Create prompts with varying number of role transitions
    test_cases = [
        # 1 boundary
        StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are helpful."),
            PromptSegment(role=Role.USER, content=b"Hello."),
        ]),
        # 2 boundaries
        StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are helpful."),
            PromptSegment(role=Role.USER, content=b"Hello."),
            PromptSegment(role=Role.ASSISTANT, content=b"Hi there!"),
        ]),
        # 4 boundaries (multi-turn)
        StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are helpful."),
            PromptSegment(role=Role.USER, content=b"Hello."),
            PromptSegment(role=Role.ASSISTANT, content=b"Hi there!"),
            PromptSegment(role=Role.USER, content=b"How are you?"),
            PromptSegment(role=Role.ASSISTANT, content=b"I'm doing well."),
        ]),
    ]

    results_baseline = []
    results_lil = []
    results_lil_hhc = []

    for i, prompt in enumerate(tqdm(test_cases, desc="Boundary tests")):
        num_boundaries = len(prompt.segments) - 1

        # Baseline
        raw_bytes = prompt.to_text().encode()
        enc_baseline = tokenizer.encode(raw_bytes)

        # LIL
        lil_result = lil.pack(prompt)
        enc_lil = tokenizer.encode(lil_result.data)

        # Verify reversibility
        unpacked = lil.unpack(lil_result.data)
        lossless = unpacked.to_text() == prompt.to_text()

        results_baseline.append({
            "num_segments": len(prompt.segments),
            "num_boundaries": num_boundaries,
            "bytes": len(raw_bytes),
            "tokens": len(enc_baseline.tokens),
            "bytes_per_token": len(raw_bytes) / max(len(enc_baseline.tokens), 1),
        })

        results_lil.append({
            "num_segments": len(prompt.segments),
            "num_boundaries": num_boundaries,
            "bytes": len(lil_result.data),
            "tokens": len(enc_lil.tokens),
            "bytes_per_token": len(lil_result.data) / max(len(enc_lil.tokens), 1),
            "lil_ratio": lil_result.size_ratio,
            "lossless": lossless,
        })

        # LIL + HHC
        if tokenizer_hhc:
            enc_hhc = tokenizer_hhc.encode(lil_result.data)
            results_lil_hhc.append({
                "num_boundaries": num_boundaries,
                "tokens": len(enc_hhc.tokens),
            })

    return ExperimentResult(
        name="role_boundary",
        baseline_metrics={
            "results": results_baseline,
            "avg_bytes_per_token": float(np.mean([r["bytes_per_token"] for r in results_baseline])),
        },
        lil_metrics={
            "results": results_lil,
            "avg_bytes_per_token": float(np.mean([r["bytes_per_token"] for r in results_lil])),
            "all_lossless": all(r["lossless"] for r in results_lil),
        },
        lil_hhc_metrics={
            "results": results_lil_hhc,
        } if results_lil_hhc else None,
    )


def run_efficiency_experiment(
    lil: LanguageInterfaceLayer,
    tokenizer: UniversalTokenizer,
    tokenizer_hhc: UniversalTokenizer | None = None,
) -> ExperimentResult:
    """Experiment 3: Interface efficiency.

    Reports E2E BPB, structural BPB, and token count deltas.
    """
    print("\n=== Experiment 3: Interface Efficiency ===")

    test_prompts = create_test_prompts()

    results_baseline = []
    results_lil = []
    results_lil_hhc = []

    for name, prompt in tqdm(test_prompts, desc="Efficiency tests"):
        # Baseline
        raw_bytes = prompt.to_text().encode()
        enc_baseline = tokenizer.encode(raw_bytes)
        dec_baseline = tokenizer.decode(enc_baseline)

        baseline_metrics = compute_metrics(
            enc_baseline, raw_bytes,
            vocab_size=tokenizer.config.codebook.num_codes
        )

        # LIL
        lil_result = lil.pack(prompt)
        enc_lil = tokenizer.encode(lil_result.data)
        dec_lil = tokenizer.decode(enc_lil)

        lil_metrics = compute_metrics(
            enc_lil, lil_result.data,
            vocab_size=tokenizer.config.codebook.num_codes
        )

        # Verify full round-trip
        if dec_lil.data == lil_result.data:
            final_prompt = lil.unpack(dec_lil.data)
            full_lossless = final_prompt.to_text() == prompt.to_text()
        else:
            full_lossless = False

        results_baseline.append({
            "name": name,
            "bytes": len(raw_bytes),
            "tokens": len(enc_baseline.tokens),
            "e2e_bpb": baseline_metrics.end_to_end_bpb,
            "structural_bpb": baseline_metrics.structural_bpb,
            "lossless": dec_baseline.data == raw_bytes,
        })

        results_lil.append({
            "name": name,
            "original_bytes": lil_result.original_size,
            "packed_bytes": len(lil_result.data),
            "tokens": len(enc_lil.tokens),
            "e2e_bpb": lil_metrics.end_to_end_bpb,
            "structural_bpb": lil_metrics.structural_bpb,
            "lil_ratio": lil_result.size_ratio,
            "full_lossless": full_lossless,
        })

        # LIL + HHC
        if tokenizer_hhc:
            enc_hhc = tokenizer_hhc.encode(lil_result.data)
            hhc_metrics = compute_metrics(
                enc_hhc, lil_result.data,
                vocab_size=tokenizer_hhc.config.codebook.num_codes
            )
            results_lil_hhc.append({
                "name": name,
                "tokens": len(enc_hhc.tokens),
                "e2e_bpb": hhc_metrics.end_to_end_bpb,
                "structural_bpb": hhc_metrics.structural_bpb,
            })

    # Compute averages
    def avg(lst, key):
        return float(np.mean([r[key] for r in lst])) if lst else 0

    return ExperimentResult(
        name="efficiency",
        baseline_metrics={
            "results": results_baseline,
            "avg_e2e_bpb": avg(results_baseline, "e2e_bpb"),
            "avg_structural_bpb": avg(results_baseline, "structural_bpb"),
            "avg_tokens": avg(results_baseline, "tokens"),
            "lossless_rate": sum(1 for r in results_baseline if r["lossless"]) / len(results_baseline),
        },
        lil_metrics={
            "results": results_lil,
            "avg_e2e_bpb": avg(results_lil, "e2e_bpb"),
            "avg_structural_bpb": avg(results_lil, "structural_bpb"),
            "avg_tokens": avg(results_lil, "tokens"),
            "avg_lil_ratio": avg(results_lil, "lil_ratio"),
            "full_lossless_rate": sum(1 for r in results_lil if r["full_lossless"]) / len(results_lil),
        },
        lil_hhc_metrics={
            "results": results_lil_hhc,
            "avg_e2e_bpb": avg(results_lil_hhc, "e2e_bpb"),
            "avg_structural_bpb": avg(results_lil_hhc, "structural_bpb"),
            "avg_tokens": avg(results_lil_hhc, "tokens"),
        } if results_lil_hhc else None,
    )


def main():
    parser = argparse.ArgumentParser(description="LIL evaluation experiments")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/cpu_small.toml"),
                        help="Baseline tokenizer config")
    parser.add_argument("--config-hhc", type=Path, default=Path("configs/cpu_small_hhc.toml"),
                        help="HHC tokenizer config (optional)")
    parser.add_argument("--output", "-o", type=Path, default=Path("eval/results/lil"),
                        help="Output directory")
    parser.add_argument("--no-macros", action="store_true",
                        help="Disable macro compaction in LIL")

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize LIL
    lil = LanguageInterfaceLayer(enable_macros=not args.no_macros)
    print(f"LIL initialized (macros: {not args.no_macros})")
    print(f"  Registered macros: {len(lil.registry)}")

    # Initialize tokenizers
    if args.config.exists():
        config = load_config(args.config)
        tokenizer = UniversalTokenizer(config)
        print(f"Loaded baseline tokenizer: {args.config}")
    else:
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    tokenizer_hhc = None
    if args.config_hhc.exists():
        config_hhc = load_config(args.config_hhc)
        tokenizer_hhc = UniversalTokenizer(config_hhc)
        print(f"Loaded HHC tokenizer: {args.config_hhc}")

    # Run experiments
    experiments = []

    exp1 = run_prompt_extension_experiment(lil, tokenizer, tokenizer_hhc)
    experiments.append(exp1)
    print(f"\nExp 1 - Baseline avg churn: {exp1.baseline_metrics['avg_token_churn']:.4f}")
    print(f"Exp 1 - LIL avg churn: {exp1.lil_metrics['avg_token_churn']:.4f}")

    exp2 = run_role_boundary_experiment(lil, tokenizer, tokenizer_hhc)
    experiments.append(exp2)
    print(f"\nExp 2 - Baseline avg bytes/token: {exp2.baseline_metrics['avg_bytes_per_token']:.2f}")
    print(f"Exp 2 - LIL avg bytes/token: {exp2.lil_metrics['avg_bytes_per_token']:.2f}")
    print(f"Exp 2 - All lossless: {exp2.lil_metrics['all_lossless']}")

    exp3 = run_efficiency_experiment(lil, tokenizer, tokenizer_hhc)
    experiments.append(exp3)
    print(f"\nExp 3 - Baseline E2E BPB: {exp3.baseline_metrics['avg_e2e_bpb']:.2f}")
    print(f"Exp 3 - LIL E2E BPB: {exp3.lil_metrics['avg_e2e_bpb']:.2f}")
    print(f"Exp 3 - Full lossless rate: {exp3.lil_metrics['full_lossless_rate']:.0%}")

    # Save results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": str(args.config),
        "config_hhc": str(args.config_hhc) if tokenizer_hhc else None,
        "macros_enabled": not args.no_macros,
        "num_macros": len(lil.registry),
        "experiments": {
            exp.name: {
                "baseline": exp.baseline_metrics,
                "lil": exp.lil_metrics,
                "lil_hhc": exp.lil_hhc_metrics,
            }
            for exp in experiments
        },
    }

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed metrics
    with open(args.output / "metrics.jsonl", "w") as f:
        for exp in experiments:
            f.write(json.dumps({"experiment": exp.name, "type": "baseline", **exp.baseline_metrics}) + "\n")
            f.write(json.dumps({"experiment": exp.name, "type": "lil", **exp.lil_metrics}) + "\n")
            if exp.lil_hhc_metrics:
                f.write(json.dumps({"experiment": exp.name, "type": "lil_hhc", **exp.lil_hhc_metrics}) + "\n")

    print(f"\n=== Results saved to {args.output} ===")

    # Generate report
    def fmt(metrics, key, fmt_spec=".2f"):
        """Format metric with fallback to N/A."""
        if metrics is None:
            return "N/A"
        val = metrics.get(key)
        if val is None:
            return "N/A"
        return f"{val:{fmt_spec}}"

    report_lines = [
        "# LIL Evaluation Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Config:** {args.config}",
        f"**Macros:** {'Enabled' if not args.no_macros else 'Disabled'} ({len(lil.registry)} registered)",
        "",
        "## Summary",
        "",
        "| Metric | Baseline | LIL | LIL+HHC |",
        "|--------|----------|-----|---------|",
        f"| E2E BPB | {exp3.baseline_metrics['avg_e2e_bpb']:.2f} | {exp3.lil_metrics['avg_e2e_bpb']:.2f} | {fmt(exp3.lil_hhc_metrics, 'avg_e2e_bpb')} |",
        f"| Structural BPB | {exp3.baseline_metrics['avg_structural_bpb']:.2f} | {exp3.lil_metrics['avg_structural_bpb']:.2f} | {fmt(exp3.lil_hhc_metrics, 'avg_structural_bpb')} |",
        f"| Avg Tokens | {exp3.baseline_metrics['avg_tokens']:.1f} | {exp3.lil_metrics['avg_tokens']:.1f} | {fmt(exp3.lil_hhc_metrics, 'avg_tokens', '.1f')} |",
        f"| Lossless Rate | {exp3.baseline_metrics['lossless_rate']:.0%} | {exp3.lil_metrics['full_lossless_rate']:.0%} | N/A |",
        "",
        "## Experiment 1: Prompt Extension Stability",
        "",
        f"- **Baseline avg token churn:** {exp1.baseline_metrics['avg_token_churn']:.4f}",
        f"- **LIL avg token churn:** {exp1.lil_metrics['avg_token_churn']:.4f}",
        f"- **Improvement:** {(1 - exp1.lil_metrics['avg_token_churn']/max(exp1.baseline_metrics['avg_token_churn'], 0.001)):.1%}",
        "",
        "## Experiment 2: Role Boundary Robustness",
        "",
        f"- **All LIL transforms lossless:** {exp2.lil_metrics['all_lossless']}",
        f"- **LIL ratio:** {exp3.lil_metrics['avg_lil_ratio']:.2%}",
        "",
        "## Experiment 3: Interface Efficiency",
        "",
        "Trade-off analysis: LIL may increase token count to achieve structural benefits.",
        "",
    ]

    with open(args.output / "report.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
