# Artifact Checklist

This document describes how to reproduce all results in the paper "Testing the Platonic Representation Hypothesis via Representation-Controlled Tokenization."

## System Requirements

- **Python**: 3.12+
- **OS**: Linux, macOS, or Windows with WSL
- **Hardware**: CPU-only (GPU optional, not required)
- **Memory**: 8GB RAM minimum
- **Disk**: 1GB free space

## Dependencies

All dependencies are managed via [uv](https://github.com/astral-sh/uv) and PEP 723 inline script metadata. No manual installation required.

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Deterministic Data Generation

Evaluation data is generated deterministically from fixed seeds:

```bash
# Generate core evaluation data (required for all experiments)
uv run scripts/make_eval_data.py --profile core --out data/eval

# Generate domain-shift stress data (for HHC experiments)
uv run scripts/make_eval_data.py --profile shift --out data/shift
```

Data generation is fully deterministic; the same seeds produce identical outputs.

## Reproducing Paper Results

### 1. Run Core Experiment

```bash
uv run scripts/run_experiment.py \
  --config configs/cpu_small.toml \
  --name paper_core \
  --eval data/eval
```

Outputs land in `eval/results/paper_core/`:
- `summary.json` — aggregate metrics
- `metrics.jsonl` — per-file results
- `manifests/` — reproducibility manifests

### 2. Run HHC Stability Experiment

```bash
# Baseline
uv run scripts/run_experiment.py \
  --config configs/cpu_small.toml \
  --name baseline_shift \
  --eval data/shift

# HHC-enabled
uv run scripts/run_experiment.py \
  --config configs/cpu_small_hhc.toml \
  --name hhc_shift \
  --eval data/shift
```

### 3. Run LIL Evaluation

```bash
uv run scripts/lil_eval.py --output eval/results/lil
```

### 4. Reproduce from Manifest

```bash
# Reproduce and compare (must match within tolerance)
uv run scripts/reproduce.py \
  --manifest "eval/results/paper_core/manifests/*.json" \
  --compare
```

## Output Structure

Each experiment produces an isolated run directory:

```
eval/results/<name>/
  summary.json       # Aggregate metrics
  metrics.jsonl      # Per-file detailed results
  report.md          # Human-readable report (if generated)
  manifests/         # Run manifests for reproduction
    <name>_<timestamp>.json
```

## Claims Supported

| Claim | Section | Verification |
|-------|---------|--------------|
| 100% lossless reconstruction | §3.2 | `summary.json`: `lossless_rate = 1.0` |
| ~3.35 BPB compression | §3.2 | `summary.json`: `mean_bits_per_byte` |
| HHC improves stability by 36% | §4.2 | Compare `baseline_shift` vs `hhc_shift` |
| HHC costs 10.5% compression | §4.2 | BPB difference between runs |
| LIL 100% round-trip lossless | §6.2 | `eval/results/lil/summary.json` |

## Building the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `paper/main.pdf`

## Running Tests

```bash
uv run scripts/test_all.py
```

All tests must pass for valid reproduction.

## Troubleshooting

**"No module named torch"**: Scripts use PEP 723 inline dependencies. Run with `uv run` to auto-install.

**Metric mismatch on reproduce**: Ensure identical Python version and seed configuration. Floating-point differences < 1e-6 are tolerated.

**Out of memory**: Use `configs/cpu_small.toml` instead of `configs/default.toml`.

## Contact

J. W. Miller — j.w.miller@curv.institute
