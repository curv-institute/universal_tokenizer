# Universal Lossless Tokenizer

A universal lossless tokenizer for bit-exact reconstruction of arbitrary byte streams.

## Overview

This tokenizer implements:
- **Equilibrium projection**: Tokens as attractors of span latents via contraction mapping
- **Curvature-based segmentation**: Boundary selection optimizing `bit_cost + β·curvature + γ·stability`
- **Closed-loop stability control**: Adaptive regulation of tokenization parameters
- **Lossless reconstruction**: Bit-exact decode via residual coding
- **Harmonized Hyper-Connections (HHC)**: Optional relational regularization for improved stability
- **Language Interface Layer (LIL)**: Reversible prompt structure normalization

## Installation

```bash
# Clone repository
git clone https://github.com/curv-institute/universal_tokenizer.git
cd universal_tokenizer

# Dependencies managed via uv (PEP 723 inline scripts)
# Install uv if needed:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

```bash
# Run tests
uv run scripts/test_all.py

# Encode data
uv run scripts/encode.py data.txt --output tokens.json

# Evaluate
uv run scripts/eval.py --config configs/cpu_small.toml
```

## Reproduce Paper Results

All results from the paper can be reproduced from recorded manifests.

### 1. Generate Evaluation Data

```bash
uv run scripts/make_eval_data.py --profile core --out data/eval
```

### 2. Run Experiment

```bash
uv run scripts/run_experiment.py \
  --config configs/cpu_small.toml \
  --name paper_v1 \
  --eval data/eval
```

Outputs go to `eval/results/paper_v1/`:
- `summary.json` — aggregate metrics
- `metrics.jsonl` — per-file results
- `manifests/` — reproducibility manifests

### 3. Reproduce from Manifest

```bash
uv run scripts/reproduce.py \
  --manifest "eval/results/paper_v1/manifests/*.json" \
  --compare
```

### 4. Build Paper PDF

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

See [ARTIFACT_CHECKLIST.md](ARTIFACT_CHECKLIST.md) for complete reproduction instructions.

## Project Structure

```
universal_tokenizer/
  configs/           # TOML configuration files
  scripts/           # Runnable scripts (PEP 723)
  tokenizer/         # Core library
  eval/results/      # Per-run evaluation outputs
  tests/             # Test suite
  paper/             # LaTeX manuscript and figures
```

## Configuration

Two main configurations:
- `configs/cpu_small.toml` — Baseline tokenizer
- `configs/cpu_small_hhc.toml` — HHC-enabled tokenizer

## Citation

```bibtex
@software{miller2026universal,
  author = {Miller, J. W.},
  title = {Universal Lossless Tokenizer},
  year = {2026},
  url = {https://github.com/curv-institute/universal_tokenizer}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Author

J. W. Miller, Founding Director, CURV Institute
j.w.miller@curv.institute
