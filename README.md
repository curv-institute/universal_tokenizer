# Universal Lossless Tokenizer

**Codebase for:** [Testing the Platonic Representation Hypothesis via Representation-Controlled Tokenization](https://curv.institute/publications/universal-tokenizer-prh/)

J. W. Miller, CURV Institute — January 2026

## Abstract

Modern tokenizers are treated as neutral preprocessing steps, yet they implicitly encode assumptions about representation, stability, and learnability. The Platonic Representation Hypothesis (PRH) posits that stable abstract representations exist independently of semantics and learning objectives, and that such representations can be measured and regulated.

We test this hypothesis empirically by constructing a representation-controlled stack consisting of:
- **Universal Lossless Tokenizer (ULT)** — bit-exact reconstruction of arbitrary byte streams
- **Harmonized Hyper-Connections (HHC)** — relational regularization for improved stability
- **Language Interface Layer (LIL)** — reversible prompt structure normalization

## Key Results

| Component | Finding |
|-----------|---------|
| **ULT** | 100% lossless reconstruction, 3.35 bits/byte (2.39× compression) on heterogeneous streams |
| **HHC** | +36% stability improvement at cost of +10.5% compression penalty |
| **LM Proxy** | Negative result: stability-optimized tokens are harder to learn (confirms PRH) |
| **LIL** | 100% lossless round-trip with 5% structural overhead |

The negative LM proxy result is not a failure but a confirmation: representations optimized for global stability are not necessarily aligned with local next-token predictability. This supports PRH's prediction that stability, efficiency, and learnability are distinct axes requiring explicit trade-offs.

## Installation

```bash
git clone https://github.com/curv-institute/universal_tokenizer.git
cd universal_tokenizer

# Dependencies managed via uv (PEP 723 inline scripts)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

```bash
# Run tests
uv run scripts/test_all.py

# Encode data
uv run scripts/encode.py data.txt --output tokens.json

# Evaluate baseline
uv run scripts/eval.py --config configs/cpu_small.toml

# Evaluate with HHC
uv run scripts/eval.py --config configs/cpu_small_hhc.toml
```

## Components

### Universal Lossless Tokenizer (ULT)

Operates directly on bytes with streaming support and guaranteed exact reconstruction:
- **Equilibrium projection**: Tokens as attractors via contraction mapping
- **Curvature-based segmentation**: Boundary selection optimizing `bit_cost + β·curvature + γ·stability`
- **Residual coding**: Bit-exact decode for lossless reconstruction

### Harmonized Hyper-Connections (HHC)

Relational regularization mechanism introducing bounded coupling between neighboring representations:
- Closed-loop stability controller
- Curvature diagnostics during tokenization
- Configurable stability/efficiency trade-off

### Language Interface Layer (LIL)

Reversible, deterministic transformation for structured prompts:
- Explicit role markers and segment boundaries
- 23 reversible macros for common patterns
- Tokenizer-agnostic, streaming-compatible

## Reproduce Paper Results

All results can be reproduced from recorded manifests.

### 1. Generate Evaluation Data

```bash
uv run scripts/make_eval_data.py --profile core --out data/eval
```

### 2. Run Experiments

```bash
# Baseline
uv run scripts/run_experiment.py \
  --config configs/cpu_small.toml \
  --name baseline \
  --eval data/eval

# HHC-enabled
uv run scripts/run_experiment.py \
  --config configs/cpu_small_hhc.toml \
  --name hhc \
  --eval data/eval
```

### 3. Compare Results

```bash
uv run scripts/report.py --compare eval/results/baseline eval/results/hhc
```

### 4. Reproduce from Manifest

```bash
uv run scripts/reproduce.py \
  --manifest "eval/results/*/manifests/*.json" \
  --compare
```

### 5. Build Paper PDF

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
  data/eval/         # Evaluation datasets with manifests
```

## Citation

```bibtex
@article{miller2026prh_tokenizer,
  author = {Miller, J. W.},
  title = {Testing the Platonic Representation Hypothesis via Representation-Controlled Tokenization},
  year = {2026},
  institution = {CURV Institute},
  url = {https://curv.institute/publications/universal-tokenizer-prh/}
}
```

## References

- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. arXiv:2405.07987
- Miller, J. W. (2026). Harmonized Hyper-Connections: Relational Coupling for Stable Representations. CURV Institute Technical Report.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Author

J. W. Miller, Founding Director, CURV Institute
j.w.miller@curv.institute
