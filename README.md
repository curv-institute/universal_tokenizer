# Universal Lossless Tokenizer

A RIFT-instantiated universal lossless tokenizer for bit-exact reconstruction of arbitrary byte streams.

## Overview

This tokenizer implements:
- **Equilibrium projection**: Tokens as attractors of span latents via contraction mapping
- **Curvature-based segmentation**: Boundary selection optimizing `bit_cost + β·curvature + γ·stability`
- **Harmonizer control loop**: Adaptive regulation of tokenization parameters
- **Lossless reconstruction**: Bit-exact decode via residual coding

## Installation

```bash
# Clone repository
git clone https://github.com/curv-institute/universal_tokenizer.git
cd universal_tokenizer

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate
```

## Usage

```bash
# Train tokenizer
uv run scripts/train.py --config configs/default.toml

# Tokenize data
uv run scripts/tokenize.py --input data.txt --output tokens.json

# Evaluate
uv run scripts/eval.py --config configs/default.toml

# Run tests
uv run scripts/test_all.py
```

## Version Control with jj (Jujutsu)

This project uses [jj](https://github.com/martinvonz/jj) for version control.

```bash
# Install jj
cargo install jj-cli

# Basic workflow
jj status          # View current changes
jj diff            # View detailed diff
jj commit -m "msg" # Commit changes
jj log             # View history

# Push to GitHub
jj git push
```

## Project Structure

```
universal_tokenizer/
  configs/           # TOML configuration files
  scripts/           # Runnable scripts (PEP 723)
  tokenizer/         # Core library
  eval/results/      # Evaluation outputs
  tests/             # Test suite
  paper/             # LaTeX manuscript
```

## Citation

```bibtex
@software{miller2025universal,
  author = {Miller, J. W.},
  title = {Universal Lossless Tokenizer},
  year = {2025},
  url = {https://github.com/curv-institute/universal_tokenizer}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Author

J. W. Miller, Founding Director, CURV Institute
j.w.miller@curv.institute
