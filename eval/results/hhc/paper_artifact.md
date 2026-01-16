# Paper Artifact - Universal Lossless Tokenizer

## Citation

```bibtex
@software{miller2025universal,
  author = {Miller, J. W.},
  title = {Universal Lossless Tokenizer},
  year = {2025},
  url = {https://github.com/curv-institute/universal_tokenizer}
}
```

## Key Results

| Metric | Value |
|--------|-------|
| Compression Ratio | 25.20 |
| Bits per Byte | 0.59 |
| Lossless Rate | 100.0% |
| Avg Token Length | 25.20 |

## Reproducibility

```bash
git clone https://github.com/curv-institute/universal_tokenizer
cd universal_tokenizer
uv venv .venv && source .venv/bin/activate
uv run scripts/run_experiment.py --config configs/default.toml --name paper_v0_1
```

## Environment

- Timestamp: 20260115_181607
- Data entropy: 7.716 bits/byte
- Samples evaluated: 5
