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

| Metric | Value | Description |
|--------|-------|-------------|
| **Lossless BPB** | 3.70 | End-to-end compression (primary) |
| Structural BPB | 0.35 | Token representation only |
| Compression Ratio | 28.31 | Bytes per token |
| Lossless Rate | 100.0% | Exact reconstruction |
| Avg Token Length | 28.31 | Mean bytes per token |

**Note:** Lossless BPB is the primary compression metric and includes both token IDs and residual bytes required for exact reconstruction. Structural BPB measures token representation efficiency only and should not be cited as a compression ratio.

## Reproducibility

```bash
git clone https://github.com/curv-institute/universal_tokenizer
cd universal_tokenizer
uv venv .venv && source .venv/bin/activate
uv run scripts/run_experiment.py --config configs/default.toml --name paper_v0_1
```

## Environment

- Timestamp: 20260115_204723
- Data entropy: 6.655 bits/byte
- Samples evaluated: 100
