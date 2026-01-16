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
| **Lossless BPB** | 4.83 | End-to-end compression (primary) |
| Structural BPB | 0.46 | Token representation only |
| Compression Ratio | 25.20 | Bytes per token |
| Lossless Rate | 100.0% | Exact reconstruction |
| Avg Token Length | 25.20 | Mean bytes per token |

**Note:** Lossless BPB is the primary compression metric and includes both token IDs and residual bytes required for exact reconstruction. Structural BPB measures token representation efficiency only and should not be cited as a compression ratio.

## Reproducibility

```bash
git clone https://github.com/curv-institute/universal_tokenizer
cd universal_tokenizer
uv venv .venv && source .venv/bin/activate
uv run scripts/run_experiment.py --config configs/default.toml --name paper_v0_1
```

## Environment

- Timestamp: 20260115_195804
- Data entropy: 7.716 bits/byte
- Samples evaluated: 5
