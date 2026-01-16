# Universal Lossless Tokenizer - Evaluation Report

Generated: 2026-01-15T20:14:24.029471

## Configuration

- Config: `configs/cpu_small.toml`
- Model: `None`
- Samples: 5
- Data entropy: 7.716 bits/byte

## Results

| Tokenizer | Lossless BPB* | Struct BPB | Compression | Lossless |
|-----------|---------------|------------|-------------|----------|
| universal | 0.59 | 0.59 | 25.20 | 100.0% |
| raw_bytes | 13.00 | 13.00 | 1.00 | 100.0% |
| byte_bpe | 12.01 | 12.01 | 1.08 | 100.0% |

*\*Primary compression metric*

**Metric Definitions:** End-to-end BPB (Lossless BPB) measures the true lossless encoding cost including token IDs and residual bytes. Structural BPB measures representational efficiency prior to residual correction and should not be interpreted as a standalone compression ratio. All compression claims in this work are based on end-to-end lossless BPB.

## Detailed Metrics

### universal

- Total tokens: 41
- Total bytes: 1038
- **Lossless BPB (E2E):** 0.5920
- Structural BPB: 0.5920
- Total residual bytes: 492
- Mean compression ratio: 25.2000
- Mean avg token length: 25.2000
- Mean curvature: 0.6159
- Curvature P90: 0.6249
- Mean stability: 0.2409
- Stability P10: 0.1953
- Lossless rate: 100.00%

### raw_bytes

- Total tokens: 1038
- Total bytes: 1038
- **Lossless BPB (E2E):** 13.0000
- Structural BPB: 13.0000
- Total residual bytes: 0
- Mean compression ratio: 1.0000
- Mean avg token length: 1.0000
- Mean curvature: 0.0000
- Curvature P90: 0.0000
- Mean stability: 1.0000
- Stability P10: 1.0000
- Lossless rate: 100.00%

### byte_bpe

- Total tokens: 958
- Total bytes: 1038
- **Lossless BPB (E2E):** 12.0119
- Structural BPB: 12.0119
- Total residual bytes: 0
- Mean compression ratio: 1.0834
- Mean avg token length: 1.0834
- Mean curvature: 0.0000
- Curvature P90: 0.0000
- Mean stability: 1.0000
- Stability P10: 1.0000
- Lossless rate: 100.00%
