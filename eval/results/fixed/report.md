# Universal Lossless Tokenizer - Evaluation Report

Generated: 2026-01-15T19:58:49.490513

## Configuration

- Config: `configs/cpu_small.toml`
- Model: `None`
- Samples: 5
- Data entropy: 7.716 bits/byte

## Results

| Tokenizer | Lossless BPB* | Struct BPB | Compression | Lossless |
|-----------|---------------|------------|-------------|----------|
| universal | 4.83 | 0.46 | 25.20 | 100.0% |
| raw_bytes | 8.00 | 8.00 | 1.00 | 100.0% |
| byte_bpe | 9.24 | 9.24 | 1.08 | 100.0% |

*\*Primary compression metric*

**Metric Definitions:** End-to-end BPB (Lossless BPB) measures the true lossless encoding cost including token IDs and residual bytes. Structural BPB measures representational efficiency prior to residual correction and should not be interpreted as a standalone compression ratio. All compression claims in this work are based on end-to-end lossless BPB.

## Detailed Metrics

### universal

- Total tokens: 41
- Total bytes: 1038
- **Lossless BPB (E2E):** 4.8268
- Structural BPB: 0.4554
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
- **Lossless BPB (E2E):** 8.0000
- Structural BPB: 8.0000
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
- **Lossless BPB (E2E):** 9.2400
- Structural BPB: 9.2400
- Total residual bytes: 0
- Mean compression ratio: 1.0834
- Mean avg token length: 1.0834
- Mean curvature: 0.0000
- Curvature P90: 0.0000
- Mean stability: 1.0000
- Stability P10: 1.0000
- Lossless rate: 100.00%
