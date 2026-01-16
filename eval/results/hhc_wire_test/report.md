# Universal Lossless Tokenizer - Evaluation Report

Generated: 2026-01-15T21:07:51.736777

## Configuration

- Config: `configs/cpu_small_hhc.toml`
- Model: `None`
- Samples: 100
- Data entropy: 6.655 bits/byte

## Results

| Tokenizer | Lossless BPB* | Struct BPB | Compression | Lossless |
|-----------|---------------|------------|-------------|----------|
| universal | 3.70 | 0.35 | 28.31 | 100.0% |
| raw_bytes | 8.00 | 8.00 | 1.00 | 100.0% |
| byte_bpe | 2.01 | 2.01 | 6.58 | 100.0% |

*\*Primary compression metric*

**Metric Definitions:** End-to-end BPB (Lossless BPB) measures the true lossless encoding cost including token IDs and residual bytes. Structural BPB measures representational efficiency prior to residual correction and should not be interpreted as a standalone compression ratio. All compression claims in this work are based on end-to-end lossless BPB.

## Detailed Metrics

### universal

- Total tokens: 908
- Total bytes: 25600
- **Lossless BPB (E2E):** 3.6972
- Structural BPB: 0.3547
- Total residual bytes: 10696
- Mean compression ratio: 28.3106
- Mean avg token length: 28.3106
- Mean curvature: 0.5608
- Curvature P90: 0.6788
- Mean stability: 0.3183
- Stability P10: 0.2021
- Lossless rate: 100.00%

### raw_bytes

- Total tokens: 25600
- Total bytes: 25600
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

- Total tokens: 5147
- Total bytes: 25600
- **Lossless BPB (E2E):** 2.0105
- Structural BPB: 2.0105
- Total residual bytes: 0
- Mean compression ratio: 6.5813
- Mean avg token length: 6.5813
- Mean curvature: 0.0000
- Curvature P90: 0.0000
- Mean stability: 1.0000
- Stability P10: 1.0000
- Lossless rate: 100.00%

## HHC Diagnostics

- **HHC Active:** Yes
- Active fraction: 100.0%
- Mean neighbors: 8.0
- Mean neighbor distance: 0.731
- Neighbor distance variance: 0.113
- Mean curvature delta: 0.028
- Nonzero delta fraction: 100.0%
