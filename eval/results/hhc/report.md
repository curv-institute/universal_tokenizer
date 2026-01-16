# Universal Lossless Tokenizer - Evaluation Report

Generated: 2026-01-15T18:18:35.398954

## Configuration

- Config: `configs/cpu_small_hhc.toml`
- Model: `None`
- Samples: 5
- Data entropy: 7.716 bits/byte

## Results

| Tokenizer | Compression | BPB | Lossless |
|-----------|-------------|-----|----------|
| universal | 25.20 | 0.59 | 100.0% |
| raw_bytes | 1.00 | 13.00 | 100.0% |
| byte_bpe | 1.08 | 12.01 | 100.0% |

## Detailed Metrics

### universal

- Total tokens: 41
- Total bytes: 1038
- Mean compression ratio: 25.2000
- Mean bits per byte: 0.5920
- Mean avg token length: 25.2000
- Mean curvature: 0.6159
- Curvature P90: 0.6249
- Mean stability: 0.2409
- Stability P10: 0.1953
- Lossless rate: 100.00%

### raw_bytes

- Total tokens: 1038
- Total bytes: 1038
- Mean compression ratio: 1.0000
- Mean bits per byte: 13.0000
- Mean avg token length: 1.0000
- Mean curvature: 0.0000
- Curvature P90: 0.0000
- Mean stability: 1.0000
- Stability P10: 1.0000
- Lossless rate: 100.00%

### byte_bpe

- Total tokens: 958
- Total bytes: 1038
- Mean compression ratio: 1.0834
- Mean bits per byte: 12.0119
- Mean avg token length: 1.0834
- Mean curvature: 0.0000
- Curvature P90: 0.0000
- Mean stability: 1.0000
- Stability P10: 1.0000
- Lossless rate: 100.00%
