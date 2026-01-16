# Evaluation Matrix Report

Generated: 2026-01-15T20:18:45.250512
Total runs aggregated: 6

## Comparison Table

| Tier | BASELINE E2E BPB | BASELINE Struct BPB | BASELINE Curv P90 | BASELINE Lossless | BASELINE Compress | HHC E2E BPB | HHC Struct BPB | HHC Curv P90 | HHC Lossless | HHC Compress |
|------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10mb_baseline | 3.346 | 0.316 | 0.626 | 100.0% | 31.84 | 0.000 | 0.000 | 0.000 | 0.0% | 0.00 |
| 10mb_hhc | 0.000 | 0.000 | 0.000 | 0.0% | 0.00 | 3.346 | 0.316 | 0.626 | 100.0% | 31.84 |
| 1mb_baseline | 3.346 | 0.316 | 0.626 | 100.0% | 31.84 | 0.000 | 0.000 | 0.000 | 0.0% | 0.00 |
| 1mb_hhc | 0.000 | 0.000 | 0.000 | 0.0% | 0.00 | 3.346 | 0.316 | 0.626 | 100.0% | 31.84 |
| smoke_baseline | 4.827 | 0.455 | 0.625 | 100.0% | 25.20 | 0.000 | 0.000 | 0.000 | 0.0% | 0.00 |
| smoke_hhc | 0.000 | 0.000 | 0.000 | 0.0% | 0.00 | 4.827 | 0.455 | 0.625 | 100.0% | 25.20 |

## Metric Definitions

- **E2E BPB**: End-to-end bits per byte (lossless compression, primary metric)
- **Struct BPB**: Structural bits per byte (token representation only)
- **Curv P90**: 90th percentile curvature (boundary quality)
- **Lossless**: Exact reconstruction rate
- **Compress**: Compression ratio (bytes/token)

## Detailed Results

### 10Mb_Baseline

**BASELINE** (1 runs)

- E2E BPB: 3.346 (std: 0.0000, n=1)
- Struct BPB: 0.316 (std: 0.0000, n=1)
- Curv P90: 0.626 (std: 0.0000, n=1)
- Lossless: 100.0% (std: 0.0000, n=1)
- Compress: 31.84 (std: 0.0000, n=1)

**HHC** (0 runs)

- E2E BPB: 0.000 (std: 0.0000, n=0)
- Struct BPB: 0.000 (std: 0.0000, n=0)
- Curv P90: 0.000 (std: 0.0000, n=0)
- Lossless: 0.0% (std: 0.0000, n=0)
- Compress: 0.00 (std: 0.0000, n=0)

### 10Mb_Hhc

**BASELINE** (0 runs)

- E2E BPB: 0.000 (std: 0.0000, n=0)
- Struct BPB: 0.000 (std: 0.0000, n=0)
- Curv P90: 0.000 (std: 0.0000, n=0)
- Lossless: 0.0% (std: 0.0000, n=0)
- Compress: 0.00 (std: 0.0000, n=0)

**HHC** (1 runs)

- E2E BPB: 3.346 (std: 0.0000, n=1)
- Struct BPB: 0.316 (std: 0.0000, n=1)
- Curv P90: 0.626 (std: 0.0000, n=1)
- Lossless: 100.0% (std: 0.0000, n=1)
- Compress: 31.84 (std: 0.0000, n=1)

### 1Mb_Baseline

**BASELINE** (1 runs)

- E2E BPB: 3.346 (std: 0.0000, n=1)
- Struct BPB: 0.316 (std: 0.0000, n=1)
- Curv P90: 0.626 (std: 0.0000, n=1)
- Lossless: 100.0% (std: 0.0000, n=1)
- Compress: 31.84 (std: 0.0000, n=1)

**HHC** (0 runs)

- E2E BPB: 0.000 (std: 0.0000, n=0)
- Struct BPB: 0.000 (std: 0.0000, n=0)
- Curv P90: 0.000 (std: 0.0000, n=0)
- Lossless: 0.0% (std: 0.0000, n=0)
- Compress: 0.00 (std: 0.0000, n=0)

### 1Mb_Hhc

**BASELINE** (0 runs)

- E2E BPB: 0.000 (std: 0.0000, n=0)
- Struct BPB: 0.000 (std: 0.0000, n=0)
- Curv P90: 0.000 (std: 0.0000, n=0)
- Lossless: 0.0% (std: 0.0000, n=0)
- Compress: 0.00 (std: 0.0000, n=0)

**HHC** (1 runs)

- E2E BPB: 3.346 (std: 0.0000, n=1)
- Struct BPB: 0.316 (std: 0.0000, n=1)
- Curv P90: 0.626 (std: 0.0000, n=1)
- Lossless: 100.0% (std: 0.0000, n=1)
- Compress: 31.84 (std: 0.0000, n=1)

### Smoke_Baseline

**BASELINE** (1 runs)

- E2E BPB: 4.827 (std: 0.0000, n=1)
- Struct BPB: 0.455 (std: 0.0000, n=1)
- Curv P90: 0.625 (std: 0.0000, n=1)
- Lossless: 100.0% (std: 0.0000, n=1)
- Compress: 25.20 (std: 0.0000, n=1)

**HHC** (0 runs)

- E2E BPB: 0.000 (std: 0.0000, n=0)
- Struct BPB: 0.000 (std: 0.0000, n=0)
- Curv P90: 0.000 (std: 0.0000, n=0)
- Lossless: 0.0% (std: 0.0000, n=0)
- Compress: 0.00 (std: 0.0000, n=0)

### Smoke_Hhc

**BASELINE** (0 runs)

- E2E BPB: 0.000 (std: 0.0000, n=0)
- Struct BPB: 0.000 (std: 0.0000, n=0)
- Curv P90: 0.000 (std: 0.0000, n=0)
- Lossless: 0.0% (std: 0.0000, n=0)
- Compress: 0.00 (std: 0.0000, n=0)

**HHC** (1 runs)

- E2E BPB: 4.827 (std: 0.0000, n=1)
- Struct BPB: 0.455 (std: 0.0000, n=1)
- Curv P90: 0.625 (std: 0.0000, n=1)
- Lossless: 100.0% (std: 0.0000, n=1)
- Compress: 25.20 (std: 0.0000, n=1)
