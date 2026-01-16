## HHC Trade-off Comparison

| Metric | Baseline | HHC | Delta |
|--------|----------|-----|-------|
| E2E BPB | 3.35 | 3.70 | +0.35 |
| Structural BPB | 0.32 | 0.35 | +0.04 |
| Token Churn Rate | 0.00 | 0.00 | 0.00 |
| Substring Stability Rate | 0.00 | 0.00 | 0.00 |
| Curvature P90 (tail mass) | 0.63 | 0.68 | +0.06 |
| Boundary Curvature Delta | 0.00 | 0.00 | 0.00 |
| Harmonizer Interventions | 0 | 2 | +2 |
| Throughput (KB/s) | 0.0 | 0.0 | 0.0 |

**Notes:**
- E2E BPB: End-to-end bits per byte (primary compression metric, lower is better)
- Structural BPB: Token representation efficiency (excludes residuals)
- Token Churn Rate: Inconsistency of tokenization for identical patterns (lower is better)
- Substring Stability Rate: Consistency for repeated motifs (higher is better)
- Curvature P90: 90th percentile curvature (boundary quality, lower is better)
- Boundary Curvature Delta: Difference between boundary and background curvature
- Harmonizer Interventions: Number of HHC parameter adjustments
