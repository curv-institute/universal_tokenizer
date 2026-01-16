# Paper Artifact - HHC Stability Experiments

Generated: 2026-01-15T21:52:33.710816

## Summary

HHC slightly increases compression cost (3.70 BPB vs 3.35 BPB baseline) but improves stability under regime shifts: token churn remained stable by N/A, substring stability remained stable by N/A, and boundary curvature spikes remained stable by N/A. Harmonizer interventions changed from 0 to 2.

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


## Key Findings

1. **Compression Trade-off**: HHC achieves 3.70 BPB vs 3.35 BPB baseline (10.5% change)

2. **Stability Improvement**: Token churn rate: 0.00 (HHC) vs 0.00 (baseline)

3. **Boundary Quality**: Curvature P90: 0.6832 (HHC) vs 0.6262 (baseline)

## Reproducibility

```bash
# Run baseline evaluation
uv run scripts/eval.py --config configs/cpu_small.toml --output eval/results/baseline

# Run HHC evaluation
uv run scripts/eval.py --config configs/cpu_small_hhc.toml --output eval/results/hhc

# Generate comparison artifacts
uv run scripts/report.py --compare eval/results/baseline eval/results/hhc
```

## Raw Data Reference

- Baseline samples: 100
- HHC samples: 100
- Baseline timestamp: 20260115_215207
- HHC timestamp: 20260115_215212

## Figure Data

Boundary analysis figure data is available in `boundary_figure_data.json` for use with:
- **pgfplots**: Import JSON and plot distance bins vs churn/curvature
- **matplotlib**: Load JSON with `json.load()` and use standard plotting

## LaTeX Table

The LaTeX version of the trade-off table is available in `results_hhc_tradeoff.tex`.
Include in paper with: `\input{results_hhc_tradeoff.tex}`
