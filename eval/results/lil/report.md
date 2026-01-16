# LIL Evaluation Report

**Generated:** 2026-01-16T02:37:08.079006
**Config:** configs/cpu_small.toml
**Macros:** Enabled (23 registered)

## Summary

| Metric | Baseline | LIL | LIL+HHC |
|--------|----------|-----|---------|
| E2E BPB | 3.82 | 4.02 | 4.41 |
| Structural BPB | 0.35 | 0.38 | 0.42 |
| Avg Tokens | 3.6 | 3.6 | 4.2 |
| Lossless Rate | 100% | 100% | N/A |

## Experiment 1: Prompt Extension Stability

- **Baseline avg token churn:** 2.2000
- **LIL avg token churn:** 4.2500
- **Improvement:** -93.2%

## Experiment 2: Role Boundary Robustness

- **All LIL transforms lossless:** True
- **LIL ratio:** 107.32%

## Experiment 3: Interface Efficiency

Trade-off analysis: LIL may increase token count to achieve structural benefits.
