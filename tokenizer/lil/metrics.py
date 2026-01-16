"""LIL evaluation metrics.

Metrics for evaluating the Language Interface Layer focus on RIFT principles:
- Instruction churn: stability of core instructions under context extension
- Substring stability: consistency of tokenization for identical substrings
- Curvature at boundaries: smoothness at role transitions
- Harmonizer intervention frequency: controller activity

These metrics evaluate interface optimization, NOT compression performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SegmentMetrics:
    """Metrics for a single prompt segment.

    Attributes:
        role: The segment role
        byte_count: Number of bytes
        token_count: Number of tokens
        avg_curvature: Average curvature across tokens
        boundary_curvature: Curvature at segment boundary
    """
    role: str
    byte_count: int
    token_count: int
    avg_curvature: float = 0.0
    boundary_curvature: float = 0.0


@dataclass
class ChurnMetrics:
    """Metrics for instruction churn analysis.

    Churn measures how stable a core instruction is when context changes.

    Attributes:
        core_bytes: Size of core instruction
        core_tokens_baseline: Tokens for core alone
        core_tokens_extended: Tokens for core in extended context
        token_churn_rate: Fraction of tokens that changed
        byte_position_drift: Average drift in byte positions
    """
    core_bytes: int
    core_tokens_baseline: int
    core_tokens_extended: int
    token_churn_rate: float = 0.0
    byte_position_drift: float = 0.0


@dataclass
class BoundaryMetrics:
    """Metrics for role boundary analysis.

    Attributes:
        num_boundaries: Number of role transitions
        avg_boundary_curvature: Average curvature at boundaries
        max_boundary_curvature: Maximum boundary curvature
        boundary_token_churn: Token changes near boundaries
        harmonizer_interventions: Controller interventions at boundaries
    """
    num_boundaries: int
    avg_boundary_curvature: float = 0.0
    max_boundary_curvature: float = 0.0
    boundary_token_churn: float = 0.0
    harmonizer_interventions: int = 0


@dataclass
class LILEvalMetrics:
    """Complete LIL evaluation metrics.

    Attributes:
        segments: Per-segment metrics
        churn: Instruction churn metrics
        boundaries: Boundary metrics
        total_bytes: Total input bytes
        total_tokens: Total output tokens
        e2e_bpb: End-to-end bits per byte
        structural_bpb: Structural bits per byte
        lossless: Whether reconstruction is lossless
    """
    segments: list[SegmentMetrics] = field(default_factory=list)
    churn: ChurnMetrics | None = None
    boundaries: BoundaryMetrics | None = None
    total_bytes: int = 0
    total_tokens: int = 0
    e2e_bpb: float = 0.0
    structural_bpb: float = 0.0
    lossless: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "segments": [
                {
                    "role": s.role,
                    "byte_count": s.byte_count,
                    "token_count": s.token_count,
                    "avg_curvature": s.avg_curvature,
                    "boundary_curvature": s.boundary_curvature,
                }
                for s in self.segments
            ],
            "churn": {
                "core_bytes": self.churn.core_bytes,
                "core_tokens_baseline": self.churn.core_tokens_baseline,
                "core_tokens_extended": self.churn.core_tokens_extended,
                "token_churn_rate": self.churn.token_churn_rate,
                "byte_position_drift": self.churn.byte_position_drift,
            } if self.churn else None,
            "boundaries": {
                "num_boundaries": self.boundaries.num_boundaries,
                "avg_boundary_curvature": self.boundaries.avg_boundary_curvature,
                "max_boundary_curvature": self.boundaries.max_boundary_curvature,
                "boundary_token_churn": self.boundaries.boundary_token_churn,
                "harmonizer_interventions": self.boundaries.harmonizer_interventions,
            } if self.boundaries else None,
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "e2e_bpb": self.e2e_bpb,
            "structural_bpb": self.structural_bpb,
            "lossless": self.lossless,
        }


def compute_token_churn(
    tokens_a: Sequence[int],
    tokens_b: Sequence[int],
) -> float:
    """Compute token churn between two token sequences.

    Churn is the fraction of tokens that differ between sequences
    when aligned by position.

    Args:
        tokens_a: First token sequence
        tokens_b: Second token sequence

    Returns:
        Churn rate (0 = identical, 1 = completely different)
    """
    if len(tokens_a) == 0 and len(tokens_b) == 0:
        return 0.0

    # Use edit distance for alignment-independent comparison
    # Simplified: compare overlapping portion + penalize length diff
    min_len = min(len(tokens_a), len(tokens_b))
    max_len = max(len(tokens_a), len(tokens_b))

    if max_len == 0:
        return 0.0

    matches = sum(1 for i in range(min_len) if tokens_a[i] == tokens_b[i])
    churn = 1.0 - (matches / max_len)

    return churn


def compute_boundary_curvature(
    curvatures: Sequence[float],
    boundary_positions: Sequence[int],
    window: int = 4,
) -> tuple[float, float]:
    """Compute curvature statistics at boundaries.

    Args:
        curvatures: Per-token curvature values
        boundary_positions: Token positions of role boundaries
        window: Window size around boundaries

    Returns:
        Tuple of (average boundary curvature, max boundary curvature)
    """
    if not boundary_positions or not curvatures:
        return 0.0, 0.0

    boundary_curvatures = []
    for pos in boundary_positions:
        start = max(0, pos - window)
        end = min(len(curvatures), pos + window + 1)
        if start < end:
            boundary_curvatures.extend(curvatures[start:end])

    if not boundary_curvatures:
        return 0.0, 0.0

    return float(np.mean(boundary_curvatures)), float(np.max(boundary_curvatures))


def compute_instruction_stability(
    core_tokens_alone: Sequence[int],
    core_tokens_in_context: Sequence[int],
) -> tuple[float, float]:
    """Compute stability of core instruction when extended.

    Args:
        core_tokens_alone: Tokens when encoding core alone
        core_tokens_in_context: Tokens for core part when in larger context

    Returns:
        Tuple of (churn rate, position drift)
    """
    churn = compute_token_churn(core_tokens_alone, core_tokens_in_context)

    # Position drift: how much token positions shift on average
    if len(core_tokens_alone) > 0 and len(core_tokens_in_context) > 0:
        min_len = min(len(core_tokens_alone), len(core_tokens_in_context))
        drifts = []
        for i in range(min_len):
            if core_tokens_alone[i] != core_tokens_in_context[i]:
                # Look for token in nearby positions
                found_offset = None
                for offset in range(1, 5):
                    if i + offset < len(core_tokens_in_context) and \
                       core_tokens_alone[i] == core_tokens_in_context[i + offset]:
                        found_offset = offset
                        break
                    if i - offset >= 0 and \
                       core_tokens_alone[i] == core_tokens_in_context[i - offset]:
                        found_offset = offset
                        break
                if found_offset is not None:
                    drifts.append(found_offset)

        position_drift = float(np.mean(drifts)) if drifts else 0.0
    else:
        position_drift = 0.0

    return churn, position_drift
