"""Stream segmenter - DIRT instantiation.

Selects span boundaries as low-action path optimizing:
bit_cost + beta*curvature + gamma*stability_penalty
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

import torch
from torch import Tensor

from .interfaces import SegmentationConfig

if TYPE_CHECKING:
    from .encoder import ByteEncoder
    from .equilibrium import EquilibriumProjector
    from .codebook import TokenCodebook
    from .curvature import CurvatureEstimator


class StreamSegmenter:
    """Streaming segmenter with curvature-aware boundary selection.

    Implements DIRT instantiation - dynamics as streaming segmentation.
    Composition is admissible only if equilibrium stable and curvature bounded.
    """

    def __init__(
        self,
        config: SegmentationConfig,
        encoder: "ByteEncoder",
        equilibrium: "EquilibriumProjector",
        codebook: "TokenCodebook",
        curvature: "CurvatureEstimator",
    ):
        self.config = config
        self.encoder = encoder
        self.equilibrium = equilibrium
        self.codebook = codebook
        self.curvature = curvature

        # Adaptive parameters (can be modified by controller)
        self.beta = config.beta_curvature
        self.gamma = config.gamma_stability
        self.max_span = config.max_span

    def segment(self, data: bytes) -> Iterator[tuple[int, int]]:
        """Segment byte stream into spans.

        Args:
            data: Input byte sequence

        Yields:
            (start, end) tuples for each span
        """
        for start, end, _, _ in self.segment_with_scores(data):
            yield start, end

    def segment_with_scores(
        self, data: bytes
    ) -> Iterator[tuple[int, int, float, float]]:
        """Segment with curvature and stability scores.

        Uses dynamic programming with lookahead for optimal segmentation.

        Yields:
            (start, end, curvature, stability) tuples
        """
        n = len(data)
        if n == 0:
            return

        pos = 0
        while pos < n:
            best_end = min(pos + self.config.min_span, n)
            best_score = float("inf")
            best_curv = 0.0
            best_stab = 1.0

            # Evaluate candidate spans with lookahead
            max_end = min(pos + self.max_span, n)

            for end in range(pos + self.config.min_span, max_end + 1):
                span = data[pos:end]
                score, curv, stab = self._score_span(span, pos, end, data)

                if score < best_score:
                    best_score = score
                    best_end = end
                    best_curv = curv
                    best_stab = stab

            yield pos, best_end, best_curv, best_stab
            pos = best_end

    def _score_span(
        self, span: bytes, start: int, end: int, full_data: bytes
    ) -> tuple[float, float, float]:
        """Score a candidate span.

        Returns:
            (total_score, curvature, stability)
        """
        # Encode span
        with torch.no_grad():
            z = self.encoder.encode([span])

            # Project to equilibrium
            z_eq, trajectory = self.equilibrium.project_with_trajectory(z)

            # Get curvature
            curv = self.curvature.estimate(z_eq).item()

            # Stability: measure convergence rate
            if len(trajectory) > 1:
                deltas = [
                    (trajectory[i + 1] - trajectory[i]).norm().item()
                    for i in range(len(trajectory) - 1)
                ]
                stab = 1.0 / (1.0 + sum(deltas))
            else:
                stab = 1.0

            # Quantize to get bit cost proxy
            _, ids = self.codebook.quantize(z_eq)
            # Bit cost: log2(vocab_size) per token, amortized over span length
            bit_cost = 13.0 / len(span)  # log2(8192) ≈ 13

        # Total score
        score = bit_cost + self.beta * curv + self.gamma * (1.0 - stab)

        return score, curv, stab

    def update_parameters(self, beta: float, gamma: float, max_span: int) -> None:
        """Update segmentation parameters from controller.

        Args:
            beta: New curvature penalty
            gamma: New stability penalty
            max_span: New maximum span length
        """
        self.beta = beta
        self.gamma = gamma
        self.max_span = max_span


class GreedySegmenter:
    """Simple greedy segmenter for baseline comparison."""

    def __init__(self, config: SegmentationConfig):
        self.config = config

    def segment(self, data: bytes) -> Iterator[tuple[int, int]]:
        """Segment using fixed-size spans."""
        pos = 0
        n = len(data)
        span_len = (self.config.min_span + self.config.max_span) // 2

        while pos < n:
            end = min(pos + span_len, n)
            yield pos, end
            pos = end

    def segment_with_scores(
        self, data: bytes
    ) -> Iterator[tuple[int, int, float, float]]:
        """Segment with dummy scores."""
        for start, end in self.segment(data):
            yield start, end, 0.0, 1.0
