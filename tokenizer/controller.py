"""Harmonizer controller - Curv Cycle implementation.

Closed-loop control that keeps tokenizer in stable operating regime.
Observes rolling statistics and adjusts admissibility parameters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .interfaces import ControllerConfig


@dataclass
class ControllerState:
    """Internal controller state."""

    # Rolling statistics buffers
    avg_token_lens: deque = field(default_factory=lambda: deque(maxlen=100))
    bpbs: deque = field(default_factory=lambda: deque(maxlen=100))
    curvatures: deque = field(default_factory=lambda: deque(maxlen=100))
    stabilities: deque = field(default_factory=lambda: deque(maxlen=100))
    churns: deque = field(default_factory=lambda: deque(maxlen=100))

    # Current parameters
    K_max: float = 8192.0
    beta_curv: float = 0.1
    gamma_stab: float = 0.1
    max_span_len: float = 32.0
    split_merge_rate: float = 0.0

    # Step counter
    step_count: int = 0


class HarmonizerController:
    """Harmonizer: adaptive control for tokenizer stability.

    Implements the Curv Cycle: representations persist and compose
    only if they can regulate their own curvature/stability under pressure.

    Control law:
        S_t = {avg_token_len, bpb, curvature_tail, stability_margin, token_churn}
        theta_t = {K_max, beta_curv, gamma_stab, max_span_len, split_merge_rate}
        theta_{t+1} = clamp(theta_t + G * (S_t - targets))
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.state = ControllerState()

        # Target setpoints
        self.targets = {
            "avg_token_len": config.target_avg_len,
            "bits_per_byte": config.target_bpb,
            "curvature_tail": config.target_curvature,
            "stability_margin": 0.8,  # Want high stability
            "token_churn": 0.1,  # Want low churn
        }

        # Parameter bounds
        self.bounds = {
            "K_max": (256, 65536),
            "beta_curv": (0.001, 1.0),
            "gamma_stab": (0.001, 1.0),
            "max_span_len": (4, 128),
            "split_merge_rate": (-0.1, 0.1),
        }

    def observe(self, stats: dict[str, float]) -> None:
        """Observe current statistics.

        Args:
            stats: Dictionary with keys like:
                - avg_token_len
                - bits_per_byte
                - residual_bpb
                - curvature_tail
                - stability_margin
                - token_churn
        """
        if "avg_token_len" in stats:
            self.state.avg_token_lens.append(stats["avg_token_len"])
        if "bits_per_byte" in stats:
            self.state.bpbs.append(stats["bits_per_byte"])
        if "curvature_tail" in stats:
            self.state.curvatures.append(stats["curvature_tail"])
        if "stability_margin" in stats:
            self.state.stabilities.append(stats["stability_margin"])
        if "token_churn" in stats:
            self.state.churns.append(stats["token_churn"])

    def get_parameters(self) -> dict[str, float]:
        """Get current control parameters."""
        return {
            "K_max": self.state.K_max,
            "beta_curv": self.state.beta_curv,
            "gamma_stab": self.state.gamma_stab,
            "max_span_len": self.state.max_span_len,
            "split_merge_rate": self.state.split_merge_rate,
        }

    def step(self) -> dict[str, float]:
        """Perform control step and return updated parameters."""
        self.state.step_count += 1

        # Only update at intervals
        if self.state.step_count % self.config.update_interval != 0:
            return self.get_parameters()

        # Compute current statistics
        current = self._compute_current_stats()

        # Compute errors
        errors = {k: current.get(k, 0) - self.targets.get(k, 0) for k in self.targets}

        # Apply control law
        gain = self.config.gain

        # Adjust beta_curv based on curvature
        if "curvature_tail" in errors:
            self.state.beta_curv += gain * errors["curvature_tail"]

        # Adjust gamma_stab based on stability
        if "stability_margin" in errors:
            # Negative error means stability is low, increase penalty
            self.state.gamma_stab -= gain * errors["stability_margin"]

        # Adjust max_span_len based on avg_token_len
        if "avg_token_len" in errors:
            # If tokens too short, allow longer spans
            self.state.max_span_len += gain * 10 * errors["avg_token_len"]

        # Clamp all parameters to bounds
        self._clamp_parameters()

        return self.get_parameters()

    def _compute_current_stats(self) -> dict[str, float]:
        """Compute current statistics from buffers."""
        stats = {}

        if self.state.avg_token_lens:
            stats["avg_token_len"] = sum(self.state.avg_token_lens) / len(
                self.state.avg_token_lens
            )

        if self.state.bpbs:
            stats["bits_per_byte"] = sum(self.state.bpbs) / len(self.state.bpbs)

        if self.state.curvatures:
            # 90th percentile for tail
            sorted_curv = sorted(self.state.curvatures)
            idx = int(0.9 * len(sorted_curv))
            stats["curvature_tail"] = sorted_curv[min(idx, len(sorted_curv) - 1)]

        if self.state.stabilities:
            # Minimum stability (worst case)
            stats["stability_margin"] = min(self.state.stabilities)

        if self.state.churns:
            stats["token_churn"] = sum(self.state.churns) / len(self.state.churns)

        return stats

    def _clamp_parameters(self) -> None:
        """Clamp parameters to valid bounds."""
        lo, hi = self.bounds["K_max"]
        self.state.K_max = max(lo, min(hi, self.state.K_max))

        lo, hi = self.bounds["beta_curv"]
        self.state.beta_curv = max(lo, min(hi, self.state.beta_curv))

        lo, hi = self.bounds["gamma_stab"]
        self.state.gamma_stab = max(lo, min(hi, self.state.gamma_stab))

        lo, hi = self.bounds["max_span_len"]
        self.state.max_span_len = max(lo, min(hi, self.state.max_span_len))

        lo, hi = self.bounds["split_merge_rate"]
        self.state.split_merge_rate = max(lo, min(hi, self.state.split_merge_rate))

    def state_dict(self) -> dict[str, Any]:
        """Return controller state."""
        return {
            "K_max": self.state.K_max,
            "beta_curv": self.state.beta_curv,
            "gamma_stab": self.state.gamma_stab,
            "max_span_len": self.state.max_span_len,
            "split_merge_rate": self.state.split_merge_rate,
            "step_count": self.state.step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load controller state."""
        self.state.K_max = state.get("K_max", 8192.0)
        self.state.beta_curv = state.get("beta_curv", 0.1)
        self.state.gamma_stab = state.get("gamma_stab", 0.1)
        self.state.max_span_len = state.get("max_span_len", 32.0)
        self.state.split_merge_rate = state.get("split_merge_rate", 0.0)
        self.state.step_count = state.get("step_count", 0)
