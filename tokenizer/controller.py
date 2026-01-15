"""Harmonizer controller - Curv Cycle implementation.

Closed-loop control that keeps tokenizer in stable operating regime.
Observes rolling statistics and adjusts admissibility parameters.

Supports HHC (Harmonized Hyper-Connections) when config.hhc.enabled and
config.hhc.apply_harmonizer are True. HHC tracks relational stats between
consecutive tokens and uses them to adjust control parameters.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .interfaces import ControllerConfig

if TYPE_CHECKING:
    from .interfaces import TokenizerConfig


logger = logging.getLogger(__name__)


@dataclass
class HHCState:
    """HHC-specific state for relational statistics."""

    # Rolling buffers for HHC stats
    neighbor_divergences: deque = field(default_factory=lambda: deque(maxlen=100))
    hhc_tensions: deque = field(default_factory=lambda: deque(maxlen=100))
    cross_scale_consistencies: deque = field(default_factory=lambda: deque(maxlen=100))

    # Parameter adjustment multipliers from HHC
    merge_threshold_adj: float = 1.0  # <1 = more aggressive merging
    curvature_penalty_adj: float = 1.0  # <1 = lower curvature penalty
    span_limit_adj: float = 1.0  # >1 = allow longer spans


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

    # HHC state (only used when HHC enabled)
    hhc: HHCState = field(default_factory=HHCState)


class HarmonizerController:
    """Harmonizer: adaptive control for tokenizer stability.

    Implements the Curv Cycle: representations persist and compose
    only if they can regulate their own curvature/stability under pressure.

    Control law:
        S_t = {avg_token_len, bpb, curvature_tail, stability_margin, token_churn}
        theta_t = {K_max, beta_curv, gamma_stab, max_span_len, split_merge_rate}
        theta_{t+1} = clamp(theta_t + G * (S_t - targets))

    When HHC is enabled (config.hhc.enabled and config.hhc.apply_harmonizer):
        - Tracks relational stats: neighbor_divergence, hhc_tension, cross_scale_consistency
        - Adjusts merge thresholds (more aggressive when neighbors similar)
        - Adjusts curvature penalties (lower when HHC tension is low)
        - Adjusts span limits based on cross-scale consistency
    """

    def __init__(
        self,
        config: ControllerConfig,
        *,
        full_config: TokenizerConfig | None = None,
    ):
        """Initialize the HarmonizerController.

        Args:
            config: Controller-specific configuration.
            full_config: Full tokenizer config (needed for HHC settings).
                         If None, HHC features are disabled.
        """
        self.config = config
        self._full_config = full_config
        self.state = ControllerState()

        # HHC settings - only enabled if full_config provided and HHC enabled
        self._hhc_enabled = (
            full_config is not None
            and full_config.hhc.enabled
            and full_config.hhc.apply_harmonizer
        )
        self._log_hhc_stats = (
            self._hhc_enabled
            and full_config is not None
            and full_config.hhc.log_hhc_stats
        )

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
                - neighbor_divergence (HHC): mean distance between consecutive latents
                - hhc_tension (HHC): variance of neighbor distances
                - cross_scale_consistency (HHC): stability across span lengths
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

        # HHC stats - only track when HHC is enabled
        if self._hhc_enabled:
            if "neighbor_divergence" in stats:
                self.state.hhc.neighbor_divergences.append(stats["neighbor_divergence"])
            if "hhc_tension" in stats:
                self.state.hhc.hhc_tensions.append(stats["hhc_tension"])
            if "cross_scale_consistency" in stats:
                self.state.hhc.cross_scale_consistencies.append(
                    stats["cross_scale_consistency"]
                )

    def get_parameters(self) -> dict[str, float]:
        """Get current control parameters.

        When HHC is enabled, parameters are adjusted by HHC multipliers:
        - beta_curv is scaled by curvature_penalty_adj
        - max_span_len is scaled by span_limit_adj
        - merge_threshold_adj affects split_merge_rate
        """
        if not self._hhc_enabled:
            # Baseline behavior - return raw parameters
            return {
                "K_max": self.state.K_max,
                "beta_curv": self.state.beta_curv,
                "gamma_stab": self.state.gamma_stab,
                "max_span_len": self.state.max_span_len,
                "split_merge_rate": self.state.split_merge_rate,
            }

        # HHC-adjusted parameters
        hhc = self.state.hhc
        adjusted_beta = self.state.beta_curv * hhc.curvature_penalty_adj
        adjusted_span = self.state.max_span_len * hhc.span_limit_adj
        # More aggressive merging (lower threshold) -> positive split_merge_rate
        adjusted_split_merge = self.state.split_merge_rate + (
            0.01 * (1.0 - hhc.merge_threshold_adj)
        )

        # Clamp adjusted values to bounds
        lo, hi = self.bounds["beta_curv"]
        adjusted_beta = max(lo, min(hi, adjusted_beta))
        lo, hi = self.bounds["max_span_len"]
        adjusted_span = max(lo, min(hi, adjusted_span))
        lo, hi = self.bounds["split_merge_rate"]
        adjusted_split_merge = max(lo, min(hi, adjusted_split_merge))

        return {
            "K_max": self.state.K_max,
            "beta_curv": adjusted_beta,
            "gamma_stab": self.state.gamma_stab,
            "max_span_len": adjusted_span,
            "split_merge_rate": adjusted_split_merge,
        }

    def step(self) -> dict[str, float]:
        """Perform control step and return updated parameters.

        When HHC is enabled, also updates HHC adjustment multipliers based on
        relational statistics (neighbor divergence, tension, cross-scale consistency).
        """
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

        # Update HHC adjustments if enabled
        if self._hhc_enabled:
            self._update_hhc_adjustments()

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

    def _compute_hhc_stats(self) -> dict[str, float]:
        """Compute current HHC statistics from buffers.

        Returns:
            Dictionary with:
                - mean_neighbor_divergence: average distance between consecutive latents
                - mean_hhc_tension: average variance of neighbor distances
                - mean_cross_scale_consistency: average stability across spans
        """
        stats = {}
        hhc = self.state.hhc

        if hhc.neighbor_divergences:
            stats["mean_neighbor_divergence"] = sum(hhc.neighbor_divergences) / len(
                hhc.neighbor_divergences
            )

        if hhc.hhc_tensions:
            stats["mean_hhc_tension"] = sum(hhc.hhc_tensions) / len(hhc.hhc_tensions)

        if hhc.cross_scale_consistencies:
            stats["mean_cross_scale_consistency"] = sum(
                hhc.cross_scale_consistencies
            ) / len(hhc.cross_scale_consistencies)

        return stats

    def _update_hhc_adjustments(self) -> None:
        """Update HHC adjustment multipliers based on relational statistics.

        HHC adjustments:
        1. merge_threshold_adj: Lower when neighbors are similar (low divergence)
           -> encourages more aggressive merging
        2. curvature_penalty_adj: Lower when HHC tension is low
           -> reduces curvature penalty in stable regions
        3. span_limit_adj: Higher when cross-scale consistency is high
           -> allows longer spans when stable across scales
        """
        hhc_stats = self._compute_hhc_stats()
        hhc = self.state.hhc
        max_delta = 0.25  # Max adjustment per step

        if self._full_config is not None:
            max_delta = self._full_config.hhc.max_delta_per_step

        # 1. Merge threshold adjustment based on neighbor divergence
        # Low divergence (similar neighbors) -> more aggressive merging (lower adj)
        if "mean_neighbor_divergence" in hhc_stats:
            divergence = hhc_stats["mean_neighbor_divergence"]
            # Normalize: assume typical divergence is around 1.0
            # If divergence < 1.0, reduce threshold (merge_threshold_adj < 1)
            # If divergence > 1.0, increase threshold (merge_threshold_adj > 1)
            target_adj = min(2.0, max(0.5, divergence))
            delta = target_adj - hhc.merge_threshold_adj
            delta = max(-max_delta, min(max_delta, delta))
            hhc.merge_threshold_adj += delta

        # 2. Curvature penalty adjustment based on HHC tension
        # Low tension -> lower curvature penalty (stable region)
        if "mean_hhc_tension" in hhc_stats:
            tension = hhc_stats["mean_hhc_tension"]
            # Normalize: assume typical tension is around 0.5
            # If tension < 0.5, reduce penalty (curvature_penalty_adj < 1)
            # If tension > 0.5, increase penalty (curvature_penalty_adj > 1)
            target_adj = min(2.0, max(0.5, 1.0 + (tension - 0.5)))
            delta = target_adj - hhc.curvature_penalty_adj
            delta = max(-max_delta, min(max_delta, delta))
            hhc.curvature_penalty_adj += delta

        # 3. Span limit adjustment based on cross-scale consistency
        # High consistency -> allow longer spans
        if "mean_cross_scale_consistency" in hhc_stats:
            consistency = hhc_stats["mean_cross_scale_consistency"]
            # Normalize: assume typical consistency is around 0.8
            # If consistency > 0.8, increase spans (span_limit_adj > 1)
            # If consistency < 0.8, decrease spans (span_limit_adj < 1)
            target_adj = min(1.5, max(0.75, 1.0 + (consistency - 0.8)))
            delta = target_adj - hhc.span_limit_adj
            delta = max(-max_delta, min(max_delta, delta))
            hhc.span_limit_adj += delta

        # Log HHC adjustments if enabled
        if self._log_hhc_stats:
            logger.info(
                "HHC adjustments: merge_threshold=%.3f, curvature_penalty=%.3f, "
                "span_limit=%.3f | stats: divergence=%.3f, tension=%.3f, "
                "consistency=%.3f",
                hhc.merge_threshold_adj,
                hhc.curvature_penalty_adj,
                hhc.span_limit_adj,
                hhc_stats.get("mean_neighbor_divergence", 0.0),
                hhc_stats.get("mean_hhc_tension", 0.0),
                hhc_stats.get("mean_cross_scale_consistency", 0.0),
            )

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
        state = {
            "K_max": self.state.K_max,
            "beta_curv": self.state.beta_curv,
            "gamma_stab": self.state.gamma_stab,
            "max_span_len": self.state.max_span_len,
            "split_merge_rate": self.state.split_merge_rate,
            "step_count": self.state.step_count,
        }

        # Include HHC state if HHC is enabled
        if self._hhc_enabled:
            state["hhc"] = {
                "merge_threshold_adj": self.state.hhc.merge_threshold_adj,
                "curvature_penalty_adj": self.state.hhc.curvature_penalty_adj,
                "span_limit_adj": self.state.hhc.span_limit_adj,
            }

        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load controller state."""
        self.state.K_max = state.get("K_max", 8192.0)
        self.state.beta_curv = state.get("beta_curv", 0.1)
        self.state.gamma_stab = state.get("gamma_stab", 0.1)
        self.state.max_span_len = state.get("max_span_len", 32.0)
        self.state.split_merge_rate = state.get("split_merge_rate", 0.0)
        self.state.step_count = state.get("step_count", 0)

        # Load HHC state if present and HHC is enabled
        if self._hhc_enabled and "hhc" in state:
            hhc_state = state["hhc"]
            self.state.hhc.merge_threshold_adj = hhc_state.get(
                "merge_threshold_adj", 1.0
            )
            self.state.hhc.curvature_penalty_adj = hhc_state.get(
                "curvature_penalty_adj", 1.0
            )
            self.state.hhc.span_limit_adj = hhc_state.get("span_limit_adj", 1.0)

    def get_hhc_adjustments(self) -> dict[str, float] | None:
        """Get current HHC adjustment multipliers.

        Returns:
            Dictionary with adjustment multipliers if HHC enabled, None otherwise.
        """
        if not self._hhc_enabled:
            return None

        return {
            "merge_threshold_adj": self.state.hhc.merge_threshold_adj,
            "curvature_penalty_adj": self.state.hhc.curvature_penalty_adj,
            "span_limit_adj": self.state.hhc.span_limit_adj,
        }

    @property
    def hhc_enabled(self) -> bool:
        """Return whether HHC is enabled for this controller."""
        return self._hhc_enabled
