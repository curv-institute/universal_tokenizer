"""Curvature estimation - GRIT proxy.

Implements local sensitivity/brittleness diagnostics.
Curvature correlates with unstable merges and token churn.

HHC Extension:
When HHC is enabled (config.hhc.enabled and config.hhc.apply_curvature),
curvature includes relational disharmony:
    K_hhc(z) = K_local(z) + alpha * Var_neighbors(||z - z_r||)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .interfaces import ModelConfig, HHCConfig

logger = logging.getLogger(__name__)


class CurvatureEstimator(nn.Module):
    """Curvature estimator using Jacobian-based diagnostics.

    Implements GRIT proxy - measures local sensitivity of the representation.
    High curvature indicates unstable regions prone to token churn.

    When HHC is enabled, extends curvature to include relational disharmony:
        K_hhc(z) = K_local(z) + alpha * Var_neighbors(||z - z_r||)
    """

    def __init__(
        self,
        config: ModelConfig,
        num_hutchinson_samples: int = 4,
        hhc_config: HHCConfig | None = None,
        neighbors: Tensor | None = None,
    ):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.num_samples = num_hutchinson_samples

        # HHC configuration (defaults to disabled)
        self.hhc_config = hhc_config if hhc_config is not None else HHCConfig()

        # Neighbor latents for HHC relational disharmony
        # Shape: (num_neighbors, latent_dim) or None
        self._neighbor_latents: Tensor | None = None
        if neighbors is not None:
            self.set_neighbors(neighbors)

        # HHC statistics tracking (legacy format for backward compat)
        self._hhc_stats: dict[str, list[float]] = {
            "hhc_contribution_mean": [],
            "hhc_contribution_max": [],
            "neighbor_variance_mean": [],
            "local_curvature_mean": [],
        }

        # New HHC diagnostics per-call tracking
        self._hhc_diagnostics: list[dict[str, float]] = []

        # Curvature prediction network (fast approximation)
        self.curv_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Softplus(),  # Ensure non-negative
        )

    def set_neighbors(self, neighbors: Tensor) -> None:
        """Set neighbor latents for HHC relational disharmony computation.

        Args:
            neighbors: Neighbor latent tensor (num_neighbors, latent_dim)
        """
        if neighbors.dim() != 2:
            raise ValueError(f"neighbors must be 2D, got {neighbors.dim()}D")
        if neighbors.shape[1] != self.latent_dim:
            raise ValueError(
                f"neighbors latent_dim {neighbors.shape[1]} != {self.latent_dim}"
            )
        self._neighbor_latents = neighbors.detach()

    def clear_neighbors(self) -> None:
        """Clear neighbor latents."""
        self._neighbor_latents = None

    def get_hhc_stats(self) -> dict[str, Any]:
        """Get aggregated HHC statistics.

        Returns:
            Dictionary with aggregated stats:
            - hhc_active_fraction: fraction of calls with neighbors > 0
            - hhc_mean_num_neighbors: mean number of neighbors used
            - hhc_mean_curvature_delta: mean K_hhc - K_local
            - hhc_nonzero_delta_fraction: fraction of calls with delta > 0
            - Legacy stats (hhc_contribution_mean, etc.) also included
        """
        result: dict[str, Any] = {}

        # Aggregated diagnostics from new per-call tracking
        if self._hhc_diagnostics:
            num_calls = len(self._hhc_diagnostics)
            active_calls = sum(
                1 for d in self._hhc_diagnostics if d.get("num_neighbors", 0) > 0
            )
            nonzero_delta_calls = sum(
                1 for d in self._hhc_diagnostics if d.get("hhc_curvature_delta", 0) > 0
            )

            result["hhc_active_fraction"] = active_calls / num_calls
            result["hhc_mean_num_neighbors"] = sum(
                d.get("num_neighbors", 0) for d in self._hhc_diagnostics
            ) / num_calls
            result["hhc_mean_curvature_delta"] = sum(
                d.get("hhc_curvature_delta", 0) for d in self._hhc_diagnostics
            ) / num_calls
            result["hhc_nonzero_delta_fraction"] = nonzero_delta_calls / num_calls
            result["hhc_mean_disharmony"] = sum(
                d.get("disharmony", 0) for d in self._hhc_diagnostics
            ) / num_calls
        else:
            result["hhc_active_fraction"] = 0.0
            result["hhc_mean_num_neighbors"] = 0.0
            result["hhc_mean_curvature_delta"] = 0.0
            result["hhc_nonzero_delta_fraction"] = 0.0
            result["hhc_mean_disharmony"] = 0.0

        # Include legacy stats
        result["_legacy_stats"] = self._hhc_stats.copy()

        return result

    def get_hhc_diagnostics(self) -> list[dict[str, float]]:
        """Get per-call HHC diagnostics.

        Returns:
            List of per-call diagnostic dicts with:
            - hhc_curvature_delta: K_hhc - K_local
            - num_neighbors: number of neighbors used
            - disharmony: Var_r(||z - z_r||) value
        """
        return self._hhc_diagnostics.copy()

    def reset_hhc_stats(self) -> None:
        """Reset HHC statistics tracking."""
        for key in self._hhc_stats:
            self._hhc_stats[key] = []
        self._hhc_diagnostics = []

    def _compute_disharmony(
        self, z: Tensor, neighbors: Tensor | None = None
    ) -> tuple[Tensor, int]:
        """Compute disharmony = Var_r(||z - z_r||) for neighbor latents.

        Args:
            z: Latent tensor (batch, latent_dim)
            neighbors: Optional neighbor latent tensor (num_neighbors, latent_dim).
                       If None, falls back to self._neighbor_latents.

        Returns:
            Tuple of (disharmony tensor (batch,), num_neighbors used)
        """
        # Determine which neighbors to use: explicit param > stored neighbors
        neighbor_latents = neighbors if neighbors is not None else self._neighbor_latents

        if neighbor_latents is None or neighbor_latents.shape[0] == 0:
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype), 0

        num_neighbors = neighbor_latents.shape[0]

        # Move neighbors to same device as z
        neighbor_latents = neighbor_latents.to(z.device)

        # Compute distances: (batch, num_neighbors)
        # z: (batch, latent_dim), neighbors: (num_neighbors, latent_dim)
        # Expand for broadcasting: z -> (batch, 1, latent_dim), neighbors -> (1, num_neighbors, latent_dim)
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)
        neighbors_expanded = neighbor_latents.unsqueeze(0)  # (1, num_neighbors, latent_dim)

        # L2 distances to each neighbor
        distances = (z_expanded - neighbors_expanded).norm(dim=-1)  # (batch, num_neighbors)

        # Variance across neighbors for each sample (disharmony)
        disharmony = distances.var(dim=-1)  # (batch,)

        return disharmony, num_neighbors

    def estimate(
        self, z: Tensor, context: Tensor | None = None, neighbors: Tensor | None = None
    ) -> Tensor:
        """Estimate curvature at latent points.

        Uses learned approximation for efficiency. When HHC is enabled
        (hhc_config.enabled and hhc_config.apply_curvature) and neighbors
        are provided, extends curvature with relational disharmony:
            K_hhc(z) = K_local(z) + alpha * disharmony
        where disharmony = Var_r(||z - z_r||) for each z_r in neighbors.

        Args:
            z: Latent tensor (batch, latent_dim)
            context: Optional context tensor (unused, for protocol)
            neighbors: Optional neighbor latent tensor (num_neighbors, latent_dim).
                       If None, falls back to stored neighbors from set_neighbors().

        Returns:
            Curvature scores (batch,)
        """
        # Local curvature (baseline)
        k_local = self.curv_net(z).squeeze(-1)

        # If HHC disabled, return baseline (bitwise identical)
        if not self.hhc_config.enabled or not self.hhc_config.apply_curvature:
            return k_local

        # HHC: Compute relational disharmony
        alpha = self.hhc_config.alpha_curvature
        disharmony, num_neighbors = self._compute_disharmony(z, neighbors)

        # Only add HHC contribution if we have neighbors
        if num_neighbors > 0:
            hhc_contribution = alpha * disharmony
            k_hhc = k_local + hhc_contribution
        else:
            hhc_contribution = torch.zeros_like(k_local)
            k_hhc = k_local

        # Track per-call diagnostics
        with torch.no_grad():
            curvature_delta = (k_hhc - k_local).mean().item()
            disharmony_mean = disharmony.mean().item() if num_neighbors > 0 else 0.0

            self._hhc_diagnostics.append({
                "hhc_curvature_delta": curvature_delta,
                "num_neighbors": float(num_neighbors),
                "disharmony": disharmony_mean,
            })

        # Log HHC stats if enabled (legacy format)
        if self.hhc_config.log_hhc_stats:
            with torch.no_grad():
                self._hhc_stats["hhc_contribution_mean"].append(
                    hhc_contribution.mean().item()
                )
                self._hhc_stats["hhc_contribution_max"].append(
                    hhc_contribution.max().item()
                )
                self._hhc_stats["neighbor_variance_mean"].append(
                    disharmony.mean().item() if num_neighbors > 0 else 0.0
                )
                self._hhc_stats["local_curvature_mean"].append(k_local.mean().item())

                logger.debug(
                    "HHC curvature: local=%.4f, hhc_contrib=%.4f, total=%.4f, neighbors=%d",
                    k_local.mean().item(),
                    hhc_contribution.mean().item(),
                    k_hhc.mean().item(),
                    num_neighbors,
                )

        return k_hhc

    def estimate_jacobian(
        self, z: Tensor, transform_fn: callable
    ) -> Tensor:
        """Estimate curvature via Jacobian norm (Hutchinson trace estimator).

        More accurate but slower than learned approximation.

        Args:
            z: Latent tensor (batch, latent_dim)
            transform_fn: Function mapping z -> z' (e.g., equilibrium step)

        Returns:
            Curvature estimates (batch,)
        """
        batch_size = z.shape[0]
        device = z.device

        z = z.requires_grad_(True)

        # Hutchinson trace estimator for ||J||_F^2
        trace_est = torch.zeros(batch_size, device=device)

        for _ in range(self.num_samples):
            # Random probe vector
            v = torch.randn_like(z)
            v = v / v.norm(dim=-1, keepdim=True)

            # Compute J @ v via forward-mode AD (JVP)
            z_out = transform_fn(z)
            jvp = torch.autograd.grad(
                z_out, z, grad_outputs=v, create_graph=False, retain_graph=True
            )[0]

            # ||J @ v||^2 estimates ||J||_F^2 / d
            trace_est += (jvp ** 2).sum(dim=-1)

        # Average over samples and normalize
        trace_est = trace_est / self.num_samples

        return trace_est.sqrt()  # Return Frobenius norm estimate

    def contraction_deficit(
        self, z: Tensor, z_next: Tensor, target_contraction: float = 0.9
    ) -> Tensor:
        """Compute contraction deficit (how far from target contraction).

        Args:
            z: Current latent
            z_next: Next latent after one step
            target_contraction: Desired contraction factor

        Returns:
            Deficit scores (batch,) - higher means more problematic
        """
        # Measure actual contraction
        delta = (z_next - z).norm(dim=-1)
        z_norm = z.norm(dim=-1).clamp(min=1e-6)

        actual_contraction = delta / z_norm

        # Deficit is how much we exceed target
        deficit = (actual_contraction - target_contraction).clamp(min=0)

        return deficit

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass - same as estimate()."""
        return self.estimate(z)

    def state_dict(self) -> dict[str, Any]:
        return {k: v.cpu() for k, v in super().state_dict().items()}

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        super().load_state_dict(state, strict=strict)
