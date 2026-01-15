"""Equilibrium projection - LoRE instantiation.

Implements deterministic contraction mapping for token assignment.
z_{t+1} = (1-eta) * z_t + eta * F(z_t, span_features)

With optional HHC (Harmonized Hyper-Connections) for neighbor coupling:
z* = project(z) + lambda * mean(clamp_norm(neighbors, max_norm) - z)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .interfaces import EQConfig, HHCConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class HHCStats:
    """Statistics for HHC diagnostics."""

    mean_neighbor_distance: float = 0.0
    neighbor_distance_variance: float = 0.0
    num_neighbors: int = 0
    delta_applied: float = 0.0


class EquilibriumProjector(nn.Module):
    """Equilibrium projection via contraction mapping.

    Implements LoRE instantiation - canonicalization to attractor states.

    With HHC enabled, applies weak coupling to neighbor latents:
    z* = project(z) + lambda * mean(neighbors - z)
    """

    def __init__(
        self,
        config: EQConfig,
        model_config: ModelConfig,
        hhc_config: HHCConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.latent_dim = model_config.latent_dim
        self.hhc_config = hhc_config or HHCConfig()

        # Contraction network F(z, features)
        self.contract_net = nn.Sequential(
            nn.Linear(model_config.latent_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
            nn.Linear(model_config.hidden_dim, model_config.latent_dim),
        )

        # Optional feature conditioning
        self.feature_proj = nn.Linear(model_config.latent_dim, model_config.latent_dim)

        # HHC state: sliding window of neighbor latents
        self._neighbor_window: deque[Tensor] = deque(
            maxlen=self.hhc_config.window_tokens
        )
        self._last_hhc_stats: HHCStats | None = None

    def forward(self, z: Tensor, span_features: Tensor | None = None) -> Tensor:
        """Single contraction step.

        Args:
            z: Current latent (batch, latent_dim)
            span_features: Optional conditioning

        Returns:
            Updated latent after one step
        """
        # Compute F(z)
        f_z = self.contract_net(z)

        # Add feature conditioning if provided
        if span_features is not None:
            f_z = f_z + self.feature_proj(span_features)

        # Contraction: z_{t+1} = (1-eta)*z_t + eta*F(z_t)
        eta = self.config.eta
        if self.config.use_residual:
            return (1 - eta) * z + eta * f_z
        else:
            return eta * f_z

    def set_neighbors(self, neighbors: list[Tensor] | None) -> None:
        """Set neighbor latents for HHC coupling.

        Args:
            neighbors: List of neighbor latent tensors, or None to clear.
                       Each tensor should be shape (latent_dim,) or (1, latent_dim).
        """
        self._neighbor_window.clear()
        if neighbors is not None:
            for n in neighbors:
                # Normalize shape to (latent_dim,)
                if n.dim() == 2 and n.size(0) == 1:
                    n = n.squeeze(0)
                self._neighbor_window.append(n.detach().clone())

    def add_neighbor(self, neighbor: Tensor) -> None:
        """Add a single neighbor latent to the sliding window.

        Args:
            neighbor: Neighbor latent tensor, shape (latent_dim,) or (1, latent_dim).
        """
        if neighbor.dim() == 2 and neighbor.size(0) == 1:
            neighbor = neighbor.squeeze(0)
        self._neighbor_window.append(neighbor.detach().clone())

    def clear_neighbors(self) -> None:
        """Clear all neighbor latents from the window."""
        self._neighbor_window.clear()
        self._last_hhc_stats = None

    def get_hhc_stats(self) -> HHCStats | None:
        """Get the most recent HHC statistics.

        Returns:
            HHCStats from the last project() call, or None if HHC was not applied.
        """
        return self._last_hhc_stats

    def _compute_hhc_coupling(
        self, z: Tensor, neighbors: list[Tensor] | None = None
    ) -> tuple[Tensor, HHCStats]:
        """Compute HHC neighbor coupling term.

        Applies the formula: delta = lambda_equilibrium * (z_bar - z)
        Where z_bar = mean(clamp_norm(neighbors, max_neighbor_norm))

        Args:
            z: Projected latent tensor (batch, latent_dim) or (latent_dim,)
            neighbors: Optional list of neighbor latent tensors. If provided,
                       uses these instead of the internal _neighbor_window.

        Returns:
            Tuple of (coupling delta, HHC statistics)
        """
        # Handle both batched and unbatched input
        was_unbatched = z.dim() == 1
        if was_unbatched:
            z = z.unsqueeze(0)

        device = z.device
        dtype = z.dtype

        # Use provided neighbors or fall back to internal window
        if neighbors is not None and len(neighbors) > 0:
            neighbor_list = neighbors
        elif len(self._neighbor_window) > 0:
            neighbor_list = list(self._neighbor_window)
        else:
            neighbor_list = []

        if len(neighbor_list) == 0:
            # No neighbors - return zero delta (bitwise identical to baseline)
            stats = HHCStats(
                mean_neighbor_distance=0.0,
                neighbor_distance_variance=0.0,
                num_neighbors=0,
                delta_applied=0.0,
            )
            delta = torch.zeros_like(z)
            if was_unbatched:
                delta = delta.squeeze(0)
            return delta, stats

        # Stack neighbors: (num_neighbors, latent_dim)
        # Normalize shapes to (latent_dim,) before stacking
        normalized_neighbors = []
        for n in neighbor_list:
            n_tensor = n.to(device=device, dtype=dtype)
            if n_tensor.dim() == 2 and n_tensor.size(0) == 1:
                n_tensor = n_tensor.squeeze(0)
            elif n_tensor.dim() == 0:
                continue  # Skip scalar tensors
            normalized_neighbors.append(n_tensor)

        if len(normalized_neighbors) == 0:
            stats = HHCStats(
                mean_neighbor_distance=0.0,
                neighbor_distance_variance=0.0,
                num_neighbors=0,
                delta_applied=0.0,
            )
            delta = torch.zeros_like(z)
            if was_unbatched:
                delta = delta.squeeze(0)
            return delta, stats

        neighbors_tensor = torch.stack(normalized_neighbors)  # (num_neighbors, latent_dim)
        num_neighbors = neighbors_tensor.size(0)

        # Step 1: Clamp each neighbor's norm by max_neighbor_norm
        # This prevents any single neighbor from dominating
        neighbor_norms = neighbors_tensor.norm(dim=-1, keepdim=True)  # (num_neighbors, 1)
        max_norm = self.hhc_config.max_neighbor_norm
        norm_scale = torch.clamp(max_norm / (neighbor_norms + 1e-8), max=1.0)
        neighbors_clamped = neighbors_tensor * norm_scale  # (num_neighbors, latent_dim)

        # Step 2: Compute z_bar = mean of clamped neighbors
        z_bar = neighbors_clamped.mean(dim=0, keepdim=True)  # (1, latent_dim)

        # Step 3: Compute delta = lambda * (z_bar - z)
        # z: (batch, latent_dim), z_bar: (1, latent_dim) broadcasts
        diff = z_bar - z  # (batch, latent_dim)
        delta = self.hhc_config.lambda_equilibrium * diff

        # Step 4: Clamp delta magnitude by max_delta_per_step
        delta_norm = delta.norm(dim=-1, keepdim=True)
        max_delta = self.hhc_config.max_delta_per_step
        delta_scale = torch.clamp(max_delta / (delta_norm + 1e-8), max=1.0)
        delta = delta * delta_scale

        # Compute statistics for diagnostics
        # Distance from z to each neighbor (before clamping)
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)
        neighbors_expanded = neighbors_tensor.unsqueeze(0)  # (1, num_neighbors, latent_dim)
        distances = (neighbors_expanded - z_expanded).norm(dim=-1)  # (batch, num_neighbors)

        mean_dist = distances.mean().item()
        var_dist = distances.var().item() if num_neighbors > 1 else 0.0
        delta_applied = delta.norm(dim=-1).mean().item()

        stats = HHCStats(
            mean_neighbor_distance=mean_dist,
            neighbor_distance_variance=var_dist,
            num_neighbors=num_neighbors,
            delta_applied=delta_applied,
        )

        if was_unbatched:
            delta = delta.squeeze(0)

        return delta, stats

    def project(
        self,
        z: Tensor,
        span_features: Tensor | None = None,
        neighbors: list[Tensor] | None = None,
    ) -> Tensor:
        """Project latent to equilibrium attractor.

        When HHC is enabled (config.hhc.enabled and config.hhc.apply_equilibrium),
        applies weak coupling to neighbors after projection:
        z* = project(z) + lambda * (z_bar - z)
        where z_bar = mean(clamp_norm(neighbors, max_neighbor_norm))

        Args:
            z: Initial latent tensor (batch, latent_dim)
            span_features: Optional conditioning features
            neighbors: Optional list of neighbor latent tensors for HHC coupling.
                       If provided and HHC is enabled, these are used for coupling.
                       If None, falls back to internal _neighbor_window.

        Returns:
            Equilibrium latent tensor (batch, latent_dim)
        """
        # Standard equilibrium projection
        for _ in range(self.config.num_steps):
            z_next = self.forward(z, span_features)

            # Early stopping on convergence
            if torch.allclose(z, z_next, atol=self.config.tolerance):
                z = z_next
                break
            z = z_next

        # Apply HHC coupling if enabled and neighbors are available
        if self.hhc_config.enabled and self.hhc_config.apply_equilibrium:
            delta, stats = self._compute_hhc_coupling(z, neighbors)

            # Only apply delta if we actually have neighbors
            if stats.num_neighbors > 0:
                z = z + delta

            # Store stats and optionally log
            if self.hhc_config.log_hhc_stats:
                self._last_hhc_stats = stats
                if stats.num_neighbors > 0:
                    logger.debug(
                        "HHC coupling applied: num_neighbors=%d, "
                        "mean_neighbor_distance=%.4f, delta_magnitude=%.4f",
                        stats.num_neighbors,
                        stats.mean_neighbor_distance,
                        stats.delta_applied,
                    )
                else:
                    logger.debug("HHC enabled but no neighbors provided")
            else:
                self._last_hhc_stats = stats
        else:
            self._last_hhc_stats = None

        return z

    def project_with_trajectory(
        self,
        z: Tensor,
        span_features: Tensor | None = None,
        neighbors: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Project with full trajectory for analysis.

        When HHC is enabled, the final state includes HHC coupling.
        The trajectory shows the contraction steps before HHC is applied.

        Args:
            z: Initial latent tensor (batch, latent_dim)
            span_features: Optional conditioning features
            neighbors: Optional list of neighbor latent tensors for HHC coupling

        Returns:
            Final equilibrium and list of intermediate states
        """
        trajectory = [z.clone()]

        for _ in range(self.config.num_steps):
            z = self.forward(z, span_features)
            trajectory.append(z.clone())

            # Early stopping
            if len(trajectory) > 1:
                if torch.allclose(trajectory[-1], trajectory[-2], atol=self.config.tolerance):
                    break

        # Apply HHC coupling if enabled and neighbors available
        if self.hhc_config.enabled and self.hhc_config.apply_equilibrium:
            delta, stats = self._compute_hhc_coupling(z, neighbors)

            if stats.num_neighbors > 0:
                z = z + delta
                trajectory.append(z.clone())  # Include HHC-adjusted state in trajectory

            if self.hhc_config.log_hhc_stats:
                self._last_hhc_stats = stats
                if stats.num_neighbors > 0:
                    logger.debug(
                        "HHC trajectory coupling: num_neighbors=%d, "
                        "mean_neighbor_distance=%.4f, delta_magnitude=%.4f",
                        stats.num_neighbors,
                        stats.mean_neighbor_distance,
                        stats.delta_applied,
                    )
            else:
                self._last_hhc_stats = stats
        else:
            self._last_hhc_stats = None

        return z, trajectory

    def contraction_factor(self, z: Tensor) -> Tensor:
        """Estimate local contraction factor (Lipschitz constant proxy).

        Args:
            z: Latent points (batch, latent_dim)

        Returns:
            Contraction factors (batch,)
        """
        z.requires_grad_(True)
        z_next = self.forward(z)

        # Compute Jacobian norm via random projection (Hutchinson)
        v = torch.randn_like(z)
        v = v / v.norm(dim=-1, keepdim=True)

        jvp = torch.autograd.grad(
            z_next, z, grad_outputs=v, create_graph=False, retain_graph=False
        )[0]

        return (jvp * v).sum(dim=-1).abs()

    def state_dict(self) -> dict[str, Any]:
        return {k: v.cpu() for k, v in super().state_dict().items()}

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        super().load_state_dict(state, strict=strict)
