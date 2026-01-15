"""Equilibrium projection - LoRE instantiation.

Implements deterministic contraction mapping for token assignment.
z_{t+1} = (1-eta) * z_t + eta * F(z_t, span_features)

With optional HHC (Harmonized Hyper-Connections) for neighbor coupling:
z* = project(z) + lambda * mean(neighbors - z)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .interfaces import EQConfig, HHCConfig, ModelConfig


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

    def _compute_hhc_coupling(self, z: Tensor) -> tuple[Tensor, HHCStats]:
        """Compute HHC neighbor coupling term.

        Args:
            z: Projected latent tensor (batch, latent_dim) or (latent_dim,)

        Returns:
            Tuple of (coupling delta, HHC statistics)
        """
        # Handle both batched and unbatched input
        was_unbatched = z.dim() == 1
        if was_unbatched:
            z = z.unsqueeze(0)

        batch_size = z.size(0)
        device = z.device
        dtype = z.dtype

        if len(self._neighbor_window) == 0:
            # No neighbors - return zero delta
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
        neighbors = torch.stack(list(self._neighbor_window)).to(device=device, dtype=dtype)
        num_neighbors = neighbors.size(0)

        # Compute distances from z to each neighbor
        # z: (batch, latent_dim), neighbors: (num_neighbors, latent_dim)
        # Expand for broadcasting: z becomes (batch, 1, latent_dim)
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)
        neighbors_expanded = neighbors.unsqueeze(0)  # (1, num_neighbors, latent_dim)

        # Differences: (batch, num_neighbors, latent_dim)
        diffs = neighbors_expanded - z_expanded

        # Clamp neighbor contributions by max_neighbor_norm
        diff_norms = diffs.norm(dim=-1, keepdim=True)  # (batch, num_neighbors, 1)
        max_norm = self.hhc_config.max_neighbor_norm
        scale = torch.clamp(max_norm / (diff_norms + 1e-8), max=1.0)
        diffs_clamped = diffs * scale

        # Mean coupling: lambda * mean(neighbors - z)
        mean_diff = diffs_clamped.mean(dim=1)  # (batch, latent_dim)

        # Apply lambda scaling
        delta = self.hhc_config.lambda_equilibrium * mean_diff

        # Clamp delta by max_delta_per_step
        delta_norm = delta.norm(dim=-1, keepdim=True)
        max_delta = self.hhc_config.max_delta_per_step
        delta_scale = torch.clamp(max_delta / (delta_norm + 1e-8), max=1.0)
        delta = delta * delta_scale

        # Compute statistics
        distances = diff_norms.squeeze(-1)  # (batch, num_neighbors)
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

    def project(self, z: Tensor, span_features: Tensor | None = None) -> Tensor:
        """Project latent to equilibrium attractor.

        When HHC is enabled (config.hhc.enabled and config.hhc.apply_equilibrium),
        applies weak coupling to neighbors after projection:
        z* = project(z) + lambda * mean(neighbors - z)

        Args:
            z: Initial latent tensor (batch, latent_dim)
            span_features: Optional conditioning features

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

        # Apply HHC coupling if enabled
        if self.hhc_config.enabled and self.hhc_config.apply_equilibrium:
            delta, stats = self._compute_hhc_coupling(z)
            z = z + delta

            if self.hhc_config.log_hhc_stats:
                self._last_hhc_stats = stats
        else:
            self._last_hhc_stats = None

        return z

    def project_with_trajectory(
        self, z: Tensor, span_features: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        """Project with full trajectory for analysis.

        When HHC is enabled, the final state includes HHC coupling.
        The trajectory shows the contraction steps before HHC is applied.

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

        # Apply HHC coupling if enabled
        if self.hhc_config.enabled and self.hhc_config.apply_equilibrium:
            delta, stats = self._compute_hhc_coupling(z)
            z = z + delta
            trajectory.append(z.clone())  # Include HHC-adjusted state in trajectory

            if self.hhc_config.log_hhc_stats:
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
