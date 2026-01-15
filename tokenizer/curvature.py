"""Curvature estimation - GRIT proxy.

Implements local sensitivity/brittleness diagnostics.
Curvature correlates with unstable merges and token churn.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .interfaces import ModelConfig


class CurvatureEstimator(nn.Module):
    """Curvature estimator using Jacobian-based diagnostics.

    Implements GRIT proxy - measures local sensitivity of the representation.
    High curvature indicates unstable regions prone to token churn.
    """

    def __init__(self, config: ModelConfig, num_hutchinson_samples: int = 4):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.num_samples = num_hutchinson_samples

        # Curvature prediction network (fast approximation)
        self.curv_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Softplus(),  # Ensure non-negative
        )

    def estimate(self, z: Tensor, context: Tensor | None = None) -> Tensor:
        """Estimate curvature at latent points.

        Uses learned approximation for efficiency.

        Args:
            z: Latent tensor (batch, latent_dim)
            context: Optional context tensor (unused, for protocol)

        Returns:
            Curvature scores (batch,)
        """
        return self.curv_net(z).squeeze(-1)

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
