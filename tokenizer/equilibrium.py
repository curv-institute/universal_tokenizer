"""Equilibrium projection - LoRE instantiation.

Implements deterministic contraction mapping for token assignment.
z_{t+1} = (1-eta) * z_t + eta * F(z_t, span_features)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .interfaces import EQConfig, ModelConfig


class EquilibriumProjector(nn.Module):
    """Equilibrium projection via contraction mapping.

    Implements LoRE instantiation - canonicalization to attractor states.
    """

    def __init__(self, config: EQConfig, model_config: ModelConfig):
        super().__init__()
        self.config = config
        self.latent_dim = model_config.latent_dim

        # Contraction network F(z, features)
        self.contract_net = nn.Sequential(
            nn.Linear(model_config.latent_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
            nn.Linear(model_config.hidden_dim, model_config.latent_dim),
        )

        # Optional feature conditioning
        self.feature_proj = nn.Linear(model_config.latent_dim, model_config.latent_dim)

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

    def project(self, z: Tensor, span_features: Tensor | None = None) -> Tensor:
        """Project latent to equilibrium attractor.

        Args:
            z: Initial latent tensor (batch, latent_dim)
            span_features: Optional conditioning features

        Returns:
            Equilibrium latent tensor (batch, latent_dim)
        """
        for _ in range(self.config.num_steps):
            z_next = self.forward(z, span_features)

            # Early stopping on convergence
            if torch.allclose(z, z_next, atol=self.config.tolerance):
                return z_next
            z = z_next

        return z

    def project_with_trajectory(
        self, z: Tensor, span_features: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        """Project with full trajectory for analysis.

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
