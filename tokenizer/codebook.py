"""Token codebook with EMA updates.

Maps equilibrium latents to discrete token IDs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .interfaces import CodebookConfig


class TokenCodebook(nn.Module):
    """Codebook for vector quantization.

    Maps continuous latent vectors to discrete token IDs.
    Uses EMA updates for stable training.
    """

    def __init__(self, config: CodebookConfig):
        super().__init__()
        self.config = config

        # Codebook vectors
        self.codes = nn.Parameter(
            torch.randn(config.num_codes, config.code_dim) * config.init_scale
        )

        # EMA tracking for updates
        self.register_buffer("ema_count", torch.zeros(config.num_codes))
        self.register_buffer("ema_weight", self.codes.data.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def num_codes(self) -> int:
        return self.config.num_codes

    @property
    def code_dim(self) -> int:
        return self.config.code_dim

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize latent vectors to codebook.

        Args:
            z: Latent tensor (batch, latent_dim)

        Returns:
            Tuple of (quantized vectors, token IDs)
        """
        # Compute distances to all codes
        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2*z·c
        z_sq = (z ** 2).sum(dim=-1, keepdim=True)  # (batch, 1)
        c_sq = (self.codes ** 2).sum(dim=-1)  # (num_codes,)
        dots = z @ self.codes.T  # (batch, num_codes)

        distances = z_sq + c_sq - 2 * dots  # (batch, num_codes)

        # Find nearest codes
        ids = distances.argmin(dim=-1)  # (batch,)

        # Lookup quantized vectors
        quantized = self.lookup(ids)

        return quantized, ids

    def lookup(self, ids: Tensor) -> Tensor:
        """Lookup codebook vectors by ID.

        Args:
            ids: Token ID tensor (batch,)

        Returns:
            Codebook vectors (batch, code_dim)
        """
        return F.embedding(ids, self.codes)

    def update_ema(self, z: Tensor, ids: Tensor) -> None:
        """Update codebook with EMA.

        Args:
            z: Encoded vectors (batch, code_dim)
            ids: Assigned token IDs (batch,)
        """
        if not self.training:
            return

        with torch.no_grad():
            # Count assignments
            one_hot = F.one_hot(ids, self.num_codes).float()  # (batch, num_codes)
            counts = one_hot.sum(dim=0)  # (num_codes,)

            # Sum of assigned vectors
            sums = one_hot.T @ z  # (num_codes, code_dim)

            # EMA update
            decay = self.config.ema_decay
            self.ema_count.mul_(decay).add_(counts, alpha=1 - decay)
            self.ema_weight.mul_(decay).add_(sums, alpha=1 - decay)

            # Update codes
            n = self.ema_count.unsqueeze(-1).clamp(min=1e-5)
            self.codes.data.copy_(self.ema_weight / n)

    def commitment_loss(self, z: Tensor, quantized: Tensor) -> Tensor:
        """Compute commitment loss.

        Args:
            z: Original latent vectors
            quantized: Quantized vectors (detached)

        Returns:
            Commitment loss scalar
        """
        return self.config.commitment_weight * F.mse_loss(z, quantized.detach())

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass with straight-through estimator.

        Args:
            z: Latent vectors (batch, latent_dim)

        Returns:
            Tuple of (quantized with gradients, ids, commitment loss)
        """
        quantized, ids = self.quantize(z)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # Update EMA during training
        if self.training:
            self.update_ema(z, ids)

        # Commitment loss
        commit_loss = self.commitment_loss(z, quantized)

        return quantized_st, ids, commit_loss

    def state_dict(self) -> dict[str, Any]:
        """Return codebook state."""
        return {
            "codes": self.codes.data.cpu(),
            "ema_count": self.ema_count.cpu(),
            "ema_weight": self.ema_weight.cpu(),
            "initialized": self.initialized.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load codebook state."""
        self.codes.data.copy_(state["codes"])
        self.ema_count.copy_(state["ema_count"])
        self.ema_weight.copy_(state["ema_weight"])
        self.initialized.copy_(state.get("initialized", torch.tensor(True)))
