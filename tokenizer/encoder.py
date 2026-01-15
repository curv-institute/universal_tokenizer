"""Byte span encoder - MLR instantiation.

Maps byte spans to latent representations z in R^d.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .interfaces import ModelConfig


class ByteEncoder(nn.Module):
    """Neural encoder for byte spans.

    Implements the MLR instantiation - measurement device into representation space.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Byte embedding
        self.embed = nn.Embedding(config.input_dim, config.embed_dim)

        # Position encoding (learnable)
        self.max_len = 256
        self.pos_embed = nn.Embedding(self.max_len, config.embed_dim)

        # Encoder layers
        layers = []
        in_dim = config.embed_dim
        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else config.latent_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU() if i < config.num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # Pooling projection
        self.pool_proj = nn.Linear(config.latent_dim, config.latent_dim)

    def forward(self, byte_ids: Tensor, lengths: Tensor) -> Tensor:
        """Encode batched byte sequences.

        Args:
            byte_ids: (batch, max_len) byte indices
            lengths: (batch,) actual lengths

        Returns:
            (batch, latent_dim) latent vectors
        """
        batch_size, max_len = byte_ids.shape
        device = byte_ids.device

        # Embed bytes
        x = self.embed(byte_ids)  # (batch, max_len, embed_dim)

        # Add position embeddings
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(positions)

        # Encode
        x = self.encoder(x)  # (batch, max_len, latent_dim)

        # Mean pooling with length mask
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch, max_len, 1)

        pooled = (x * mask).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
        return self.pool_proj(pooled)

    def encode(self, spans: list[bytes]) -> Tensor:
        """Encode byte spans to latent vectors.

        Args:
            spans: List of byte sequences

        Returns:
            Tensor of shape (batch, latent_dim)
        """
        if not spans:
            return torch.zeros(0, self.config.latent_dim)

        # Prepare batch
        lengths = torch.tensor([len(s) for s in spans])
        max_len = min(int(lengths.max().item()), self.max_len)

        byte_ids = torch.zeros(len(spans), max_len, dtype=torch.long)
        for i, span in enumerate(spans):
            span_len = min(len(span), max_len)
            byte_ids[i, :span_len] = torch.tensor(list(span[:span_len]), dtype=torch.long)

        # Move to same device as model
        device = next(self.parameters()).device
        byte_ids = byte_ids.to(device)
        lengths = lengths.to(device).clamp(max=max_len)

        with torch.no_grad():
            return self.forward(byte_ids, lengths)

    def state_dict(self) -> dict[str, Any]:
        """Return model state."""
        return {k: v.cpu() for k, v in super().state_dict().items()}

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        """Load model state."""
        super().load_state_dict(state, strict=strict)
