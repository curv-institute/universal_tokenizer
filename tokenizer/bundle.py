"""Tokenizer bundle - complete system assembly.

Assembles all components into the complete Universal Lossless Tokenizer.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterator, Any

import torch
from torch import Tensor

from .interfaces import (
    TokenizerConfig,
    TokenizerBundle,
    Token,
    EncodeResult,
    DecodeResult,
)
from .encoder import ByteEncoder
from .equilibrium import EquilibriumProjector
from .codebook import TokenCodebook
from .curvature import CurvatureEstimator
from .segmenter import StreamSegmenter
from .controller import HarmonizerController
from .residuals import ResidualCoder

__version__ = "0.1.0"


class UniversalTokenizer:
    """Complete Universal Lossless Tokenizer.

    Assembles all components for encoding and decoding with
    bit-exact reconstruction via residual coding.
    """

    def __init__(self, config: TokenizerConfig):
        self._config = config

        # Initialize components with HHC config
        self.encoder = ByteEncoder(config.model)
        self.equilibrium = EquilibriumProjector(
            config.equilibrium, config.model, config.hhc
        )
        self.codebook = TokenCodebook(config.codebook)
        self.curvature = CurvatureEstimator(config.model, hhc_config=config.hhc)
        self.controller = HarmonizerController(config.controller)
        self.residual_coder = ResidualCoder()

        # Segmenter needs references to other components
        self.segmenter = StreamSegmenter(
            config.segmentation,
            self.encoder,
            self.equilibrium,
            self.codebook,
            self.curvature,
        )

        # Streaming-local neighbor buffer for HHC
        # Stores (token_id, span_len, z_canonical) tuples
        self._neighbor_buffer: deque[tuple[int, int, Tensor]] = deque(
            maxlen=config.hhc.window_tokens
        )

        # Device management
        self._device = torch.device(config.train.device)
        self.to(self._device)

    @property
    def config(self) -> TokenizerConfig:
        return self._config

    def to(self, device: torch.device | str) -> "UniversalTokenizer":
        """Move all components to device."""
        device = torch.device(device)
        self._device = device

        self.encoder.to(device)
        self.equilibrium.to(device)
        self.codebook.to(device)
        self.curvature.to(device)

        return self

    def train_mode(self, mode: bool = True) -> "UniversalTokenizer":
        """Set training mode for all components."""
        self.encoder.train(mode)
        self.equilibrium.train(mode)
        self.codebook.train(mode)
        self.curvature.train(mode)
        return self

    def eval_mode(self) -> "UniversalTokenizer":
        """Set evaluation mode."""
        return self.train_mode(False)

    def _get_neighbor_latents(self) -> Tensor | None:
        """Get stacked neighbor latents from buffer.

        Returns:
            Tensor of shape (n, latent_dim) if buffer has entries,
            None if buffer is empty.
        """
        if not self._neighbor_buffer:
            return None

        # Stack the z_canonical tensors from buffer
        latents = [entry[2] for entry in self._neighbor_buffer]
        return torch.stack(latents, dim=0)

    def _update_neighbor_buffer(
        self, token_id: int, span_len: int, z_canonical: Tensor
    ) -> None:
        """Add a committed token to the neighbor buffer.

        Args:
            token_id: The token ID
            span_len: Length of the span in bytes
            z_canonical: The canonical latent tensor (latent_dim,)
        """
        # Ensure tensor is detached and on CPU for storage efficiency
        z_stored = z_canonical.detach().clone()
        if z_stored.dim() == 2 and z_stored.size(0) == 1:
            z_stored = z_stored.squeeze(0)
        self._neighbor_buffer.append((token_id, span_len, z_stored))

    def _set_neighbor_context(self) -> None:
        """Set neighbor latents on equilibrium and curvature components.

        Passes current buffer contents to components that use HHC.
        """
        if not self._config.hhc.enabled:
            return

        neighbor_latents = self._get_neighbor_latents()

        if neighbor_latents is not None:
            # Move to device for computation
            neighbor_latents = neighbor_latents.to(self._device)

            # Set on equilibrium projector
            if self._config.hhc.apply_equilibrium:
                self.equilibrium.set_neighbors(list(neighbor_latents))

            # Set on curvature estimator
            if self._config.hhc.apply_curvature:
                self.curvature.set_neighbors(neighbor_latents)
        else:
            # Clear neighbors if buffer is empty
            self.equilibrium.clear_neighbors()
            self.curvature.clear_neighbors()

    def encode(self, data: bytes) -> EncodeResult:
        """Encode bytes to tokens with lossless residuals.

        Args:
            data: Input byte sequence

        Returns:
            EncodeResult with tokens, IDs, and residuals
        """
        self.eval_mode()

        tokens = []
        all_residuals = b""

        # Clear neighbor buffer at start of encode
        if self._config.hhc.enabled:
            self._neighbor_buffer.clear()
            self.equilibrium.clear_neighbors()
            self.curvature.clear_neighbors()

        # Segment and encode
        with torch.no_grad():
            for start, end, curv, stab in self.segmenter.segment_with_scores(data):
                span = data[start:end]

                # Encode span to latent
                z = self.encoder.encode([span])

                # Set neighbor context before projection (if HHC enabled)
                if self._config.hhc.enabled:
                    self._set_neighbor_context()

                # Project to equilibrium (uses neighbors if HHC enabled)
                z_eq = self.equilibrium.project(z)

                # Quantize to token
                _, ids = self.codebook.quantize(z_eq)
                token_id = ids[0].item()

                # Update neighbor buffer with committed token (if HHC enabled)
                if self._config.hhc.enabled:
                    self._update_neighbor_buffer(token_id, len(span), z_eq)

                # For reconstruction, we need the residual
                # Since decoder doesn't exist yet, store original span
                residual = self.residual_coder.encode_residual(span, span)
                all_residuals += residual

                tokens.append(
                    Token(
                        id=token_id,
                        span=span,
                        start=start,
                        end=end,
                        residual=residual,
                        curvature=curv,
                        stability=stab,
                    )
                )

                # Update controller with stats
                self.controller.observe(
                    {
                        "avg_token_len": len(span),
                        "curvature_tail": curv,
                        "stability_margin": stab,
                    }
                )

        # Controller step
        params = self.controller.step()
        self.segmenter.update_parameters(
            params["beta_curv"],
            params["gamma_stab"],
            int(params["max_span_len"]),
        )

        return EncodeResult(
            tokens=tokens,
            ids=[t.id for t in tokens],
            residuals=all_residuals,
            metadata={
                "version": __version__,
                "num_tokens": len(tokens),
                "total_bytes": len(data),
                "controller_params": params,
            },
        )

    def decode(self, result: EncodeResult) -> DecodeResult:
        """Decode tokens to exact original bytes.

        Uses residuals for bit-exact reconstruction.

        Args:
            result: EncodeResult from encode()

        Returns:
            DecodeResult with reconstructed bytes
        """
        data = b""

        for token in result.tokens:
            # Use residual for exact reconstruction
            reconstructed = self.residual_coder.decode_residual(
                token.span,  # Approximate (same as original for now)
                token.residual,
            )
            data += reconstructed

        return DecodeResult(
            data=data,
            metadata={
                "version": __version__,
                "num_tokens": len(result.tokens),
            },
        )

    def encode_streaming(self, stream: Iterator[bytes]) -> Iterator[Token]:
        """Streaming encode.

        Args:
            stream: Iterator yielding byte chunks

        Yields:
            Token objects
        """
        self.eval_mode()
        buffer = b""

        # Clear neighbor buffer at start of new stream (if HHC enabled)
        if self._config.hhc.enabled:
            self._neighbor_buffer.clear()
            self.equilibrium.clear_neighbors()
            self.curvature.clear_neighbors()

        with torch.no_grad():
            for chunk in stream:
                buffer += chunk

                # Process complete segments
                while len(buffer) >= self._config.segmentation.min_span:
                    # Get first segment
                    segments = list(self.segmenter.segment_with_scores(buffer))
                    if not segments:
                        break

                    start, end, curv, stab = segments[0]

                    # Only yield if we have enough buffer
                    if end > len(buffer):
                        break

                    span = buffer[start:end]
                    z = self.encoder.encode([span])

                    # Set neighbor context before projection (if HHC enabled)
                    if self._config.hhc.enabled:
                        self._set_neighbor_context()

                    z_eq = self.equilibrium.project(z)
                    _, ids = self.codebook.quantize(z_eq)
                    token_id = ids[0].item()

                    # Update neighbor buffer with committed token (if HHC enabled)
                    if self._config.hhc.enabled:
                        self._update_neighbor_buffer(token_id, len(span), z_eq)

                    residual = self.residual_coder.encode_residual(span, span)

                    yield Token(
                        id=token_id,
                        span=span,
                        start=start,
                        end=end,
                        residual=residual,
                        curvature=curv,
                        stability=stab,
                    )

                    buffer = buffer[end:]

            # Flush remaining buffer
            if buffer:
                for start, end, curv, stab in self.segmenter.segment_with_scores(buffer):
                    span = buffer[start:end]
                    z = self.encoder.encode([span])

                    # Set neighbor context before projection (if HHC enabled)
                    if self._config.hhc.enabled:
                        self._set_neighbor_context()

                    z_eq = self.equilibrium.project(z)
                    _, ids = self.codebook.quantize(z_eq)
                    token_id = ids[0].item()

                    # Update neighbor buffer with committed token (if HHC enabled)
                    if self._config.hhc.enabled:
                        self._update_neighbor_buffer(token_id, len(span), z_eq)

                    residual = self.residual_coder.encode_residual(span, span)

                    yield Token(
                        id=token_id,
                        span=span,
                        start=start,
                        end=end,
                        residual=residual,
                        curvature=curv,
                        stability=stab,
                    )

    def save(self, path: Path | str) -> None:
        """Save tokenizer state."""
        bundle = self.get_bundle()
        bundle.save(path)

    @classmethod
    def load(cls, path: Path | str) -> "UniversalTokenizer":
        """Load tokenizer from path."""
        bundle = TokenizerBundle.load(path)

        tokenizer = cls(bundle.config)
        tokenizer.encoder.load_state_dict(bundle.model_state.get("encoder", {}))
        tokenizer.equilibrium.load_state_dict(bundle.model_state.get("equilibrium", {}))
        tokenizer.codebook.load_state_dict(bundle.codebook_state)
        tokenizer.curvature.load_state_dict(bundle.model_state.get("curvature", {}))
        tokenizer.controller.load_state_dict(bundle.controller_state)

        return tokenizer

    def get_bundle(self) -> TokenizerBundle:
        """Get complete tokenizer bundle."""
        return TokenizerBundle(
            version=__version__,
            config=self._config,
            model_state={
                "encoder": self.encoder.state_dict(),
                "equilibrium": self.equilibrium.state_dict(),
                "curvature": self.curvature.state_dict(),
            },
            codebook_state=self.codebook.state_dict(),
            controller_state=self.controller.state_dict(),
            metadata={"device": str(self._device)},
        )

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Get all trainable parameters."""
        yield from self.encoder.parameters()
        yield from self.equilibrium.parameters()
        yield from self.codebook.parameters()
        yield from self.curvature.parameters()
