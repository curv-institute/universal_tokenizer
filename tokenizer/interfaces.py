"""Interfaces for Universal Lossless Tokenizer.

This file is the SOURCE OF TRUTH. All implementations must conform to these
protocols and dataclasses. No implementation may violate these contracts.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol, Iterator, Any, runtime_checkable

import torch
from torch import Tensor


# =============================================================================
# A) Configuration Models
# =============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Encoder model configuration."""

    input_dim: int = 256  # Byte vocabulary size
    embed_dim: int = 128  # Embedding dimension
    hidden_dim: int = 256  # Hidden layer dimension
    latent_dim: int = 64  # Latent space dimension
    num_layers: int = 2  # Number of encoder layers
    dropout: float = 0.0  # Dropout rate (0 for determinism)


@dataclass(frozen=True)
class EQConfig:
    """Equilibrium projection configuration."""

    num_steps: int = 10  # Fixed unroll steps for contraction
    eta: float = 0.5  # Contraction rate: z_{t+1} = (1-eta)*z_t + eta*F(z_t)
    tolerance: float = 1e-6  # Convergence tolerance
    use_residual: bool = True  # Use residual connections


@dataclass(frozen=True)
class CodebookConfig:
    """Codebook configuration."""

    num_codes: int = 8192  # Number of tokens/attractors
    code_dim: int = 64  # Must match latent_dim
    init_scale: float = 0.1  # Initialization scale
    ema_decay: float = 0.99  # EMA decay for codebook updates
    commitment_weight: float = 0.25  # Commitment loss weight


@dataclass(frozen=True)
class SegmentationConfig:
    """Segmentation/dynamics configuration."""

    min_span: int = 1  # Minimum span length (bytes)
    max_span: int = 32  # Maximum span length (bytes)
    beta_curvature: float = 0.1  # Curvature penalty weight
    gamma_stability: float = 0.1  # Stability penalty weight
    lookahead: int = 4  # Lookahead for boundary selection


@dataclass(frozen=True)
class ControllerConfig:
    """Harmonizer/controller configuration."""

    target_avg_len: float = 4.0  # Target average token length
    target_bpb: float = 1.0  # Target bits per byte
    target_curvature: float = 0.5  # Target curvature bound
    gain: float = 0.01  # Control gain
    update_interval: int = 100  # Steps between updates


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_steps: int = 10000
    warmup_steps: int = 100
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    seed: int = 42
    device: str = "cpu"  # "cpu" or "cuda"


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 64
    max_samples: int = 1000
    seed: int = 42


@dataclass(frozen=True)
class HHCConfig:
    """Harmonized Hyper-Connections configuration."""

    enabled: bool = False
    apply_equilibrium: bool = True
    apply_curvature: bool = True
    apply_harmonizer: bool = True
    window_tokens: int = 8
    overlap_bytes: int = 16
    adjacency: bool = True
    lambda_equilibrium: float = 0.05
    alpha_curvature: float = 0.10
    max_neighbor_norm: float = 10.0
    max_delta_per_step: float = 0.25
    log_hhc_stats: bool = True


@dataclass
class TokenizerConfig:
    """Complete tokenizer configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    equilibrium: EQConfig = field(default_factory=EQConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hhc: HHCConfig = field(default_factory=HHCConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenizerConfig:
        """Create from dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            equilibrium=EQConfig(**d.get("equilibrium", {})),
            codebook=CodebookConfig(**d.get("codebook", {})),
            segmentation=SegmentationConfig(**d.get("segmentation", {})),
            controller=ControllerConfig(**d.get("controller", {})),
            train=TrainConfig(**d.get("train", {})),
            eval=EvalConfig(**d.get("eval", {})),
            hhc=HHCConfig(**d.get("hhc", {})),
        )


# =============================================================================
# C) Tokenization I/O
# =============================================================================


@dataclass
class Token:
    """A single token with metadata."""

    id: int  # Token ID (codebook index)
    span: bytes  # Original byte span
    start: int  # Start position in input
    end: int  # End position in input
    residual: bytes = b""  # Residual for lossless reconstruction
    curvature: float = 0.0  # Curvature at this token
    stability: float = 1.0  # Stability score


@dataclass
class EncodeResult:
    """Result of encoding a byte sequence."""

    tokens: list[Token]  # Token sequence
    ids: list[int]  # Token IDs only
    residuals: bytes  # Concatenated residuals
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def compression_ratio(self) -> float:
        if not self.tokens:
            return 1.0
        total_bytes = sum(len(t.span) for t in self.tokens)
        return total_bytes / self.num_tokens if self.num_tokens > 0 else 1.0


@dataclass
class DecodeResult:
    """Result of decoding tokens back to bytes."""

    data: bytes  # Reconstructed byte sequence
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# B) Bundle Model
# =============================================================================


@dataclass
class TokenizerBundle:
    """Complete tokenizer state for save/load."""

    version: str
    config: TokenizerConfig
    model_state: dict[str, Any]
    codebook_state: dict[str, Any]
    controller_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        """Save bundle to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": self.version,
                "config": self.config.to_dict(),
                "model_state": self.model_state,
                "codebook_state": self.codebook_state,
                "controller_state": self.controller_state,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> TokenizerBundle:
        """Load bundle from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            version=data["version"],
            config=TokenizerConfig.from_dict(data["config"]),
            model_state=data["model_state"],
            codebook_state=data["codebook_state"],
            controller_state=data.get("controller_state", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# D) Component Protocols
# =============================================================================


@runtime_checkable
class Encoder(Protocol):
    """Protocol for byte span encoder.

    Maps byte spans to latent representations z in R^d.
    This is the MLR instantiation - the measurement device into representation space.
    """

    @abstractmethod
    def encode(self, spans: list[bytes]) -> Tensor:
        """Encode byte spans to latent vectors.

        Args:
            spans: List of byte sequences

        Returns:
            Tensor of shape (batch, latent_dim)
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return model state."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load model state."""
        ...


@runtime_checkable
class EquilibriumProjector(Protocol):
    """Protocol for equilibrium projection.

    Implements LoRE instantiation as deterministic contraction mapping:
    z_{t+1} = (1-eta) * z_t + eta * F(z_t, span_features)
    """

    @abstractmethod
    def project(self, z: Tensor, span_features: Tensor | None = None) -> Tensor:
        """Project latent to equilibrium attractor.

        Args:
            z: Initial latent tensor (batch, latent_dim)
            span_features: Optional conditioning features

        Returns:
            Equilibrium latent tensor (batch, latent_dim)
        """
        ...

    @abstractmethod
    def project_with_trajectory(
        self, z: Tensor, span_features: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        """Project with full trajectory for analysis.

        Returns:
            Final equilibrium and list of intermediate states
        """
        ...


@runtime_checkable
class Codebook(Protocol):
    """Protocol for token codebook.

    Maps equilibrium latents to discrete token IDs.
    """

    @abstractmethod
    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize latent vectors to codebook.

        Args:
            z: Latent tensor (batch, latent_dim)

        Returns:
            Tuple of (quantized vectors, token IDs)
        """
        ...

    @abstractmethod
    def lookup(self, ids: Tensor) -> Tensor:
        """Lookup codebook vectors by ID.

        Args:
            ids: Token ID tensor (batch,)

        Returns:
            Codebook vectors (batch, code_dim)
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return codebook state."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load codebook state."""
        ...


@runtime_checkable
class CurvatureEstimator(Protocol):
    """Protocol for curvature/inconsistency estimation.

    Implements GRIT proxy as local sensitivity diagnostic.
    Curvature should correlate with unstable merges and token churn.
    """

    @abstractmethod
    def estimate(self, z: Tensor, context: Tensor | None = None) -> Tensor:
        """Estimate curvature at latent points.

        Args:
            z: Latent tensor (batch, latent_dim)
            context: Optional context tensor

        Returns:
            Curvature scores (batch,)
        """
        ...


@runtime_checkable
class ResidualCoder(Protocol):
    """Protocol for residual coding.

    Encodes/decodes residual information for lossless reconstruction.
    """

    @abstractmethod
    def encode_residual(self, original: bytes, reconstructed: bytes) -> bytes:
        """Encode residual between original and reconstructed.

        Args:
            original: Original byte sequence
            reconstructed: Reconstructed from token

        Returns:
            Residual bytes
        """
        ...

    @abstractmethod
    def decode_residual(self, reconstructed: bytes, residual: bytes) -> bytes:
        """Decode using residual for exact reconstruction.

        Args:
            reconstructed: Reconstructed from token
            residual: Residual bytes

        Returns:
            Exact original bytes
        """
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Protocol for stream segmentation.

    Implements DIRT instantiation - selecting span boundaries as low-action path.
    Optimizes: bit_cost + beta*curvature + gamma*stability_penalty
    """

    @abstractmethod
    def segment(self, data: bytes) -> Iterator[tuple[int, int]]:
        """Segment byte stream into spans.

        Args:
            data: Input byte sequence

        Yields:
            (start, end) tuples for each span
        """
        ...

    @abstractmethod
    def segment_with_scores(
        self, data: bytes
    ) -> Iterator[tuple[int, int, float, float]]:
        """Segment with curvature and stability scores.

        Yields:
            (start, end, curvature, stability) tuples
        """
        ...


@runtime_checkable
class Controller(Protocol):
    """Protocol for Harmonizer controller.

    The closed-loop control layer that keeps the tokenizer in stable operation.
    Observes rolling statistics and adjusts admissibility parameters.
    """

    @abstractmethod
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
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, float]:
        """Get current control parameters.

        Returns:
            Dictionary with keys like:
                - K_max (max codebook size active)
                - beta_curv (curvature penalty)
                - gamma_stab (stability penalty)
                - max_span_len
                - split_merge_rate
        """
        ...

    @abstractmethod
    def step(self) -> dict[str, float]:
        """Perform control step and return updated parameters."""
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return controller state."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load controller state."""
        ...


# =============================================================================
# E) BaselineTokenizer Protocol
# =============================================================================


@runtime_checkable
class BaselineTokenizer(Protocol):
    """Protocol for baseline tokenizers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tokenizer name."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""
        ...

    @abstractmethod
    def encode(self, data: bytes) -> EncodeResult:
        """Encode bytes to tokens."""
        ...

    @abstractmethod
    def decode(self, result: EncodeResult) -> DecodeResult:
        """Decode tokens to bytes."""
        ...


# =============================================================================
# Full Tokenizer Protocol
# =============================================================================


@runtime_checkable
class UniversalTokenizer(Protocol):
    """Protocol for the complete Universal Lossless Tokenizer."""

    @property
    @abstractmethod
    def config(self) -> TokenizerConfig:
        """Get tokenizer configuration."""
        ...

    @abstractmethod
    def encode(self, data: bytes) -> EncodeResult:
        """Encode bytes to tokens with lossless residuals."""
        ...

    @abstractmethod
    def decode(self, result: EncodeResult) -> DecodeResult:
        """Decode tokens to exact original bytes."""
        ...

    @abstractmethod
    def encode_streaming(self, stream: Iterator[bytes]) -> Iterator[Token]:
        """Streaming encode."""
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save tokenizer state."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> UniversalTokenizer:
        """Load tokenizer from path."""
        ...

    @abstractmethod
    def get_bundle(self) -> TokenizerBundle:
        """Get complete tokenizer bundle."""
        ...
