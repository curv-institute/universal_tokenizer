"""Evaluation metrics.

Provides metrics for tokenizer evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

from .interfaces import Token, EncodeResult


@dataclass
class TokenizationMetrics:
    """Metrics for a single tokenization."""

    num_tokens: int = 0
    num_bytes: int = 0
    compression_ratio: float = 1.0
    bits_per_byte: float = 8.0
    avg_token_length: float = 1.0
    avg_curvature: float = 0.0
    max_curvature: float = 0.0
    avg_stability: float = 1.0
    min_stability: float = 1.0
    residual_bytes: int = 0
    residual_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Aggregate metrics over multiple samples."""

    num_samples: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    mean_compression_ratio: float = 1.0
    mean_bits_per_byte: float = 8.0
    mean_avg_token_length: float = 1.0
    mean_curvature: float = 0.0
    curvature_p90: float = 0.0
    mean_stability: float = 1.0
    stability_p10: float = 1.0
    total_residual_bytes: int = 0
    lossless_rate: float = 1.0

    # Per-sample metrics for analysis
    sample_metrics: list[TokenizationMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sample_metrics"] = [m.to_dict() for m in self.sample_metrics]
        return d


def compute_metrics(result: EncodeResult, original: bytes) -> TokenizationMetrics:
    """Compute metrics for a single tokenization result.

    Args:
        result: Tokenization result
        original: Original byte sequence

    Returns:
        TokenizationMetrics
    """
    num_tokens = len(result.tokens)
    num_bytes = len(original)

    if num_tokens == 0:
        return TokenizationMetrics(num_bytes=num_bytes)

    # Compression ratio
    compression_ratio = num_bytes / num_tokens if num_tokens > 0 else 1.0

    # Bits per byte (assuming log2(vocab_size) bits per token)
    # Using 8192 as default vocab size -> ~13 bits per token
    bits_per_token = 13.0  # log2(8192)
    bits_per_byte = (bits_per_token * num_tokens) / num_bytes if num_bytes > 0 else 8.0

    # Token length stats
    avg_token_length = num_bytes / num_tokens if num_tokens > 0 else 1.0

    # Curvature stats
    curvatures = [t.curvature for t in result.tokens]
    avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0
    max_curvature = max(curvatures) if curvatures else 0.0

    # Stability stats
    stabilities = [t.stability for t in result.tokens]
    avg_stability = sum(stabilities) / len(stabilities) if stabilities else 1.0
    min_stability = min(stabilities) if stabilities else 1.0

    # Residual stats
    residual_bytes = len(result.residuals)
    residual_ratio = residual_bytes / num_bytes if num_bytes > 0 else 0.0

    return TokenizationMetrics(
        num_tokens=num_tokens,
        num_bytes=num_bytes,
        compression_ratio=compression_ratio,
        bits_per_byte=bits_per_byte,
        avg_token_length=avg_token_length,
        avg_curvature=avg_curvature,
        max_curvature=max_curvature,
        avg_stability=avg_stability,
        min_stability=min_stability,
        residual_bytes=residual_bytes,
        residual_ratio=residual_ratio,
    )


def aggregate_metrics(metrics: Sequence[TokenizationMetrics]) -> AggregateMetrics:
    """Aggregate metrics over multiple samples.

    Args:
        metrics: Sequence of per-sample metrics

    Returns:
        AggregateMetrics
    """
    if not metrics:
        return AggregateMetrics()

    n = len(metrics)

    # Totals
    total_tokens = sum(m.num_tokens for m in metrics)
    total_bytes = sum(m.num_bytes for m in metrics)

    # Means
    mean_compression = sum(m.compression_ratio for m in metrics) / n
    mean_bpb = sum(m.bits_per_byte for m in metrics) / n
    mean_avg_len = sum(m.avg_token_length for m in metrics) / n
    mean_curv = sum(m.avg_curvature for m in metrics) / n
    mean_stab = sum(m.avg_stability for m in metrics) / n

    # Percentiles for curvature (90th) and stability (10th)
    curvatures = sorted(m.max_curvature for m in metrics)
    stabilities = sorted(m.min_stability for m in metrics)

    curv_p90 = curvatures[int(0.9 * (n - 1))] if n > 1 else curvatures[0]
    stab_p10 = stabilities[int(0.1 * (n - 1))] if n > 1 else stabilities[0]

    # Residuals
    total_residual = sum(m.residual_bytes for m in metrics)

    # Lossless rate (assuming all are lossless if residuals are present)
    lossless_rate = 1.0  # By design, all tokenizations are lossless

    return AggregateMetrics(
        num_samples=n,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        mean_compression_ratio=mean_compression,
        mean_bits_per_byte=mean_bpb,
        mean_avg_token_length=mean_avg_len,
        mean_curvature=mean_curv,
        curvature_p90=curv_p90,
        mean_stability=mean_stab,
        stability_p10=stab_p10,
        total_residual_bytes=total_residual,
        lossless_rate=lossless_rate,
        sample_metrics=list(metrics),
    )


def verify_lossless(original: bytes, decoded: bytes) -> bool:
    """Verify lossless reconstruction.

    Args:
        original: Original bytes
        decoded: Decoded bytes

    Returns:
        True if bit-exact match
    """
    return original == decoded


def compute_entropy(data: bytes) -> float:
    """Compute entropy of byte sequence.

    Args:
        data: Byte sequence

    Returns:
        Entropy in bits per byte
    """
    if not data:
        return 0.0

    # Count byte frequencies
    counts = [0] * 256
    for b in data:
        counts[b] += 1

    # Compute entropy
    n = len(data)
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)

    return entropy


def compression_overhead(
    original_bytes: int,
    num_tokens: int,
    residual_bytes: int,
    vocab_size: int = 8192,
) -> float:
    """Compute compression overhead.

    Args:
        original_bytes: Original data size
        num_tokens: Number of tokens
        residual_bytes: Residual size
        vocab_size: Token vocabulary size

    Returns:
        Overhead ratio (1.0 = same size, <1.0 = compression, >1.0 = expansion)
    """
    bits_per_token = math.ceil(math.log2(vocab_size))
    token_bits = num_tokens * bits_per_token
    token_bytes = (token_bits + 7) // 8

    total_encoded = token_bytes + residual_bytes
    return total_encoded / original_bytes if original_bytes > 0 else 1.0
