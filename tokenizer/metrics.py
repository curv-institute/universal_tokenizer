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
    bits_per_byte: float = 8.0  # DEPRECATED: Use end_to_end_bpb instead
    end_to_end_bpb: float = 8.0  # PRIMARY: Full compression metric including residuals
    structural_bpb: float = 8.0  # SECONDARY: Token representation efficiency (excludes residuals)
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
    mean_bits_per_byte: float = 8.0  # DEPRECATED: Use mean_end_to_end_bpb instead
    mean_end_to_end_bpb: float = 8.0  # PRIMARY: Full compression metric including residuals
    mean_structural_bpb: float = 8.0  # SECONDARY: Token representation efficiency
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


def compute_end_to_end_bpb(
    num_tokens: int,
    residual_bytes: int,
    original_bytes: int,
    vocab_size: int = 8192,
) -> float:
    """Compute end-to-end bits per byte including residuals.

    This is the PRIMARY metric for compression claims. It represents the
    actual bits needed to losslessly represent the original data.

    Formula: (ceil(log2(vocab_size)) * num_tokens + residual_bytes * 8) / original_bytes

    IMPORTANT: This metric MUST be >= empirical entropy for valid lossless compression.
    Values below entropy indicate a bug in the computation.

    Args:
        num_tokens: Number of tokens in encoding
        residual_bytes: Number of residual bytes for lossless reconstruction
        original_bytes: Original data size in bytes
        vocab_size: Token vocabulary size (default 8192)

    Returns:
        End-to-end bits per byte (must be >= entropy for valid compression)
    """
    if original_bytes == 0:
        return 8.0

    bits_per_token = math.ceil(math.log2(vocab_size))  # 13 bits for vocab_size=8192
    token_bits = bits_per_token * num_tokens
    residual_bits = residual_bytes * 8

    return (token_bits + residual_bits) / original_bytes


def compute_structural_bpb(
    num_tokens: int,
    original_bytes: int,
    vocab_size: int = 8192,
) -> float:
    """Compute structural bits per byte (token representation efficiency).

    This is a SECONDARY metric that measures how efficiently the tokenizer
    represents the data structure, EXCLUDING residual correction bytes.

    Formula: (ceil(log2(vocab_size)) * num_tokens) / original_bytes

    NOTE: This metric CAN be less than entropy - it is NOT a compression claim.
    It measures representational efficiency before residual correction.

    Args:
        num_tokens: Number of tokens in encoding
        original_bytes: Original data size in bytes
        vocab_size: Token vocabulary size (default 8192)

    Returns:
        Structural bits per byte (can be < entropy, not a compression claim)
    """
    if original_bytes == 0:
        return 8.0

    bits_per_token = math.ceil(math.log2(vocab_size))  # 13 bits for vocab_size=8192
    token_bits = bits_per_token * num_tokens

    return token_bits / original_bytes


def compute_metrics(
    result: EncodeResult,
    original: bytes,
    vocab_size: int = 8192,
) -> TokenizationMetrics:
    """Compute metrics for a single tokenization result.

    Args:
        result: Tokenization result
        original: Original byte sequence
        vocab_size: Token vocabulary size (default 8192)

    Returns:
        TokenizationMetrics
    """
    num_tokens = len(result.tokens)
    num_bytes = len(original)
    residual_bytes = len(result.residuals)

    if num_tokens == 0:
        return TokenizationMetrics(num_bytes=num_bytes)

    # Compression ratio
    compression_ratio = num_bytes / num_tokens if num_tokens > 0 else 1.0

    # Compute both BPB metrics using the standalone functions
    end_to_end_bpb = compute_end_to_end_bpb(
        num_tokens, residual_bytes, num_bytes, vocab_size
    )
    structural_bpb = compute_structural_bpb(num_tokens, num_bytes, vocab_size)

    # DEPRECATED: Keep bits_per_byte for backward compatibility
    # This maps to end_to_end_bpb (the correct lossless metric)
    bits_per_byte = end_to_end_bpb

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
    residual_ratio = residual_bytes / num_bytes if num_bytes > 0 else 0.0

    return TokenizationMetrics(
        num_tokens=num_tokens,
        num_bytes=num_bytes,
        compression_ratio=compression_ratio,
        bits_per_byte=bits_per_byte,
        end_to_end_bpb=end_to_end_bpb,
        structural_bpb=structural_bpb,
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
    mean_end_to_end_bpb = sum(m.end_to_end_bpb for m in metrics) / n
    mean_structural_bpb = sum(m.structural_bpb for m in metrics) / n
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
        mean_end_to_end_bpb=mean_end_to_end_bpb,
        mean_structural_bpb=mean_structural_bpb,
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
    """Compute compression overhead ratio including residuals.

    This computes the actual storage overhead by comparing the total encoded
    size (token bits packed into bytes + residual bytes) against the original
    data size. This properly accounts for residuals in the compression claim.

    Formula:
        token_bits = ceil(log2(vocab_size)) * num_tokens
        token_bytes = ceil(token_bits / 8)
        total_encoded = token_bytes + residual_bytes
        overhead = total_encoded / original_bytes

    Args:
        original_bytes: Original data size in bytes
        num_tokens: Number of tokens in encoding
        residual_bytes: Number of residual bytes for lossless reconstruction
        vocab_size: Token vocabulary size (default 8192)

    Returns:
        Overhead ratio:
        - 1.0 = same size as original
        - <1.0 = compression achieved
        - >1.0 = expansion (encoded larger than original)
    """
    bits_per_token = math.ceil(math.log2(vocab_size))  # 13 bits for vocab_size=8192
    token_bits = num_tokens * bits_per_token
    token_bytes = (token_bits + 7) // 8  # Round up to whole bytes

    total_encoded = token_bytes + residual_bytes
    return total_encoded / original_bytes if original_bytes > 0 else 1.0
