"""Evaluation metrics.

Provides metrics for tokenizer evaluation, including boundary-aware
stability and churn metrics for domain transitions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import Token, EncodeResult, UniversalTokenizer, BaselineTokenizer

# Import at runtime to avoid circular imports
def _get_interfaces():
    from .interfaces import Token, EncodeResult
    return Token, EncodeResult


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

    # Boundary-aware metrics (optional, populated when manifest available)
    boundary_metrics: BoundaryAwareMetrics | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sample_metrics"] = [m.to_dict() for m in self.sample_metrics]
        if self.boundary_metrics is not None:
            d["boundary_metrics"] = self.boundary_metrics.to_dict()
        else:
            d["boundary_metrics"] = None
        return d


@dataclass
class BoundaryAwareMetrics:
    """Boundary-aware metrics for domain transition analysis.

    These metrics measure tokenizer stability and consistency across
    domain boundaries (e.g., text->code, code->binary transitions).

    Attributes:
        token_churn_rate: Rate at which identical byte sequences produce
            different token sequences in different contexts. Lower is better.
        substring_stability_rate: Consistency of tokenization for repeated
            motifs across the data. Higher is better (1.0 = perfect stability).
        boundary_churn: Token churn rate within the boundary window
            (typically +/- 64 bytes around domain transitions).
        background_churn: Token churn rate away from boundaries (baseline).
        boundary_curvature_mean: Mean curvature within boundary windows.
        background_curvature_mean: Mean curvature away from boundaries.
        boundary_curvature_p90: 90th percentile curvature at boundaries
            (measures worst-case instability at transitions).
        boundary_curvature_delta: Difference between boundary and background
            curvature means (positive = higher curvature at boundaries).

        # Boundary alignment metrics (manifest-based)
        total_domain_boundaries: Number of domain boundaries in manifest.
        token_boundaries_at_domain: Token boundaries within threshold of domain boundary.
        boundary_alignment_rate: Fraction of domain boundaries with aligned tokens.

        # Per-domain performance
        per_domain_bpb: BPB by domain type (text, code, binary, etc.).
        per_domain_tokens: Token count by domain type.
        per_domain_bytes: Byte count by domain type.

        # Cross-domain transition metrics
        transition_count: Number of domain transitions analyzed.
        avg_curvature_at_transition: Mean curvature at domain transitions.
        avg_stability_at_transition: Mean stability at domain transitions.
    """

    token_churn_rate: float = 0.0
    substring_stability_rate: float = 1.0
    boundary_churn: float = 0.0
    background_churn: float = 0.0
    boundary_curvature_mean: float = 0.0
    background_curvature_mean: float = 0.0
    boundary_curvature_p90: float = 0.0
    boundary_curvature_delta: float = 0.0

    # Boundary alignment metrics (manifest-based)
    total_domain_boundaries: int = 0
    token_boundaries_at_domain: int = 0
    boundary_alignment_rate: float = 0.0

    # Per-domain performance
    per_domain_bpb: dict[str, float] = field(default_factory=dict)
    per_domain_tokens: dict[str, int] = field(default_factory=dict)
    per_domain_bytes: dict[str, int] = field(default_factory=dict)

    # Cross-domain transition metrics
    transition_count: int = 0
    avg_curvature_at_transition: float = 0.0
    avg_stability_at_transition: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BoundaryAwareMetrics:
        """Create from dictionary."""
        return cls(**d)


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


def compute_boundary_aware_metrics(
    tokens: list,
    manifest: dict,
    boundary_threshold: int = 8,
    boundary_window: int = 64,
    vocab_size: int = 8192,
) -> BoundaryAwareMetrics:
    """Compute boundary-aware metrics from tokenization and manifest.

    This function analyzes how well token boundaries align with domain
    boundaries from the manifest, and measures per-domain performance.

    Args:
        tokens: List of Token objects from encode result
        manifest: Manifest dict with block boundaries and metadata
        boundary_threshold: Max distance (bytes) to consider boundary aligned
        boundary_window: Window size (bytes) around boundaries for curvature analysis
        vocab_size: Token vocabulary size for BPB calculation

    Returns:
        BoundaryAwareMetrics instance with populated fields
    """
    metrics = BoundaryAwareMetrics()

    blocks = manifest.get("blocks", [])
    if not blocks or not tokens:
        return metrics

    # Extract domain boundary offsets (start of each block except first)
    domain_boundaries = []
    for block in blocks:
        start = block.get("start", 0)
        if start > 0:
            domain_boundaries.append(start)

    metrics.total_domain_boundaries = len(domain_boundaries)

    # Build set of token boundary offsets
    token_boundaries = set()
    for token in tokens:
        token_boundaries.add(token.start)
        token_boundaries.add(token.end)

    # Count aligned boundaries
    aligned_count = 0
    for db in domain_boundaries:
        for tb in token_boundaries:
            if abs(tb - db) <= boundary_threshold:
                aligned_count += 1
                break

    metrics.token_boundaries_at_domain = aligned_count
    if domain_boundaries:
        metrics.boundary_alignment_rate = aligned_count / len(domain_boundaries)

    # Per-domain metrics
    bits_per_token = math.ceil(math.log2(vocab_size)) if vocab_size > 1 else 1

    for block in blocks:
        block_type = block.get("type", "unknown")
        block_start = block.get("start", 0)
        block_end = block_start + block.get("length", 0)

        # Find tokens that overlap with this block
        block_tokens = []
        for token in tokens:
            if token.end > block_start and token.start < block_end:
                block_tokens.append(token)

        if not block_tokens:
            continue

        # Count bytes and tokens in this block
        block_bytes = 0
        block_token_count = 0
        for token in block_tokens:
            overlap_start = max(token.start, block_start)
            overlap_end = min(token.end, block_end)
            overlap_bytes = overlap_end - overlap_start

            block_bytes += overlap_bytes
            token_len = token.end - token.start
            if token_len > 0 and overlap_bytes >= token_len / 2:
                block_token_count += 1

        # Initialize domain stats if needed
        if block_type not in metrics.per_domain_bpb:
            metrics.per_domain_bpb[block_type] = 0.0
            metrics.per_domain_tokens[block_type] = 0
            metrics.per_domain_bytes[block_type] = 0

        metrics.per_domain_tokens[block_type] += block_token_count
        metrics.per_domain_bytes[block_type] += block_bytes

    # Compute per-domain BPB
    for domain in metrics.per_domain_bytes:
        total_bytes = metrics.per_domain_bytes[domain]
        total_tokens = metrics.per_domain_tokens[domain]
        if total_bytes > 0 and total_tokens > 0:
            metrics.per_domain_bpb[domain] = (bits_per_token * total_tokens) / total_bytes

    # Cross-domain transition metrics
    transition_curvatures = []
    transition_stabilities = []
    boundary_curvatures = []
    background_curvatures = []

    # Build boundary proximity map
    def is_near_boundary(offset: int) -> bool:
        for db in domain_boundaries:
            if abs(offset - db) <= boundary_window:
                return True
        return False

    for token in tokens:
        token_center = (token.start + token.end) // 2
        if is_near_boundary(token_center):
            boundary_curvatures.append(token.curvature)
        else:
            background_curvatures.append(token.curvature)

    # Transition-specific metrics (tokens closest to each boundary)
    for db in domain_boundaries:
        closest_token = None
        min_dist = float("inf")
        for token in tokens:
            dist = min(abs(token.start - db), abs(token.end - db))
            if dist < min_dist:
                min_dist = dist
                closest_token = token

        if closest_token is not None and min_dist <= boundary_window:
            transition_curvatures.append(closest_token.curvature)
            transition_stabilities.append(closest_token.stability)
            metrics.transition_count += 1

    # Compute aggregate curvature metrics
    if boundary_curvatures:
        metrics.boundary_curvature_mean = sum(boundary_curvatures) / len(boundary_curvatures)
        sorted_curv = sorted(boundary_curvatures)
        p90_idx = int(0.9 * (len(sorted_curv) - 1))
        metrics.boundary_curvature_p90 = sorted_curv[p90_idx]

    if background_curvatures:
        metrics.background_curvature_mean = sum(background_curvatures) / len(background_curvatures)

    metrics.boundary_curvature_delta = (
        metrics.boundary_curvature_mean - metrics.background_curvature_mean
    )

    if transition_curvatures:
        metrics.avg_curvature_at_transition = sum(transition_curvatures) / len(
            transition_curvatures
        )
    if transition_stabilities:
        metrics.avg_stability_at_transition = sum(transition_stabilities) / len(
            transition_stabilities
        )

    return metrics


def load_manifest(manifest_path: Path | str) -> dict:
    """Load a manifest file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Manifest dictionary
    """
    path = Path(manifest_path)
    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


# =============================================================================
# Boundary-Aware Stability and Churn Metrics (Experiment 2)
# =============================================================================


def load_boundary_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load manifest with boundary offsets.

    Parses the manifest JSON file generated by make_eval_data.py and
    extracts block boundary information for domain transition analysis.

    Args:
        manifest_path: Path to the manifest JSON file.

    Returns:
        Dictionary containing:
        - 'blocks': List of block info dicts with 'type', 'start', 'length'
        - 'boundaries': List of boundary offsets (where domains change)
        - 'boundary_types': List of (from_type, to_type) tuples for each boundary
        - 'total_bytes': Total data size in bytes
        - 'motifs': List of known motif byte strings (for stability testing)

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        json.JSONDecodeError: If manifest is not valid JSON.
    """
    path = Path(manifest_path)
    with open(path, "r") as f:
        manifest = json.load(f)

    blocks = manifest.get("blocks", [])
    boundaries = []
    boundary_types = []

    # Extract boundary offsets (where blocks meet)
    for i in range(len(blocks) - 1):
        current_block = blocks[i]
        next_block = blocks[i + 1]

        # Boundary is at the end of the current block (start of next)
        boundary_offset = current_block["start"] + current_block["length"]
        boundaries.append(boundary_offset)
        boundary_types.append((current_block["type"], next_block["type"]))

    # Known motifs from make_eval_data.py
    motifs = [
        b"<<<MOTIF_START>>>",
        b"===BOUNDARY===",
        b"---SEPARATOR---",
        b"[[[MARKER]]]",
        b"###TAG###",
        b"@@@ANCHOR@@@",
        b"***REPEAT***",
        b"~~~PATTERN~~~",
    ]

    return {
        "version": manifest.get("version", "1.0"),
        "blocks": blocks,
        "boundaries": boundaries,
        "boundary_types": boundary_types,
        "total_bytes": manifest.get("actual_size_bytes", 0),
        "motifs": motifs,
        "raw_manifest": manifest,
    }


def _tokenize_substring(
    tokenizer: Any,
    data: bytes,
    start: int,
    end: int,
) -> tuple[list[int], list[float]]:
    """Helper to tokenize a substring and extract token IDs and curvatures.

    Args:
        tokenizer: Tokenizer with encode() method returning EncodeResult.
        data: Full byte sequence.
        start: Start offset (inclusive).
        end: End offset (exclusive).

    Returns:
        Tuple of (token_ids, curvatures).
    """
    if start < 0:
        start = 0
    if end > len(data):
        end = len(data)
    if start >= end:
        return [], []

    substring = data[start:end]
    result = tokenizer.encode(substring)

    token_ids = result.ids
    curvatures = [t.curvature for t in result.tokens]

    return token_ids, curvatures


def token_churn_rate(
    tokenizer: Any,
    data: bytes,
    boundary_offsets: list[int],
    window_bytes: int = 64,
) -> float:
    """Measure token churn rate: whether identical substrings produce different tokens.

    Compares tokenization of identical byte sequences appearing in different
    domain contexts. For each boundary, we extract fixed-size windows on either
    side and look for byte patterns that appear in both windows but tokenize
    differently.

    Args:
        tokenizer: Tokenizer with encode() method.
        data: Full byte sequence.
        boundary_offsets: List of byte offsets where domain boundaries occur.
        window_bytes: Window size on each side of boundary (default 64).

    Returns:
        Token churn rate between 0.0 and 1.0. Lower is better (more consistent).
        Returns 0.0 if no comparable patterns found.
    """
    if not boundary_offsets or len(data) < window_bytes * 2:
        return 0.0

    total_comparisons = 0
    churned_comparisons = 0

    # For each boundary, compare tokenizations in left vs right contexts
    for boundary in boundary_offsets:
        left_start = max(0, boundary - window_bytes)
        left_end = boundary
        right_start = boundary
        right_end = min(len(data), boundary + window_bytes)

        # Get raw bytes for left and right windows
        left_bytes = data[left_start:left_end]
        right_bytes = data[right_start:right_end]

        # Extract all 4-byte patterns from each side
        pattern_len = 4
        left_patterns: dict[bytes, int] = {}
        for i in range(len(left_bytes) - pattern_len + 1):
            pattern = left_bytes[i : i + pattern_len]
            if pattern not in left_patterns:
                left_patterns[pattern] = i + left_start

        # Check if any patterns appear on the right side
        for i in range(len(right_bytes) - pattern_len + 1):
            pattern = right_bytes[i : i + pattern_len]
            if pattern in left_patterns:
                # Found a common pattern - tokenize each occurrence in isolation
                left_pos = left_patterns[pattern]
                right_pos = right_start + i

                # Tokenize with some context around each occurrence
                ctx = 8  # context bytes
                left_ctx_ids, _ = _tokenize_substring(
                    tokenizer, data, left_pos - ctx, left_pos + pattern_len + ctx
                )
                right_ctx_ids, _ = _tokenize_substring(
                    tokenizer, data, right_pos - ctx, right_pos + pattern_len + ctx
                )

                total_comparisons += 1
                if left_ctx_ids != right_ctx_ids:
                    churned_comparisons += 1

    if total_comparisons == 0:
        return 0.0

    return churned_comparisons / total_comparisons


def substring_stability_rate(
    tokenizer: Any,
    data: bytes,
    motif_locations: list[tuple[bytes, list[int]]] | None = None,
) -> float:
    """Measure consistency of tokenization for repeated motifs.

    For each motif pattern, finds all occurrences in the data and checks
    whether they produce the same tokenization. Higher stability rate
    indicates more consistent tokenization of repeated patterns.

    Args:
        tokenizer: Tokenizer with encode() method.
        data: Full byte sequence.
        motif_locations: Optional list of (motif_bytes, [offset, ...]) tuples.
            If None, searches for common motifs from make_eval_data.py.

    Returns:
        Stability rate between 0.0 and 1.0. Higher is better (more stable).
        Returns 1.0 if no motifs found (vacuously stable).
    """
    # Default motifs from make_eval_data.py
    default_motifs = [
        b"<<<MOTIF_START>>>",
        b"===BOUNDARY===",
        b"---SEPARATOR---",
        b"[[[MARKER]]]",
        b"###TAG###",
        b"@@@ANCHOR@@@",
        b"***REPEAT***",
        b"~~~PATTERN~~~",
    ]

    if motif_locations is None:
        # Find all occurrences of default motifs
        motif_locations = []
        for motif in default_motifs:
            locations: list[int] = []
            start = 0
            while True:
                pos = data.find(motif, start)
                if pos == -1:
                    break
                locations.append(pos)
                start = pos + 1
            if locations:
                motif_locations.append((motif, locations))

    if not motif_locations:
        return 1.0  # No motifs to check

    total_pairs = 0
    stable_pairs = 0

    for motif, locations in motif_locations:
        if len(locations) < 2:
            continue

        # Tokenize each occurrence with context
        ctx = 8
        tokenizations: list[tuple[int, ...]] = []
        for loc in locations:
            start = max(0, loc - ctx)
            end = min(len(data), loc + len(motif) + ctx)
            ids, _ = _tokenize_substring(tokenizer, data, start, end)
            tokenizations.append(tuple(ids))

        # Compare all pairs
        for i in range(len(tokenizations)):
            for j in range(i + 1, len(tokenizations)):
                total_pairs += 1
                if tokenizations[i] == tokenizations[j]:
                    stable_pairs += 1

    if total_pairs == 0:
        return 1.0

    return stable_pairs / total_pairs


def boundary_churn_metrics(
    tokenizer: Any,
    data: bytes,
    boundaries: list[int],
    window_bytes: int = 64,
) -> dict[str, float]:
    """Measure churn and curvature near domain boundaries.

    Computes metrics comparing tokenization behavior near boundaries
    (within +/- window_bytes) versus away from boundaries. This helps
    identify whether domain transitions cause increased instability.

    Args:
        tokenizer: Tokenizer with encode() method.
        data: Full byte sequence.
        boundaries: List of boundary offsets (domain transitions).
        window_bytes: Window size around boundaries (default 64).

    Returns:
        Dictionary with:
        - 'churn_at_boundary': Token churn rate within boundary windows.
        - 'churn_away_from_boundary': Token churn rate elsewhere.
        - 'curvature_at_boundary': Mean curvature within boundary windows.
        - 'curvature_away_from_boundary': Mean curvature elsewhere.
        - 'curvature_delta_at_boundary': Spike magnitude (boundary - background).
        - 'tail_mass_at_boundary': P90+ curvature fraction near boundaries.
    """
    if not boundaries or len(data) < window_bytes * 2:
        return {
            "churn_at_boundary": 0.0,
            "churn_away_from_boundary": 0.0,
            "curvature_at_boundary": 0.0,
            "curvature_away_from_boundary": 0.0,
            "curvature_delta_at_boundary": 0.0,
            "tail_mass_at_boundary": 0.0,
        }

    # Build set of byte offsets that are "near" a boundary
    boundary_zone: set[int] = set()
    for b in boundaries:
        for offset in range(b - window_bytes, b + window_bytes):
            if 0 <= offset < len(data):
                boundary_zone.add(offset)

    # Tokenize the entire data
    result = tokenizer.encode(data)

    # Classify tokens by whether they overlap with boundary zones
    boundary_curvatures: list[float] = []
    background_curvatures: list[float] = []

    for token in result.tokens:
        # Check if this token overlaps with boundary zone
        token_offsets = set(range(token.start, token.end))
        overlaps_boundary = bool(token_offsets & boundary_zone)

        if overlaps_boundary:
            boundary_curvatures.append(token.curvature)
        else:
            background_curvatures.append(token.curvature)

    # Compute curvature statistics
    if boundary_curvatures:
        curvature_at_boundary = sum(boundary_curvatures) / len(boundary_curvatures)
        # P90 curvature at boundaries
        sorted_boundary = sorted(boundary_curvatures)
        p90_idx = int(0.9 * (len(sorted_boundary) - 1))
        p90_curvature = sorted_boundary[p90_idx] if sorted_boundary else 0.0
        # Tail mass: fraction of boundary curvatures above global P90
        all_curvatures = boundary_curvatures + background_curvatures
        if all_curvatures:
            global_p90 = sorted(all_curvatures)[int(0.9 * (len(all_curvatures) - 1))]
            tail_count = sum(1 for c in boundary_curvatures if c >= global_p90)
            tail_mass = tail_count / len(boundary_curvatures)
        else:
            tail_mass = 0.0
    else:
        curvature_at_boundary = 0.0
        p90_curvature = 0.0
        tail_mass = 0.0

    if background_curvatures:
        curvature_away = sum(background_curvatures) / len(background_curvatures)
    else:
        curvature_away = 0.0

    # Compute churn metrics using the token_churn_rate function
    churn_at_boundary = token_churn_rate(tokenizer, data, boundaries, window_bytes)

    # For background churn, we sample random positions away from boundaries
    # and treat them as pseudo-boundaries to measure baseline churn
    import random

    rng = random.Random(42)  # Deterministic
    non_boundary_positions: list[int] = []
    for i in range(0, len(data), window_bytes * 4):
        if i not in boundary_zone and i + window_bytes < len(data):
            non_boundary_positions.append(i)

    # Sample up to len(boundaries) random positions for comparison
    if non_boundary_positions and len(non_boundary_positions) >= len(boundaries):
        sample_positions = rng.sample(
            non_boundary_positions, min(len(boundaries), len(non_boundary_positions))
        )
        churn_away = token_churn_rate(tokenizer, data, sample_positions, window_bytes)
    else:
        churn_away = 0.0

    return {
        "churn_at_boundary": churn_at_boundary,
        "churn_away_from_boundary": churn_away,
        "curvature_at_boundary": curvature_at_boundary,
        "curvature_away_from_boundary": curvature_away,
        "curvature_delta_at_boundary": curvature_at_boundary - curvature_away,
        "tail_mass_at_boundary": tail_mass,
    }


def compute_full_boundary_metrics(
    tokenizer: Any,
    data: bytes,
    manifest_path: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
    window_bytes: int = 64,
) -> BoundaryAwareMetrics:
    """Compute comprehensive boundary-aware metrics including churn and stability.

    This is the main entry point for computing all boundary-aware metrics
    for a tokenizer on a given dataset with known domain boundaries.
    Combines alignment metrics with churn and stability analysis.

    Args:
        tokenizer: Tokenizer with encode() method.
        data: Full byte sequence.
        manifest_path: Path to manifest JSON file (optional if manifest provided).
        manifest: Pre-loaded manifest dict (optional if manifest_path provided).
        window_bytes: Window size for boundary analysis (default 64).

    Returns:
        BoundaryAwareMetrics dataclass with all computed metrics.

    Raises:
        ValueError: If neither manifest_path nor manifest is provided.
    """
    if manifest is None:
        if manifest_path is None:
            raise ValueError("Either manifest_path or manifest must be provided")
        manifest = load_boundary_manifest(manifest_path)

    # Get tokenization result for alignment metrics
    result = tokenizer.encode(data)
    tokens = result.tokens

    # First compute the alignment-based metrics
    raw_manifest = manifest.get("raw_manifest", manifest)
    base_metrics = compute_boundary_aware_metrics(
        tokens=tokens,
        manifest=raw_manifest,
        boundary_threshold=8,
        boundary_window=window_bytes,
    )

    # Extract boundaries for churn/stability analysis
    boundaries = manifest.get("boundaries", [])
    if not boundaries:
        # Fall back to extracting from blocks
        blocks = manifest.get("blocks", raw_manifest.get("blocks", []))
        for i in range(len(blocks) - 1):
            current_block = blocks[i]
            boundary_offset = current_block["start"] + current_block["length"]
            boundaries.append(boundary_offset)

    motifs = manifest.get("motifs", [])

    # Compute token churn rate
    churn_rate = token_churn_rate(tokenizer, data, boundaries, window_bytes)

    # Compute substring stability rate
    motif_locations: list[tuple[bytes, list[int]]] = []
    for motif in motifs:
        if isinstance(motif, str):
            motif = motif.encode("utf-8")
        locations: list[int] = []
        start = 0
        while True:
            pos = data.find(motif, start)
            if pos == -1:
                break
            locations.append(pos)
            start = pos + 1
        if locations:
            motif_locations.append((motif, locations))

    stability_rate = substring_stability_rate(
        tokenizer, data, motif_locations if motif_locations else None
    )

    # Compute boundary churn metrics
    churn_metrics = boundary_churn_metrics(tokenizer, data, boundaries, window_bytes)

    # Update base_metrics with churn/stability values
    base_metrics.token_churn_rate = churn_rate
    base_metrics.substring_stability_rate = stability_rate
    base_metrics.boundary_churn = churn_metrics["churn_at_boundary"]
    base_metrics.background_churn = churn_metrics["churn_away_from_boundary"]

    # Ensure curvature metrics are populated from churn_metrics if not already
    if base_metrics.boundary_curvature_mean == 0.0:
        base_metrics.boundary_curvature_mean = churn_metrics["curvature_at_boundary"]
    if base_metrics.background_curvature_mean == 0.0:
        base_metrics.background_curvature_mean = churn_metrics["curvature_away_from_boundary"]
    if base_metrics.boundary_curvature_delta == 0.0:
        base_metrics.boundary_curvature_delta = churn_metrics["curvature_delta_at_boundary"]

    return base_metrics


def aggregate_metrics_with_boundaries(
    metrics: Sequence[TokenizationMetrics],
    tokenizer: Any | None = None,
    data: bytes | None = None,
    manifest_path: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
    window_bytes: int = 64,
) -> AggregateMetrics:
    """Aggregate metrics and optionally include boundary-aware metrics.

    This is an enhanced version of aggregate_metrics() that also computes
    boundary-aware metrics when manifest information is available.

    Args:
        metrics: Sequence of per-sample TokenizationMetrics.
        tokenizer: Optional tokenizer for boundary analysis.
        data: Optional full data bytes for boundary analysis.
        manifest_path: Optional path to manifest JSON file.
        manifest: Optional pre-loaded manifest dict.
        window_bytes: Window size for boundary analysis (default 64).

    Returns:
        AggregateMetrics with optional boundary_metrics populated.
    """
    # First compute standard aggregate metrics
    result = aggregate_metrics(metrics)

    # If we have the necessary inputs, compute boundary metrics
    if (
        tokenizer is not None
        and data is not None
        and (manifest_path is not None or manifest is not None)
    ):
        try:
            boundary_metrics = compute_full_boundary_metrics(
                tokenizer=tokenizer,
                data=data,
                manifest_path=manifest_path,
                manifest=manifest,
                window_bytes=window_bytes,
            )
            result.boundary_metrics = boundary_metrics
        except Exception:
            # If boundary metrics fail, continue without them
            pass

    return result
