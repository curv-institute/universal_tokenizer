"""Residual coding for lossless reconstruction.

Encodes/decodes residual information to achieve bit-exact reconstruction.
"""

from __future__ import annotations

import zlib
from typing import Any


class ResidualCoder:
    """Residual coder for lossless byte reconstruction.

    Since tokens represent spans approximately, residuals capture
    the exact difference for bit-exact reconstruction.
    """

    def __init__(self, compression_level: int = 6):
        """Initialize residual coder.

        Args:
            compression_level: zlib compression level (0-9)
        """
        self.compression_level = compression_level

    def encode_residual(self, original: bytes, reconstructed: bytes) -> bytes:
        """Encode residual between original and reconstructed.

        For lossless tokenization, we simply store the original span
        since reconstruction from tokens alone is lossy.

        Args:
            original: Original byte sequence
            reconstructed: Reconstructed from token (may be approximate)

        Returns:
            Residual bytes (compressed)
        """
        # XOR-based residual if lengths match
        if len(original) == len(reconstructed):
            residual = bytes(a ^ b for a, b in zip(original, reconstructed))
            # Check if XOR is beneficial (lots of zeros)
            if residual.count(0) > len(residual) // 2:
                compressed = zlib.compress(residual, self.compression_level)
                if len(compressed) < len(original):
                    return b"\x01" + compressed  # XOR marker

        # Fall back to storing original compressed
        compressed = zlib.compress(original, self.compression_level)
        return b"\x00" + compressed  # Direct marker

    def decode_residual(self, reconstructed: bytes, residual: bytes) -> bytes:
        """Decode using residual for exact reconstruction.

        Args:
            reconstructed: Reconstructed from token
            residual: Residual bytes

        Returns:
            Exact original bytes
        """
        if not residual:
            return reconstructed

        marker = residual[0]
        data = residual[1:]

        if marker == 0x00:
            # Direct storage
            return zlib.decompress(data)
        elif marker == 0x01:
            # XOR residual
            xor_data = zlib.decompress(data)
            if len(xor_data) != len(reconstructed):
                # Fallback if lengths don't match
                return zlib.decompress(data)
            return bytes(a ^ b for a, b in zip(reconstructed, xor_data))
        else:
            raise ValueError(f"Unknown residual marker: {marker}")


class SimpleResidualCoder:
    """Simple residual coder that just stores original bytes."""

    def encode_residual(self, original: bytes, reconstructed: bytes) -> bytes:
        """Simply return the original bytes."""
        return original

    def decode_residual(self, reconstructed: bytes, residual: bytes) -> bytes:
        """Simply return the residual (original bytes)."""
        return residual if residual else reconstructed


class LengthPrefixedResidualCoder:
    """Residual coder with length prefixing for streaming."""

    def encode_residual(self, original: bytes, reconstructed: bytes) -> bytes:
        """Encode with length prefix."""
        length = len(original)
        # Use variable-length encoding for length
        if length < 128:
            prefix = bytes([length])
        elif length < 16384:
            prefix = bytes([0x80 | (length & 0x7F), length >> 7])
        else:
            prefix = bytes([
                0x80 | (length & 0x7F),
                0x80 | ((length >> 7) & 0x7F),
                length >> 14,
            ])
        return prefix + original

    def decode_residual(self, reconstructed: bytes, residual: bytes) -> bytes:
        """Decode with length prefix."""
        if not residual:
            return reconstructed

        # Decode variable-length prefix
        pos = 0
        length = 0
        shift = 0

        while pos < len(residual):
            byte = residual[pos]
            length |= (byte & 0x7F) << shift
            pos += 1
            if not (byte & 0x80):
                break
            shift += 7

        return residual[pos : pos + length]
