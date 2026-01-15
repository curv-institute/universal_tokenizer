"""Baseline tokenizers for comparison.

Provides simple baselines:
- Raw byte tokenizer (256 symbols)
- Byte-level BPE baseline
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .interfaces import Token, EncodeResult, DecodeResult


class RawByteTokenizer:
    """Raw byte tokenizer - 256 symbols, one per byte.

    The simplest possible tokenizer. Each byte is its own token.
    Compression ratio is always 1.0.
    """

    @property
    def name(self) -> str:
        return "raw_bytes"

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, data: bytes) -> EncodeResult:
        """Encode bytes to tokens (one token per byte)."""
        tokens = []
        for i, byte in enumerate(data):
            tokens.append(
                Token(
                    id=byte,
                    span=bytes([byte]),
                    start=i,
                    end=i + 1,
                    residual=b"",
                    curvature=0.0,
                    stability=1.0,
                )
            )

        return EncodeResult(
            tokens=tokens,
            ids=[t.id for t in tokens],
            residuals=b"",
            metadata={"tokenizer": self.name},
        )

    def decode(self, result: EncodeResult) -> DecodeResult:
        """Decode tokens to bytes."""
        data = bytes(result.ids)
        return DecodeResult(data=data, metadata={"tokenizer": self.name})


class ByteBPETokenizer:
    """Simple byte-level BPE tokenizer.

    Learns merges from data, falls back to raw bytes for OOV.
    """

    def __init__(self, vocab_size: int = 1024, min_frequency: int = 2):
        self._name = "byte_bpe"
        self._vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Initialize with byte vocabulary
        self.byte_to_id: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Learned merges
        self.merges: list[tuple[bytes, bytes]] = []
        self.next_id = 256

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def train(self, data: bytes) -> None:
        """Train BPE on data."""
        # Start with byte-level tokens
        tokens = [bytes([b]) for b in data]

        while self.next_id < self._vocab_size:
            # Count pairs
            pairs = Counter()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1

            if not pairs:
                break

            # Find most frequent pair
            best_pair, freq = pairs.most_common(1)[0]
            if freq < self.min_frequency:
                break

            # Merge
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.byte_to_id[new_token] = self.next_id
            self.id_to_bytes[self.next_id] = new_token
            self.next_id += 1

            # Apply merge to tokens
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

    def encode(self, data: bytes) -> EncodeResult:
        """Encode bytes using learned merges."""
        # Start with bytes
        tokens = [bytes([b]) for b in data]

        # Apply merges in order
        for a, b in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to Token objects
        result_tokens = []
        pos = 0
        for token_bytes in tokens:
            token_id = self.byte_to_id.get(token_bytes, -1)
            if token_id == -1:
                # Fallback to raw bytes (shouldn't happen)
                for b in token_bytes:
                    result_tokens.append(
                        Token(
                            id=b,
                            span=bytes([b]),
                            start=pos,
                            end=pos + 1,
                            residual=b"",
                        )
                    )
                    pos += 1
            else:
                result_tokens.append(
                    Token(
                        id=token_id,
                        span=token_bytes,
                        start=pos,
                        end=pos + len(token_bytes),
                        residual=b"",
                    )
                )
                pos += len(token_bytes)

        return EncodeResult(
            tokens=result_tokens,
            ids=[t.id for t in result_tokens],
            residuals=b"",
            metadata={"tokenizer": self.name, "num_merges": len(self.merges)},
        )

    def decode(self, result: EncodeResult) -> DecodeResult:
        """Decode tokens to bytes."""
        data = b""
        for token_id in result.ids:
            if token_id in self.id_to_bytes:
                data += self.id_to_bytes[token_id]
            else:
                # Unknown token, skip or use replacement
                data += b"?"
        return DecodeResult(data=data, metadata={"tokenizer": self.name})

    def state_dict(self) -> dict[str, Any]:
        """Return tokenizer state."""
        return {
            "merges": [(a.hex(), b.hex()) for a, b in self.merges],
            "vocab_size": self._vocab_size,
            "next_id": self.next_id,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load tokenizer state."""
        self._vocab_size = state["vocab_size"]
        self.next_id = state["next_id"]
        self.merges = [
            (bytes.fromhex(a), bytes.fromhex(b)) for a, b in state["merges"]
        ]

        # Rebuild vocabulary
        self.byte_to_id = {bytes([i]): i for i in range(256)}
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        for i, (a, b) in enumerate(self.merges):
            new_token = a + b
            token_id = 256 + i
            self.byte_to_id[new_token] = token_id
            self.id_to_bytes[token_id] = new_token
