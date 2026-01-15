"""Data loading utilities.

Provides data loading for training and evaluation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence


class ByteDataset:
    """Simple byte dataset for training."""

    def __init__(
        self,
        data: bytes | None = None,
        path: Path | str | None = None,
        chunk_size: int = 256,
    ):
        """Initialize dataset.

        Args:
            data: Raw bytes (if provided directly)
            path: Path to file to load
            chunk_size: Size of chunks to yield
        """
        if data is not None:
            self.data = data
        elif path is not None:
            with open(path, "rb") as f:
                self.data = f.read()
        else:
            self.data = b""

        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return max(1, len(self.data) // self.chunk_size)

    def __getitem__(self, idx: int) -> bytes:
        start = idx * self.chunk_size
        end = start + self.chunk_size
        return self.data[start:end]

    def __iter__(self) -> Iterator[bytes]:
        for i in range(len(self)):
            yield self[i]

    def random_chunks(self, n: int, seed: int | None = None) -> list[bytes]:
        """Get n random chunks."""
        if seed is not None:
            random.seed(seed)

        if len(self.data) < self.chunk_size:
            return [self.data] * n

        chunks = []
        for _ in range(n):
            start = random.randint(0, len(self.data) - self.chunk_size)
            chunks.append(self.data[start : start + self.chunk_size])
        return chunks


class BatchIterator:
    """Batch iterator for training."""

    def __init__(
        self,
        dataset: ByteDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self._indices: list[int] = []
        self._pos = 0

    def __iter__(self) -> Iterator[list[bytes]]:
        self._indices = list(range(len(self.dataset)))
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self._indices)
        self._pos = 0
        return self

    def __next__(self) -> list[bytes]:
        if self._pos >= len(self._indices):
            raise StopIteration

        batch_indices = self._indices[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size

        return [self.dataset[i] for i in batch_indices]


def load_text_file(path: Path | str, encoding: str = "utf-8") -> bytes:
    """Load text file as bytes."""
    with open(path, "r", encoding=encoding) as f:
        return f.read().encode(encoding)


def load_binary_file(path: Path | str) -> bytes:
    """Load binary file."""
    with open(path, "rb") as f:
        return f.read()


def generate_random_bytes(n: int, seed: int | None = None) -> bytes:
    """Generate n random bytes."""
    if seed is not None:
        random.seed(seed)
    return bytes(random.randint(0, 255) for _ in range(n))


def generate_text_like(n: int, seed: int | None = None) -> bytes:
    """Generate text-like bytes (ASCII printable + newlines)."""
    if seed is not None:
        random.seed(seed)

    chars = (
        list(range(32, 127)) + [10, 10, 10]  # ASCII printable + extra newlines
    )
    return bytes(random.choice(chars) for _ in range(n))


class StreamingDataLoader:
    """Streaming data loader for large files."""

    def __init__(
        self,
        paths: Sequence[Path | str],
        chunk_size: int = 4096,
        shuffle_files: bool = True,
        seed: int | None = None,
    ):
        self.paths = [Path(p) for p in paths]
        self.chunk_size = chunk_size
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self) -> Iterator[bytes]:
        paths = list(self.paths)
        if self.shuffle_files:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(paths)

        for path in paths:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
