#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Deterministic evaluation data generator for Universal Tokenizer.

Generates diverse data blocks in a cyclic pattern to force regime shifts:
text -> code -> json -> binary -> logs -> text -> code -> ...

Supports multiple scripts (Chinese, Arabic, Cyrillic, etc.) and includes
repeated motifs for stability testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


# ============================================================================
# Profile definitions
# ============================================================================

PROFILES = {
    "smoke": {"min_bytes": 1024, "max_bytes": 10 * 1024},  # 1-10 KB
    "core": {"min_bytes": 1024 * 1024, "max_bytes": 10 * 1024 * 1024},  # 1-10 MB
    "stress": {"min_bytes": 50 * 1024 * 1024, "max_bytes": 200 * 1024 * 1024},  # 50-200 MB
}


# ============================================================================
# Deterministic random generator (seeded)
# ============================================================================


class DeterministicRandom:
    """Wrapper around random module for deterministic generation."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: list) -> any:
        return self._rng.choice(seq)

    def choices(self, seq: list, k: int) -> list:
        return self._rng.choices(seq, k=k)

    def shuffle(self, seq: list) -> None:
        self._rng.shuffle(seq)

    def random(self) -> float:
        return self._rng.random()

    def randbytes(self, n: int) -> bytes:
        """Generate n random bytes deterministically."""
        # Use struct to generate bytes from random integers
        result = bytearray()
        while len(result) < n:
            # Generate 8 random bytes at a time using 64-bit integers
            val = self._rng.getrandbits(64)
            result.extend(struct.pack("<Q", val))
        return bytes(result[:n])


# ============================================================================
# Text corpora (multilingual with non-ASCII)
# ============================================================================

# English text samples
ENGLISH_SAMPLES = [
    "The quick brown fox jumps over the lazy dog. ",
    "Machine learning models process vast amounts of data efficiently. ",
    "Artificial intelligence continues to transform industries worldwide. ",
    "Natural language processing enables computers to understand human text. ",
    "Deep neural networks have revolutionized computer vision applications. ",
    "Tokenization is a fundamental step in text processing pipelines. ",
    "The algorithm converges after several thousand iterations. ",
    "Distributed computing allows parallel processing across clusters. ",
]

# Chinese text samples (Simplified)
CHINESE_SAMPLES = [
    "\u4eba\u5de5\u667a\u80fd\u6b63\u5728\u6539\u53d8\u6211\u4eec\u7684\u751f\u6d3b\u65b9\u5f0f\u3002",
    "\u6df1\u5ea6\u5b66\u4e60\u662f\u673a\u5668\u5b66\u4e60\u7684\u4e00\u4e2a\u5206\u652f\u3002",
    "\u81ea\u7136\u8bed\u8a00\u5904\u7406\u8ba9\u8ba1\u7b97\u673a\u7406\u89e3\u4eba\u7c7b\u8bed\u8a00\u3002",
    "\u6570\u636e\u79d1\u5b66\u662f\u4e00\u4e2a\u8de8\u5b66\u79d1\u7684\u7814\u7a76\u9886\u57df\u3002",
    "\u7f16\u7a0b\u662f\u4e00\u9879\u91cd\u8981\u7684\u6280\u80fd\u3002",
]

# Arabic text samples (right-to-left script)
ARABIC_SAMPLES = [
    "\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a \u064a\u063a\u064a\u0631 \u0627\u0644\u0639\u0627\u0644\u0645. ",
    "\u0627\u0644\u062a\u0639\u0644\u0645 \u0627\u0644\u0622\u0644\u064a \u064a\u062a\u0637\u0648\u0631 \u0628\u0633\u0631\u0639\u0629. ",
    "\u0645\u0639\u0627\u0644\u062c\u0629 \u0627\u0644\u0644\u063a\u0629 \u0627\u0644\u0637\u0628\u064a\u0639\u064a\u0629 \u0645\u0647\u0645\u0629. ",
    "\u0627\u0644\u0628\u0631\u0645\u062c\u0629 \u0645\u0647\u0627\u0631\u0629 \u0623\u0633\u0627\u0633\u064a\u0629. ",
]

# Cyrillic text samples (Russian)
CYRILLIC_SAMPLES = [
    "\u0418\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442 \u043c\u0435\u043d\u044f\u0435\u0442 \u043c\u0438\u0440. ",
    "\u041c\u0430\u0448\u0438\u043d\u043d\u043e\u0435 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u0440\u0430\u0437\u0432\u0438\u0432\u0430\u0435\u0442\u0441\u044f \u0431\u044b\u0441\u0442\u0440\u043e. ",
    "\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0435\u0441\u0442\u0435\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0433\u043e \u044f\u0437\u044b\u043a\u0430 \u0432\u0430\u0436\u043d\u0430. ",
    "\u041f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 - \u043a\u043b\u044e\u0447\u0435\u0432\u043e\u0439 \u043d\u0430\u0432\u044b\u043a. ",
]

# Japanese text samples (Hiragana, Katakana, Kanji mix)
JAPANESE_SAMPLES = [
    "\u4eba\u5de5\u77e5\u80fd\u306f\u4e16\u754c\u3092\u5909\u3048\u3066\u3044\u307e\u3059\u3002",
    "\u6a5f\u68b0\u5b66\u7fd2\u306f\u6025\u901f\u306b\u767a\u5c55\u3057\u3066\u3044\u307e\u3059\u3002",
    "\u30c7\u30a3\u30fc\u30d7\u30e9\u30fc\u30cb\u30f3\u30b0\u306f\u91cd\u8981\u3067\u3059\u3002",
    "\u30d7\u30ed\u30b0\u30e9\u30df\u30f3\u30b0\u306f\u57fa\u672c\u7684\u306a\u30b9\u30ad\u30eb\u3067\u3059\u3002",
]

# Korean text samples (Hangul)
KOREAN_SAMPLES = [
    "\uc778\uacf5\uc9c0\ub2a5\uc774 \uc138\uc0c1\uc744 \ubc14\uafb8\uace0 \uc788\uc2b5\ub2c8\ub2e4. ",
    "\uba38\uc2e0\ub7ec\ub2dd\uc740 \ube60\ub974\uac8c \ubc1c\uc804\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4. ",
    "\ub525\ub7ec\ub2dd\uc740 \uc911\uc694\ud55c \uae30\uc220\uc785\ub2c8\ub2e4. ",
    "\ud504\ub85c\uadf8\ub798\ubc0d\uc740 \uae30\ubcf8\uc801\uc778 \uae30\uc220\uc785\ub2c8\ub2e4. ",
]

# Hebrew text samples
HEBREW_SAMPLES = [
    "\u05d1\u05d9\u05e0\u05d4 \u05de\u05dc\u05d0\u05db\u05d5\u05ea\u05d9\u05ea \u05de\u05e9\u05e0\u05d4 \u05d0\u05ea \u05d4\u05e2\u05d5\u05dc\u05dd. ",
    "\u05dc\u05de\u05d9\u05d3\u05ea \u05de\u05db\u05d5\u05e0\u05d4 \u05de\u05ea\u05e4\u05ea\u05d7\u05ea \u05de\u05d4\u05e8. ",
    "\u05ea\u05db\u05e0\u05d5\u05ea \u05d4\u05d9\u05d0 \u05de\u05d9\u05d5\u05de\u05e0\u05d5\u05ea \u05d7\u05e9\u05d5\u05d1\u05d4. ",
]

# Greek text samples
GREEK_SAMPLES = [
    "\u0397 \u03c4\u03b5\u03c7\u03bd\u03b7\u03c4\u03ae \u03bd\u03bf\u03b7\u03bc\u03bf\u03c3\u03cd\u03bd\u03b7 \u03b1\u03bb\u03bb\u03ac\u03b6\u03b5\u03b9 \u03c4\u03bf\u03bd \u03ba\u03cc\u03c3\u03bc\u03bf. ",
    "\u0397 \u03bc\u03b7\u03c7\u03b1\u03bd\u03b9\u03ba\u03ae \u03bc\u03ac\u03b8\u03b7\u03c3\u03b7 \u03b5\u03be\u03b5\u03bb\u03af\u03c3\u03c3\u03b5\u03c4\u03b1\u03b9 \u03b3\u03c1\u03ae\u03b3\u03bf\u03c1\u03b1. ",
    "\u039f \u03c0\u03c1\u03bf\u03b3\u03c1\u03b1\u03bc\u03bc\u03b1\u03c4\u03b9\u03c3\u03bc\u03cc\u03c2 \u03b5\u03af\u03bd\u03b1\u03b9 \u03c3\u03b7\u03bc\u03b1\u03bd\u03c4\u03b9\u03ba\u03cc\u03c2. ",
]


# ============================================================================
# Source code templates
# ============================================================================

PYTHON_TEMPLATES = [
    '''def process_data(data: list[int]) -> dict[str, float]:
    """Process input data and return statistics."""
    if not data:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return {"mean": mean, "std": variance ** 0.5}

''',
    '''class TokenEncoder:
    """Encode tokens to integer IDs."""

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self._cache: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        tokens = text.split()
        return [hash(t) % self.vocab_size for t in tokens]

''',
    '''async def fetch_data(url: str, timeout: float = 30.0) -> bytes:
    """Fetch data from URL with timeout."""
    import asyncio
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as response:
            return await response.read()

''',
    '''@dataclass
class Config:
    """Training configuration."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_steps: int = 10000
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

''',
]

RUST_TEMPLATES = [
    '''use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    vocab_size: usize,
}

impl Tokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_size,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| self.vocab.get(w).copied().unwrap_or(0))
            .collect()
    }
}

''',
    '''fn process_chunk(data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut output = Vec::with_capacity(data.len());
    for &byte in data {
        if byte.is_ascii_alphanumeric() {
            output.push(byte);
        } else {
            output.push(b'_');
        }
    }
    Ok(output)
}

''',
    '''#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_steps: usize,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 32,
            max_steps: 10000,
            seed: 42,
        }
    }
}

''',
]

C_TEMPLATES = [
    '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* data;
    size_t len;
    size_t cap;
} Buffer;

Buffer* buffer_new(size_t initial_cap) {
    Buffer* buf = malloc(sizeof(Buffer));
    buf->data = malloc(initial_cap);
    buf->len = 0;
    buf->cap = initial_cap;
    return buf;
}

void buffer_free(Buffer* buf) {
    free(buf->data);
    free(buf);
}

''',
    '''int process_tokens(const int* tokens, size_t n, int* output) {
    if (tokens == NULL || output == NULL) {
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        output[i] = tokens[i] ^ 0xFF;
    }
    return 0;
}

''',
    '''#define MAX_VOCAB_SIZE 65536
#define HASH_SEED 0x9e3779b9

static inline uint32_t hash_string(const char* str) {
    uint32_t h = HASH_SEED;
    while (*str) {
        h = h * 31 + (unsigned char)*str++;
    }
    return h % MAX_VOCAB_SIZE;
}

''',
]


# ============================================================================
# JSONL templates
# ============================================================================


def generate_jsonl_record(rng: DeterministicRandom, record_id: int) -> str:
    """Generate a single JSONL record."""
    record_types = [
        lambda: {
            "id": record_id,
            "type": "user",
            "name": f"user_{rng.randint(1000, 9999)}",
            "email": f"user{rng.randint(1, 1000)}@example.com",
            "score": rng.randint(0, 100) / 10.0,
            "active": rng.choice([True, False]),
            "tags": rng.choices(["ml", "nlp", "cv", "rl", "data"], k=rng.randint(1, 3)),
        },
        lambda: {
            "id": record_id,
            "type": "event",
            "timestamp": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z",
            "action": rng.choice(["click", "view", "purchase", "signup", "logout"]),
            "user_id": rng.randint(1, 10000),
            "metadata": {"source": rng.choice(["web", "mobile", "api"]), "version": f"{rng.randint(1, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 99)}"},
        },
        lambda: {
            "id": record_id,
            "type": "model",
            "name": f"model_v{rng.randint(1, 100)}",
            "params": rng.randint(1, 1000) * 1_000_000,
            "metrics": {
                "accuracy": rng.randint(70, 99) / 100.0,
                "f1_score": rng.randint(60, 95) / 100.0,
                "latency_ms": rng.randint(10, 500),
            },
            "config": {"hidden_dim": rng.choice([256, 512, 768, 1024]), "num_layers": rng.randint(4, 24), "dropout": rng.choice([0.0, 0.1, 0.2])},
        },
    ]
    record = rng.choice(record_types)()
    return json.dumps(record, ensure_ascii=False) + "\n"


# ============================================================================
# Log/CSV templates
# ============================================================================

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_COMPONENTS = ["tokenizer", "encoder", "decoder", "trainer", "data_loader", "optimizer", "scheduler"]
LOG_MESSAGES = [
    "Processing batch {batch_id} of {total_batches}",
    "Loss: {loss:.4f}, Accuracy: {acc:.2%}",
    "Checkpoint saved to {path}",
    "Loading model from {path}",
    "Initializing {component} with {params} parameters",
    "Memory usage: {mem_mb:.1f} MB",
    "Epoch {epoch}/{total_epochs} completed in {time:.2f}s",
    "Validation metrics: loss={loss:.4f}, acc={acc:.4f}",
]


def generate_log_line(rng: DeterministicRandom, line_id: int) -> str:
    """Generate a single log line."""
    level = rng.choice(LOG_LEVELS)
    component = rng.choice(LOG_COMPONENTS)
    msg_template = rng.choice(LOG_MESSAGES)

    # Fill in template variables
    msg = msg_template.format(
        batch_id=rng.randint(1, 1000),
        total_batches=rng.randint(100, 10000),
        loss=rng.random() * 5,
        acc=rng.random(),
        path=f"/checkpoints/model_{rng.randint(1, 100)}.pt",
        component=component,
        params=rng.randint(1000, 1000000),
        mem_mb=rng.random() * 16000,
        epoch=rng.randint(1, 100),
        total_epochs=100,
        time=rng.random() * 3600,
    )

    timestamp = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d}"
    return f"{timestamp} [{level:8s}] {component}: {msg}\n"


def generate_csv_line(rng: DeterministicRandom, line_id: int) -> str:
    """Generate a single CSV line."""
    return f"{line_id},{rng.randint(1, 10000)},{rng.random():.6f},{rng.random():.6f},{rng.choice(['train', 'val', 'test'])},{rng.randint(0, 255)}\n"


# ============================================================================
# Repeated motifs (for stability testing)
# ============================================================================

MOTIFS = [
    b"<<<MOTIF_START>>>",
    b"===BOUNDARY===",
    b"---SEPARATOR---",
    b"[[[MARKER]]]",
    b"###TAG###",
    b"@@@ANCHOR@@@",
    b"***REPEAT***",
    b"~~~PATTERN~~~",
]


# ============================================================================
# Block generators
# ============================================================================


@dataclass
class BlockInfo:
    """Information about a generated block."""

    block_type: str
    start_offset: int
    length: int
    metadata: dict = field(default_factory=dict)


def generate_text_block(rng: DeterministicRandom, target_size: int) -> tuple[bytes, dict]:
    """Generate multilingual natural language text."""
    all_samples = ENGLISH_SAMPLES + CHINESE_SAMPLES + ARABIC_SAMPLES + CYRILLIC_SAMPLES + JAPANESE_SAMPLES + KOREAN_SAMPLES + HEBREW_SAMPLES + GREEK_SAMPLES

    text_parts = []
    current_size = 0

    while current_size < target_size:
        sample = rng.choice(all_samples)
        text_parts.append(sample)
        current_size += len(sample.encode("utf-8"))

    text = "".join(text_parts)
    data = text.encode("utf-8")[:target_size]

    return data, {"languages": ["en", "zh", "ar", "ru", "ja", "ko", "he", "el"]}


def generate_code_block(rng: DeterministicRandom, target_size: int) -> tuple[bytes, dict]:
    """Generate source code (Python, Rust, C)."""
    templates = [("python", PYTHON_TEMPLATES), ("rust", RUST_TEMPLATES), ("c", C_TEMPLATES)]

    code_parts = []
    current_size = 0
    languages_used = set()

    while current_size < target_size:
        lang, lang_templates = rng.choice(templates)
        template = rng.choice(lang_templates)
        code_parts.append(template)
        current_size += len(template.encode("utf-8"))
        languages_used.add(lang)

    code = "".join(code_parts)
    data = code.encode("utf-8")[:target_size]

    return data, {"languages": list(languages_used)}


def generate_jsonl_block(rng: DeterministicRandom, target_size: int) -> tuple[bytes, dict]:
    """Generate JSONL structured data."""
    lines = []
    current_size = 0
    record_id = 0

    while current_size < target_size:
        line = generate_jsonl_record(rng, record_id)
        lines.append(line)
        current_size += len(line.encode("utf-8"))
        record_id += 1

    data = "".join(lines).encode("utf-8")[:target_size]

    return data, {"num_records": record_id}


def generate_binary_block(rng: DeterministicRandom, target_size: int) -> tuple[bytes, dict]:
    """Generate uniformly random binary data."""
    data = rng.randbytes(target_size)
    return data, {"random": True}


def generate_logs_block(rng: DeterministicRandom, target_size: int) -> tuple[bytes, dict]:
    """Generate log/CSV data."""
    lines = []
    current_size = 0
    line_id = 0

    # Alternate between log lines and CSV lines
    use_logs = rng.choice([True, False])

    if use_logs:
        while current_size < target_size:
            line = generate_log_line(rng, line_id)
            lines.append(line)
            current_size += len(line.encode("utf-8"))
            line_id += 1
        format_type = "log"
    else:
        # Add CSV header
        header = "id,user_id,score,confidence,split,label\n"
        lines.append(header)
        current_size += len(header)

        while current_size < target_size:
            line = generate_csv_line(rng, line_id)
            lines.append(line)
            current_size += len(line.encode("utf-8"))
            line_id += 1
        format_type = "csv"

    data = "".join(lines).encode("utf-8")[:target_size]

    return data, {"format": format_type, "num_lines": line_id}


def add_motifs(data: bytes, rng: DeterministicRandom) -> bytes:
    """Insert repeated motifs at random positions for stability testing."""
    result = bytearray(data)
    num_motifs = max(1, len(data) // 10000)  # ~1 motif per 10KB

    for _ in range(num_motifs):
        motif = rng.choice(MOTIFS)
        # Insert at a random position
        pos = rng.randint(0, max(0, len(result) - 1))
        result[pos : pos + len(motif)] = motif

    return bytes(result)


# ============================================================================
# Main generator
# ============================================================================

# Block type cycle: text -> code -> json -> binary -> logs -> repeat
BLOCK_TYPES = ["text", "code", "jsonl", "binary", "logs"]

# Domain shift cycle for stress testing: text -> binary -> code -> json -> logs -> repeat
SHIFT_DOMAIN_CYCLE = ["text", "binary", "code", "jsonl", "logs"]

BLOCK_GENERATORS: dict[str, Callable[[DeterministicRandom, int], tuple[bytes, dict]]] = {
    "text": generate_text_block,
    "code": generate_code_block,
    "jsonl": generate_jsonl_block,
    "binary": generate_binary_block,
    "logs": generate_logs_block,
}


def generate_eval_data(
    seed: int,
    total_size: int,
    add_stability_motifs: bool = True,
) -> tuple[bytes, list[BlockInfo]]:
    """Generate deterministic evaluation data.

    Args:
        seed: Random seed for reproducibility
        total_size: Target total size in bytes
        add_stability_motifs: Whether to add repeated motifs

    Returns:
        Tuple of (data bytes, list of block info)
    """
    rng = DeterministicRandom(seed)
    blocks: list[BlockInfo] = []
    data_parts: list[bytes] = []
    current_offset = 0

    # Determine block sizes (vary between 1KB and 100KB per block)
    min_block = 1024
    max_block = min(100 * 1024, total_size // 5)  # At least 5 blocks
    if max_block < min_block:
        max_block = min_block

    block_idx = 0

    while current_offset < total_size:
        # Determine block type (cycle through types)
        block_type = BLOCK_TYPES[block_idx % len(BLOCK_TYPES)]
        generator = BLOCK_GENERATORS[block_type]

        # Determine block size
        remaining = total_size - current_offset
        block_size = min(rng.randint(min_block, max_block), remaining)

        # Generate block
        block_data, metadata = generator(rng, block_size)

        # Add motifs to non-binary blocks
        if add_stability_motifs and block_type != "binary" and len(block_data) > 1000:
            block_data = add_motifs(block_data, rng)

        # Record block info
        blocks.append(
            BlockInfo(
                block_type=block_type,
                start_offset=current_offset,
                length=len(block_data),
                metadata=metadata,
            )
        )

        data_parts.append(block_data)
        current_offset += len(block_data)
        block_idx += 1

    data = b"".join(data_parts)

    return data, blocks


# ============================================================================
# Repeated motifs for cross-domain reappearance
# ============================================================================

CROSS_DOMAIN_MOTIFS = [
    b"UNIVERSAL_TOKENIZER_MOTIF_ALPHA",
    b"CROSS_DOMAIN_PATTERN_BETA",
    b"REPEATED_SEQUENCE_GAMMA",
    b"STABILITY_CHECK_DELTA",
    b"BOUNDARY_MARKER_EPSILON",
]


def generate_cross_domain_motifs(rng: DeterministicRandom, num_motifs: int = 10) -> list[bytes]:
    """Generate a set of unique motifs that will be repeated across domains."""
    motifs = []
    for i in range(num_motifs):
        base_motif = rng.choice(CROSS_DOMAIN_MOTIFS)
        # Add a unique suffix to each motif
        suffix = f"_{i:04d}_{rng.randint(1000, 9999)}".encode("utf-8")
        motifs.append(base_motif + suffix)
    return motifs


@dataclass
class BoundaryInfo:
    """Information about a domain boundary."""
    offset: int
    from_domain: str
    to_domain: str


@dataclass
class MotifLocation:
    """Information about where a motif appears in the data."""
    motif_id: int
    motif_bytes: str  # hex representation for JSON serialization
    locations: list[dict]  # list of {offset: int, domain: str, block_index: int}


def generate_shift_dataset(
    size_mb: int,
    seed: int = 42,
) -> tuple[bytes, list[BlockInfo], list[BoundaryInfo], list[MotifLocation]]:
    """Generate a domain-shift stress stream dataset.

    This dataset is designed to stress-test tokenizers with repeated regime shifts
    between different data domains. It includes repeated motifs that appear across
    different domains to test stability.

    Args:
        size_mb: Target size in megabytes (10-50 MB typical)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (data bytes, block info list, boundary info list, motif locations)
    """
    rng = DeterministicRandom(seed)
    total_size = size_mb * 1024 * 1024

    # Scale block sizes based on dataset size to ensure multiple domain transitions
    # For small datasets (<=2MB), use smaller blocks (16KB-64KB)
    # For medium datasets (<=5MB), use medium blocks (64KB-256KB)
    # For large datasets (>5MB), use standard blocks (256KB-1MB)
    if size_mb <= 2:
        min_block_size = 16 * 1024  # 16 KB
        max_block_size = 64 * 1024  # 64 KB
    elif size_mb <= 5:
        min_block_size = 64 * 1024  # 64 KB
        max_block_size = 256 * 1024  # 256 KB
    else:
        min_block_size = 256 * 1024  # 256 KB
        max_block_size = 1024 * 1024  # 1 MB

    # Generate cross-domain motifs that will be inserted into blocks
    num_motifs = 10
    motifs = generate_cross_domain_motifs(rng, num_motifs)
    motif_locations: list[MotifLocation] = [
        MotifLocation(
            motif_id=i,
            motif_bytes=motifs[i].hex(),
            locations=[]
        )
        for i in range(num_motifs)
    ]

    blocks: list[BlockInfo] = []
    boundaries: list[BoundaryInfo] = []
    data_parts: list[bytes] = []
    current_offset = 0
    block_idx = 0
    prev_domain: str | None = None

    while current_offset < total_size:
        # Cycle through domains in the shift pattern
        domain = SHIFT_DOMAIN_CYCLE[block_idx % len(SHIFT_DOMAIN_CYCLE)]
        generator = BLOCK_GENERATORS[domain]

        # Determine block size (256KB to 1MB)
        remaining = total_size - current_offset
        block_size = min(rng.randint(min_block_size, max_block_size), remaining)

        # Ensure we don't create tiny final blocks
        if remaining - block_size < min_block_size // 2:
            block_size = remaining

        # Generate block
        block_data, metadata = generator(rng, block_size)

        # Insert cross-domain motifs into non-binary blocks
        if domain != "binary" and len(block_data) > 1000:
            block_data = bytearray(block_data)
            # Insert 1-3 motifs per block
            num_insertions = rng.randint(1, 3)
            for _ in range(num_insertions):
                motif_idx = rng.randint(0, num_motifs - 1)
                motif = motifs[motif_idx]
                # Choose insertion position (avoid very start/end)
                insert_pos = rng.randint(100, max(101, len(block_data) - len(motif) - 100))
                # Insert motif (overwrite to maintain size)
                if insert_pos + len(motif) <= len(block_data):
                    block_data[insert_pos:insert_pos + len(motif)] = motif
                    # Record motif location
                    motif_locations[motif_idx].locations.append({
                        "offset": current_offset + insert_pos,
                        "domain": domain,
                        "block_index": block_idx,
                    })
            block_data = bytes(block_data)

        # Record boundary if domain changed
        if prev_domain is not None and prev_domain != domain:
            boundaries.append(BoundaryInfo(
                offset=current_offset,
                from_domain=prev_domain,
                to_domain=domain,
            ))

        # Record block info
        blocks.append(BlockInfo(
            block_type=domain,
            start_offset=current_offset,
            length=len(block_data),
            metadata=metadata,
        ))

        data_parts.append(block_data)
        current_offset += len(block_data)
        prev_domain = domain
        block_idx += 1

    data = b"".join(data_parts)

    return data, blocks, boundaries, motif_locations


def create_shift_manifest(
    seed: int,
    size_mb: int,
    actual_size: int,
    blocks: list[BlockInfo],
    boundaries: list[BoundaryInfo],
    motif_locations: list[MotifLocation],
    output_path: Path,
    data_hash: str,
) -> dict:
    """Create a manifest for the shift dataset with boundary information."""
    # Compute block summary
    block_summary = {}
    for block in blocks:
        bt = block.block_type
        if bt not in block_summary:
            block_summary[bt] = {"count": 0, "total_bytes": 0}
        block_summary[bt]["count"] += 1
        block_summary[bt]["total_bytes"] += block.length

    # Compute boundary transition summary
    boundary_summary = {}
    for b in boundaries:
        key = f"{b.from_domain}->{b.to_domain}"
        boundary_summary[key] = boundary_summary.get(key, 0) + 1

    manifest = {
        "version": "1.0",
        "type": "shift_dataset",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "target_size_mb": size_mb,
        "target_size_bytes": size_mb * 1024 * 1024,
        "actual_size_bytes": actual_size,
        "sha256": data_hash,
        "num_blocks": len(blocks),
        "num_boundaries": len(boundaries),
        "block_summary": block_summary,
        "boundary_summary": boundary_summary,
        "domain_cycle": SHIFT_DOMAIN_CYCLE,
        "blocks": [
            {
                "type": b.block_type,
                "start": b.start_offset,
                "length": b.length,
                "metadata": b.metadata,
            }
            for b in blocks
        ],
        "boundaries": [
            {
                "offset": b.offset,
                "from_domain": b.from_domain,
                "to_domain": b.to_domain,
            }
            for b in boundaries
        ],
        "motifs": [
            {
                "motif_id": m.motif_id,
                "motif_bytes": m.motif_bytes,
                "occurrences": len(m.locations),
                "locations": m.locations,
            }
            for m in motif_locations
        ],
        "output_file": str(output_path),
    }

    return manifest


def create_manifest(
    seed: int,
    total_size: int,
    actual_size: int,
    profile: str,
    blocks: list[BlockInfo],
    output_path: Path,
) -> dict:
    """Create a manifest describing the generated data."""
    # Compute hash of data
    block_summary = {}
    for block in blocks:
        bt = block.block_type
        if bt not in block_summary:
            block_summary[bt] = {"count": 0, "total_bytes": 0}
        block_summary[bt]["count"] += 1
        block_summary[bt]["total_bytes"] += block.length

    manifest = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "profile": profile,
        "target_size_bytes": total_size,
        "actual_size_bytes": actual_size,
        "num_blocks": len(blocks),
        "block_summary": block_summary,
        "blocks": [
            {
                "type": b.block_type,
                "start": b.start_offset,
                "length": b.length,
                "metadata": b.metadata,
            }
            for b in blocks
        ],
        "output_file": str(output_path),
    }

    return manifest


def generate_shift_data_main(args: argparse.Namespace) -> None:
    """Generate shift dataset (called when --shift flag is used)."""
    size_mb = args.shift_size_mb

    # Output paths
    output_dir = Path("data/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"heldout_shift_{size_mb}mb.bin"
    manifest_path = output_dir / f"heldout_shift_{size_mb}mb_manifest.json"

    print(f"Generating domain-shift stress stream dataset...")
    print(f"  Size: {size_mb} MB")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_path}")
    print(f"  Domain cycle: {' -> '.join(SHIFT_DOMAIN_CYCLE)}")

    # Generate shift dataset
    data, blocks, boundaries, motif_locations = generate_shift_dataset(
        size_mb=size_mb,
        seed=args.seed,
    )

    # Compute hash
    data_hash = hashlib.sha256(data).hexdigest()
    print(f"  SHA256: {data_hash[:16]}...")

    # Write data
    with open(output_path, "wb") as f:
        f.write(data)

    print(f"  Written {len(data):,} bytes to {output_path}")

    # Create and save manifest
    manifest = create_shift_manifest(
        seed=args.seed,
        size_mb=size_mb,
        actual_size=len(data),
        blocks=blocks,
        boundaries=boundaries,
        motif_locations=motif_locations,
        output_path=output_path,
        data_hash=data_hash,
    )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Manifest saved to {manifest_path}")

    # Print summary
    print(f"\nDataset composition:")
    print(f"  Total blocks: {len(blocks)}")
    print(f"  Domain boundaries: {len(boundaries)}")

    print("\nBlock distribution:")
    for bt, info in manifest["block_summary"].items():
        pct = info["total_bytes"] / len(data) * 100
        print(f"  {bt:8s}: {info['count']:4d} blocks, {info['total_bytes']:,} bytes ({pct:.1f}%)")

    print("\nBoundary transitions:")
    for transition, count in manifest["boundary_summary"].items():
        print(f"  {transition}: {count}")

    # Count total motif occurrences
    total_motif_occurrences = sum(len(m.locations) for m in motif_locations)
    print(f"\nCross-domain motifs:")
    print(f"  Unique motifs: {len(motif_locations)}")
    print(f"  Total occurrences: {total_motif_occurrences}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic evaluation data for Universal Tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path for generated data (required unless --shift is used)",
    )
    parser.add_argument(
        "--size-bytes",
        type=int,
        default=None,
        help="Exact size in bytes (overrides profile min/max)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["smoke", "core", "stress"],
        default="core",
        help="Size profile: smoke (1-10KB), core (1-10MB), stress (50-200MB)",
    )
    parser.add_argument(
        "--no-motifs",
        action="store_true",
        help="Disable stability motifs",
    )

    # Shift dataset arguments
    parser.add_argument(
        "--shift",
        action="store_true",
        help="Generate domain-shift stress stream dataset instead of standard eval data",
    )
    parser.add_argument(
        "--shift-size-mb",
        type=int,
        default=10,
        help="Size of shift dataset in megabytes (default: 10)",
    )

    args = parser.parse_args()

    # Handle shift dataset generation
    if args.shift:
        generate_shift_data_main(args)
        return

    # Standard eval data generation requires --out
    if args.out is None:
        parser.error("--out is required when not using --shift")

    # Determine target size
    if args.size_bytes is not None:
        total_size = args.size_bytes
    else:
        profile = PROFILES[args.profile]
        # Use seed to deterministically pick size within range
        rng = DeterministicRandom(args.seed)
        total_size = rng.randint(profile["min_bytes"], profile["max_bytes"])

    print(f"Generating {total_size:,} bytes of evaluation data...")
    print(f"  Seed: {args.seed}")
    print(f"  Profile: {args.profile}")
    print(f"  Output: {args.out}")

    # Generate data
    data, blocks = generate_eval_data(
        seed=args.seed,
        total_size=total_size,
        add_stability_motifs=not args.no_motifs,
    )

    # Compute hash
    data_hash = hashlib.sha256(data).hexdigest()
    print(f"  SHA256: {data_hash[:16]}...")

    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Write data
    with open(args.out, "wb") as f:
        f.write(data)

    print(f"  Written {len(data):,} bytes to {args.out}")

    # Create and save manifest
    manifest_dir = Path("data/eval/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = create_manifest(
        seed=args.seed,
        total_size=total_size,
        actual_size=len(data),
        profile=args.profile,
        blocks=blocks,
        output_path=args.out,
    )
    manifest["sha256"] = data_hash

    manifest_name = f"manifest_{args.out.stem}_seed{args.seed}.json"
    manifest_path = manifest_dir / manifest_name

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Manifest saved to {manifest_path}")

    # Print block summary
    print("\nBlock composition:")
    for bt, info in manifest["block_summary"].items():
        pct = info["total_bytes"] / len(data) * 100
        print(f"  {bt:8s}: {info['count']:4d} blocks, {info['total_bytes']:,} bytes ({pct:.1f}%)")


if __name__ == "__main__":
    main()
