"""Tests for encode/decode roundtrip."""

import pytest
import torch
import random
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.config import load_config
from tokenizer.bundle import UniversalTokenizer
from tokenizer.baselines import RawByteTokenizer, ByteBPETokenizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestRoundtrip:
    @pytest.fixture
    def tokenizer(self):
        config = load_config(Path("configs/cpu_small.toml"))
        return UniversalTokenizer(config)

    def test_simple_roundtrip(self, tokenizer):
        data = b"hello world"
        encoded = tokenizer.encode(data)
        decoded = tokenizer.decode(encoded)
        assert decoded.data == data

    def test_binary_roundtrip(self, tokenizer):
        data = bytes(range(256))
        encoded = tokenizer.encode(data)
        decoded = tokenizer.decode(encoded)
        assert decoded.data == data

    def test_empty_input(self, tokenizer):
        data = b""
        encoded = tokenizer.encode(data)
        decoded = tokenizer.decode(encoded)
        assert decoded.data == data

    def test_null_bytes(self, tokenizer):
        data = b"\x00" * 50
        encoded = tokenizer.encode(data)
        decoded = tokenizer.decode(encoded)
        assert decoded.data == data

    def test_unicode(self, tokenizer):
        data = "unicode: \u00e9\u00e0\u00f9 \u4e2d\u6587".encode("utf-8")
        encoded = tokenizer.encode(data)
        decoded = tokenizer.decode(encoded)
        assert decoded.data == data


class TestBaselinesRoundtrip:
    def test_raw_bytes(self):
        tok = RawByteTokenizer()
        data = b"test data 123"
        encoded = tok.encode(data)
        decoded = tok.decode(encoded)
        assert decoded.data == data

    def test_bpe(self):
        tok = ByteBPETokenizer(vocab_size=512)
        train_data = b"the quick brown fox " * 10
        tok.train(train_data)

        data = b"the quick fox"
        encoded = tok.encode(data)
        decoded = tok.decode(encoded)
        assert decoded.data == data


class TestDeterminism:
    def test_same_output_same_seed(self):
        config = load_config(Path("configs/cpu_small.toml"))

        set_seed(42)
        tok1 = UniversalTokenizer(config)
        data = b"determinism test"
        enc1 = tok1.encode(data)

        set_seed(42)
        tok2 = UniversalTokenizer(config)
        enc2 = tok2.encode(data)

        assert enc1.ids == enc2.ids


class TestStreaming:
    def test_streaming_encode(self):
        config = load_config(Path("configs/cpu_small.toml"))
        tok = UniversalTokenizer(config)

        data = b"streaming test data " * 3

        def chunks():
            for i in range(0, len(data), 10):
                yield data[i:i+10]

        tokens = list(tok.encode_streaming(chunks()))
        assert len(tokens) > 0

        # Verify coverage
        covered = sum(len(t.span) for t in tokens)
        assert covered == len(data)
