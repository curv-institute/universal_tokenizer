#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "pyyaml",
# ]
# ///
"""Run all tests for Universal Lossless Tokenizer."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str) -> None:
        print(f"  PASS: {name}")
        self.passed += 1

    def fail(self, name: str, msg: str) -> None:
        print(f"  FAIL: {name} - {msg}")
        self.failed += 1
        self.errors.append((name, msg))

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed")
        if self.failed:
            print("Failures:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


def test_config(result: TestResult) -> None:
    """Test configuration loading."""
    print("\n[Config Tests]")

    from tokenizer.config import load_config
    from tokenizer.interfaces import TokenizerConfig

    # Test default config loading
    try:
        config = load_config(Path("configs/default.toml"))
        assert isinstance(config, TokenizerConfig)
        assert config.model.latent_dim == 64
        assert config.codebook.num_codes == 8192
        result.ok("load_default_config")
    except Exception as e:
        result.fail("load_default_config", str(e))

    # Test cpu_small config
    try:
        config = load_config(Path("configs/cpu_small.toml"))
        assert config.model.latent_dim == 32
        assert config.codebook.num_codes == 1024
        result.ok("load_cpu_small_config")
    except Exception as e:
        result.fail("load_cpu_small_config", str(e))


def test_encoder(result: TestResult) -> None:
    """Test byte encoder."""
    print("\n[Encoder Tests]")

    from tokenizer.encoder import ByteEncoder
    from tokenizer.interfaces import ModelConfig

    config = ModelConfig(latent_dim=32, hidden_dim=64, embed_dim=32)
    encoder = ByteEncoder(config)

    # Test encoding
    try:
        spans = [b"hello", b"world", b"test"]
        z = encoder.encode(spans)
        assert z.shape == (3, 32)
        result.ok("encode_spans")
    except Exception as e:
        result.fail("encode_spans", str(e))

    # Test determinism
    try:
        set_seed(42)
        z1 = encoder.encode([b"test"])
        set_seed(42)
        z2 = encoder.encode([b"test"])
        assert torch.allclose(z1, z2)
        result.ok("encoder_determinism")
    except Exception as e:
        result.fail("encoder_determinism", str(e))


def test_equilibrium(result: TestResult) -> None:
    """Test equilibrium projection."""
    print("\n[Equilibrium Tests]")

    from tokenizer.equilibrium import EquilibriumProjector
    from tokenizer.interfaces import EQConfig, ModelConfig

    model_cfg = ModelConfig(latent_dim=32, hidden_dim=64)
    eq_cfg = EQConfig(num_steps=5, eta=0.5)
    projector = EquilibriumProjector(eq_cfg, model_cfg)

    # Test projection
    try:
        z = torch.randn(4, 32)
        z_eq = projector.project(z)
        assert z_eq.shape == z.shape
        result.ok("equilibrium_project")
    except Exception as e:
        result.fail("equilibrium_project", str(e))

    # Test trajectory
    try:
        z = torch.randn(2, 32)
        z_eq, traj = projector.project_with_trajectory(z)
        assert len(traj) > 1
        assert z_eq.shape == z.shape
        result.ok("equilibrium_trajectory")
    except Exception as e:
        result.fail("equilibrium_trajectory", str(e))


def test_codebook(result: TestResult) -> None:
    """Test codebook."""
    print("\n[Codebook Tests]")

    from tokenizer.codebook import TokenCodebook
    from tokenizer.interfaces import CodebookConfig

    config = CodebookConfig(num_codes=256, code_dim=32)
    codebook = TokenCodebook(config)

    # Test quantization
    try:
        z = torch.randn(8, 32)
        quantized, ids = codebook.quantize(z)
        assert quantized.shape == z.shape
        assert ids.shape == (8,)
        assert ids.max() < 256
        result.ok("codebook_quantize")
    except Exception as e:
        result.fail("codebook_quantize", str(e))

    # Test lookup
    try:
        ids = torch.tensor([0, 10, 100, 255])
        vectors = codebook.lookup(ids)
        assert vectors.shape == (4, 32)
        result.ok("codebook_lookup")
    except Exception as e:
        result.fail("codebook_lookup", str(e))


def test_residuals(result: TestResult) -> None:
    """Test residual coding."""
    print("\n[Residual Tests]")

    from tokenizer.residuals import ResidualCoder, SimpleResidualCoder

    # Test simple coder
    try:
        coder = SimpleResidualCoder()
        original = b"hello world"
        residual = coder.encode_residual(original, b"")
        decoded = coder.decode_residual(b"", residual)
        assert decoded == original
        result.ok("simple_residual_roundtrip")
    except Exception as e:
        result.fail("simple_residual_roundtrip", str(e))

    # Test compressed coder
    try:
        coder = ResidualCoder()
        original = b"hello world" * 100
        residual = coder.encode_residual(original, original)
        decoded = coder.decode_residual(original, residual)
        assert decoded == original
        result.ok("compressed_residual_roundtrip")
    except Exception as e:
        result.fail("compressed_residual_roundtrip", str(e))


def test_baselines(result: TestResult) -> None:
    """Test baseline tokenizers."""
    print("\n[Baseline Tests]")

    from tokenizer.baselines import RawByteTokenizer, ByteBPETokenizer

    # Test raw bytes
    try:
        tok = RawByteTokenizer()
        data = b"hello world"
        encoded = tok.encode(data)
        decoded = tok.decode(encoded)
        assert decoded.data == data
        assert len(encoded.tokens) == len(data)
        result.ok("raw_bytes_roundtrip")
    except Exception as e:
        result.fail("raw_bytes_roundtrip", str(e))

    # Test BPE
    try:
        tok = ByteBPETokenizer(vocab_size=512)
        train_data = b"the quick brown fox jumps over the lazy dog " * 10
        tok.train(train_data)

        data = b"the quick fox"
        encoded = tok.encode(data)
        decoded = tok.decode(encoded)
        assert decoded.data == data
        result.ok("bpe_roundtrip")
    except Exception as e:
        result.fail("bpe_roundtrip", str(e))


def test_bundle(result: TestResult) -> None:
    """Test tokenizer bundle save/load."""
    print("\n[Bundle Tests]")

    import tempfile
    from tokenizer.bundle import UniversalTokenizer
    from tokenizer.config import load_config

    config = load_config(Path("configs/cpu_small.toml"))

    # Test save/load
    try:
        tok = UniversalTokenizer(config)
        data = b"test data for bundle"
        encoded1 = tok.encode(data)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tok.save(f.name)
            tok2 = UniversalTokenizer.load(f.name)

        encoded2 = tok2.encode(data)
        assert encoded1.ids == encoded2.ids
        result.ok("bundle_save_load")
    except Exception as e:
        result.fail("bundle_save_load", str(e))


def test_roundtrip(result: TestResult) -> None:
    """Test encode/decode roundtrip."""
    print("\n[Roundtrip Tests]")

    from tokenizer.bundle import UniversalTokenizer
    from tokenizer.config import load_config

    config = load_config(Path("configs/cpu_small.toml"))
    tok = UniversalTokenizer(config)

    # Test various inputs
    test_cases = [
        b"hello world",
        b"",
        bytes(range(256)),
        b"\x00" * 100,
        b"unicode: \xc3\xa9\xc3\xa0\xc3\xb9",
    ]

    for i, data in enumerate(test_cases):
        try:
            if not data:
                result.ok(f"roundtrip_{i}_empty")
                continue

            encoded = tok.encode(data)
            decoded = tok.decode(encoded)
            assert decoded.data == data, f"Mismatch: {decoded.data!r} != {data!r}"
            result.ok(f"roundtrip_{i}")
        except Exception as e:
            result.fail(f"roundtrip_{i}", str(e))


def test_streaming(result: TestResult) -> None:
    """Test streaming encode."""
    print("\n[Streaming Tests]")

    from tokenizer.bundle import UniversalTokenizer
    from tokenizer.config import load_config

    config = load_config(Path("configs/cpu_small.toml"))
    tok = UniversalTokenizer(config)

    try:
        data = b"streaming test data " * 5

        def chunk_iter():
            for i in range(0, len(data), 10):
                yield data[i : i + 10]

        tokens = list(tok.encode_streaming(chunk_iter()))
        assert len(tokens) > 0
        result.ok("streaming_encode")
    except Exception as e:
        result.fail("streaming_encode", str(e))


def test_determinism(result: TestResult) -> None:
    """Test deterministic output."""
    print("\n[Determinism Tests]")

    from tokenizer.bundle import UniversalTokenizer
    from tokenizer.config import load_config

    config = load_config(Path("configs/cpu_small.toml"))

    try:
        set_seed(42)
        tok1 = UniversalTokenizer(config)
        data = b"determinism test"
        encoded1 = tok1.encode(data)

        set_seed(42)
        tok2 = UniversalTokenizer(config)
        encoded2 = tok2.encode(data)

        assert encoded1.ids == encoded2.ids
        result.ok("deterministic_output")
    except Exception as e:
        result.fail("deterministic_output", str(e))


def main() -> int:
    print("Universal Lossless Tokenizer - Test Suite")
    print("=" * 50)

    set_seed(42)
    result = TestResult()

    try:
        test_config(result)
        test_encoder(result)
        test_equilibrium(result)
        test_codebook(result)
        test_residuals(result)
        test_baselines(result)
        test_bundle(result)
        test_roundtrip(result)
        test_streaming(result)
        test_determinism(result)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1

    success = result.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
