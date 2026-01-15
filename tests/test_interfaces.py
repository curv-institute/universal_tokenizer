"""Tests for interfaces module."""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.interfaces import (
    TokenizerConfig,
    ModelConfig,
    EQConfig,
    CodebookConfig,
    Token,
    EncodeResult,
    DecodeResult,
    TokenizerBundle,
)


def test_model_config_defaults():
    """Test ModelConfig has correct defaults."""
    config = ModelConfig()
    assert config.input_dim == 256
    assert config.embed_dim == 128
    assert config.latent_dim == 64


def test_tokenizer_config_to_dict():
    """Test config serialization."""
    config = TokenizerConfig()
    d = config.to_dict()
    assert "model" in d
    assert "equilibrium" in d
    assert d["model"]["latent_dim"] == 64


def test_tokenizer_config_from_dict():
    """Test config deserialization."""
    d = {"model": {"latent_dim": 32}}
    config = TokenizerConfig.from_dict(d)
    assert config.model.latent_dim == 32


def test_token_dataclass():
    """Test Token dataclass."""
    token = Token(
        id=42,
        span=b"hello",
        start=0,
        end=5,
    )
    assert token.id == 42
    assert token.span == b"hello"
    assert token.curvature == 0.0


def test_encode_result_properties():
    """Test EncodeResult properties."""
    tokens = [
        Token(id=1, span=b"ab", start=0, end=2),
        Token(id=2, span=b"cd", start=2, end=4),
    ]
    result = EncodeResult(tokens=tokens, ids=[1, 2], residuals=b"")

    assert result.num_tokens == 2
    assert result.compression_ratio == 2.0


def test_decode_result():
    """Test DecodeResult."""
    result = DecodeResult(data=b"hello")
    assert result.data == b"hello"


def test_bundle_save_load(tmp_path):
    """Test TokenizerBundle save/load."""
    config = TokenizerConfig()
    bundle = TokenizerBundle(
        version="0.1.0",
        config=config,
        model_state={"test": torch.zeros(10)},
        codebook_state={"codes": torch.randn(100, 64)},
    )

    path = tmp_path / "bundle.pt"
    bundle.save(path)

    loaded = TokenizerBundle.load(path)
    assert loaded.version == "0.1.0"
    assert loaded.config.model.latent_dim == 64
