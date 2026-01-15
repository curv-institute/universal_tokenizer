"""Tests for tokenizer components."""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.interfaces import ModelConfig, EQConfig, CodebookConfig
from tokenizer.encoder import ByteEncoder
from tokenizer.equilibrium import EquilibriumProjector
from tokenizer.codebook import TokenCodebook
from tokenizer.curvature import CurvatureEstimator


class TestByteEncoder:
    def test_encode_single(self):
        config = ModelConfig(latent_dim=32, hidden_dim=64, embed_dim=32)
        encoder = ByteEncoder(config)

        z = encoder.encode([b"hello"])
        assert z.shape == (1, 32)

    def test_encode_batch(self):
        config = ModelConfig(latent_dim=32, hidden_dim=64, embed_dim=32)
        encoder = ByteEncoder(config)

        z = encoder.encode([b"hello", b"world", b"test"])
        assert z.shape == (3, 32)

    def test_encode_empty(self):
        config = ModelConfig(latent_dim=32, hidden_dim=64, embed_dim=32)
        encoder = ByteEncoder(config)

        z = encoder.encode([])
        assert z.shape == (0, 32)


class TestEquilibriumProjector:
    def test_project(self):
        model_cfg = ModelConfig(latent_dim=32, hidden_dim=64)
        eq_cfg = EQConfig(num_steps=5, eta=0.5)
        proj = EquilibriumProjector(eq_cfg, model_cfg)

        z = torch.randn(4, 32)
        z_eq = proj.project(z)
        assert z_eq.shape == (4, 32)

    def test_trajectory(self):
        model_cfg = ModelConfig(latent_dim=32, hidden_dim=64)
        eq_cfg = EQConfig(num_steps=5, eta=0.5)
        proj = EquilibriumProjector(eq_cfg, model_cfg)

        z = torch.randn(2, 32)
        z_eq, traj = proj.project_with_trajectory(z)

        assert len(traj) >= 2
        assert z_eq.shape == (2, 32)


class TestTokenCodebook:
    def test_quantize(self):
        config = CodebookConfig(num_codes=256, code_dim=32)
        codebook = TokenCodebook(config)

        z = torch.randn(8, 32)
        quantized, ids = codebook.quantize(z)

        assert quantized.shape == (8, 32)
        assert ids.shape == (8,)
        assert ids.min() >= 0
        assert ids.max() < 256

    def test_lookup(self):
        config = CodebookConfig(num_codes=256, code_dim=32)
        codebook = TokenCodebook(config)

        ids = torch.tensor([0, 10, 100])
        vectors = codebook.lookup(ids)
        assert vectors.shape == (3, 32)

    def test_forward(self):
        config = CodebookConfig(num_codes=256, code_dim=32)
        codebook = TokenCodebook(config)
        codebook.train()

        z = torch.randn(4, 32)
        quantized, ids, loss = codebook(z)

        assert quantized.shape == (4, 32)
        assert ids.shape == (4,)
        assert loss.ndim == 0  # scalar


class TestCurvatureEstimator:
    def test_estimate(self):
        config = ModelConfig(latent_dim=32, hidden_dim=64)
        estimator = CurvatureEstimator(config)

        z = torch.randn(4, 32)
        curv = estimator.estimate(z)

        assert curv.shape == (4,)
        assert (curv >= 0).all()  # Non-negative due to Softplus
