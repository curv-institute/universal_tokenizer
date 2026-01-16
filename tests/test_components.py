"""Tests for tokenizer components."""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.interfaces import ModelConfig, EQConfig, CodebookConfig, ControllerConfig
from tokenizer.encoder import ByteEncoder
from tokenizer.equilibrium import EquilibriumProjector
from tokenizer.codebook import TokenCodebook
from tokenizer.curvature import CurvatureEstimator
from tokenizer.controller import (
    HarmonizerController,
    InterventionRecord,
    analyze_intervention_correlation,
)


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


class TestHarmonizerTracing:
    """Tests for HarmonizerController intervention tracing (Experiment 3)."""

    def test_tracing_disabled_by_default(self):
        """Tracing should be disabled by default for minimal overhead."""
        config = ControllerConfig(update_interval=1)
        controller = HarmonizerController(config)

        assert controller.get_intervention_count() == 0
        assert controller.get_intervention_trace() == []

    def test_enable_tracing_lazy_init(self):
        """Tracing should only initialize the list when enabled."""
        config = ControllerConfig(update_interval=1)
        controller = HarmonizerController(config)

        # Initially no trace list
        assert controller._intervention_trace is None

        # Enable tracing
        controller.enable_tracing(True)
        assert controller._intervention_trace is not None
        assert controller._intervention_trace == []

    def test_record_intervention(self):
        """Interventions should be recorded when tracing is enabled."""
        config = ControllerConfig(update_interval=1, gain=0.1)
        controller = HarmonizerController(config)
        controller.enable_tracing(True)

        # Observe some stats that will trigger an intervention
        controller.observe({"avg_token_len": 10.0, "curvature_tail": 1.0, "stability_margin": 0.5})
        controller.set_byte_offset(1000)
        controller.step()

        # Should have recorded an intervention
        trace = controller.get_intervention_trace()
        assert len(trace) >= 1
        record = trace[0]
        assert "step" in record
        assert "byte_offset" in record
        assert "adjustments" in record
        assert "trigger_metrics" in record
        assert "hhc_stats" in record

    def test_clear_trace(self):
        """clear_trace should reset the intervention list."""
        config = ControllerConfig(update_interval=1, gain=0.1)
        controller = HarmonizerController(config)
        controller.enable_tracing(True)

        controller.observe({"avg_token_len": 10.0, "curvature_tail": 1.0})
        controller.step()
        assert controller.get_intervention_count() > 0

        controller.clear_trace()
        assert controller.get_intervention_count() == 0

    def test_byte_offset_tracking(self):
        """Byte offset should be properly recorded in interventions."""
        config = ControllerConfig(update_interval=1, gain=0.1)
        controller = HarmonizerController(config)
        controller.enable_tracing(True)

        controller.observe({"avg_token_len": 10.0, "curvature_tail": 1.0})
        controller.set_byte_offset(5000)
        controller.step()

        trace = controller.get_intervention_trace()
        if trace:  # Only check if an intervention was recorded
            assert trace[0]["byte_offset"] == 5000


class TestInterventionCorrelationAnalysis:
    """Tests for correlation analysis between interventions and boundaries."""

    def test_empty_inputs(self):
        """Should handle empty inputs gracefully."""
        result = analyze_intervention_correlation([], [], window_bytes=256)
        assert result["interventions_near_boundary"] == 0
        assert result["interventions_away_from_boundary"] == 0
        assert result["correlation_coefficient"] == 0.0

    def test_empty_boundaries(self):
        """Should handle case with interventions but no boundaries."""
        trace = [
            {"byte_offset": 100, "step": 1, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},
            {"byte_offset": 500, "step": 2, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},
        ]
        result = analyze_intervention_correlation(trace, [], window_bytes=256)
        assert result["interventions_near_boundary"] == 0
        assert result["interventions_away_from_boundary"] == 2

    def test_all_near_boundary(self):
        """Should detect all interventions near boundaries."""
        trace = [
            {"byte_offset": 100, "step": 1, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},
            {"byte_offset": 105, "step": 2, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},
        ]
        boundaries = [100]
        result = analyze_intervention_correlation(trace, boundaries, window_bytes=256)
        assert result["interventions_near_boundary"] == 2
        assert result["interventions_away_from_boundary"] == 0
        assert result["boundary_triggered_fraction"] == 1.0

    def test_all_away_from_boundary(self):
        """Should detect all interventions away from boundaries."""
        trace = [
            {"byte_offset": 1000, "step": 1, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},
            {"byte_offset": 2000, "step": 2, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},
        ]
        boundaries = [100]
        result = analyze_intervention_correlation(trace, boundaries, window_bytes=256)
        assert result["interventions_near_boundary"] == 0
        assert result["interventions_away_from_boundary"] == 2
        assert result["boundary_triggered_fraction"] == 0.0

    def test_mixed_near_and_away(self):
        """Should correctly classify mixed interventions."""
        trace = [
            {"byte_offset": 100, "step": 1, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},  # near 100
            {"byte_offset": 1000, "step": 2, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},  # away
            {"byte_offset": 510, "step": 3, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},  # near 500
        ]
        boundaries = [100, 500]
        result = analyze_intervention_correlation(trace, boundaries, window_bytes=50)
        assert result["interventions_near_boundary"] == 2
        assert result["interventions_away_from_boundary"] == 1
        assert abs(result["boundary_triggered_fraction"] - 2/3) < 0.01

    def test_boundary_coverage(self):
        """Should track which boundaries have nearby interventions."""
        trace = [
            {"byte_offset": 100, "step": 1, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},  # near first
            {"byte_offset": 105, "step": 2, "adjustments": {}, "trigger_metrics": {}, "hhc_stats": {}},  # also near first
        ]
        boundaries = [100, 500, 1000]  # 3 boundaries, only 1 has nearby interventions
        result = analyze_intervention_correlation(trace, boundaries, window_bytes=50)
        assert abs(result["boundary_coverage"] - 1/3) < 0.01
