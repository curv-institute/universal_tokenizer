"""Configuration loading and validation.

Loads TOML configs and resolves defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .interfaces import (
    TokenizerConfig,
    ModelConfig,
    EQConfig,
    CodebookConfig,
    SegmentationConfig,
    ControllerConfig,
    TrainConfig,
    EvalConfig,
)


def load_config(path: Path | str) -> TokenizerConfig:
    """Load and validate configuration from TOML file.

    Args:
        path: Path to TOML config file

    Returns:
        Validated TokenizerConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return _parse_config(data)


def _parse_config(data: dict[str, Any]) -> TokenizerConfig:
    """Parse config dictionary into TokenizerConfig."""
    # Extract sections with defaults
    model_data = data.get("model", {})
    eq_data = data.get("equilibrium", {})
    codebook_data = data.get("codebook", {})
    seg_data = data.get("segmentation", {})
    ctrl_data = data.get("controller", {})
    train_data = data.get("train", {})
    eval_data = data.get("eval", {})

    # Build config with validation
    model = ModelConfig(**model_data)
    eq = EQConfig(**eq_data)
    codebook = CodebookConfig(**codebook_data)
    seg = SegmentationConfig(**seg_data)
    ctrl = ControllerConfig(**ctrl_data)
    train = TrainConfig(**train_data)
    eval_cfg = EvalConfig(**eval_data)

    config = TokenizerConfig(
        model=model,
        equilibrium=eq,
        codebook=codebook,
        segmentation=seg,
        controller=ctrl,
        train=train,
        eval=eval_cfg,
    )

    # Validate consistency
    _validate_config(config)

    return config


def _validate_config(config: TokenizerConfig) -> None:
    """Validate configuration consistency."""
    # Codebook dimension must match latent dimension
    if config.codebook.code_dim != config.model.latent_dim:
        raise ValueError(
            f"codebook.code_dim ({config.codebook.code_dim}) must match "
            f"model.latent_dim ({config.model.latent_dim})"
        )

    # Segmentation bounds
    if config.segmentation.min_span > config.segmentation.max_span:
        raise ValueError(
            f"segmentation.min_span ({config.segmentation.min_span}) must be <= "
            f"segmentation.max_span ({config.segmentation.max_span})"
        )

    # EQ parameters
    if not 0 < config.equilibrium.eta <= 1:
        raise ValueError(f"equilibrium.eta must be in (0, 1], got {config.equilibrium.eta}")

    if config.equilibrium.num_steps < 1:
        raise ValueError(f"equilibrium.num_steps must be >= 1, got {config.equilibrium.num_steps}")


def merge_configs(base: TokenizerConfig, override: dict[str, Any]) -> TokenizerConfig:
    """Merge override values into base config.

    Args:
        base: Base configuration
        override: Dictionary of override values

    Returns:
        New config with overrides applied
    """
    base_dict = base.to_dict()

    # Deep merge
    for key, value in override.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key].update(value)
        else:
            base_dict[key] = value

    return _parse_config(base_dict)
