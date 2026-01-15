#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "pyyaml",
#   "tqdm",
# ]
# ///
"""Train the Universal Lossless Tokenizer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.config import load_config
from tokenizer.bundle import UniversalTokenizer
from tokenizer.data import ByteDataset, BatchIterator, generate_text_like
from tokenizer.metrics import compute_metrics


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    config_path: Path,
    data_path: Path | None = None,
    output_dir: Path = Path("checkpoints"),
    name: str | None = None,
) -> None:
    """Train tokenizer.

    Args:
        config_path: Path to config TOML
        data_path: Path to training data (generates synthetic if None)
        output_dir: Output directory for checkpoints
        name: Experiment name
    """
    # Load config
    config = load_config(config_path)
    set_seed(config.train.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = name or f"train_{timestamp}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if data_path and data_path.exists():
        print(f"Loading data from {data_path}")
        with open(data_path, "rb") as f:
            data = f.read()
    else:
        print("Generating synthetic training data")
        data = generate_text_like(100_000, seed=config.train.seed)

    dataset = ByteDataset(data=data, chunk_size=256)
    loader = BatchIterator(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        seed=config.train.seed,
    )

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = UniversalTokenizer(config)
    tokenizer.train_mode()

    # Optimizer
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # Training loop
    print(f"Training for {config.train.max_steps} steps...")
    step = 0
    total_loss = 0.0
    pbar = tqdm(total=config.train.max_steps, desc="Training")

    while step < config.train.max_steps:
        for batch in loader:
            if step >= config.train.max_steps:
                break

            optimizer.zero_grad()

            # Process batch
            batch_loss = 0.0
            for chunk in batch:
                # Encode spans
                spans = [chunk[i : i + 8] for i in range(0, len(chunk) - 7, 4)]
                if not spans:
                    continue

                z = tokenizer.encoder.encode(spans)
                z_eq = tokenizer.equilibrium.project(z)
                _, ids, commit_loss = tokenizer.codebook(z_eq)

                # Loss is commitment loss (VQ-VAE style)
                batch_loss += commit_loss

            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
                optimizer.step()

                total_loss += batch_loss.item()

            step += 1
            pbar.update(1)

            # Logging
            if step % 100 == 0:
                avg_loss = total_loss / 100
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                total_loss = 0.0

            # Checkpoint
            if step % config.train.checkpoint_interval == 0:
                ckpt_path = exp_dir / f"checkpoint_{step}.pt"
                tokenizer.save(ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")

    pbar.close()

    # Save final model
    final_path = exp_dir / "final.pt"
    tokenizer.save(final_path)
    print(f"Saved final model to {final_path}")

    # Save training info
    info = {
        "config": str(config_path),
        "steps": step,
        "timestamp": timestamp,
        "name": exp_name,
    }
    with open(exp_dir / "train_info.json", "w") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Universal Tokenizer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )

    args = parser.parse_args()
    train(args.config, args.data, args.output, args.name)


if __name__ == "__main__":
    main()
