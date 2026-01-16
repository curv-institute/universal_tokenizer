#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
# ]
# ///
"""Minimal autoregressive LM proxy training script."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TokenDataset(Dataset):
    """Dataset that creates sequences of fixed length from a token array."""

    def __init__(self, tokens: torch.Tensor, context_len: int):
        self.tokens = tokens
        self.context_len = context_len
        # We need context_len + 1 tokens to get input and target
        self.num_sequences = len(tokens) - context_len

    def __len__(self) -> int:
        return max(0, self.num_sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Input is tokens[idx:idx+context_len], target is tokens[idx+1:idx+context_len+1]
        x = self.tokens[idx : idx + self.context_len]
        y = self.tokens[idx + 1 : idx + self.context_len + 1]
        return x, y


class TinyLM(nn.Module):
    """Minimal decoder-only transformer language model."""

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        context_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_len = context_len

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_len, hidden_size)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        # Causal mask (registered as buffer so it moves with the model)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Get embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = tok_emb + pos_emb

        # Apply transformer with causal masking
        # TransformerDecoder expects memory input, we use the input itself as memory
        # with appropriate masking for decoder-only behavior
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        h = self.transformer(h, h, tgt_mask=causal_mask, memory_mask=causal_mask)

        # Project to vocabulary
        logits = self.output_proj(h)
        return logits


def load_tokens(path: Path) -> torch.Tensor:
    """Load tokens from .npy or .pt file."""
    if path.suffix == ".npy":
        tokens = np.load(path)
        tokens = torch.from_numpy(tokens).long()
    elif path.suffix == ".pt":
        tokens = torch.load(path, weights_only=True)
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(f"Expected tensor in {path}, got {type(tokens)}")
        tokens = tokens.long()
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Flatten if multi-dimensional
    tokens = tokens.flatten()
    return tokens


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(
    tokens_path: Path,
    output_dir: Path,
    steps: int = 10000,
    batch_size: int = 32,
    seed: int = 42,
    vocab_size: int = 1024,
    context_len: int = 512,
):
    """Run training loop."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading tokens from {tokens_path}")
    tokens = load_tokens(tokens_path)
    print(f"Loaded {len(tokens)} tokens")

    # Validate vocab size
    max_token = tokens.max().item()
    if max_token >= vocab_size:
        print(f"Warning: max token {max_token} >= vocab_size {vocab_size}, adjusting vocab_size")
        vocab_size = max_token + 1

    # Create dataset and dataloader
    dataset = TokenDataset(tokens, context_len)
    if len(dataset) == 0:
        raise ValueError(f"Not enough tokens ({len(tokens)}) for context length {context_len}")

    print(f"Dataset size: {len(dataset)} sequences")

    # Use a generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,  # For determinism
    )

    # Create model
    model = TinyLM(
        vocab_size=vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        context_len=context_len,
        dropout=0.0,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    loss_log_path = output_dir / "loss_log.jsonl"
    start_time = time.time()
    step = 0
    epoch = 0
    running_loss = 0.0

    model.train()
    pbar = tqdm(total=steps, desc="Training")

    with open(loss_log_path, "w") as loss_log:
        while step < steps:
            epoch += 1
            for x, y in dataloader:
                if step >= steps:
                    break

                x = x.to(device)
                y = y.to(device)

                # Forward pass
                optimizer.zero_grad()
                logits = model(x)

                # Compute loss
                # Reshape for cross entropy: (batch * seq, vocab) and (batch * seq,)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

                # Backward pass
                loss.backward()
                optimizer.step()

                step += 1
                running_loss += loss.item()

                # Logging
                if step % 100 == 0:
                    avg_loss = running_loss / 100
                    log_entry = {
                        "step": step,
                        "loss": avg_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    loss_log.write(json.dumps(log_entry) + "\n")
                    loss_log.flush()
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    running_loss = 0.0

                # Checkpointing
                if step % 2000 == 0:
                    checkpoint_path = output_dir / f"checkpoint_{step}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "vocab_size": vocab_size,
                            "context_len": context_len,
                        },
                        checkpoint_path,
                    )
                    print(f"\nSaved checkpoint to {checkpoint_path}")

                pbar.update(1)

    pbar.close()
    elapsed_time = time.time() - start_time

    # Final evaluation loss (on a fresh batch)
    model.eval()
    total_eval_loss = 0.0
    eval_batches = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_eval_loss += loss.item()
            eval_batches += 1
            if eval_batches >= 10:
                break

    final_loss = total_eval_loss / max(eval_batches, 1)

    # Save final metrics
    final_metrics = {
        "final_loss": final_loss,
        "total_steps": step,
        "elapsed_time_seconds": elapsed_time,
        "vocab_size": vocab_size,
        "context_len": context_len,
        "batch_size": batch_size,
        "seed": seed,
    }

    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Time elapsed: {elapsed_time:.1f}s")
    print(f"Results saved to {output_dir}")

    # Save final checkpoint
    final_checkpoint_path = output_dir / "checkpoint_final.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "context_len": context_len,
        },
        final_checkpoint_path,
    )
    print(f"Saved final checkpoint to {final_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a minimal autoregressive LM")
    parser.add_argument(
        "--tokens",
        type=Path,
        required=True,
        help="Path to token file (.npy or .pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of training steps (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="Vocabulary size (default: 1024)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=512,
        help="Context length (default: 512)",
    )

    args = parser.parse_args()

    train(
        tokens_path=args.tokens,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        vocab_size=args.vocab_size,
        context_len=args.context_len,
    )


if __name__ == "__main__":
    main()
