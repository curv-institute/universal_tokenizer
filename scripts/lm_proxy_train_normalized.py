#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
# ]
# ///
"""Byte-normalized LM proxy training script.

This script normalizes for potential confounds by:
1. Setting context length to achieve target bytes per context (not tokens)
2. Training for a fixed byte budget (not steps)
3. Reporting bytes processed for fairness verification
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ByteNormalizedDataset(Dataset):
    """Dataset that normalizes by bytes, not tokens."""

    def __init__(self, tokens: torch.Tensor, context_tokens: int, bytes_per_token: float):
        self.tokens = tokens
        self.context_tokens = context_tokens
        self.bytes_per_token = bytes_per_token
        self.num_sequences = max(0, len(tokens) - context_tokens)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        x = self.tokens[idx : idx + self.context_tokens]
        y = self.tokens[idx + 1 : idx + self.context_tokens + 1]
        # Return bytes represented by this sample
        bytes_in_sample = len(x) * self.bytes_per_token
        return x, y, bytes_in_sample


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

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_len, hidden_size)

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
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = tok_emb + pos_emb

        mask = self.causal_mask[:seq_len, :seq_len]
        h = self.transformer(h, h, tgt_mask=mask)
        logits = self.output_proj(h)
        return logits


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_normalized(
    tokens_path: Path,
    info_path: Path,
    output_dir: Path,
    target_context_bytes: int = 2048,
    target_total_bytes: int = 500_000,
    batch_size: int = 8,
    seed: int = 42,
):
    """Run byte-normalized training."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load token info
    with open(info_path) as f:
        info = json.load(f)

    vocab_size = info["vocab_size"]
    bytes_per_token = info["bytes_per_token"]
    num_bytes = info["num_bytes"]
    hhc_enabled = info.get("hhc_enabled", False)

    print(f"Token info: vocab={vocab_size}, bytes_per_token={bytes_per_token:.2f}")
    print(f"HHC enabled: {hhc_enabled}")

    # Load tokens
    tokens = np.load(tokens_path)
    tokens = torch.from_numpy(tokens).long().flatten()
    print(f"Loaded {len(tokens)} tokens representing {num_bytes} bytes")

    # Compute normalized context length (target bytes / bytes_per_token)
    context_tokens = max(32, min(512, round(target_context_bytes / bytes_per_token)))
    actual_context_bytes = context_tokens * bytes_per_token
    print(f"Context: {context_tokens} tokens = {actual_context_bytes:.0f} bytes (target: {target_context_bytes})")

    # Validate
    max_token = tokens.max().item()
    if max_token >= vocab_size:
        vocab_size = max_token + 1

    # Create dataset
    dataset = ByteNormalizedDataset(tokens, context_tokens, bytes_per_token)
    if len(dataset) == 0:
        raise ValueError(f"Not enough tokens for context length {context_tokens}")

    bytes_per_sample = context_tokens * bytes_per_token
    samples_for_budget = max(1, int(target_total_bytes / bytes_per_sample))

    print(f"Dataset: {len(dataset)} sequences, {bytes_per_sample:.0f} bytes/sample")
    print(f"Target: {target_total_bytes} bytes = ~{samples_for_budget} samples")

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,
    )

    # Create model with normalized context
    model = TinyLM(
        vocab_size=vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        context_len=context_tokens,
        dropout=0.0,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop - stop by bytes, not steps
    loss_log = []
    start_time = time.time()

    total_bytes_processed = 0
    total_tokens_processed = 0
    step = 0
    epoch = 0

    model.train()
    pbar = tqdm(total=target_total_bytes, desc="Bytes", unit="B", unit_scale=True)

    while total_bytes_processed < target_total_bytes:
        epoch += 1
        for batch_x, batch_y, batch_bytes in dataloader:
            if total_bytes_processed >= target_total_bytes:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()

            batch_bytes_total = batch_bytes.sum().item()
            total_bytes_processed += batch_bytes_total
            total_tokens_processed += batch_x.numel()
            step += 1

            pbar.update(batch_bytes_total)

            # Log every 10 steps
            if step % 10 == 0:
                loss_log.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "bytes_processed": total_bytes_processed,
                    "tokens_processed": total_tokens_processed,
                })
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    pbar.close()
    elapsed = time.time() - start_time
    final_loss = loss.item()

    # Final metrics
    print(f"\nTraining complete!")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Steps: {step}")
    print(f"  Epochs: {epoch}")
    print(f"  Bytes processed: {total_bytes_processed:,}")
    print(f"  Tokens processed: {total_tokens_processed:,}")
    print(f"  Time: {elapsed:.1f}s")

    # Save results
    results = {
        "final_loss": final_loss,
        "steps": step,
        "epochs": epoch,
        "bytes_processed": total_bytes_processed,
        "tokens_processed": total_tokens_processed,
        "time_elapsed_s": elapsed,
        "seed": seed,
        "hhc_enabled": hhc_enabled,
        "normalization": {
            "target_context_bytes": target_context_bytes,
            "actual_context_bytes": actual_context_bytes,
            "context_tokens": context_tokens,
            "bytes_per_token": bytes_per_token,
            "target_total_bytes": target_total_bytes,
            "bytes_per_sample": bytes_per_sample,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "loss_log.jsonl", "w") as f:
        for entry in loss_log:
            f.write(json.dumps(entry) + "\n")

    print(f"Results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Byte-normalized LM proxy training")
    parser.add_argument("--tokens", type=Path, required=True, help="Path to tokens .npy file")
    parser.add_argument("--info", type=Path, required=True, help="Path to token info .json file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--context-bytes", type=int, default=2048, help="Target context bytes")
    parser.add_argument("--total-bytes", type=int, default=500_000, help="Total bytes to train on")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_normalized(
        tokens_path=args.tokens,
        info_path=args.info,
        output_dir=args.output,
        target_context_bytes=args.context_bytes,
        target_total_bytes=args.total_bytes,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
