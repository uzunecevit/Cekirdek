#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 1.1: Re-Training (Yeni 64 Vocab ile)

Yeni vocab: 64 karakter (a-z, Türkçe, rakamlar, operatörler)
Hedef: loss < 2.5, spike oranı %10-20
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training config
EPOCHS = 5
BATCH_SIZE = 64
BLOCK_SIZE = 32
D_MODEL = 64
N_LAYER = 2
N_HEAD = 4
D_FF = 128
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4

# SNN config
THRESHOLD = 0.3
AMPLITUDE = 2.0
DECAY = 0.1

# ──────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────


def load_vocab():
    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    return vocab["vocab_size"], vocab["char2idx"], vocab["idx2char"]


def load_samples():
    path = os.path.join(DATA_DIR, "pretrain_samples.txt")
    with open(path, encoding="utf-8") as f:
        samples = [line.strip() for line in f if line.strip()]
    return samples


class CharDataset(Dataset):
    def __init__(self, samples, char2idx, block_size):
        self.char2idx = char2idx
        self.block_size = block_size
        self.sequences = []
        for s in samples:
            tokens = (
                [char2idx["<BOS>"]]
                + [char2idx[c] for c in s if c in char2idx]
                + [char2idx["<BOS>"]]
            )
            for i in range(len(tokens) - 1):
                chunk = tokens[i : i + block_size + 1]
                if len(chunk) > 1:
                    self.sequences.append(chunk)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(
            seq[1:], dtype=torch.long
        )


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    px = torch.zeros(len(xs), max_len, dtype=torch.long)
    py = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        px[i, : len(x)] = x
        py[i, : len(y)] = y
    return px, py


# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────


def train():
    print(f"Device: {DEVICE}")

    vocab_size, char2idx, idx2char = load_vocab()
    idx2char = {int(k): v for k, v in idx2char.items()}
    samples = load_samples()

    # Train/val split
    import random

    random.seed(42)
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = CharDataset(train_samples, char2idx, BLOCK_SIZE)
    val_ds = CharDataset(val_samples, char2idx, BLOCK_SIZE)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Vocab: {vocab_size}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = SpikingLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        d_ff=D_FF,
        block_size=BLOCK_SIZE,
        threshold=THRESHOLD,
        decay=DECAY,
        amplitude=AMPLITUDE,
        use_surrogate=True,
        ternary=True,
        fast_weight_fc2=True,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_dl)
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _, loss = model(xb, targets=yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model.eval()
        val_loss = 0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                _, loss = model(xb, targets=yb)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= n_batches

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}",
            end="",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": vocab_size,
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "config": {
                        "d_model": D_MODEL,
                        "n_layer": N_LAYER,
                        "n_head": N_HEAD,
                        "d_ff": D_FF,
                        "block_size": BLOCK_SIZE,
                    },
                },
                os.path.join(CKPT_DIR, "spiking_lm_v2.pt"),
            )
            print(f" *", end="")
        print()

    # Inference
    print("\n--- Generated Samples ---")
    model.eval()
    test_prompts = ["<BOS>", "koş", "3+4=", "ev"]
    for prompt in test_prompts:
        tokens = [char2idx["<BOS>"]] + [char2idx[c] for c in prompt if c in char2idx]
        idx = torch.tensor([tokens], device=DEVICE)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=16, temperature=0.8)
        generated = "".join(
            idx2char[t.item()]
            for t in out[0, 1:]
            if t.item() != char2idx["<BOS>"] and t.item() < vocab_size
        )
        print(f"  '{prompt}' → '{generated}'")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved: {os.path.join(CKPT_DIR, 'spiking_lm_v2.pt')}")


if __name__ == "__main__":
    train()
