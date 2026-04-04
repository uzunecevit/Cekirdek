#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 1: Temel SNN Eğitimi
Karakter düzeyinde isim üretme (makemore tarzı).
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "../../VİCDAN_HEBBIAN/micro/names.txt"
)
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
VOCAB_SIZE = None
D_MODEL = 64
N_LAYER = 2
N_HEAD = 4
D_FF = 128
BLOCK_SIZE = 16
THRESHOLD = 0.3  # Optimal spike rate
DECAY = 0.3
USE_SURROGATE = True

# Training
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
MAX_SAMPLES = 5000  # 205K → 5K (hız için)

# ──────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────


class NameDataset(Dataset):
    def __init__(self, names, chars, block_size):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.BOS = len(chars)
        self.vocab_size = len(chars) + 1
        self.block_size = block_size
        self.sequences = []
        for name in names:
            tokens = [self.BOS] + [self.stoi[c] for c in name] + [self.BOS]
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


def load_data(path):
    import random

    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    random.seed(42)
    random.shuffle(names)
    split = int(len(names) * 0.9)
    return names[:split], names[split:]


# ──────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────


def train():
    print(f"Device: {DEVICE}")
    print(f"Surrogate gradient: {USE_SURROGATE}")
    print(f"Threshold: {THRESHOLD}, Decay: {DECAY}")

    # Data
    train_names, val_names = load_data(DATA_PATH)
    train_names = train_names[:MAX_SAMPLES]  # Hız için küçült
    chars = sorted(set("".join(train_names)))
    train_ds = NameDataset(train_names, chars, BLOCK_SIZE)
    val_ds = NameDataset(val_names, chars, BLOCK_SIZE)
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

    global VOCAB_SIZE
    VOCAB_SIZE = train_ds.vocab_size
    print(f"Vocab: {VOCAB_SIZE}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = SpikingLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        d_ff=D_FF,
        block_size=BLOCK_SIZE,
        threshold=THRESHOLD,
        decay=DECAY,
        use_surrogate=USE_SURROGATE,
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
            logits, loss = model(xb, targets=yb)
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

        spike_rate = model.spike_rate()
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f} | spike: {spike_rate:.1%}",
            end="",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": VOCAB_SIZE,
                    "chars": chars,
                    "stoi": train_ds.stoi,
                    "itos": {i: c for c, i in train_ds.stoi.items()},
                    "BOS": train_ds.BOS,
                    "config": {
                        "d_model": D_MODEL,
                        "n_layer": N_LAYER,
                        "n_head": N_HEAD,
                        "d_ff": D_FF,
                        "block_size": BLOCK_SIZE,
                    },
                },
                os.path.join(CKPT_DIR, "spiking_lm.pt"),
            )
            print(f" *", end="")
        print()

    # Inference
    print("\n--- Generated Names ---")
    model.eval()
    itos = {i: c for c, i in train_ds.stoi.items()}
    BOS = train_ds.BOS
    with torch.no_grad():
        for i in range(10):
            idx = torch.tensor([[BOS]], device=DEVICE)
            out = model.generate(idx, max_new_tokens=16, temperature=0.8)
            name = "".join(
                itos[t.item()]
                for t in out[0, 1:]
                if t.item() != BOS and t.item() < len(chars)
            )
            print(f"  {i + 1:2d}: {name}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved: {os.path.join(CKPT_DIR, 'spiking_lm.pt')}")


if __name__ == "__main__":
    train()
