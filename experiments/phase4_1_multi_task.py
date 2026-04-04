#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 4.1: Multi-Task Stability Test

Sıralı öğrenme protokolü:
1. Baseline loss'ları ölç (her task için)
2. Task B (Türkçe) → 5 epoch STDP + consolidation
3. Task C (Matematik) → 5 epoch STDP + consolidation
4. Forgetting ölçümü: Her task sonrası tüm task'ların loss'unu tekrar ölç
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
EPOCHS = 5
BATCH_SIZE = 32
BLOCK_SIZE = 32
LR = 0.0005
CONSOLIDATION_ALPHA = 0.05
SURPRISE_THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────


def load_vocab():
    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    return vocab["vocab_size"], vocab["char2idx"], vocab["idx2char"]


class CharDataset(Dataset):
    def __init__(self, lines, char2idx, block_size):
        self.char2idx = char2idx
        self.block_size = block_size
        self.sequences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            tokens = (
                [char2idx["<BOS>"]]
                + [char2idx[c] for c in line if c in char2idx]
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


def load_lines(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_dataloader(lines, char2idx, batch_size=BATCH_SIZE):
    ds = CharDataset(lines, char2idx, BLOCK_SIZE)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    ), len(ds)


def load_model():
    path = os.path.join(CKPT_DIR, "spiking_lm_v2.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    config = ckpt["config"]
    model = SpikingLM(
        vocab_size=ckpt["vocab_size"],
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        d_ff=config["d_ff"],
        block_size=config["block_size"],
        threshold=0.3,
        decay=0.1,
        amplitude=2.0,
        use_surrogate=True,
        ternary=True,
        fast_weight_fc2=True,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _, loss = model(xb, targets=yb)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def compute_surprise(probs):
    import math

    eps = 1e-9
    H = -(probs * torch.log(probs + eps)).sum()
    H_max = math.log(probs.size(-1))
    return (H / H_max).item()


def apply_stdp_gated(model, x, y, lr, surprise_threshold, ffn_lr=0.0):
    """
    STDP update: LM Head + FFN fc2 (layer-wise).
    """
    with torch.no_grad():
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        if len(x) < 2:
            return 0.0, 0.0

        idx = torch.tensor([x], device=DEVICE)
        target = torch.tensor([y], device=DEVICE)
        logits, loss = model(idx, targets=target)
        vocab_size = logits.size(-1)
        last_probs = F.softmax(logits[0, -1, :], dim=-1)
        surprise = compute_surprise(last_probs)

        if surprise > surprise_threshold:
            for t in range(logits.size(1)):
                probs = F.softmax(logits[0, t, :], dim=-1)
                target_token = y[t]
                target_onehot = torch.zeros(vocab_size, device=DEVICE)
                target_onehot[target_token] = 1.0
                token_idx = x[min(t, len(x) - 1)]
                pre = model.wte.weight[token_idx]
                error = target_onehot - probs
                dw = lr * error.unsqueeze(1) * pre.unsqueeze(0)
                model.lm_head_fast.add_(dw)
                model.lm_head_fast.clamp_(-1.0, 1.0)

                # FFN fc2 STDP update
                if ffn_lr > 0:
                    for block in model.blocks:
                        fc2 = block.ffn.fc2
                        if fc2.fast_weight and block.ffn._pre_activation is not None:
                            # pre: (1, d_ff), post: (1, d_model)
                            pre_act = block.ffn._pre_activation[0]  # (d_ff,)
                            post_act = block.ffn._post_activation[0]  # (d_model,)

                            # fc2 fast weight: (d_model, d_ff)
                            # ΔW = ffn_lr * outer(post_act, pre_act)
                            # Ama direction belirlemek için error sinyali kullan
                            # Basit: post_act * pre_act^T * |error| (magnitude)
                            error_mag = error.abs().mean()
                            dw_ffn = ffn_lr * error_mag * torch.outer(post_act, pre_act)
                            fc2.weight_fast.add_(dw_ffn)
                            fc2.weight_fast.clamp_(-1.0, 1.0)

        return loss.item(), surprise


def train_task(model, dataloader, task_name, epochs=EPOCHS, ffn_lr=0.0):
    """Bir task üzerinde STDP + consolidation ile eğit."""
    print(f"\n  📚 {task_name} eğitimi ({epochs} epoch), ffn_lr={ffn_lr}...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            for b in range(xb.size(0)):
                # -100 padding pozisyonlarını bul (yb'den)
                mask = yb[b] != -100
                if mask.sum() < 2:
                    continue
                x = xb[b][mask].tolist()
                y = yb[b][mask].tolist()
                if len(x) < 2 or len(y) < 2:
                    continue
                # Uzunlukları eşitle
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                loss, _ = apply_stdp_gated(model, x, y, LR, SURPRISE_THRESHOLD, ffn_lr=ffn_lr)
                epoch_loss += loss
                n_batches += 1

        # Epoch sonu consolidation
        report = model.consolidate(alpha=CONSOLIDATION_ALPHA, threshold=0.5)
        avg_loss = epoch_loss / max(n_batches, 1)
        consolidated = report.get("total_consolidated", 0)
        print(
            f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, consolidated={consolidated}"
        )


def main():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 4.1: Multi-Task Stability")
    print("=" * 70)

    vocab_size, char2idx, idx2char = load_vocab()
    idx2char = {int(k): v for k, v in idx2char.items()}

    # Dataloader'lar
    names_lines = load_lines("pretrain_samples.txt")
    turkish_lines = load_lines("turkish.txt")
    math_lines = load_lines("math.txt")

    names_dl, names_n = make_dataloader(names_lines, char2idx)
    turkish_dl, turkish_n = make_dataloader(turkish_lines, char2idx)
    math_dl, math_n = make_dataloader(math_lines, char2idx)

    print(f"\n  Dataset boyutları:")
    print(f"    İsimler: {names_n} sequence")
    print(f"    Türkçe: {turkish_n} sequence")
    print(f"    Matematik: {math_n} sequence")

    # Model yükle
    model, ckpt = load_model()
    print(f"\n  Model yüklendi: spiking_lm_v2.pt")

    # ── Adım 1: Baseline ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Baseline Loss'lar")
    print(f"{'=' * 70}")

    baseline = {
        "names": evaluate_loss(model, names_dl),
        "turkish": evaluate_loss(model, turkish_dl),
        "math": evaluate_loss(model, math_dl),
    }

    print(f"  İsimler:   {baseline['names']:.4f}")
    print(f"  Türkçe:    {baseline['turkish']:.4f}")
    print(f"  Matematik: {baseline['math']:.4f}")

    # ── Adım 2: Türkçe Öğrenme ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Türkçe Öğrenme")
    print(f"{'=' * 70}")

    train_task(model, turkish_dl, "Türkçe")

    # Post-Türkçe loss'lar
    post_turkish = {
        "names": evaluate_loss(model, names_dl),
        "turkish": evaluate_loss(model, turkish_dl),
        "math": evaluate_loss(model, math_dl),
    }

    print(f"\n  Post-Türkçe Loss'lar:")
    print(
        f"    İsimler:   {post_turkish['names']:.4f} ({(post_turkish['names'] / baseline['names'] - 1) * 100:+.1f}%)"
    )
    print(
        f"    Türkçe:    {post_turkish['turkish']:.4f} ({(post_turkish['turkish'] / baseline['turkish'] - 1) * 100:+.1f}%)"
    )
    print(
        f"    Matematik: {post_turkish['math']:.4f} ({(post_turkish['math'] / baseline['math'] - 1) * 100:+.1f}%)"
    )

    # ── Adım 3: Matematik Öğrenme ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Matematik Öğrenme")
    print(f"{'=' * 70}")

    train_task(model, math_dl, "Matematik")

    # Post-Matematik loss'lar
    post_math = {
        "names": evaluate_loss(model, names_dl),
        "turkish": evaluate_loss(model, turkish_dl),
        "math": evaluate_loss(model, math_dl),
    }

    print(f"\n  Post-Matematik Loss'lar:")
    print(
        f"    İsimler:   {post_math['names']:.4f} ({(post_math['names'] / baseline['names'] - 1) * 100:+.1f}%)"
    )
    print(
        f"    Türkçe:    {post_math['turkish']:.4f} ({(post_math['turkish'] / baseline['turkish'] - 1) * 100:+.1f}%)"
    )
    print(
        f"    Matematik: {post_math['math']:.4f} ({(post_math['math'] / baseline['math'] - 1) * 100:+.1f}%)"
    )

    # ── Özet ──
    print(f"\n{'=' * 70}")
    print("FORGETTING RAPORU")
    print(f"{'=' * 70}")

    names_forgetting = (post_math["names"] / baseline["names"] - 1) * 100
    turkish_forgetting = (post_math["turkish"] / baseline["turkish"] - 1) * 100

    print(f"  İsimler forgetting:   {names_forgetting:+.1f}%")
    print(f"  Türkçe forgetting:    {turkish_forgetting:+.1f}%")

    if names_forgetting < 20 and turkish_forgetting < 20:
        print(f"\n  ✅ BAŞARILI: Catastrophic forgetting < %20")
    else:
        print(f"\n  ⚠️ Forgetting yüksek: Consolidation ayarı gözden geçirilmeli")

    # Kaydet
    results = {
        "baseline": baseline,
        "post_turkish": post_turkish,
        "post_math": post_math,
        "names_forgetting_pct": names_forgetting,
        "turkish_forgetting_pct": turkish_forgetting,
    }

    result_path = os.path.join(DATA_DIR, "..", "experiments", "phase4_1_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")


if __name__ == "__main__":
    main()
