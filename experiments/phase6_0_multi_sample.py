#!/usr/bin/env python3
"""
VİCDAN — Hybrid v0.1: Multi-Sample Training (Doğru Alignment)

Birden fazla matematik ifadesi ile training.
Consolidation sadece training sonunda.
"""

import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model import SpikingLM
from src.hybrid import hybrid_step

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 32


def load_vocab():
    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    return vocab["vocab_size"], vocab["char2idx"], vocab["idx2char"]


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
    return model, ckpt


def idx2char_map(raw):
    return {str(k): v for k, v in raw.items()}


def get_next_token(model, prompt, char2idx, idx2char):
    model.eval()
    tokens = [char2idx.get(c, 0) for c in prompt if c in char2idx]
    idx = torch.tensor([tokens], device=DEVICE)
    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5 = probs.topk(5)
        return [
            (
                idx2char.get(str(top5.indices[i].item()), "?"),
                top5.values[i].item(),
            )
            for i in range(5)
        ]


def main():
    print("=" * 70)
    print("VİCDAN — Hybrid v0.1: Multi-Sample Training")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    idx2char = idx2char_map(idx2char_raw)
    model, _ = load_model()

    samples = [
        ("3+4=", "7"),
        ("5-2=", "3"),
        ("1+1=", "2"),
        ("9-3=", "6"),
        ("2*3=", "6"),
        ("7+8=", "15"),
        ("4*4=", "16"),
        ("0+5=", "5"),
    ]

    # Training öncesi
    print("\n  Training Öncesi:")
    for prompt, expected in samples:
        top5 = get_next_token(model, prompt, char2idx, idx2char)
        top1 = top5[0][0]
        correct = top1 == expected[0]
        status = "✅" if correct else "❌"
        print(f"    {status} {prompt!r} → top1={top1!r} (beklenen: {expected!r})")

    # Training: 5 epoch, her epoch tüm sample'lar
    print(f"\n  Training: 5 epoch, LR=0.01, consolidation SONRA")
    LR = 0.01

    for epoch in range(5):
        epoch_loss = 0
        for prompt, expected in samples:
            result = hybrid_step(model, prompt, char2idx, lr=LR)
            if result["loss"] is not None:
                epoch_loss += result["loss"]
        avg_loss = epoch_loss / len(samples)
        print(f"    Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

    # Consolidation
    print(f"\n  Consolidating...")
    report = model.consolidate(alpha=0.1, threshold=0.01)
    print(f"    Consolidated: {report.get('total_consolidated', 0)} layers")

    # Training sonrası
    print("\n  Training Sonrası:")
    correct_count = 0
    for prompt, expected in samples:
        top5 = get_next_token(model, prompt, char2idx, idx2char)
        top1 = top5[0][0]
        correct = top1 == expected[0]
        if correct:
            correct_count += 1
        status = "✅" if correct else "❌"
        print(f"    {status} {prompt!r} → top1={top1!r} (beklenen: {expected!r})")

    print(
        f"\n  Accuracy: {correct_count}/{len(samples)} ({correct_count / len(samples) * 100:.0f}%)"
    )

    # Generalization
    print(f"\n  Generalization:")
    gen_samples = [
        ("6+1=", "7"),
        ("8-5=", "3"),
        ("3*2=", "6"),
    ]
    for prompt, expected in gen_samples:
        top5 = get_next_token(model, prompt, char2idx, idx2char)
        top1 = top5[0][0]
        correct = top1 == expected[0]
        status = "✅" if correct else "❌"
        print(f"    {status} {prompt!r} → top1={top1!r} (beklenen: {expected!r})")


if __name__ == "__main__":
    main()
