#!/usr/bin/env python3
"""
VİCDAN — Hybrid v0.1: Direct Backprop Test

STDP yerine doğrudan backprop ile "3+4=7" öğrenebilir miyiz?
Bu, sorunun STDP'de mi yoksa model kapasitesinde mi olduğunu gösterir.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.model import SpikingLM

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


def idx2char_map(idx2char_raw):
    return {str(k): v for k, v in idx2char_raw.items()}


def get_next_token(
    model, prompt: str, char2idx: dict, idx2char: dict
) -> list[tuple[str, float]]:
    model.eval()
    tokens = [char2idx.get(c, 0) for c in prompt if c in char2idx]
    idx = torch.tensor([tokens], device=DEVICE)

    with torch.no_grad():
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top5 = probs.topk(5)

        result = []
        for i in range(5):
            idx_i = top5.indices[0, i].item()
            prob_i = top5.values[0, i].item()
            token = idx2char.get(str(idx_i), f"[{idx_i}]")
            result.append((token, prob_i))
        return result


def main():
    print("=" * 70)
    print("VİCDAN — Hybrid v0.1: Direct Backprop Test")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    idx2char = idx2char_map(idx2char_raw)

    model, ckpt = load_model()
    print(f"\n  Model yüklendi")

    # Input: "3+4=", Target: "7"
    prompt = "3+4="
    target_char = "7"
    target_idx = char2idx[target_char]

    input_tokens = [char2idx[c] for c in prompt]
    # Hedef: input'tan sonra target_char gelmeli
    # Yani model "3+4=" gördüğünde "7" üretmeli
    # Loss: cross entropy son pozisyonda
    x = torch.tensor([input_tokens], device=DEVICE)
    y = torch.tensor([input_tokens[1:] + [target_idx]], device=DEVICE)

    print(f"  Input: {prompt!r} → Target: {target_char!r} (idx={target_idx})")

    # Training öncesi
    print(f"\n  Training Öncesi:")
    top5 = get_next_token(model, prompt, char2idx, idx2char)
    for token, prob in top5:
        marker = " <-- HEDEF" if token == target_char else ""
        print(f"    {token!r}: {prob:.4f}{marker}")

    # Sadece LM head parametrelerini optimize et
    optimizer = optim.Adam(
        [
            {"params": model.lm_head_static.parameters(), "lr": 0.01},
            {"params": model.lm_head_fast, "lr": 0.01},
        ]
    )

    print(f"\n  Training: 500 adım, Adam, LR=0.01 (sadece LM head)")

    for i in range(500):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            top5 = get_next_token(model, prompt, char2idx, idx2char)
            target_prob = next((p for t, p in top5 if t == target_char), 0.0)
            top1 = top5[0][0] if top5 else "?"
            print(
                f"    Step {i + 1:3d}: loss={loss.item():.4f}, top1={top1!r}, target_prob={target_prob:.4f}"
            )

    # Training sonrası
    print(f"\n  Training Sonrası:")
    top5 = get_next_token(model, prompt, char2idx, idx2char)
    for token, prob in top5:
        marker = " <-- HEDEF" if token == target_char else ""
        print(f"    {token!r}: {prob:.4f}{marker}")

    # Full model ile de deneyelim
    print(f"\n  --- Full model training (tüm parametreler) ---")

    model2, _ = load_model()
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

    for i in range(500):
        optimizer2.zero_grad()
        logits, loss = model2(x, targets=y)
        loss.backward()
        optimizer2.step()

        if (i + 1) % 50 == 0:
            top5 = get_next_token(model2, prompt, char2idx, idx2char)
            target_prob = next((p for t, p in top5 if t == target_char), 0.0)
            top1 = top5[0][0] if top5 else "?"
            print(
                f"    Step {i + 1:3d}: loss={loss.item():.4f}, top1={top1!r}, target_prob={target_prob:.4f}"
            )

    print(f"\n  Full model Training Sonrası:")
    top5 = get_next_token(model2, prompt, char2idx, idx2char)
    for token, prob in top5:
        marker = " <-- HEDEF" if token == target_char else ""
        print(f"    {token!r}: {prob:.4f}{marker}")


if __name__ == "__main__":
    main()
