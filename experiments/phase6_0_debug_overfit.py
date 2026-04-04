#!/usr/bin/env python3
"""
VİCDAN — Hybrid v0.1: Debug Single Example Overfit

Tek bir örnek ("3+4=7") ile model overfit edebiliyor mu?
"""

import os
import sys
import json

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
    model.eval()
    return model, ckpt


def idx2char_map(idx2char_raw):
    return {str(k): v for k, v in idx2char_raw.items()}


def get_next_token(
    model, prompt: str, char2idx: dict, idx2char: dict
) -> tuple[str, float]:
    """Modelin bir sonraki token'ını ve olasılığını döndür."""
    model.eval()
    tokens = [char2idx.get(c, 0) for c in prompt if c in char2idx]
    if not tokens:
        return "?", 0.0

    idx = torch.tensor([tokens], device=DEVICE)

    with torch.no_grad():
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top5 = probs.topk(5)

        top5_tokens = []
        for i in range(5):
            idx_i = top5.indices[0, i].item()
            prob_i = top5.values[0, i].item()
            token = idx2char.get(str(idx_i), f"[{idx_i}]")
            top5_tokens.append((token, prob_i))

        return top5_tokens


def main():
    print("=" * 70)
    print("VİCDAN — Hybrid v0.1: Single Example Overfit Test")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    idx2char = idx2char_map(idx2char_raw)

    model, ckpt = load_model()
    print(f"\n  Model yüklendi: spiking_lm_v2.pt")

    # Tek örnek: "3+4=" → "7"
    prompt = "3+4="
    target = "7"
    target_idx = char2idx.get("7", -1)
    print(f"  Hedef: {prompt!r} → {target!r} (idx={target_idx})")

    # Training öncesi
    print(f"\n  Training Öncesi:")
    top5 = get_next_token(model, prompt, char2idx, idx2char)
    for token, prob in top5:
        marker = " <-- HEDEF" if token == target else ""
        print(f"    {token!r}: {prob:.4f}{marker}")

    # Training: Tek örnek, 200 iterasyon + consolidation
    print(f"\n  Training: 200 iterasyon, LR=0.01 (consolidation her 50 adımda)")

    char2idx_dict = char2idx

    for i in range(200):
        result = hybrid_step(model, prompt, char2idx_dict, lr=0.01)

        if (i + 1) % 50 == 0:
            model.consolidate(alpha=0.05, threshold=0.01)

            top5 = get_next_token(model, prompt, char2idx, idx2char)
            target_prob = next((p for t, p in top5 if t == target), 0.0)
            top1 = top5[0][0] if top5 else "?"
            print(
                f"    Step {i + 1:3d}: loss={result['loss']:.4f}, top1={top1!r}, target_prob={target_prob:.4f}"
            )

    # Training sonrası
    print(f"\n  Training Sonrası:")
    top5 = get_next_token(model, prompt, char2idx, idx2char)
    for token, prob in top5:
        marker = " <-- HEDEF" if token == target else ""
        print(f"    {token!r}: {prob:.4f}{marker}")

    # Generate ile tam test
    print(f"\n  Generate Test:")
    model.eval()
    tokens = [char2idx.get(c, 0) for c in prompt if c in char2idx]
    idx = torch.tensor([tokens], device=DEVICE)

    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=3)
        result_chars = []
        for t in range(
            tokens[-1].shape[0] if hasattr(tokens[-1], "shape") else 0,
            generated.shape[1],
        ):
            c = idx2char.get(str(generated[0, t].item()), "?")
            result_chars.append(c)
        print(f"    {prompt!r} → {''.join(result_chars)!r}")

    # Fast weights durumu
    fast_norm = model.lm_head_fast.norm().item()
    static_norm = model.lm_head_static.weight.norm().item()
    print(f"\n  W_fast norm: {fast_norm:.4f}")
    print(f"  W_static norm: {static_norm:.4f}")


if __name__ == "__main__":
    main()
