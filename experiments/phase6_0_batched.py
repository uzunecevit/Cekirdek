#!/usr/bin/env python3
"""
VİCDAN — Hybrid v0.1: Batched STDP Update

Tüm sample'ların update'lerini topla, bir kere uygula.
Bu, mini-batch gradient descent'e eşdeğer.
"""

import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model import SpikingLM
from src.intent import extract_intent
from src.engine import run_engine

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


def encode_sequence(text, char2idx):
    return [char2idx.get(c, 0) for c in text if c in char2idx]


def get_next_token(model, prompt, char2idx, idx2char):
    model.eval()
    tokens = [char2idx.get(c, 0) for c in prompt if c in char2idx]
    idx = torch.tensor([tokens], device=DEVICE)
    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5 = probs.topk(5)
        return [
            (idx2char.get(str(top5.indices[i].item()), "?"), top5.values[i].item())
            for i in range(5)
        ]


def batched_stdp_step(model, samples, char2idx, lr):
    """
    Tüm sample'lar için update topla, bir kere uygula.
    """
    device = next(model.parameters()).device
    vocab_size = model.lm_head_static.weight.size(0)

    total_dw = torch.zeros_like(model.lm_head_fast)
    total_loss = 0
    n_valid = 0

    with torch.no_grad():
        for prompt, expected in samples:
            result = run_engine(extract_intent(prompt))
            if result is None:
                continue

            target_text = prompt + str(result)
            x = encode_sequence(prompt, char2idx)
            y_full = encode_sequence(target_text, char2idx)
            y = y_full[1:]  # next token alignment

            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            if len(x) < 2:
                continue

            idx = torch.tensor([x], device=device)
            target = torch.tensor([y], device=device)
            logits, loss = model(idx, targets=target)
            total_loss += loss.item()
            n_valid += 1

            probs = F.softmax(logits[0, :, :], dim=-1)
            t = logits.size(1) - 1
            target_token = y[-1]
            pre = model._last_hidden[0, t]

            target_onehot = torch.zeros(vocab_size, device=device)
            target_onehot[target_token] = 1.0
            error = target_onehot - probs[t]

            dw = lr * error.unsqueeze(1) * pre.unsqueeze(0)
            total_dw.add_(dw)

    # Toplu update
    model.lm_head_fast.add_(total_dw)
    model.lm_head_fast.clamp_(-10.0, 10.0)

    return total_loss / max(n_valid, 1), n_valid


def main():
    print("=" * 70)
    print("VİCDAN — Hybrid v0.1: Batched STDP Update")
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

    print("\n  Training Öncesi:")
    for prompt, expected in samples:
        top5 = get_next_token(model, prompt, char2idx, idx2char)
        top1 = top5[0][0]
        correct = top1 == expected[0]
        status = "✅" if correct else "❌"
        print(f"    {status} {prompt!r} → top1={top1!r} (beklenen: {expected!r})")

    # Training: 500 epoch, batched update
    print(f"\n  Training: 500 epoch, batched STDP, LR=0.01")
    LR = 0.01

    for epoch in range(500):
        avg_loss, n = batched_stdp_step(model, samples, char2idx, lr=LR)

        if (epoch + 1) % 50 == 0:
            model.consolidate(alpha=0.05, threshold=0.01)
            correct_count = 0
            for prompt, expected in samples:
                top5 = get_next_token(model, prompt, char2idx, idx2char)
                if top5[0][0] == expected[0]:
                    correct_count += 1
            print(
                f"    Epoch {epoch + 1:3d}: loss={avg_loss:.4f}, accuracy={correct_count}/{len(samples)}"
            )

    # Consolidation
    model.consolidate(alpha=0.2, threshold=0.01)

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
