#!/usr/bin/env python3
"""
VİCDAN — Phase 6.1: Intent-Driven Execution

LM head artık sonuç token'ı DEĞİL, intent (operatör) üretiyor.

Eski: "3+4=" → "7"  (ezberleme, generalization yok)
Yeni: "3+4=" → "+"  (intent: ADD → engine hesaplar → 7)

Genelleme testi:
  Training: "3+4=" → "+", "5-2=" → "-", "2*3=" → "*"
  Test:     "6+1=" → "+"  (görmediği örnek, doğru intent?)
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.model import SpikingLM
from src.engine import run_engine
from src.intent import extract_intent

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 32

# Intent token'ları (vocab'da zaten var)
OP_TOKENS = {"+": "ADD", "-": "SUB", "*": "MUL"}


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


def encode(text, char2idx):
    return [char2idx.get(c, 0) for c in text if c in char2idx]


def get_intent_token(prompt: str) -> str:
    """Input'tan intent (operatör) token'ını çıkar."""
    for op in ["+", "-", "*"]:
        if op in prompt:
            return op
    return None


def predict_intent(
    model, prompt: str, char2idx: dict, idx2char: dict
) -> tuple[str, float]:
    """Modelin intent tahmini (son token'ın top-1'i)."""
    model.eval()
    tokens = encode(prompt, char2idx)
    idx = torch.tensor([tokens], device=DEVICE)
    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top1_idx = probs.argmax().item()
        top1_prob = probs[top1_idx].item()
        top1_token = idx2char.get(str(top1_idx), "?")
        return top1_token, top1_prob


def build_intent_batch(samples: list[str], char2idx: dict):
    """
    Intent training için batch oluştur.

    Input: "3+4=" → Target: "+" (operatör token'ı)
    """
    all_x = []
    all_y = []

    for prompt in samples:
        op = get_intent_token(prompt)
        if op is None:
            continue

        x = encode(prompt, char2idx)
        y = encode(prompt[:-1] + op, char2idx)  # Son token = operatör
        # Ya da daha basit: son token direkt op
        y = x[:-1] + [char2idx[op]]  # Son karakteri op ile değiştir

        all_x.append(x)
        all_y.append(y)

    if not all_x:
        return None, None

    max_len = max(len(x) for x in all_x)
    B = len(all_x)

    padded_x = torch.zeros(B, max_len, dtype=torch.long, device=DEVICE)
    padded_y = torch.full((B, max_len), -100, dtype=torch.long, device=DEVICE)

    for i, (x, y) in enumerate(zip(all_x, all_y)):
        padded_x[i, : len(x)] = torch.tensor(x, device=DEVICE)
        padded_y[i, : len(y)] = torch.tensor(y, device=DEVICE)

    return padded_x, padded_y


def train_intent(
    model, train_samples: list[str], char2idx: dict, epochs: int, lr: float
) -> list[dict]:
    """
    Intent classification eğitimi.

    Her epoch: tüm sample'lar batch olarak, backprop sadece LM head'e.
    """
    history = []

    # Sadece LM head'i optimize et
    optimizer = optim.Adam(
        [
            {"params": [model.lm_head_static.weight], "lr": lr},
            {"params": [model.lm_head_bias], "lr": lr},
        ]
    )

    for epoch in range(epochs):
        # Shuffle
        random.shuffle(train_samples)

        x_batch, y_batch = build_intent_batch(train_samples, char2idx)
        if x_batch is None:
            continue

        optimizer.zero_grad()
        logits, loss = model(x_batch, targets=y_batch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            # Accuracy hesapla
            correct = 0
            for prompt in train_samples:
                op = get_intent_token(prompt)
                if op is None:
                    continue
                pred, _ = predict_intent(
                    model, prompt, char2idx, {str(v): k for k, v in char2idx.items()}
                )
                if pred == op:
                    correct += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "accuracy": correct,
                    "total": len(train_samples),
                }
            )

    return history


def execute_intent(prompt: str) -> int | None:
    """Intent → Engine → Result pipeline."""
    intent = extract_intent(prompt)
    if intent is None:
        return None
    return run_engine(intent)


def main():
    print("=" * 70)
    print("VİCDAN — Phase 6.1: Intent-Driven Execution")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    idx2char = idx2char_map(idx2char_raw)

    model, ckpt = load_model()
    print(f"\n  Model yüklendi: spiking_lm_v2.pt")
    print(
        f"  Vocab: {vocab_size}, Params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Training samples (seen) — dengeli: 6 ADD, 6 SUB, 6 MUL
    train_samples = [
        # ADD
        "3+4=",
        "1+1=",
        "7+8=",
        "0+5=",
        "2+6=",
        "5+3=",
        # SUB
        "5-2=",
        "9-3=",
        "8-1=",
        "6-4=",
        "7-5=",
        "10-3=",
        # MUL
        "2*3=",
        "4*4=",
        "1*5=",
        "3*3=",
        "6*2=",
        "7*1=",
    ]

    # Test samples (unseen - generalization)
    test_samples = [
        "6+1=",
        "8-5=",
        "3*2=",
        "2+9=",
        "10-4=",
        "9-7=",
        "5*3=",
        "4+6=",
        "12-8=",
        "2*4=",
    ]

    # ── Adım 1: Training Öncesi ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training Öncesi Intent Tahminleri")
    print(f"{'=' * 70}")

    for prompt in train_samples:
        op = get_intent_token(prompt)
        pred, prob = predict_intent(model, prompt, char2idx, idx2char)
        correct = pred == op
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → intent={pred!r} (beklenen: {op!r}, {OP_TOKENS[op]}) prob={prob:.3f}"
        )

    # ── Adım 2: Intent Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Intent Classification Training")
    print(f"{'=' * 70}")

    history = train_intent(model, train_samples, char2idx, epochs=200, lr=0.005)

    print(f"\n  Training Progress:")
    for h in history:
        print(
            f"    Epoch {h['epoch']:3d}: loss={h['loss']:.4f}, intent_acc={h['accuracy']}/{h['total']}"
        )

    # ── Adım 3: Training Sonrası (Seen) ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Training Sonrası — Seen Samples")
    print(f"{'=' * 70}")

    seen_correct = 0
    for prompt in train_samples:
        op = get_intent_token(prompt)
        pred, prob = predict_intent(model, prompt, char2idx, idx2char)
        correct = pred == op
        if correct:
            seen_correct += 1
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → intent={pred!r} ({OP_TOKENS.get(pred, '?')}) prob={prob:.3f}"
        )

    seen_acc = seen_correct / len(train_samples) * 100
    print(f"\n  Seen Accuracy: {seen_correct}/{len(train_samples)} ({seen_acc:.0f}%)")

    # ── Adım 4: Generalization (Unseen) ──
    print(f"\n{'=' * 70}")
    print("ADIM 4: Generalization — Unseen Samples")
    print(f"{'=' * 70}")

    unseen_correct = 0
    for prompt in test_samples:
        op = get_intent_token(prompt)
        pred, prob = predict_intent(model, prompt, char2idx, idx2char)
        correct = pred == op
        if correct:
            unseen_correct += 1
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → intent={pred!r} ({OP_TOKENS.get(pred, '?')}) prob={prob:.3f}"
        )

    unseen_acc = unseen_correct / len(test_samples) * 100
    print(
        f"\n  Unseen Accuracy: {unseen_correct}/{len(test_samples)} ({unseen_acc:.0f}%)"
    )

    # ── Adım 5: Full Pipeline (Intent → Engine) ──
    print(f"\n{'=' * 70}")
    print("ADIM 5: Full Pipeline — Intent → Engine → Result")
    print(f"{'=' * 70}")

    all_samples = train_samples + test_samples
    pipeline_correct = 0

    for prompt in all_samples:
        expected_result = execute_intent(prompt)
        pred_intent, prob = predict_intent(model, prompt, char2idx, idx2char)

        # Predicted intent'ten result hesapla
        if pred_intent in OP_TOKENS:
            # Intent doğruysa, engine ile hesapla
            intent = extract_intent(prompt)
            if intent:
                intent["op"] = OP_TOKENS[pred_intent]
                result = run_engine(intent)
            else:
                result = None
        else:
            result = None

        correct = result == expected_result
        if correct:
            pipeline_correct += 1
        status = "✅" if correct else "❌"
        marker = " (seen)" if prompt in train_samples else " (unseen)"
        print(
            f"  {status} {prompt!r} → intent={pred_intent!r} → result={result} (beklenen: {expected_result}){marker}"
        )

    pipeline_acc = pipeline_correct / len(all_samples) * 100
    print(
        f"\n  Pipeline Accuracy: {pipeline_correct}/{len(all_samples)} ({pipeline_acc:.0f}%)"
    )

    # ── Sonuçları Kaydet ──
    results = {
        "config": {"epochs": 200, "lr": 0.005},
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "pipeline_accuracy": pipeline_acc,
        "history": history,
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase6_1_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")

    # ── Özet ──
    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Seen intent accuracy:    {seen_acc:.0f}%")
    print(f"  Unseen intent accuracy:  {unseen_acc:.0f}%")
    print(f"  Full pipeline accuracy:  {pipeline_acc:.0f}%")

    if unseen_acc >= 80:
        print(f"\n  ✅ BAŞARILI: Intent generalization çalışıyor!")
    elif unseen_acc >= 50:
        print(f"\n  ⚠️ Kısmi başarı: Daha fazla eğitim gerekli")
    else:
        print(f"\n  ❌ Generalization başarısız")


if __name__ == "__main__":
    main()
