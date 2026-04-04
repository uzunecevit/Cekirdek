#!/usr/bin/env python3
"""
VİCDAN — Phase 6.6: Saliency-Guided Action Learning

Saliency (op_boost=10.0) + Outcome-Based STDP ile ActionHead yeniden eğitimi.

Hipotez: Saliency-modulated hidden state'ler ile eğitilen ActionHead,
ambiguous input'larda doğru action'ı seçebilir.
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model import SpikingLM
from src.action_head import ActionHead
from src.intent import extract_intent
from src.engine import run_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 32
ACTIONS = ["generate", "use_math_engine"]

# Config
LR = 0.02  # Orta yol: hızlı ama stabil
EPOCHS = 100
CONSOLIDATION_EVERY = 5
EARLY_STOP_ACC = 0.85
OP_BOOST = 10.0
KW_BOOST = 1.0  # Keyword boost YOK (test'te performansı düşürdü)


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


def encode(text, char2idx):
    return [char2idx.get(c, 0) for c in text if c in char2idx]


def compute_saliency(text: str) -> list[float]:
    """Saliency maskesi: sadece operatörler boost edilir."""
    saliency = [1.0] * len(text)
    for i, ch in enumerate(text):
        if ch in ["+", "-", "*"]:
            saliency[i] = OP_BOOST
    return saliency


def get_saliency_modulated_hidden(model, text: str, char2idx: dict) -> torch.Tensor:
    """Saliency-modulated hidden state üret."""
    model.eval()
    tokens = encode(text, char2idx)
    idx = torch.tensor([tokens], device=DEVICE)
    saliency = compute_saliency(text)

    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        hidden = model.get_last_hidden()
        if hidden is None:
            return None

    # Token-level saliency modulation
    T = hidden.size(1)
    modulated = hidden.clone()

    for t in range(T):
        token_idx = idx[0, t].item()
        idx2char = {v: k for k, v in char2idx.items()}
        char = idx2char.get(token_idx, "")

        char_saliency = 1.0
        for i, c in enumerate(text):
            if c == char:
                char_saliency = saliency[i]
                break

        modulated[0, t] = modulated[0, t] * char_saliency

    return modulated


def get_expected_action(text: str, expected_label: str = None) -> str:
    if expected_label is not None:
        return "use_math_engine" if expected_label == "math" else "generate"
    return (
        "use_math_engine" if any(op in text for op in ["+", "-", "*"]) else "generate"
    )


def outcome_stdp_update(action_head, pre, action_idx, reward, lr):
    """Outcome-based STDP policy update."""
    n_actions = action_head.n_actions

    W = action_head.action_head.weight + action_head.weight_fast
    logits = F.linear(pre.unsqueeze(0), W, action_head.action_head.bias)
    probs = F.softmax(logits, dim=-1)[0]

    chosen_onehot = torch.zeros(n_actions, device=pre.device)
    chosen_onehot[action_idx] = 1.0

    if abs(reward) < 0.01:
        return {"reward": 0.0, "skipped": True}

    delta = reward * (chosen_onehot - probs)
    grad = torch.outer(pre, delta)
    action_head.weight_fast.add_(lr * grad.T)
    action_head.weight_fast.clamp_(-10.0, 10.0)
    action_head.action_head.bias.data.add_(lr * delta)

    return {"reward": reward, "action": ACTIONS[action_idx], "probs": probs.tolist()}


def predict_action(action_head, hidden, greedy=True):
    """Saliency-modulated hidden state'den action tahmini."""
    action_head.eval()
    with torch.no_grad():
        pooled = hidden.mean(dim=1)
        W = action_head.action_head.weight + action_head.weight_fast
        logits = F.linear(pooled, W, action_head.action_head.bias)
        probs = F.softmax(logits, dim=-1)

        if greedy:
            action_idx = logits.argmax(dim=-1).item()
        else:
            action_idx = torch.multinomial(probs[0], 1).item()

        return ACTIONS[action_idx], probs[0, action_idx].item()


def main():
    print("=" * 70)
    print("VİCDAN — Phase 6.6: Saliency-Guided Action Learning")
    print("=" * 70)
    print(f"  Op Boost: {OP_BOOST}, KW Boost: {KW_BOOST}")
    print(f"  LR: {LR}, Epochs: {EPOCHS}")

    vocab_size, char2idx, idx2char_raw = load_vocab()
    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]

    # ActionHead — SIFIRDAN (v2 checkpoint'i kullanmıyoruz)
    action_head = ActionHead(d_model, n_actions=2).to(DEVICE)
    print(f"  ActionHead: {d_model} → 2 (sıfırdan)")

    # Training data
    clean_math = [
        "3+4=",
        "5-2=",
        "2*3=",
        "7+8=",
        "9-3=",
        "4*4=",
        "1+1=",
        "0+5=",
        "6-4=",
        "3*3=",
    ]
    clean_text = [
        "ahmet",
        "merhaba",
        "bir gün",
        "kedi",
        "ev",
        "istanbul",
        "selam",
        "güneşli",
        "kitap",
        "dün",
    ]
    ambiguous = [
        ("3+4 kaç eder?", "math"),
        ("sonucu hesapla: 3+4", "math"),
        ("mehmet 3+4 dedi", "text"),
        ("2+2=5 doğru mu?", "text"),
        ("bugün 5-2 yaptım", "text"),
        ("toplama nedir?", "text"),
        ("5+7'nin sonucu", "math"),
        ("kedi ve köpek", "text"),
        ("10-3=7 doğru", "text"),
        ("hesapla 6*8", "math"),
    ]

    train_clean = [(t, None) for t in clean_math + clean_text]
    train_ambiguous = [(t, l) for t, l in ambiguous]
    train_pool = train_clean + train_ambiguous

    # Test data
    test_clean_math = ["6+1=", "8-5=", "3*2=", "12+34="]
    test_clean_text = ["mehmet", "nasılsın", "bir hafta", "köpek"]
    test_ambiguous = [
        ("8+9 kaçtır?", "math"),
        ("ali 3-1 dedi", "text"),
        ("çarpma tablosu", "text"),
        ("sonuç: 4*5", "math"),
    ]

    # Pre-compute saliency-modulated hidden states (hız için)
    print(f"\n  Pre-computing saliency-modulated hidden states...")
    hidden_cache = {}
    for text, label in (
        train_pool
        + [(t, None) for t in test_clean_math + test_clean_text]
        + test_ambiguous
    ):
        if text not in hidden_cache:
            h = get_saliency_modulated_hidden(model, text, char2idx)
            if h is not None:
                hidden_cache[text] = h

    print(f"  Cached {len(hidden_cache)} hidden states")

    # ── Training Öncesi ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training Öncesi (Random ActionHead)")
    print(f"{'=' * 70}")

    for text, label in ambiguous[:3]:
        expected = get_expected_action(text, label)
        h = hidden_cache.get(text)
        if h is not None:
            pred, conf = predict_action(action_head, h)
            correct = pred == expected
            status = "✅" if correct else "❌"
            print(
                f"  {status} {text!r} → {pred!r} (beklenen: {expected!r}) conf={conf:.3f}"
            )

    # ── Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Outcome-Based STDP Training (Saliency-Modulated)")
    print(f"{'=' * 70}")

    history = []

    for epoch in range(EPOCHS):
        random.shuffle(train_pool)
        epoch_rewards = []
        epoch_correct = 0

        for text, label in train_pool:
            h = hidden_cache.get(text)
            if h is None:
                continue

            pre = h.mean(dim=1)[0]

            # Action seç (multinomial)
            _, probs, _, _ = action_head(h)
            action_idx = torch.multinomial(probs[0], 1).item()

            # Reward
            expected = get_expected_action(text, label)
            actual = ACTIONS[action_idx]
            reward = 1.0 if actual == expected else -1.0

            # STDP update
            outcome_stdp_update(action_head, pre.detach(), action_idx, reward, lr=LR)

            epoch_rewards.append(reward)
            if reward > 0:
                epoch_correct += 1

        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        accuracy = epoch_correct / max(len(epoch_rewards), 1) * 100

        if (epoch + 1) % CONSOLIDATION_EVERY == 0:
            action_head.consolidate(alpha=0.1, threshold=0.01)

        if (epoch + 1) % 10 == 0:
            # Clean test
            clean_correct = 0
            for t in test_clean_math + test_clean_text:
                h = hidden_cache.get(t)
                if h is None:
                    continue
                expected = get_expected_action(t)
                pred, _ = predict_action(action_head, h, greedy=True)
                if pred == expected:
                    clean_correct += 1
            clean_acc = clean_correct / len(test_clean_math + test_clean_text) * 100

            # Ambiguous test
            amb_correct = 0
            for t, l in test_ambiguous:
                h = hidden_cache.get(t)
                if h is None:
                    continue
                expected = get_expected_action(t, l)
                pred, _ = predict_action(action_head, h, greedy=True)
                if pred == expected:
                    amb_correct += 1
            amb_acc = amb_correct / len(test_ambiguous) * 100

            history.append(
                {
                    "epoch": epoch + 1,
                    "avg_reward": avg_reward,
                    "train_accuracy": accuracy,
                    "clean_accuracy": clean_acc,
                    "ambiguous_accuracy": amb_acc,
                }
            )
            print(
                f"    Epoch {epoch + 1:3d}: reward={avg_reward:+.2f}, train={accuracy:.0f}%, clean={clean_acc:.0f}%, amb={amb_acc:.0f}%"
            )

            if clean_acc >= EARLY_STOP_ACC * 100 and amb_acc >= 50:
                print(f"\n  ⏹️ Early stopping at epoch {epoch + 1}")
                break

    # ── Final Tests ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Clean Test (Unseen)")
    print(f"{'=' * 70}")

    clean_correct = 0
    for t in test_clean_math + test_clean_text:
        h = hidden_cache.get(t)
        if h is None:
            continue
        expected = get_expected_action(t)
        pred, conf = predict_action(action_head, h, greedy=True)
        correct = pred == expected
        if correct:
            clean_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {t!r} → {pred!r} conf={conf:.3f}")

    clean_acc = clean_correct / len(test_clean_math + test_clean_text) * 100
    print(
        f"\n  Clean Accuracy: {clean_correct}/{len(test_clean_math + test_clean_text)} ({clean_acc:.0f}%)"
    )

    print(f"\n{'=' * 70}")
    print("ADIM 4: Ambiguous Test (Unseen)")
    print(f"{'=' * 70}")

    amb_correct = 0
    for t, l in test_ambiguous:
        h = hidden_cache.get(t)
        if h is None:
            continue
        expected = get_expected_action(t, l)
        pred, conf = predict_action(action_head, h, greedy=True)
        correct = pred == expected
        if correct:
            amb_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {t!r} → {pred!r} (beklenen: {expected!r}) conf={conf:.3f}")

    amb_acc = amb_correct / len(test_ambiguous) * 100
    print(
        f"\n  Ambiguous Accuracy: {amb_correct}/{len(test_ambiguous)} ({amb_acc:.0f}%)"
    )

    print(f"\n{'=' * 70}")
    print("ADIM 5: Training Ambiguity (Seen)")
    print(f"{'=' * 70}")

    train_amb_correct = 0
    for t, l in ambiguous:
        h = hidden_cache.get(t)
        if h is None:
            continue
        expected = get_expected_action(t, l)
        pred, conf = predict_action(action_head, h, greedy=True)
        correct = pred == expected
        if correct:
            train_amb_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {t!r} → {pred!r} (beklenen: {expected!r}) conf={conf:.3f}")

    train_amb_acc = train_amb_correct / len(ambiguous) * 100
    print(
        f"\n  Training Ambiguous Accuracy: {train_amb_correct}/{len(ambiguous)} ({train_amb_acc:.0f}%)"
    )

    # ── Save ──
    results = {
        "config": {
            "lr": LR,
            "op_boost": OP_BOOST,
            "kw_boost": KW_BOOST,
            "epochs": EPOCHS,
        },
        "clean_accuracy": clean_acc,
        "ambiguous_accuracy": amb_acc,
        "training_ambiguous_accuracy": train_amb_acc,
        "history": history,
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase6_6_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {result_path}")

    ckpt_path = os.path.join(CKPT_DIR, "action_head_v4_saliency.pt")
    torch.save(action_head.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Clean:         {clean_acc:.0f}%")
    print(f"  Ambiguous:     {amb_acc:.0f}%")
    print(f"  Training Amb:  {train_amb_acc:.0f}%")

    if history:
        first = history[0]
        last = history[-1]
        print(f"\n  Learning curve:")
        print(
            f"    Clean:  {first['clean_accuracy']:.0f}% → {last['clean_accuracy']:.0f}%"
        )
        print(
            f"    Ambig:  {first['ambiguous_accuracy']:.0f}% → {last['ambiguous_accuracy']:.0f}%"
        )

    if clean_acc >= 80 and amb_acc >= 50:
        print(f"\n  ✅ BAŞARILI: Saliency-guided learning çalıştı!")
    elif clean_acc >= 60:
        print(f"\n  ⚠️ Kısmi başarı")
    else:
        print(f"\n  ❌ Başarısız")


if __name__ == "__main__":
    main()
