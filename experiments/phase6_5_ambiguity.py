#!/usr/bin/env python3
"""
VİCDAN — Phase 6.5: Ambiguity Injection (Belirsizlikte Öğrenme)

Steril ortamdan gerçek dünyaya:
1. Belirsiz input'lar (ambiguous)
2. Ambiguous reward = 0 (öğrenme sinyali yok)
3. Sistem gri alanları yönetmeyi öğrensin

Phase 6.4: "3+4=" → math (kolay)
Phase 6.5: "3+4 kaç eder?" → ??? (zor)
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
LR = 0.05
EPOCHS = 100
CONSOLIDATION_EVERY = 5
EARLY_STOP_ACC = 0.85

# ── Data ──

# Clean (net)
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

# Ambiguous (belirsiz)
ambiguous = [
    ("3+4 kaç eder?", "math"),  # Math ama text formunda
    ("sonucu hesapla: 3+4", "math"),  # Math ama farklı syntax
    ("mehmet 3+4 dedi", "text"),  # Text içinde math (alıntı)
    ("2+2=5 doğru mu?", "text"),  # Math sorusu, doğrulama
    ("bugün 5-2 yaptım", "text"),  # Text içinde math expression
    ("toplama nedir?", "text"),  # Math konusu ama text
    ("5+7'nin sonucu", "math"),  # Math ima
    ("kedi ve köpek", "text"),  # Pure text
    ("10-3=7 doğru", "text"),  # Math statement
    ("hesapla 6*8", "math"),  # Math command
]

# Test (unseen)
test_clean_math = ["6+1=", "8-5=", "3*2=", "12+34="]
test_clean_text = ["mehmet", "nasılsın", "bir hafta", "köpek"]
test_ambiguous = [
    ("8+9 kaçtır?", "math"),
    ("ali 3-1 dedi", "text"),
    ("çarpma tablosu", "text"),
    ("sonuç: 4*5", "math"),
]


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


def is_math(text: str) -> bool:
    """Basit math detection (ground truth için)."""
    return any(op in text for op in ["+", "-", "*"])


def is_math_like(text: str) -> bool:
    """
    Ambiguity-aware math detection.
    Clean math → True
    Ambiguous math → True
    Text içinde math → False (alıntı)
    """
    # Doğrudan math pattern
    if any(text.endswith(op + "=") for op in ["+", "-", "*"]):
        return True
    # "hesapla", "sonuç", "kaç" gibi math indicator'ları
    math_words = ["hesapla", "sonuç", "kaç", "çarp", "topla", "çıkar"]
    if any(w in text.lower() for w in math_words):
        # Ama alıntı değilse
        if "dedi" not in text and "doğru" not in text:
            return True
    return False


def get_expected_action(text: str, expected_label: str = None) -> str:
    """
    Ground truth action.

    Clean math → use_math_engine
    Clean text → generate
    Ambiguous → expected_label'e göre
    """
    if expected_label is not None:
        return "use_math_engine" if expected_label == "math" else "generate"
    return "use_math_engine" if is_math(text) else "generate"


def evaluate_outcome_ambiguous(
    action_idx: int, text: str, expected_label: str = None
) -> float:
    """
    Ambiguity-aware reward.

    Clean math + engine → +1
    Clean text + generate → +1
    Ambiguous math + engine → +1
    Ambiguous text + generate → +1
    Yanlış tool → -1
    """
    expected = get_expected_action(text, expected_label)
    actual = ACTIONS[action_idx]

    if actual == expected:
        return +1.0
    else:
        return -1.0


def outcome_stdp_update(action_head, pre, action_idx, reward, lr):
    """Outcome-based STDP policy update."""
    n_actions = action_head.n_actions

    W = action_head.action_head.weight + action_head.weight_fast
    logits = F.linear(pre.unsqueeze(0), W, action_head.action_head.bias)
    probs = F.softmax(logits, dim=-1)[0]

    chosen_onehot = torch.zeros(n_actions, device=pre.device)
    chosen_onehot[action_idx] = 1.0

    # reward = 0 → öğrenme yok (ambiguity)
    if abs(reward) < 0.01:
        return {"reward": 0.0, "action": ACTIONS[action_idx], "skipped": True}

    delta = reward * (chosen_onehot - probs)
    grad = torch.outer(pre, delta)
    action_head.weight_fast.add_(lr * grad.T)
    action_head.weight_fast.clamp_(-10.0, 10.0)
    action_head.action_head.bias.data.add_(lr * delta)

    return {
        "reward": reward,
        "action": ACTIONS[action_idx],
        "probs": probs.tolist(),
        "grad_norm": grad.norm().item(),
    }


def predict_action(action_head, model, text, char2idx, greedy=False):
    """Action tahmini."""
    model.eval()
    action_head.eval()
    tokens = encode(text, char2idx)
    idx = torch.tensor([tokens], device=DEVICE)

    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        hidden = model.get_last_hidden()
        if hidden is None:
            return "generate", 0.0

        _, probs, action, confidence = action_head(hidden)

        if greedy:
            action_idx = action.item()
        else:
            action_idx = torch.multinomial(probs[0], 1).item()

        return ACTIONS[action_idx], probs[0, action_idx].item()


def main():
    print("=" * 70)
    print("VİCDAN — Phase 6.5: Ambiguity Injection")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]

    # ActionHead — sıfırdan (önceki checkpoint'i yükleme, kendi öğrensin)
    action_head = ActionHead(d_model, n_actions=2).to(DEVICE)

    print(f"\n  Model: SpikingLM (76K params)")
    print(
        f"  ActionHead: {d_model} → 2 ({sum(p.numel() for p in action_head.parameters()):,} params)"
    )
    print(f"  Clean samples: {len(clean_math) + len(clean_text)}")
    print(f"  Ambiguous samples: {len(ambiguous)}")
    print(f"  LR: {LR}, Epochs: {EPOCHS}")

    # ── Adım 1: Training Öncesi ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training Öncesi")
    print(f"{'=' * 70}")

    print("\n  Clean:")
    for prompt in clean_math[:2] + clean_text[:2]:
        expected = get_expected_action(prompt)
        pred, conf = predict_action(action_head, model, prompt, char2idx)
        correct = pred == expected
        status = "✅" if correct else "❌"
        print(
            f"    {status} {prompt!r} → {pred!r} (beklenen: {expected!r}) conf={conf:.3f}"
        )

    print("\n  Ambiguous:")
    for prompt, label in ambiguous[:4]:
        expected = get_expected_action(prompt, label)
        pred, conf = predict_action(action_head, model, prompt, char2idx)
        correct = pred == expected
        status = "✅" if correct else "❌"
        print(
            f"    {status} {prompt!r} → {pred!r} (beklenen: {expected!r}) conf={conf:.3f}"
        )

    # ── Adım 2: Training (Clean + Ambiguous) ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Training — Clean + Ambiguous")
    print(f"{'=' * 70}")

    # Training pool: clean + ambiguous
    train_clean = [(t, None) for t in clean_math + clean_text]  # None = auto-detect
    train_ambiguous = [(t, l) for t, l in ambiguous]
    train_pool = train_clean + train_ambiguous

    history = []

    for epoch in range(EPOCHS):
        random.shuffle(train_pool)
        epoch_rewards = []
        epoch_correct = 0
        epoch_skipped = 0

        for text, label in train_pool:
            tokens = encode(text, char2idx)
            idx = torch.tensor([tokens], device=DEVICE)

            with torch.no_grad():
                model.eval()
                model_logits, _ = model(idx[:, -BLOCK_SIZE:])
                hidden = model.get_last_hidden()
                if hidden is None:
                    continue

                pre = hidden.mean(dim=1)[0]
                _, probs, _, _ = action_head(hidden)

            # Action seç (multinomial — exploration)
            action_idx = torch.multinomial(probs[0], 1).item()

            # Reward (ambiguous-aware)
            expected = get_expected_action(text, label)
            actual = ACTIONS[action_idx]
            reward = 1.0 if actual == expected else -1.0

            # STDP update
            update_info = outcome_stdp_update(
                action_head, pre.detach(), action_idx, reward, lr=LR
            )
            if update_info.get("skipped"):
                epoch_skipped += 1
            else:
                epoch_rewards.append(reward)
                if reward > 0:
                    epoch_correct += 1

        # Accuracy (sadece reward verilen örnekler)
        n_rewarded = len(epoch_rewards)
        accuracy = epoch_correct / max(n_rewarded, 1) * 100
        avg_reward = sum(epoch_rewards) / max(n_rewarded, 1)

        # Consolidation
        if (epoch + 1) % CONSOLIDATION_EVERY == 0:
            action_head.consolidate(alpha=0.1, threshold=0.01)

        # Test her 10 epoch
        if (epoch + 1) % 10 == 0:
            # Clean test
            clean_correct = 0
            for prompt in test_clean_math + test_clean_text:
                expected = get_expected_action(prompt)
                pred, _ = predict_action(
                    action_head, model, prompt, char2idx, greedy=True
                )
                if pred == expected:
                    clean_correct += 1
            clean_acc = clean_correct / len(test_clean_math + test_clean_text) * 100

            # Ambiguous test
            amb_correct = 0
            for prompt, label in test_ambiguous:
                expected = get_expected_action(prompt, label)
                pred, _ = predict_action(
                    action_head, model, prompt, char2idx, greedy=True
                )
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
                    "skipped": epoch_skipped,
                }
            )
            print(
                f"    Epoch {epoch + 1:3d}: reward={avg_reward:+.2f}, clean={clean_acc:.0f}%, amb={amb_acc:.0f}%, skipped={epoch_skipped}"
            )

            # Early stopping
            if clean_acc >= EARLY_STOP_ACC * 100 and amb_acc >= 50:
                print(f"\n  ⏹️ Early stopping at epoch {epoch + 1}")
                break

    # ── Adım 3: Clean Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Clean Test (Unseen)")
    print(f"{'=' * 70}")

    clean_correct = 0
    all_test = test_clean_math + test_clean_text
    for prompt in all_test:
        expected = get_expected_action(prompt)
        pred, conf = predict_action(action_head, model, prompt, char2idx, greedy=True)
        correct = pred == expected
        if correct:
            clean_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → {pred!r} conf={conf:.3f}")

    clean_acc = clean_correct / len(all_test) * 100
    print(f"\n  Clean Accuracy: {clean_correct}/{len(all_test)} ({clean_acc:.0f}%)")

    # ── Adım 4: Ambiguous Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 4: Ambiguous Test (Unseen)")
    print(f"{'=' * 70}")

    amb_correct = 0
    for prompt, label in test_ambiguous:
        expected = get_expected_action(prompt, label)
        pred, conf = predict_action(action_head, model, prompt, char2idx, greedy=True)
        correct = pred == expected
        if correct:
            amb_correct += 1
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → {pred!r} (beklenen: {expected!r}) conf={conf:.3f}"
        )

    amb_acc = amb_correct / len(test_ambiguous) * 100
    print(
        f"\n  Ambiguous Accuracy: {amb_correct}/{len(test_ambiguous)} ({amb_acc:.0f}%)"
    )

    # ── Adım 5: Training Ambiguity Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 5: Training Ambiguity (Seen)")
    print(f"{'=' * 70}")

    train_amb_correct = 0
    for prompt, label in ambiguous:
        expected = get_expected_action(prompt, label)
        pred, conf = predict_action(action_head, model, prompt, char2idx, greedy=True)
        correct = pred == expected
        if correct:
            train_amb_correct += 1
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → {pred!r} (beklenen: {expected!r}) conf={conf:.3f}"
        )

    train_amb_acc = train_amb_correct / len(ambiguous) * 100
    print(
        f"\n  Training Ambiguous Accuracy: {train_amb_correct}/{len(ambiguous)} ({train_amb_acc:.0f}%)"
    )

    # ── Sonuçları Kaydet ──
    results = {
        "config": {"epochs": EPOCHS, "lr": LR, "ambiguous_samples": len(ambiguous)},
        "clean_accuracy": clean_acc,
        "ambiguous_accuracy": amb_acc,
        "training_ambiguous_accuracy": train_amb_acc,
        "history": history,
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase6_5_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")

    # Checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "action_head_v3_ambiguity.pt")
    torch.save(action_head.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # ── Özet ──
    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Clean accuracy:         {clean_acc:.0f}%")
    print(f"  Ambiguous accuracy:     {amb_acc:.0f}%")
    print(f"  Training ambiguous:     {train_amb_acc:.0f}%")

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
        print(f"\n  ✅ BAŞARILI: Ambiguity injection çalıştı!")
    elif clean_acc >= 60:
        print(f"\n  ⚠️ Kısmi başarı: Clean OK, ambiguous zor")
    else:
        print(f"\n  ❌ Ambiguity learning başarısız")


if __name__ == "__main__":
    main()
