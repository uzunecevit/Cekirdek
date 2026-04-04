#!/usr/bin/env python3
"""
VİCDAN — Phase 6.3: Tool-Use Reinforcement (STDP ile Karar Öğrenme)

SNN'e ne zaman tool kullanacağını STDP ile öğretir.

Training:
  Math input → correct_action=1 (use_math_engine) → reward=+1
  Text input → correct_action=0 (generate) → reward=+1
  Yanlış seçim → reward=-1

Test:
  Seen + Unseen samples ile action accuracy ölçümü
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
    return any(op in text for op in ["+", "-", "*"])


def get_correct_action(text: str) -> int:
    """Ground truth: math=1 (engine), text=0 (generate)."""
    return 1 if is_math(text) else 0


def predict_action(action_head, model, text: str, char2idx: dict) -> tuple[str, float]:
    """Model + action head ile action tahmini."""
    model.eval()
    action_head.eval()
    tokens = encode(text, char2idx)
    idx = torch.tensor([tokens], device=DEVICE)

    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        hidden = model.get_last_hidden()
        if hidden is None:
            return "generate", 0.0

        _, _, action, confidence = action_head(hidden)
        action_name = ACTIONS[action.item()]
        conf = confidence.item()

    return action_name, conf


def main():
    print("=" * 70)
    print("VİCDAN — Phase 6.3: Tool-Use Reinforcement (STDP)")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]

    # Action Head
    action_head = ActionHead(d_model, n_actions=2).to(DEVICE)

    print(f"\n  Model: SpikingLM (76K params)")
    print(
        f"  ActionHead: {d_model} → 2 ({sum(p.numel() for p in action_head.parameters()):,} params)"
    )

    # Training data
    train_math = [
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
        "2+6=",
        "8-1=",
    ]
    train_text = [
        "ahmet",
        "merhaba",
        "bir gün",
        "kedi",
        "ev",
        "istanbul",
        "selam",
        "güneşli",
        "kitap",
        "yürümek",
        "arkadaş",
        "dün",
    ]
    train_samples = train_math + train_text

    # Test data (unseen)
    test_math = ["6+1=", "8-5=", "3*2=", "12+34=", "10-4=", "5*3="]
    test_text = ["mehmet", "nasılsın", "bir hafta", "köpek", "okul", "yağmurlu"]
    test_samples = test_math + test_text

    # ── Adım 1: Training Öncesi ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training Öncesi Action Tahminleri")
    print(f"{'=' * 70}")

    for prompt in train_samples[:6]:
        correct_action = ACTIONS[get_correct_action(prompt)]
        pred, conf = predict_action(action_head, model, prompt, char2idx)
        correct = pred == correct_action
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → action={pred!r} (beklenen: {correct_action!r}) conf={conf:.3f}"
        )

    # ── Adım 2: STDP Reinforcement Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: STDP Reinforcement Training")
    print(f"{'=' * 70}")

    EPOCHS = 50
    LR = 0.01
    history = []

    for epoch in range(EPOCHS):
        random.shuffle(train_samples)
        epoch_rewards = []
        epoch_correct = 0

        for prompt in train_samples:
            correct_action = get_correct_action(prompt)

            # Forward
            tokens = encode(prompt, char2idx)
            idx = torch.tensor([tokens], device=DEVICE)

            with torch.no_grad():
                model.eval()
                model_logits, _ = model(idx[:, -BLOCK_SIZE:])
                hidden = model.get_last_hidden()
                if hidden is None:
                    continue

                _, _, predicted_action, _ = action_head(hidden)
                predicted = predicted_action.item()

            # Reward
            if predicted == correct_action:
                reward = 1.0
                epoch_correct += 1
            else:
                reward = -1.0

            # STDP update
            pre = hidden.mean(dim=1)[0]  # (d_model,)
            update_info = action_head.stdp_update(pre, correct_action, reward, lr=LR)
            epoch_rewards.append(reward)

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        accuracy = epoch_correct / len(train_samples) * 100

        if (epoch + 1) % 10 == 0:
            # Consolidation
            action_head.consolidate(alpha=0.05, threshold=0.01)

            # Test accuracy
            test_correct = 0
            for prompt in train_samples:
                correct_action = ACTIONS[get_correct_action(prompt)]
                pred, _ = predict_action(action_head, model, prompt, char2idx)
                if pred == correct_action:
                    test_correct += 1
            test_acc = test_correct / len(train_samples) * 100

            history.append(
                {
                    "epoch": epoch + 1,
                    "avg_reward": avg_reward,
                    "train_accuracy": accuracy,
                    "test_accuracy": test_acc,
                }
            )
            print(
                f"    Epoch {epoch + 1:3d}: reward={avg_reward:+.3f}, train_acc={accuracy:.0f}%, test_acc={test_acc:.0f}%"
            )

    # ── Adım 3: Seen Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Seen Samples Test")
    print(f"{'=' * 70}")

    seen_correct = 0
    for prompt in train_samples:
        correct_action = ACTIONS[get_correct_action(prompt)]
        pred, conf = predict_action(action_head, model, prompt, char2idx)
        correct = pred == correct_action
        if correct:
            seen_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → action={pred!r} conf={conf:.3f}")

    seen_acc = seen_correct / len(train_samples) * 100
    print(f"\n  Seen Accuracy: {seen_correct}/{len(train_samples)} ({seen_acc:.0f}%)")

    # ── Adım 4: Unseen Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 4: Unseen Samples Test (Generalization)")
    print(f"{'=' * 70}")

    unseen_correct = 0
    for prompt in test_samples:
        correct_action = ACTIONS[get_correct_action(prompt)]
        pred, conf = predict_action(action_head, model, prompt, char2idx)
        correct = pred == correct_action
        if correct:
            unseen_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → action={pred!r} conf={conf:.3f}")

    unseen_acc = unseen_correct / len(test_samples) * 100
    print(
        f"\n  Unseen Accuracy: {unseen_correct}/{len(test_samples)} ({unseen_acc:.0f}%)"
    )

    # ── Adım 5: Full Pipeline ──
    print(f"\n{'=' * 70}")
    print("ADIM 5: Full Pipeline — Action → Execute")
    print(f"{'=' * 70}")

    all_test = test_samples
    pipeline_correct = 0

    for prompt in all_test:
        expected_action = ACTIONS[get_correct_action(prompt)]
        pred_action, conf = predict_action(action_head, model, prompt, char2idx)

        # Execute
        if pred_action == "use_math_engine":
            intent = extract_intent(prompt)
            result = run_engine(intent)
            output = str(result) if result is not None else "ERROR"
        else:
            output = "[generate]"

        correct = pred_action == expected_action
        if correct:
            pipeline_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → {pred_action!r} → {output}")

    pipeline_acc = pipeline_correct / len(all_test) * 100
    print(
        f"\n  Pipeline Accuracy: {pipeline_correct}/{len(all_test)} ({pipeline_acc:.0f}%)"
    )

    # ── Sonuçları Kaydet ──
    results = {
        "config": {"epochs": EPOCHS, "lr": LR},
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "pipeline_accuracy": pipeline_acc,
        "history": history,
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase6_3_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")

    # Checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "action_head_v1.pt")
    torch.save(action_head.state_dict(), ckpt_path)
    print(f"  ActionHead checkpoint kaydedildi: {ckpt_path}")

    # ── Özet ──
    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Seen action accuracy:    {seen_acc:.0f}%")
    print(f"  Unseen action accuracy:  {unseen_acc:.0f}%")
    print(f"  Pipeline accuracy:       {pipeline_acc:.0f}%")

    if unseen_acc >= 80:
        print(f"\n  ✅ BAŞARILI: STDP ile tool selection öğrenildi!")
    elif unseen_acc >= 50:
        print(f"\n  ⚠️ Kısmi başarı: Daha fazla eğitim gerekli")
    else:
        print(f"\n  ❌ STDP decision learning başarısız")


if __name__ == "__main__":
    main()
