#!/usr/bin/env python3
"""
VİCDAN — Phase 6.4: Outcome-Based STDP (Label-Free)

Label YOK. Reward sonuçtan üretilir. Sistem yanılıp düzeltir.

Phase 6.3: correct_action label vardı → supervised
Phase 6.4: label yok → outcome-based → gerçek RL

STDP formülü:
  ΔW = lr × reward × (chosen_onehot - probs) × pre^T
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
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
EARLY_STOP_ACC = 0.90


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


def execute_action(action_idx: int, text: str, char2idx: dict, model) -> tuple:
    """
    Action'ı çalıştır.

    Returns:
        (output, success)
    """
    if action_idx == 1:  # use_math_engine
        intent = extract_intent(text)
        if intent is None:
            return None, False
        result = run_engine(intent)
        return str(result) if result is not None else None, result is not None
    else:  # generate
        tokens = encode(text, char2idx)
        idx = torch.tensor([tokens], device=DEVICE)
        with torch.no_grad():
            model.eval()
            generated = model.generate(idx, max_new_tokens=3)
        output_chars = []
        for t in range(len(tokens), generated.shape[1]):
            idx2char = {str(v): k for k, v in char2idx.items()}
            c = idx2char.get(str(generated[0, t].item()), "?")
            if c == "\n":
                break
            output_chars.append(c)
        return "".join(output_chars), True


def evaluate_outcome(action_idx: int, text: str, output, success: bool) -> float:
    """
    Outcome-based reward. Label YOK.

    Math input + engine → +1 (doğru tool)
    Math input + generate → -1 (yanlış tool)
    Text input + generate → +1 (doğru tool)
    Text input + engine → -1 (yanlış tool)
    """
    if action_idx == 1:  # use_math_engine
        intent = extract_intent(text)
        if intent is None:
            return -1.0  # Math değil, yanlış tool
        return +1.0  # Engine her zaman doğru sonuç üretir
    else:  # generate
        if is_math(text):
            return -1.0  # Math input'a generate = yanlış tool
        return +1.0  # Text input'a generate = doğru tool


def outcome_stdp_update(
    action_head, pre: torch.Tensor, action_idx: int, reward: float, lr: float
):
    """
    Outcome-based STDP policy update.

    ΔW = lr × reward × (chosen_onehot - probs) × pre^T

    reward > 0 → seçilen action güçlenir
    reward < 0 → seçilen action zayıflar
    """
    n_actions = action_head.n_actions

    W = action_head.action_head.weight + action_head.weight_fast
    logits = F.linear(pre.unsqueeze(0), W, action_head.action_head.bias)
    probs = F.softmax(logits, dim=-1)[0]

    chosen_onehot = torch.zeros(n_actions, device=pre.device)
    chosen_onehot[action_idx] = 1.0

    # Policy gradient benzeri: reward × (chosen - probs)
    delta = reward * (chosen_onehot - probs)

    grad = torch.outer(pre, delta)
    action_head.weight_fast.add_(lr * grad.T)
    action_head.weight_fast.clamp_(-10.0, 10.0)

    # Bias update
    action_head.action_head.bias.data.add_(lr * delta)

    return {
        "reward": reward,
        "action": ACTIONS[action_idx],
        "probs": probs.tolist(),
        "grad_norm": grad.norm().item(),
    }


def predict_action(
    action_head, model, text: str, char2idx: dict, greedy: bool = False
) -> tuple:
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

        action_name = ACTIONS[action_idx]
        conf = probs[0, action_idx].item()

    return action_name, conf


def main():
    print("=" * 70)
    print("VİCDAN — Phase 6.4: Outcome-Based STDP (Label-Free)")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]

    action_head = ActionHead(d_model, n_actions=2).to(DEVICE)

    print(f"\n  Model: SpikingLM (76K params)")
    print(
        f"  ActionHead: {d_model} → 2 ({sum(p.numel() for p in action_head.parameters()):,} params)"
    )
    print(f"  LR: {LR}, Epochs: {EPOCHS}")
    print(f"  NO LABELS — Reward = outcome")

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
        "dün",
    ]
    train_samples = train_math + train_text

    # Test data (unseen)
    test_math = ["6+1=", "8-5=", "3*2=", "12+34="]
    test_text = ["mehmet", "nasılsın", "bir hafta", "köpek"]
    test_samples = test_math + test_text

    # ── Adım 1: Training Öncesi ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training Öncesi (Random Policy)")
    print(f"{'=' * 70}")

    for prompt in train_samples[:4]:
        pred, conf = predict_action(action_head, model, prompt, char2idx, greedy=False)
        expected = "use_math_engine" if is_math(prompt) else "generate"
        correct = pred == expected
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → action={pred!r} (beklenen: {expected!r}) conf={conf:.3f}"
        )

    # ── Adım 2: Outcome-Based Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Outcome-Based STDP Training (NO LABELS)")
    print(f"{'=' * 70}")

    history = []

    for epoch in range(EPOCHS):
        random.shuffle(train_samples)
        epoch_rewards = []
        epoch_correct = 0

        for prompt in train_samples:
            # Forward
            tokens = encode(prompt, char2idx)
            idx = torch.tensor([tokens], device=DEVICE)

            with torch.no_grad():
                model.eval()
                model_logits, _ = model(idx[:, -BLOCK_SIZE:])
                hidden = model.get_last_hidden()
                if hidden is None:
                    continue

                pre = hidden.mean(dim=1)[0]
                _, probs, _, _ = action_head(hidden)

            # Action seç (exploration: multinomial)
            action_idx = torch.multinomial(probs[0], 1).item()
            action_name = ACTIONS[action_idx]

            # Execute
            output, success = execute_action(action_idx, prompt, char2idx, model)

            # Reward (outcome-based, NO LABEL)
            reward = evaluate_outcome(action_idx, prompt, output, success)

            # STDP update
            outcome_stdp_update(action_head, pre.detach(), action_idx, reward, lr=LR)

            epoch_rewards.append(reward)
            if reward > 0:
                epoch_correct += 1

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        accuracy = epoch_correct / len(train_samples) * 100

        # Consolidation
        if (epoch + 1) % CONSOLIDATION_EVERY == 0:
            action_head.consolidate(alpha=0.1, threshold=0.01)

        # Unseen test
        unseen_acc = 0.0
        if (epoch + 1) % 10 == 0:
            unseen_correct = 0
            for prompt in test_samples:
                pred, _ = predict_action(
                    action_head, model, prompt, char2idx, greedy=True
                )
                expected = "use_math_engine" if is_math(prompt) else "generate"
                if pred == expected:
                    unseen_correct += 1
            unseen_acc = unseen_correct / len(test_samples) * 100

            history.append(
                {
                    "epoch": epoch + 1,
                    "avg_reward": avg_reward,
                    "train_accuracy": accuracy,
                    "unseen_accuracy": unseen_acc,
                }
            )
            print(
                f"    Epoch {epoch + 1:3d}: reward={avg_reward:+.2f}, train={accuracy:.0f}%, unseen={unseen_acc:.0f}%"
            )

        # Early stopping
        if accuracy / 100 >= EARLY_STOP_ACC and unseen_acc >= EARLY_STOP_ACC:
            print(f"\n  ⏹️ Early stopping at epoch {epoch + 1}")
            break

    # ── Adım 3: Seen Test (Greedy) ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Seen Samples Test (Greedy)")
    print(f"{'=' * 70}")

    seen_correct = 0
    for prompt in train_samples:
        expected = "use_math_engine" if is_math(prompt) else "generate"
        pred, conf = predict_action(action_head, model, prompt, char2idx, greedy=True)
        correct = pred == expected
        if correct:
            seen_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → {pred!r} conf={conf:.3f}")

    seen_acc = seen_correct / len(train_samples) * 100
    print(f"\n  Seen Accuracy: {seen_correct}/{len(train_samples)} ({seen_acc:.0f}%)")

    # ── Adım 4: Unseen Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 4: Unseen Samples Test (Generalization)")
    print(f"{'=' * 70}")

    unseen_correct = 0
    for prompt in test_samples:
        expected = "use_math_engine" if is_math(prompt) else "generate"
        pred, conf = predict_action(action_head, model, prompt, char2idx, greedy=True)
        correct = pred == expected
        if correct:
            unseen_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → {pred!r} conf={conf:.3f}")

    unseen_acc = unseen_correct / len(test_samples) * 100
    print(
        f"\n  Unseen Accuracy: {unseen_correct}/{len(test_samples)} ({unseen_acc:.0f}%)"
    )

    # ── Adım 5: Full Pipeline ──
    print(f"\n{'=' * 70}")
    print("ADIM 5: Full Pipeline — Action → Execute → Outcome")
    print(f"{'=' * 70}")

    all_test = test_samples
    pipeline_correct = 0

    for prompt in all_test:
        expected = "use_math_engine" if is_math(prompt) else "generate"
        pred, conf = predict_action(action_head, model, prompt, char2idx, greedy=True)

        action_idx = ACTIONS.index(pred)
        output, success = execute_action(action_idx, prompt, char2idx, model)

        correct = pred == expected
        if correct:
            pipeline_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → {pred!r} → {output}")

    pipeline_acc = pipeline_correct / len(all_test) * 100
    print(
        f"\n  Pipeline Accuracy: {pipeline_correct}/{len(all_test)} ({pipeline_acc:.0f}%)"
    )

    # ── Sonuçları Kaydet ──
    results = {
        "config": {"epochs": EPOCHS, "lr": LR, "no_labels": True},
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "pipeline_accuracy": pipeline_acc,
        "history": history,
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase6_4_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")

    # Checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "action_head_v2_outcome.pt")
    torch.save(action_head.state_dict(), ckpt_path)
    print(f"  ActionHead checkpoint: {ckpt_path}")

    # ── Özet ──
    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Seen accuracy:    {seen_acc:.0f}%")
    print(f"  Unseen accuracy:  {unseen_acc:.0f}%")
    print(f"  Pipeline:         {pipeline_acc:.0f}%")

    # Learning curve analizi
    if history:
        first_reward = history[0]["avg_reward"]
        last_reward = history[-1]["avg_reward"]
        print(f"\n  Reward trend: {first_reward:+.2f} → {last_reward:+.2f}")
        if last_reward > first_reward:
            print(f"  ✅ Reward artıyor — sistem öğreniyor")
        else:
            print(f"  ❌ Reward düşüyor — öğrenme yok")

    if unseen_acc >= 80:
        print(f"\n  ✅ BAŞARILI: Outcome-based STDP çalışıyor!")
    elif unseen_acc >= 50:
        print(f"\n  ⚠️ Kısmi başarı: Daha fazla eğitim gerekli")
    else:
        print(f"\n  ❌ Outcome-based learning başarısız")


if __name__ == "__main__":
    main()
