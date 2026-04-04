#!/usr/bin/env python3
"""
VİCDAN — Phase 6.7: Multi-Step Reasoning Pipeline (3 Action)

3 action: generate, compute, verify

- "3+4=" → compute → engine → "7"
- "10-3=7 doğru" → verify → engine → True/False → "Evet, doğru"
- "merhaba" → generate → SNN text
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
from src.intent import extract_intent
from src.engine import run_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 32
ACTIONS = ["generate", "compute", "verify"]
N_ACTIONS = 3

# Config
LR = 0.02
EPOCHS = 100
CONSOLIDATION_EVERY = 5
EARLY_STOP_ACC = 0.85
OP_BOOST = 10.0


class ActionHead3(nn.Module):
    """3-action ActionHead with STDP fast weights."""

    def __init__(self, d_model, n_actions=N_ACTIONS):
        super().__init__()
        self.action_head = nn.Linear(d_model, n_actions)
        self.n_actions = n_actions
        self.register_buffer("weight_fast", torch.zeros(n_actions, d_model))

    def forward(self, hidden_state):
        pooled = hidden_state.mean(dim=1)
        W = self.action_head.weight + self.weight_fast
        logits = F.linear(pooled, W, self.action_head.bias)
        probs = F.softmax(logits, dim=-1)
        action = logits.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        return logits, probs, action, confidence

    def consolidate(self, alpha=0.05, threshold=0.01):
        fast_norm = self.weight_fast.norm().item()
        static_norm = self.action_head.weight.norm().item()
        importance = fast_norm / max(static_norm, 1e-8)
        if importance > threshold:
            self.action_head.weight.data = (
                1 - alpha
            ) * self.action_head.weight.data + alpha * self.weight_fast
            self.weight_fast.zero_()
            return {"status": "consolidated", "importance": importance}
        return {"status": "skipped", "importance": importance}


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
    saliency = [1.0] * len(text)
    for i, ch in enumerate(text):
        if ch in ["+", "-", "*"]:
            saliency[i] = OP_BOOST
    return saliency


def get_saliency_modulated_hidden(model, text: str, char2idx: dict) -> torch.Tensor:
    model.eval()
    tokens = encode(text, char2idx)
    idx = torch.tensor([tokens], device=DEVICE)
    saliency = compute_saliency(text)

    with torch.no_grad():
        model(idx[:, -BLOCK_SIZE:])
        hidden = model.get_last_hidden()
        if hidden is None:
            return None

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


def is_math_expression(text: str) -> bool:
    """Sadece math expression mi? (örn: "3+4=")"""
    return any(text.endswith(op + "=") for op in ["+", "-", "*"])


def is_math_statement(text: str) -> bool:
    """Math statement + doğrulama? (örn: "10-3=7 doğru")"""
    has_operator = any(op in text for op in ["+", "-", "*"])
    has_equals = "=" in text
    has_verification = any(w in text.lower() for w in ["doğru", "yanlış", "mı", "mi"])
    return has_operator and has_equals and has_verification


def get_expected_action(text: str, label: str = None) -> str:
    """
    Ground truth action.

    Clean math expression → compute
    Math statement (doğrulama) → verify
    Text → generate
    """
    if label is not None:
        return label

    if is_math_statement(text):
        return "verify"
    if is_math_expression(text) or any(op in text for op in ["+", "-", "*"]):
        return "compute"
    return "generate"


def execute_action(
    action: str, text: str, char2idx: dict, model, fast: bool = False
) -> str:
    """Action'ı çalıştır."""
    if action == "compute":
        intent = extract_intent(text)
        if intent is None:
            return "[parse error]"
        result = run_engine(intent)
        return str(result) if result is not None else "[error]"

    elif action == "verify":
        intent = extract_intent(text)
        if intent is None:
            return "[parse error]"
        result = run_engine(intent)
        import re

        match = re.search(r"=(\d+)", text)
        if match:
            expected = int(match.group(1))
            is_correct = result == expected
            return f"{'Evet' if is_correct else 'Hayır'}, {result}"
        return str(result)

    else:  # generate
        if fast:
            return "[text]"  # Training sırasında placeholder
        tokens = encode(text, char2idx)
        idx = torch.tensor([tokens], device=DEVICE)
        with torch.no_grad():
            model.eval()
            generated = model.generate(idx, max_new_tokens=5)
        idx2char = {str(v): k for k, v in char2idx.items()}
        output = []
        for t in range(len(tokens), generated.shape[1]):
            c = idx2char.get(str(generated[0, t].item()), "?")
            if c == "\n":
                break
            output.append(c)
        return "".join(output)


def evaluate_action(action: str, text: str, output: str) -> float:
    """Outcome-based reward."""
    expected = get_expected_action(text)

    if action == expected:
        # Doğru action seçildi
        if action == "compute":
            intent = extract_intent(text)
            if intent is None:
                return -1.0
            expected_result = run_engine(intent)
            if output == str(expected_result):
                return +1.0
            return -0.5  # Doğru action ama yanlış sonuç

        elif action == "verify":
            # Verify action'ı her zaman doğru sonuç üretir
            return +1.0

        else:  # generate
            return +1.0
    else:
        return -1.0


def outcome_stdp_update(action_head, pre, action_idx, reward, lr):
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
    print("VİCDAN — Phase 6.7: Multi-Step Reasoning (3 Action)")
    print("=" * 70)
    print(f"  Actions: {ACTIONS}")
    print(f"  Op Boost: {OP_BOOST}, LR: {LR}")

    vocab_size, char2idx, idx2char_raw = load_vocab()
    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]

    action_head = ActionHead3(d_model).to(DEVICE)
    print(
        f"  ActionHead: {d_model} → {N_ACTIONS} ({sum(p.numel() for p in action_head.parameters()):,} params)"
    )

    # Training data
    train_compute = [
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
        "3+4 kaç eder?",  # Ambiguous → compute
        ("hesapla 6*8", "compute"),
        ("sonuç: 4*5", "compute"),
        ("8+9 kaçtır?", "compute"),
        ("5+7'nin sonucu", "compute"),
    ]
    train_verify = [
        "10-3=7 doğru",
        "2+2=5 doğru mu?",
        "6-3=3 doğru",
        "3*4=12 doğru",
    ]
    train_generate = [
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
        "mehmet 3+4 dedi",  # Alıntı → generate
        ("bugün 5-2 yaptım", "generate"),  # Text içinde math
        ("toplama nedir?", "generate"),  # Math konusu ama text
        ("kedi ve köpek", "generate"),
    ]

    # Flatten
    train_pool = []
    for item in train_compute:
        if isinstance(item, tuple):
            train_pool.append((item[0], item[1]))
        else:
            train_pool.append((item, "compute"))
    for item in train_verify:
        train_pool.append((item, "verify"))
    for item in train_generate:
        if isinstance(item, tuple):
            train_pool.append((item[0], item[1]))
        else:
            train_pool.append((item, "generate"))

    # Test data
    test_compute = ["6+1=", "8-5=", "3*2=", "12+34="]
    test_verify = ["5-3=2 doğru", "4*4=16 doğru mu?"]
    test_generate = ["mehmet", "nasılsın", "bir hafta", "köpek"]

    # Pre-compute hidden states
    print(f"\n  Pre-computing hidden states...")
    hidden_cache = {}
    all_texts = [t for t, _ in train_pool] + test_compute + test_verify + test_generate
    for text in all_texts:
        if text not in hidden_cache:
            h = get_saliency_modulated_hidden(model, text, char2idx)
            if h is not None:
                hidden_cache[text] = h
    print(f"  Cached {len(hidden_cache)} hidden states")

    # ── Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Outcome-Based STDP Training (3 Action)")
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
            _, probs, _, _ = action_head(h)
            action_idx = torch.multinomial(probs[0], 1).item()
            action = ACTIONS[action_idx]

            # Execute
            output = execute_action(action, text, char2idx, model, fast=True)

            # Reward
            reward = evaluate_action(action, text, output)

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
            # Per-action accuracy
            compute_correct = 0
            compute_total = 0
            verify_correct = 0
            verify_total = 0
            generate_correct = 0
            generate_total = 0

            for text, label in train_pool:
                h = hidden_cache.get(text)
                if h is None:
                    continue
                pred, _ = predict_action(action_head, h, greedy=True)
                if label == "compute":
                    compute_total += 1
                    if pred == "compute":
                        compute_correct += 1
                elif label == "verify":
                    verify_total += 1
                    if pred == "verify":
                        verify_correct += 1
                elif label == "generate":
                    generate_total += 1
                    if pred == "generate":
                        generate_correct += 1

            compute_acc = compute_correct / max(compute_total, 1) * 100
            verify_acc = verify_correct / max(verify_total, 1) * 100
            generate_acc = generate_correct / max(generate_total, 1) * 100

            history.append(
                {
                    "epoch": epoch + 1,
                    "avg_reward": avg_reward,
                    "compute_accuracy": compute_acc,
                    "verify_accuracy": verify_acc,
                    "generate_accuracy": generate_acc,
                }
            )
            print(
                f"    Epoch {epoch + 1:3d}: reward={avg_reward:+.2f}, compute={compute_acc:.0f}%, verify={verify_acc:.0f}%, generate={generate_acc:.0f}%"
            )

            if compute_acc >= 80 and verify_acc >= 80 and generate_acc >= 80:
                print(f"\n  ⏹️ Early stopping at epoch {epoch + 1}")
                break

    # ── Test: Compute ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Compute Test (Unseen)")
    print(f"{'=' * 70}")

    for text in test_compute:
        h = hidden_cache.get(text)
        if h is None:
            continue
        pred, conf = predict_action(action_head, h, greedy=True)
        output = execute_action(pred, text, char2idx, model)
        correct = pred == "compute"
        status = "✅" if correct else "❌"
        print(f"  {status} {text!r} → {pred!r} (conf={conf:.3f}) → {output!r}")

    # ── Test: Verify ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Verify Test (Unseen)")
    print(f"{'=' * 70}")

    for text in test_verify:
        h = hidden_cache.get(text)
        if h is None:
            continue
        pred, conf = predict_action(action_head, h, greedy=True)
        output = execute_action(pred, text, char2idx, model)
        correct = pred == "verify"
        status = "✅" if correct else "❌"
        print(f"  {status} {text!r} → {pred!r} (conf={conf:.3f}) → {output!r}")

    # ── Test: Generate ──
    print(f"\n{'=' * 70}")
    print("ADIM 4: Generate Test (Unseen)")
    print(f"{'=' * 70}")

    for text in test_generate:
        h = hidden_cache.get(text)
        if h is None:
            continue
        pred, conf = predict_action(action_head, h, greedy=True)
        output = execute_action(pred, text, char2idx, model)
        correct = pred == "generate"
        status = "✅" if correct else "❌"
        print(f"  {status} {text!r} → {pred!r} (conf={conf:.3f}) → {output!r}")

    # ── Full Pipeline Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 5: Full Pipeline")
    print(f"{'=' * 70}")

    all_test = (
        [(t, "compute") for t in test_compute]
        + [(t, "verify") for t in test_verify]
        + [(t, "generate") for t in test_generate]
    )

    pipeline_correct = 0
    for text, expected in all_test:
        h = hidden_cache.get(text)
        if h is None:
            continue
        pred, conf = predict_action(action_head, h, greedy=True)
        output = execute_action(pred, text, char2idx, model)
        correct = pred == expected
        if correct:
            pipeline_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {text!r} → {pred!r} → {output!r}")

    pipeline_acc = pipeline_correct / len(all_test) * 100

    # ── Save ──
    results = {
        "config": {"lr": LR, "op_boost": OP_BOOST, "actions": ACTIONS},
        "pipeline_accuracy": pipeline_acc,
        "history": history,
    }
    result_path = os.path.join(os.path.dirname(__file__), "phase6_7_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {result_path}")

    ckpt_path = os.path.join(CKPT_DIR, "action_head_v5_multi_action.pt")
    torch.save(action_head.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Pipeline accuracy: {pipeline_acc:.0f}%")

    if pipeline_acc >= 80:
        print(f"\n  ✅ BAŞARILI: Multi-action reasoning çalışıyor!")
    elif pipeline_acc >= 60:
        print(f"\n  ⚠️ Kısmi başarı")
    else:
        print(f"\n  ❌ Başarısız")


if __name__ == "__main__":
    main()
