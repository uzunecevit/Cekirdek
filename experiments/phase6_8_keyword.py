#!/usr/bin/env python3
"""
VİCDAN — Phase 6.8: Keyword-Enhanced Action Selection

Verify için keyword boost + genişletilmiş verify dataset.
"""

import os
import sys
import json
import random
import re

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
OP_BOOST = 10.0
VERIFY_BOOST = 5.0  # Verify keyword boost


class ActionHead3(nn.Module):
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
            return {"status": "consolidated"}
        return {"status": "skipped"}


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
    """Operatör + verify keyword boost."""
    saliency = [1.0] * len(text)
    for i, ch in enumerate(text):
        if ch in ["+", "-", "*"]:
            saliency[i] = OP_BOOST
    # Verify keywords
    verify_keywords = ["doğru", "yanlış", "mı", "mi", "mu", "mü"]
    for kw in verify_keywords:
        start = 0
        while True:
            idx = text.lower().find(kw, start)
            if idx == -1:
                break
            for j in range(idx, min(idx + len(kw), len(text))):
                saliency[j] = max(saliency[j], VERIFY_BOOST)
            start = idx + 1
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


def get_expected_action(text: str, label: str = None) -> str:
    if label is not None:
        return label
    has_verify = any(
        kw in text.lower() for kw in ["doğru", "yanlış", "mı", "mi", "mu", "mü"]
    )
    has_operator = any(op in text for op in ["+", "-", "*"])
    if has_operator and has_verify:
        return "verify"
    if has_operator:
        return "compute"
    return "generate"


def execute_action(
    action: str, text: str, char2idx: dict, model, fast: bool = False
) -> str:
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
        expected = intent.get("expected")
        if expected is not None:
            is_correct = result == expected
            return (
                f"{'Evet' if is_correct else 'Hayır'}, {result} (beklenen: {expected})"
            )
        # Fallback: try to extract expected from text
        import re

        match = re.search(r"=(\d+)", text)
        if match:
            expected = int(match.group(1))
            is_correct = result == expected
            return f"{'Evet' if is_correct else 'Hayır'}, {result}"
        return str(result)
    else:
        if fast:
            return "[text]"
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
    expected = get_expected_action(text)
    if action == expected:
        if action == "compute":
            intent = extract_intent(text)
            if intent is None:
                return -1.0
            expected_result = run_engine(intent)
            return +1.0 if output == str(expected_result) else -0.5
        elif action == "verify":
            return +1.0
        else:
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
    return {"reward": reward, "action": ACTIONS[action_idx]}


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
    print("VİCDAN — Phase 6.8: Keyword-Enhanced Action Selection")
    print("=" * 70)
    print(f"  Actions: {ACTIONS}")
    print(f"  Op Boost: {OP_BOOST}, Verify Boost: {VERIFY_BOOST}")
    print(f"  LR: {LR}")

    vocab_size, char2idx, _ = load_vocab()
    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]
    action_head = ActionHead3(d_model).to(DEVICE)

    # ── Training Data (Genişletilmiş) ──
    train_pool = [
        # Compute (18 örnek)
        ("3+4=", "compute"),
        ("5-2=", "compute"),
        ("2*3=", "compute"),
        ("7+8=", "compute"),
        ("9-3=", "compute"),
        ("4*4=", "compute"),
        ("1+1=", "compute"),
        ("0+5=", "compute"),
        ("6-4=", "compute"),
        ("3*3=", "compute"),
        ("8+9 kaçtır?", "compute"),
        ("sonuç: 4*5", "compute"),
        ("5+7'nin sonucu", "compute"),
        ("hesapla 6*8", "compute"),
        ("12+34=", "compute"),
        ("10-7=", "compute"),
        ("6*9=", "compute"),
        ("15-8=", "compute"),
        # Verify (20 örnek — genişletildi)
        ("10-3=7 doğru", "verify"),
        ("2+2=5 doğru mu?", "verify"),
        ("6-3=3 doğru", "verify"),
        ("3*4=12 doğru", "verify"),
        ("5-3=2 doğru mu?", "verify"),
        ("8+1=9 doğru", "verify"),
        ("7-4=3 doğru mu?", "verify"),
        ("2*5=10 doğru", "verify"),
        ("9-6=3 doğru", "verify"),
        ("4+4=8 doğru mu?", "verify"),
        ("3*3=9 doğru", "verify"),
        ("10-5=5 doğru", "verify"),
        ("6+2=8 doğru mu?", "verify"),
        ("8-3=5 doğru", "verify"),
        ("4*2=8 doğru", "verify"),
        ("7+3=10 doğru mu?", "verify"),
        ("9-2=7 doğru", "verify"),
        ("5*2=10 doğru", "verify"),
        ("1+9=10 doğru mu?", "verify"),
        ("6-1=5 doğru", "verify"),
        # Generate (16 örnek)
        ("ahmet", "generate"),
        ("merhaba", "generate"),
        ("bir gün", "generate"),
        ("kedi", "generate"),
        ("ev", "generate"),
        ("istanbul", "generate"),
        ("selam", "generate"),
        ("güneşli", "generate"),
        ("kitap", "generate"),
        ("dün", "generate"),
        ("mehmet 3+4 dedi", "generate"),
        ("bugün 5-2 yaptım", "generate"),
        ("toplama nedir?", "generate"),
        ("kedi ve köpek", "generate"),
        ("nasılsın", "generate"),
        ("arkadaş", "generate"),
    ]

    # Test data
    test_compute = [("6+1=", "compute"), ("8-5=", "compute"), ("3*2=", "compute")]
    test_verify = [
        ("5-3=2 doğru", "verify"),
        ("4*4=16 doğru mu?", "verify"),
        ("3+7=10 doğru", "verify"),
    ]
    test_generate = [
        ("mehmet", "generate"),
        ("nasılsın", "generate"),
        ("bir hafta", "generate"),
    ]
    all_test = test_compute + test_verify + test_generate

    # Pre-compute
    print(f"\n  Pre-computing hidden states...")
    hidden_cache = {}
    for text, _ in train_pool + all_test:
        if text not in hidden_cache:
            h = get_saliency_modulated_hidden(model, text, char2idx)
            if h is not None:
                hidden_cache[text] = h
    print(f"  Cached {len(hidden_cache)} hidden states")

    # ── Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training")
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
            output = execute_action(action, text, char2idx, model, fast=True)
            reward = evaluate_action(action, text, output)
            outcome_stdp_update(action_head, pre.detach(), action_idx, reward, lr=LR)
            epoch_rewards.append(reward)
            if reward > 0:
                epoch_correct += 1

        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        accuracy = epoch_correct / max(len(epoch_rewards), 1) * 100

        if (epoch + 1) % CONSOLIDATION_EVERY == 0:
            action_head.consolidate(alpha=0.1, threshold=0.01)

        if (epoch + 1) % 10 == 0:
            comp_c = sum(
                1
                for t, l in train_pool
                if l == "compute"
                and predict_action(action_head, hidden_cache[t], True)[0] == "compute"
            )
            comp_t = sum(1 for t, l in train_pool if l == "compute")
            ver_c = sum(
                1
                for t, l in train_pool
                if l == "verify"
                and predict_action(action_head, hidden_cache[t], True)[0] == "verify"
            )
            ver_t = sum(1 for t, l in train_pool if l == "verify")
            gen_c = sum(
                1
                for t, l in train_pool
                if l == "generate"
                and predict_action(action_head, hidden_cache[t], True)[0] == "generate"
            )
            gen_t = sum(1 for t, l in train_pool if l == "generate")

            history.append(
                {
                    "epoch": epoch + 1,
                    "avg_reward": avg_reward,
                    "compute": f"{comp_c}/{comp_t}",
                    "verify": f"{ver_c}/{ver_t}",
                    "generate": f"{gen_c}/{gen_t}",
                }
            )
            print(
                f"    Epoch {epoch + 1:3d}: reward={avg_reward:+.2f}, compute={comp_c}/{comp_t}, verify={ver_c}/{ver_t}, generate={gen_c}/{gen_t}"
            )

            if comp_c == comp_t and ver_c == ver_t and gen_c == gen_t:
                print(f"  ⏹️ Early stopping at epoch {epoch + 1}")
                break

    # ── Tests ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Test Results")
    print(f"{'=' * 70}")

    pipeline_correct = 0
    for text, expected in all_test:
        h = hidden_cache.get(text)
        if h is None:
            continue
        pred, conf = predict_action(action_head, h, greedy=True)
        output = execute_action(pred, text, char2idx, model, fast=True)
        correct = pred == expected
        if correct:
            pipeline_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {text!r} → {pred!r} (conf={conf:.3f}) → {output!r}")

    pipeline_acc = pipeline_correct / len(all_test) * 100

    # Per-action
    comp_test = [(t, l) for t, l in all_test if l == "compute"]
    ver_test = [(t, l) for t, l in all_test if l == "verify"]
    gen_test = [(t, l) for t, l in all_test if l == "generate"]

    comp_acc = (
        sum(
            1
            for t, l in comp_test
            if predict_action(action_head, hidden_cache[t], True)[0] == l
        )
        / max(len(comp_test), 1)
        * 100
    )
    ver_acc = (
        sum(
            1
            for t, l in ver_test
            if predict_action(action_head, hidden_cache[t], True)[0] == l
        )
        / max(len(ver_test), 1)
        * 100
    )
    gen_acc = (
        sum(
            1
            for t, l in gen_test
            if predict_action(action_head, hidden_cache[t], True)[0] == l
        )
        / max(len(gen_test), 1)
        * 100
    )

    print(f"\n  Compute: {comp_acc:.0f}%")
    print(f"  Verify:  {ver_acc:.0f}%")
    print(f"  Generate: {gen_acc:.0f}%")
    print(f"  Pipeline: {pipeline_acc:.0f}%")

    # Save
    results = {
        "config": {"lr": LR, "op_boost": OP_BOOST, "verify_boost": VERIFY_BOOST},
        "compute_accuracy": comp_acc,
        "verify_accuracy": ver_acc,
        "generate_accuracy": gen_acc,
        "pipeline_accuracy": pipeline_acc,
        "history": history,
    }
    result_path = os.path.join(os.path.dirname(__file__), "phase6_8_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {result_path}")

    ckpt_path = os.path.join(CKPT_DIR, "action_head_v6_keyword.pt")
    torch.save(action_head.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    if pipeline_acc >= 90:
        print(f"\n  ✅ BAŞARILI: Keyword-enhanced action selection çalıştı!")
    elif pipeline_acc >= 70:
        print(f"\n  ⚠️ Kısmi başarı")
    else:
        print(f"\n  ❌ Başarısız")


if __name__ == "__main__":
    main()
