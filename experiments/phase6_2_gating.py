#!/usr/bin/env python3
"""
VİCDAN — Phase 6.2: Task Gating + Confidence Routing

SNN input'un task'ını sınıflandırır:
  - "math" → Symbolic Engine
  - "text" → SNN Generate

Training: TaskClassifier (Linear + softmax)
Test: Seen + Unseen accuracy + confidence routing
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
from src.task_gating import TaskClassifier
from src.intent import extract_intent
from src.engine import run_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 32

TASKS = ["math", "text"]
TASK2IDX = {"math": 0, "text": 1}


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


def is_math(text: str) -> bool:
    """Basit math detection (ground truth için)."""
    return any(op in text for op in ["+", "-", "*"])


def build_batch(samples: list[str], labels: list[int], char2idx: dict):
    """Task classification için batch oluştur."""
    all_x = []
    all_y = []

    for text, label in zip(samples, labels):
        x = encode(text, char2idx)
        if len(x) < 2:
            continue
        all_x.append(x)
        all_y.append(label)

    if not all_x:
        return None, None

    max_len = max(len(x) for x in all_x)
    B = len(all_x)

    padded_x = torch.zeros(B, max_len, dtype=torch.long, device=DEVICE)
    padded_y = torch.tensor(all_y, dtype=torch.long, device=DEVICE)

    for i, x in enumerate(all_x):
        padded_x[i, : len(x)] = torch.tensor(x, device=DEVICE)

    return padded_x, padded_y


def predict_task(classifier, model, text: str, char2idx: dict) -> tuple[str, float]:
    """Model + classifier ile task tahmini."""
    model.eval()
    classifier.eval()
    tokens = encode(text, char2idx)
    idx = torch.tensor([tokens], device=DEVICE)

    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        hidden = model.get_last_hidden()  # (B, T, d_model)
        if hidden is None:
            return "text", 0.0

        task_logits, confidence = classifier(hidden)
        task_idx = task_logits.argmax(dim=-1).item()
        conf = confidence.item()

    return TASKS[task_idx], conf


def main():
    print("=" * 70)
    print("VİCDAN — Phase 6.2: Task Gating + Confidence Routing")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    idx2char = idx2char_map(idx2char_raw)

    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]

    # Task Classifier
    classifier = TaskClassifier(d_model, n_tasks=2).to(DEVICE)

    print(f"\n  Model yüklendi: spiking_lm_v2.pt")
    print(f"  TaskClassifier: {d_model} → 2 (math, text)")
    print(
        f"  TaskClassifier params: {sum(p.numel() for p in classifier.parameters()):,}"
    )

    # Training data (seen)
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
    train_labels = [TASK2IDX["math"]] * len(train_math) + [TASK2IDX["text"]] * len(
        train_text
    )

    # Test data (unseen)
    test_math = [
        "6+1=",
        "8-5=",
        "3*2=",
        "12+34=",
        "10-4=",
        "5*3=",
    ]
    test_text = [
        "mehmet",
        "nasılsın",
        "bir hafta",
        "köpek",
        "okul",
        "yağmurlu",
    ]

    test_samples = test_math + test_text
    test_labels = [TASK2IDX["math"]] * len(test_math) + [TASK2IDX["text"]] * len(
        test_text
    )

    # ── Adım 1: Training Öncesi ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training Öncesi Task Tahminleri")
    print(f"{'=' * 70}")

    for prompt in train_samples[:6]:
        expected = "math" if is_math(prompt) else "text"
        pred, conf = predict_task(classifier, model, prompt, char2idx)
        correct = pred == expected
        status = "✅" if correct else "❌"
        print(
            f"  {status} {prompt!r} → task={pred!r} (beklenen: {expected!r}) conf={conf:.3f}"
        )

    # ── Adım 2: Task Classification Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Task Classification Training")
    print(f"{'=' * 70}")

    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    EPOCHS = 100

    history = []
    for epoch in range(EPOCHS):
        # Shuffle
        combined = list(zip(train_samples, train_labels))
        random.shuffle(combined)
        train_samples_shuffled = [s for s, _ in combined]
        train_labels_shuffled = [l for _, l in combined]

        x_batch, y_batch = build_batch(
            train_samples_shuffled, train_labels_shuffled, char2idx
        )
        if x_batch is None:
            continue

        optimizer.zero_grad()

        # Forward through SNN
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_batch)
            hidden = model.get_last_hidden()

        # Forward through classifier
        task_logits, _ = classifier(hidden)
        loss = F.cross_entropy(task_logits, y_batch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            # Accuracy
            correct = 0
            for s, l in zip(train_samples_shuffled, train_labels_shuffled):
                pred, _ = predict_task(classifier, model, s, char2idx)
                if TASK2IDX[pred] == l:
                    correct += 1
            acc = correct / len(train_samples_shuffled) * 100
            history.append({"epoch": epoch + 1, "loss": loss.item(), "accuracy": acc})
            print(
                f"    Epoch {epoch + 1:3d}: loss={loss.item():.4f}, train_acc={acc:.0f}%"
            )

    # ── Adım 3: Seen Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Seen Samples Test")
    print(f"{'=' * 70}")

    seen_correct = 0
    for prompt in train_samples:
        expected = "math" if is_math(prompt) else "text"
        pred, conf = predict_task(classifier, model, prompt, char2idx)
        correct = pred == expected
        if correct:
            seen_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → task={pred!r} conf={conf:.3f}")

    seen_acc = seen_correct / len(train_samples) * 100
    print(f"\n  Seen Accuracy: {seen_correct}/{len(train_samples)} ({seen_acc:.0f}%)")

    # ── Adım 4: Unseen Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 4: Unseen Samples Test (Generalization)")
    print(f"{'=' * 70}")

    unseen_correct = 0
    for prompt in test_samples:
        expected = "math" if is_math(prompt) else "text"
        pred, conf = predict_task(classifier, model, prompt, char2idx)
        correct = pred == expected
        if correct:
            unseen_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → task={pred!r} conf={conf:.3f}")

    unseen_acc = unseen_correct / len(test_samples) * 100
    print(
        f"\n  Unseen Accuracy: {unseen_correct}/{len(test_samples)} ({unseen_acc:.0f}%)"
    )

    # ── Adım 5: Full Pipeline — Task Routing ──
    print(f"\n{'=' * 70}")
    print("ADIM 5: Full Pipeline — Task Routing + Engine")
    print(f"{'=' * 70}")

    CONFIDENCE_THRESHOLD = 0.7
    all_test = test_math + test_text
    pipeline_correct = 0

    for prompt in all_test:
        expected_task = "math" if is_math(prompt) else "text"
        pred_task, conf = predict_task(classifier, model, prompt, char2idx)

        # Routing
        if conf > CONFIDENCE_THRESHOLD and pred_task == "math":
            # Engine kullan
            intent = extract_intent(prompt)
            result = run_engine(intent)
            output = str(result) if result is not None else "ERROR"
        else:
            # SNN generate (simüle: task tahmini)
            output = f"[{pred_task}]"

        correct = pred_task == expected_task
        if correct:
            pipeline_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → task={pred_task!r} conf={conf:.3f} → {output}")

    pipeline_acc = pipeline_correct / len(all_test) * 100
    print(
        f"\n  Pipeline Accuracy: {pipeline_correct}/{len(all_test)} ({pipeline_acc:.0f}%)"
    )

    # ── Sonuçları Kaydet ──
    results = {
        "config": {
            "epochs": EPOCHS,
            "lr": 0.001,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "pipeline_accuracy": pipeline_acc,
        "history": history,
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase6_2_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")

    # ── TaskClassifier Checkpoint Kaydet ──
    ckpt_path = os.path.join(CKPT_DIR, "task_classifier_v1.pt")
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(classifier.state_dict(), ckpt_path)
    print(f"  TaskClassifier checkpoint kaydedildi: {ckpt_path}")

    # ── Özet ──
    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Seen task accuracy:    {seen_acc:.0f}%")
    print(f"  Unseen task accuracy:  {unseen_acc:.0f}%")
    print(f"  Pipeline accuracy:     {pipeline_acc:.0f}%")

    if unseen_acc >= 80:
        print(f"\n  ✅ BAŞARILI: Task gating generalization çalışıyor!")
    elif unseen_acc >= 50:
        print(f"\n  ⚠️ Kısmi başarı: Daha fazla eğitim veya data gerekli")
    else:
        print(f"\n  ❌ Generalization başarısız")


if __name__ == "__main__":
    main()
