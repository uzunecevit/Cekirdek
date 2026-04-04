#!/usr/bin/env python3
"""
VİCDAN — Hybrid v0.1: Neuro-Symbolic Training Demo

LM Head = Backprop/Adam (interference-free)
Internal = STDP (online, biyolojik öğrenme)

Protokol:
1. Model yükle (spiking_lm_v2.pt)
2. Öncesi generate testi
3. Hybrid training (backprop + STDP)
4. Sonrası generate testi
5. Consolidation + kalıcılık testi
6. Generalization testi
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model import SpikingLM
from src.hybrid import hybrid_step, hybrid_train_loop

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
EPOCHS = 200
LR = 0.005
CONSOLIDATION_ALPHA = 0.05
CONSOLIDATION_THRESHOLD = 0.01
BLOCK_SIZE = 32
BATCHED = True  # Mini-batch gradient descent (interference-free)


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


def idx2char_map(idx2char_raw):
    return {str(k): v for k, v in idx2char_raw.items()}


def get_next_token(
    model, prompt: str, char2idx: dict, idx2char: dict
) -> list[tuple[str, float]]:
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


def generate_text(
    model, prompt: str, char2idx: dict, idx2char: dict, max_tokens: int = 5
) -> str:
    model.eval()
    tokens = [char2idx.get(c, 0) for c in prompt if c in char2idx]
    idx = torch.tensor([tokens], device=DEVICE)
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_tokens)
        result_chars = []
        for t in range(len(tokens), generated.shape[1]):
            c = idx2char.get(str(generated[0, t].item()), "?")
            result_chars.append(c)
    return prompt + "".join(result_chars)


def main():
    print("=" * 70)
    print("VİCDAN — Hybrid v0.1: LM Head = Backprop, Internal = STDP")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    idx2char = idx2char_map(idx2char_raw)

    model, ckpt = load_model()
    print(f"\n  Model yüklendi: spiking_lm_v2.pt")
    print(
        f"  Vocab: {vocab_size}, Params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Training samples
    train_samples = [
        "3+4=",
        "5-2=",
        "1+1=",
        "9-3=",
        "2*3=",
        "7+8=",
        "4*4=",
        "0+5=",
    ]

    test_cases = [
        ("3+4=", "7"),
        ("5-2=", "3"),
        ("1+1=", "2"),
        ("9-3=", "6"),
        ("2*3=", "6"),
        ("7+8=", "15"),
        ("4*4=", "16"),
        ("0+5=", "5"),
    ]

    # ── Adım 1: Öncesi Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 1: Training Öncesi")
    print(f"{'=' * 70}")

    for prompt, expected in test_cases:
        top5 = get_next_token(model, prompt, char2idx, idx2char)
        top1 = top5[0][0]
        correct = top1 == expected[0]
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → top1={top1!r} (beklenen: {expected!r})")

    # ── Adım 2: Hybrid Training ──
    print(f"\n{'=' * 70}")
    print("ADIM 2: Hybrid Training (Backprop LM Head + STDP Internal)")
    print(f"{'=' * 70}")

    history = hybrid_train_loop(
        model,
        train_samples,
        char2idx,
        epochs=EPOCHS,
        lr=LR,
        consolidation_alpha=CONSOLIDATION_ALPHA,
        consolidation_threshold=CONSOLIDATION_THRESHOLD,
        consolidation_every=20,
        ffn_lr=0.0,
        batched=BATCHED,
    )

    print(f"\n  Training Progress:")
    for h in history:
        if h["epoch"] % 20 == 0 or h["epoch"] <= 5:
            correct_count = 0
            for prompt, expected in test_cases:
                top5 = get_next_token(model, prompt, char2idx, idx2char)
                if top5[0][0] == expected[0]:
                    correct_count += 1
            print(
                f"    Epoch {h['epoch']:3d}: loss={h['loss']:.4f}, acc={correct_count}/{len(test_cases)}, consolidated={h['consolidated']}"
            )

    # ── Adım 3: Sonrası Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 3: Training Sonrası")
    print(f"{'=' * 70}")

    correct_count = 0
    for prompt, expected in test_cases:
        top5 = get_next_token(model, prompt, char2idx, idx2char)
        top1 = top5[0][0]
        correct = top1 == expected[0]
        if correct:
            correct_count += 1
        status = "✅" if correct else "❌"
        probs_str = ", ".join([f"{t}={p:.3f}" for t, p in top5[:3]])
        print(f"  {status} {prompt!r} → top1={top1!r} [{probs_str}]")

    accuracy = correct_count / len(test_cases) * 100
    print(f"\n  Accuracy: {correct_count}/{len(test_cases)} ({accuracy:.0f}%)")

    # ── Adım 4: Generate Test ──
    print(f"\n{'=' * 70}")
    print("ADIM 4: Generate Test")
    print(f"{'=' * 70}")

    for prompt, expected in test_cases[:4]:
        output = generate_text(model, prompt, char2idx, idx2char, max_tokens=3)
        print(f"  {prompt!r} → {output!r}")

    # ── Adım 5: Generalization ──
    print(f"\n{'=' * 70}")
    print("ADIM 5: Generalization (eğitimde olmayan örnekler)")
    print(f"{'=' * 70}")

    gen_samples = [
        ("6+1=", "7"),
        ("8-5=", "3"),
        ("3*2=", "6"),
        ("2+9=", "11"),
        ("10-4=", "6"),
    ]

    gen_correct = 0
    for prompt, expected in gen_samples:
        top5 = get_next_token(model, prompt, char2idx, idx2char)
        top1 = top5[0][0]
        correct = top1 == expected[0]
        if correct:
            gen_correct += 1
        status = "✅" if correct else "❌"
        print(f"  {status} {prompt!r} → top1={top1!r} (beklenen: {expected!r})")

    gen_accuracy = gen_correct / len(gen_samples) * 100
    print(f"\n  Generalization: {gen_correct}/{len(gen_samples)} ({gen_accuracy:.0f}%)")

    # ── Sonuçları Kaydet ──
    results = {
        "config": {
            "epochs": EPOCHS,
            "lr": LR,
            "consolidation_alpha": CONSOLIDATION_ALPHA,
        },
        "training": {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(test_cases),
            "history": history,
        },
        "generalization": {
            "accuracy": gen_accuracy,
            "correct": gen_correct,
            "total": len(gen_samples),
        },
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase6_0_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")

    # ── Özet ──
    print(f"\n{'=' * 70}")
    print("ÖZET")
    print(f"{'=' * 70}")
    print(f"  Training accuracy:  {accuracy:.0f}%")
    print(f"  Generalization:     {gen_accuracy:.0f}%")

    if accuracy >= 80:
        print(f"\n  ✅ BAŞARILI: Hybrid v0.1 (Backprop LM Head) çalışıyor!")
    elif accuracy >= 50:
        print(f"\n  ⚠️ Kısmi başarı: Daha fazla eğitim veya LR ayarı gerekli")
    else:
        print(f"\n  ❌ Düşük accuracy: Mimari gözden geçirilmeli")


if __name__ == "__main__":
    main()
