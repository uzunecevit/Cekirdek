#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 4.2: Layer-wise FFN STDP

Hedef: FFN fc2 katmanını plastikleştirerek matematiksel mantığı öğrenmek.
Test: Matematik dataset üzerinde LM Head + FFN fc2 STDP
"""

import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.phase4_1_multi_task import (
    load_model,
    load_vocab,
    load_lines,
    make_dataloader,
    evaluate_loss,
    apply_stdp_gated,
    train_task,
    compute_surprise,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 5
BATCH_SIZE = 32
FFN_LR_VALUES = [0.0, 0.001, 0.005, 0.01]


def main():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 4.2: Layer-wise FFN STDP")
    print("=" * 70)

    vocab_size, char2idx, idx2char = load_vocab()
    idx2char = {int(k): v for k, v in idx2char.items()}

    # Dataset'ler
    names_lines = load_lines("pretrain_samples.txt")
    turkish_lines = load_lines("turkish.txt")
    math_lines = load_lines("math.txt")

    names_dl, names_n = make_dataloader(names_lines, char2idx)
    turkish_dl, turkish_n = make_dataloader(turkish_lines, char2idx)
    math_dl, math_n = make_dataloader(math_lines, char2idx)

    print(f"\n  Dataset boyutları:")
    print(f"    İsimler: {names_n}, Türkçe: {turkish_n}, Matematik: {math_n}")

    results = {}

    for ffn_lr in FFN_LR_VALUES:
        print(f"\n{'=' * 70}")
        print(f"TEST: FFN LR = {ffn_lr}")
        print(f"{'=' * 70}")

        # Model yükle (her test için temiz başla)
        model, ckpt = load_model()

        # Baseline
        baseline_math = evaluate_loss(model, math_dl)
        baseline_names = evaluate_loss(model, names_dl)
        baseline_turkish = evaluate_loss(model, turkish_dl)

        print(
            f"  Baseline: math={baseline_math:.4f}, names={baseline_names:.4f}, turkish={baseline_turkish:.4f}"
        )

        # Matematik eğitimi (FFN STDP ile)
        train_task(model, math_dl, "Matematik", epochs=EPOCHS, ffn_lr=ffn_lr)

        # Sonuçlar
        post_math = evaluate_loss(model, math_dl)
        post_names = evaluate_loss(model, names_dl)
        post_turkish = evaluate_loss(model, turkish_dl)

        math_change = (post_math - baseline_math) / baseline_math * 100
        names_forgetting = (post_names - baseline_names) / baseline_names * 100
        turkish_forgetting = (post_turkish - baseline_turkish) / baseline_turkish * 100

        print(f"\n  Sonuçlar:")
        print(
            f"    Matematik: {baseline_math:.4f} → {post_math:.4f} ({math_change:+.1f}%)"
        )
        print(f"    İsimler forgetting: {names_forgetting:+.1f}%")
        print(f"    Türkçe forgetting: {turkish_forgetting:+.1f}%")

        # FFN fast weight normları
        for li, block in enumerate(model.blocks):
            fc2 = block.ffn.fc2
            if fc2.fast_weight:
                norm = fc2.weight_fast.norm().item()
                print(f"    Block {li} FFN fc2 W_fast norm: {norm:.4f}")

        results[f"ffn_lr_{ffn_lr}"] = {
            "baseline_math": baseline_math,
            "post_math": post_math,
            "math_change_pct": math_change,
            "names_forgetting_pct": names_forgetting,
            "turkish_forgetting_pct": turkish_forgetting,
        }

    # Özet tablo
    print(f"\n{'=' * 70}")
    print("ÖZET TABLO")
    print(f"{'=' * 70}")
    print(f"{'FFN LR':>10} | {'Math Δ%':>10} | {'Names Fgt%':>10} | {'Turk Fgt%':>10}")
    print(f"{'-' * 70}")
    for key, val in results.items():
        ffn_lr = float(key.split("_")[-1])
        print(
            f"{ffn_lr:10.4f} | {val['math_change_pct']:9.1f}% | {val['names_forgetting_pct']:9.1f}% | {val['turkish_forgetting_pct']:9.1f}%"
        )

    # En iyi sonucu bul
    best = min(results.items(), key=lambda x: x[1]["math_change_pct"])
    print(f"\n  En iyi: {best[0]} → Math değişim: {best[1]['math_change_pct']:+.1f}%")

    if best[1]["math_change_pct"] < 0:
        print(f"  ✅ FFN STDP matematik öğrenmeyi başardı!")
    else:
        print(f"  ❌ FFN STDP matematik öğrenmeyi başaramadı")

    # Kaydet
    result_path = os.path.join(os.path.dirname(__file__), "phase4_2_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar: {result_path}")


if __name__ == "__main__":
    main()
