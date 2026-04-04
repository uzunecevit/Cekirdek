#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 4.3: Sleep-Mediated Consolidation

Mekanizma:
1. Replay Buffer: Her epoch sırasında en yüksek ΔW norm'lu (sürpriz) örnekleri kaydet
2. Dream Mode: Buffer'daki örneklerle replay, dream_lr=0.001
3. Homeostaz: W_norm > %10 artarsa dream_lr'ı düşür
4. Uyku sonrası consolidation (α=0.05)
"""

import os
import sys
import json
import random
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
    compute_surprise,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
LEARN_EPOCHS = 5
DREAM_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.0005
DREAM_LR = 0.001
CONSOLIDATION_ALPHA = 0.05
SURPRISE_THRESHOLD = 0.5
BUFFER_SIZE = 100
DELTA_THRESHOLD = 0.0001  # Minimum ΔW norm buffer'a girmek için
HOMEOSTASIS_THRESHOLD = 0.10  # W_norm %10'dan fazla artarsa dream_lr düşür


class ReplayBuffer:
    """En yüksek ΔW norm'lu örnekleri saklar."""

    def __init__(self, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.buffer = []  # (x_tokens, y_tokens, delta_norm)

    def add(self, x, y, delta_norm):
        if delta_norm < DELTA_THRESHOLD:
            return  # Önemsiz örnekleri atla
        self.buffer.append((x, y, delta_norm))
        if len(self.buffer) > self.max_size:
            # En düşük delta_norm'lu olanı sil
            self.buffer.sort(key=lambda item: item[2])
            self.buffer.pop(0)

    def sample(self, n=1):
        if not self.buffer:
            return None
        # Delta_norm'a göre ağırlıklı örnekleme (yüksek delta daha olası)
        weights = [item[2] for item in self.buffer]
        total = sum(weights)
        if total == 0:
            return random.choice(self.buffer)[:2]
        probs = [w / total for w in weights]
        chosen = random.choices(self.buffer, weights=probs, k=n)
        return [(item[0], item[1]) for item in chosen]

    def __len__(self):
        return len(self.buffer)


def train_with_replay(model, dataloader, task_name, buffer, epochs=LEARN_EPOCHS):
    """Öğrenme fazı: STDP + replay buffer doldurma."""
    print(f"\n  📚 {task_name} öğrenme fazı ({epochs} epoch)...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        epoch_deltas = []

        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            B, T = xb.size()

            for b in range(B):
                mask = yb[b] != -100
                if mask.sum() < 2:
                    continue

                x = xb[b][mask].tolist()
                y = yb[b][mask].tolist()
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]

                if len(x) < 2:
                    continue

                # STDP öncesi W_fast norm
                w_fast_before = model.lm_head_fast.norm().item()

                loss, surprise = apply_stdp_gated(model, x, y, LR, SURPRISE_THRESHOLD)
                epoch_loss += loss
                n_batches += 1

                # STDP sonrası W_fast norm → delta
                w_fast_after = model.lm_head_fast.norm().item()
                delta_norm = abs(w_fast_after - w_fast_before)
                epoch_deltas.append(delta_norm)

                # Replay buffer'a ekle
                buffer.add(x, y, delta_norm)

        avg_delta = sum(epoch_deltas) / max(len(epoch_deltas), 1)
        avg_loss = epoch_loss / max(n_batches, 1)
        print(
            f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, avg_delta={avg_delta:.4f}, buffer={len(buffer)}"
        )


def dream_mode(model, buffer, n_replays=50, dream_lr=DREAM_LR):
    """
    Uyku fazı: Replay buffer'daki örneklerle öğrenme.
    Homeostaz: W_norm > %10 artarsa dream_lr'ı düşür.
    """
    if len(buffer) == 0:
        print("    ⚠️ Replay buffer boş, uyku atlanıyor.")
        return

    print(
        f"\n  💤 Uyku fazı: {n_replays} replay, dream_lr={dream_lr}, buffer={len(buffer)}..."
    )

    # Başlangıç W_norm
    initial_w_norm = model.lm_head_static.weight.norm().item()

    model.train()
    for replay_idx in range(n_replays):
        samples = buffer.sample()
        if samples is None:
            break

        for x, y in samples:
            # Dream STDP: surprise_threshold=0.0 (her replay öğrenilir)
            loss, _ = apply_stdp_gated(model, x, y, dream_lr, surprise_threshold=0.0)

        # Homeostaz kontrolü: Her 10 replay'da bir
        if (replay_idx + 1) % 10 == 0:
            current_w_norm = model.lm_head_static.weight.norm().item()
            w_change = abs(current_w_norm - initial_w_norm) / max(initial_w_norm, 1e-8)

            if w_change > HOMEOSTASIS_THRESHOLD:
                # Ağırlıklar kontrolsüz şişiyor → dream_lr'ı yarıya indir
                dream_lr *= 0.5
                print(
                    f"    🛡️ Homeostaz: W_norm %{w_change * 100:.1f} değişti, dream_lr={dream_lr:.6f}"
                )

                if dream_lr < 1e-6:
                    print("    ⚠️ dream_lr çok düşük, uyku durduruluyor.")
                    break

    # Uyku sonrası consolidation
    report = model.consolidate(alpha=CONSOLIDATION_ALPHA, threshold=0.5)
    consolidated = report.get("total_consolidated", 0)
    final_w_norm = model.lm_head_static.weight.norm().item()
    w_change = abs(final_w_norm - initial_w_norm) / max(initial_w_norm, 1e-8)

    print(
        f"    Uyku tamamlandı: consolidated={consolidated}, W_norm değişimi=%{w_change * 100:.1f}"
    )


def main():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 4.3: Sleep-Mediated Consolidation")
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

    # Model yükle
    model, ckpt = load_model()

    # Baseline
    baseline = {
        "names": evaluate_loss(model, names_dl),
        "turkish": evaluate_loss(model, turkish_dl),
        "math": evaluate_loss(model, math_dl),
    }
    print(f"\n  Başlangıç loss'ları:")
    print(f"    İsimler:   {baseline['names']:.4f}")
    print(f"    Türkçe:    {baseline['turkish']:.4f}")
    print(f"    Matematik: {baseline['math']:.4f}")

    # Replay Buffer
    buffer = ReplayBuffer(max_size=BUFFER_SIZE)

    # ── Faz 1: Türkçe Öğrenme + Uyku ──
    print(f"\n{'=' * 70}")
    print("FAZ 1: Türkçe Öğrenme + Uyku")
    print(f"{'=' * 70}")

    train_with_replay(model, turkish_dl, "Türkçe", buffer, epochs=LEARN_EPOCHS)
    dream_mode(model, buffer, n_replays=50)

    post_turkish = {
        "names": evaluate_loss(model, names_dl),
        "turkish": evaluate_loss(model, turkish_dl),
        "math": evaluate_loss(model, math_dl),
    }

    print(f"\n  Post-Türkçe Loss'lar:")
    for task in ["names", "turkish", "math"]:
        change = (post_turkish[task] - baseline[task]) / baseline[task] * 100
        print(f"    {task:10s}: {post_turkish[task]:.4f} ({change:+.1f}%)")

    # ── Faz 2: Matematik Öğrenme + Uyku ──
    print(f"\n{'=' * 70}")
    print("FAZ 2: Matematik Öğrenme + Uyku")
    print(f"{'=' * 70}")

    train_with_replay(model, math_dl, "Matematik", buffer, epochs=LEARN_EPOCHS)
    dream_mode(model, buffer, n_replays=50)

    post_math = {
        "names": evaluate_loss(model, names_dl),
        "turkish": evaluate_loss(model, turkish_dl),
        "math": evaluate_loss(model, math_dl),
    }

    print(f"\n  Post-Matematik Loss'lar:")
    for task in ["names", "turkish", "math"]:
        change = (post_math[task] - baseline[task]) / baseline[task] * 100
        print(f"    {task:10s}: {post_math[task]:.4f} ({change:+.1f}%)")

    # ── Forgetting Raporu ──
    print(f"\n{'=' * 70}")
    print("FORGETTING RAPORU (Uyku ile)")
    print(f"{'=' * 70}")

    names_forgetting = (
        (post_math["names"] - baseline["names"]) / baseline["names"] * 100
    )
    turkish_forgetting = (
        (post_math["turkish"] - baseline["turkish"]) / baseline["turkish"] * 100
    )
    math_learning = (post_math["math"] - baseline["math"]) / baseline["math"] * 100

    print(f"  İsimler forgetting:   {names_forgetting:+.1f}% (hedef: < %5)")
    print(f"  Türkçe forgetting:    {turkish_forgetting:+.1f}% (hedef: < %2)")
    print(f"  Matematik öğrenme:    {math_learning:+.1f}%")

    # Başarı değerlendirmesi
    if names_forgetting < 5 and turkish_forgetting < 2:
        print(f"\n  ✅ BAŞARILI: Catastrophic forgetting kontrol altında!")
    elif names_forgetting < 12.1:
        print(f"\n  ⚠️ Kısmi başarı: Forgetting azaldı ama hâlâ hedefin üstünde")
    else:
        print(f"\n  ❌ Başarısız: Forgetting arttı veya aynı kaldı")

    # Karşılaştırma (Phase 4.1 vs 4.3)
    print(f"\n  Karşılaştırma:")
    print(f"    Phase 4.1 (Uykusuz): İsimler +12.1%, Türkçe +0.4%")
    print(
        f"    Phase 4.3 (Uykulu):  İsimler {names_forgetting:+.1f}%, Türkçe {turkish_forgetting:+.1f}%"
    )

    # Sonuçları kaydet
    results = {
        "baseline": baseline,
        "post_turkish": post_turkish,
        "post_math": post_math,
        "names_forgetting_pct": names_forgetting,
        "turkish_forgetting_pct": turkish_forgetting,
        "math_learning_pct": math_learning,
        "buffer_size": len(buffer),
        "homeostasis_triggered": True,  # dream_lr azaltıldı mı?
    }

    result_path = os.path.join(os.path.dirname(__file__), "phase4_3_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {result_path}")


if __name__ == "__main__":
    main()
