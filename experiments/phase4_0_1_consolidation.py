#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 4.0.1: Episodic Memory Leak Testi

Mekanizma: Kısa vadeli öğrenmeyi (W_fast) kalıcı hafızaya (W_static) sızdır.
Formül: W_static = (1-α)·W_static + α·W_fast

Test: 1000 iterasyon öğren → consolidate → val loss karşılaştır
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_STEPS = 1000
LR = 0.0005
N_SAMPLES = 50
THRESHOLD = 0.3
AMPLITUDE = 2.0
DECAY = 0.1
SURPRISE_THRESHOLD = 0.5

# Consolidation parametreleri
CONSOLIDATION_INTERVAL = 200  # Her 200 adımda bir consolidate
CONSOLIDATION_ALPHA = 0.1  # 0.05 → 0.1 (Maksimum Basınç)

MAX_W_NORM = 5.0
MAX_LOSS = 5.0


def load_model(threshold=THRESHOLD, amplitude=AMPLITUDE, decay=DECAY):
    path = os.path.join(CKPT_DIR, "spiking_lm.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    config = ckpt["config"]
    model = SpikingLM(
        vocab_size=ckpt["vocab_size"],
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        d_ff=config["d_ff"],
        block_size=config["block_size"],
        threshold=threshold,
        decay=decay,
        amplitude=amplitude,
        use_surrogate=False,
        ternary=True,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, ckpt


def load_dataset(ckpt):
    import random

    names_path = os.path.join(
        os.path.dirname(__file__), "../../VİCDAN_HEBBIAN/micro/names.txt"
    )
    with open(names_path) as f:
        names = [line.strip() for line in f if line.strip()]
    random.seed(42)
    random.shuffle(names)
    stoi = ckpt["stoi"]
    BOS = ckpt["BOS"]
    sequences = []
    for name in names[:N_SAMPLES]:
        tokens = [BOS] + [stoi[c] for c in name] + [BOS]
        if len(tokens) >= 2:
            sequences.append((tokens[:-1], tokens[1:]))
    return sequences[:N_SAMPLES]


def compute_loss(model, sequences):
    losses = []
    with torch.no_grad():
        for x, y in sequences:
            idx = torch.tensor([x], device=DEVICE)
            target = torch.tensor([y], device=DEVICE)
            _, loss = model(idx, targets=target)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def compute_surprise(probs):
    eps = 1e-9
    H = -(probs * torch.log(probs + eps)).sum()
    H_max = math.log(probs.size(-1))
    return (H / H_max).item()


def apply_stdp_gated(model, x, y, lr, surprise_threshold):
    with torch.no_grad():
        idx = torch.tensor([x], device=DEVICE)
        target = torch.tensor([y], device=DEVICE)
        logits, loss = model(idx, targets=target)
        vocab_size = logits.size(-1)

        last_probs = F.softmax(logits[0, -1, :], dim=-1)
        surprise = compute_surprise(last_probs)

        if surprise > surprise_threshold:
            for t in range(logits.size(1)):
                probs = F.softmax(logits[0, t, :], dim=-1)
                target_token = y[t]
                target_onehot = torch.zeros(vocab_size, device=DEVICE)
                target_onehot[target_token] = 1.0
                token_idx = x[min(t, len(x) - 1)]
                pre = model.wte.weight[token_idx]
                error = target_onehot - probs
                dw = lr * error.unsqueeze(1) * pre.unsqueeze(0)

                # LM Head fast weight update
                model.lm_head_fast.add_(dw)
                model.lm_head_fast.clamp_(-1.0, 1.0)

                # FFN fc2 fast weight update (her block için)
                for block in model.blocks:
                    fc2 = block.ffn.fc2
                    if fc2.fast_weight:
                        # fc2'nin input'u fc1 çıkışı, ama biz embedding'i kullanalım
                        # Basit: aynı dw'yi fc2 fast weight'e uygula (shape uyuşmuyor, skip)
                        pass

        return loss.item(), surprise


def main():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 4.0.1: Episodic Memory Leak")
    print(f"Config: threshold={THRESHOLD}, amplitude={AMPLITUDE}, decay={DECAY}")
    print(
        f"Consolidation: every {CONSOLIDATION_INTERVAL} steps, alpha={CONSOLIDATION_ALPHA}"
    )
    print("=" * 70)

    model, ckpt = load_model()
    sequences = load_dataset(ckpt)

    # Başlangıç loss (öğrenme öncesi)
    initial_loss = compute_loss(model, sequences)
    print(f"\n  Başlangıç loss: {initial_loss:.4f}")
    print(f"  W_static norm: {model.lm_head_static.weight.norm().item():.4f}")
    print(f"  W_fast norm: {model.lm_head_fast.norm().item():.4f}")

    # Öğrenme + consolidation döngüsü
    consolidation_reports = []

    for step in range(1, N_STEPS + 1):
        x, y = sequences[step % len(sequences)]
        loss, surprise = apply_stdp_gated(model, x, y, LR, SURPRISE_THRESHOLD)

        # Kill-switch
        w_norm = model.lm_head_fast.norm().item()
        if w_norm > MAX_W_NORM:
            print(f"\n  🚨 KILL-SWITCH: W_fast_norm={w_norm:.2f} > {MAX_W_NORM}")
            break
        if loss > MAX_LOSS:
            print(f"\n  🚨 KILL-SWITCH: Loss={loss:.4f} > {MAX_LOSS}")
            break

        # Consolidation
        if step % CONSOLIDATION_INTERVAL == 0:
            reports = model.consolidate(alpha=CONSOLIDATION_ALPHA, threshold=0.5)
            consolidation_reports.append(reports)

            current_loss = compute_loss(model, sequences)
            print(
                f"  Step {step:4d}: loss={current_loss:.4f}, W_fast={w_norm:.4f}, "
                f"consolidated={reports.get('total_consolidated', 0)}"
            )

    # Final loss
    final_loss = compute_loss(model, sequences)
    change = (final_loss - initial_loss) / initial_loss * 100

    print(f"\n{'=' * 70}")
    print("SONUÇ")
    print(f"{'=' * 70}")
    print(f"  Başlangıç loss: {initial_loss:.4f}")
    print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Değişim:        {change:+.1f}%")
    print(f"  W_static norm:  {model.lm_head_static.weight.norm().item():.4f}")
    print(f"  W_fast norm:    {model.lm_head_fast.norm().item():.4f}")
    print(f"  Consolidations: {model.consolidation_count}")

    # Consolidation raporları
    print(f"\n  Consolidation Raporları:")
    for i, r in enumerate(consolidation_reports):
        print(f"    #{i + 1}: consolidated={reports[i].get('total_consolidated', 0)}")

    # Değerlendirme
    if change < -5.0:
        print(
            f"\n  ✅ BAŞARILI: Öğrenme kalıcı hale geldi (loss {abs(change):.1f}% düştü)"
        )
    elif change < 0:
        print(f"\n  ⚠️ Pozitif etki var ama zayıf (loss {abs(change):.1f}% düştü)")
    else:
        print(f"\n  ❌ Öğrenme kalıcı değil (loss {change:.1f}% arttı)")


if __name__ == "__main__":
    main()
