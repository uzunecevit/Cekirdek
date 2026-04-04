#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 3.1.2: Amplitude Sweep (Aşama 2)

Sabitler: threshold=0.3, decay=0.3
Değişken: amplitude = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

Hedef: Düşük spike rate'i yüksek amplitude ile kompanse et.
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_STEPS = 100
LR = 0.0005
N_SAMPLES = 50
AMPLITUDES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
THRESHOLD = 0.3  # Phase 3.1.1'den optimal


def load_model(threshold=THRESHOLD, amplitude=1.0):
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
        decay=0.3,
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


def apply_stdp(model, x, y, lr):
    with torch.no_grad():
        idx = torch.tensor([x], device=DEVICE)
        target = torch.tensor([y], device=DEVICE)
        logits, loss = model(idx, targets=target)
        vocab_size = logits.size(-1)
        for t in range(logits.size(1)):
            probs = F.softmax(logits[0, t, :], dim=-1)
            target_token = y[t]
            target_onehot = torch.zeros(vocab_size, device=DEVICE)
            target_onehot[target_token] = 1.0
            token_idx = x[min(t, len(x) - 1)]
            pre = model.wte.weight[token_idx]
            error = target_onehot - probs
            dw = lr * error.unsqueeze(1) * pre.unsqueeze(0)
            model.lm_head.weight.add_(dw)
            model.lm_head.weight.clamp_(-1.0, 1.0)
        return loss.item()


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 3.1.2: Amplitude Sweep")
    print(f"Threshold: {THRESHOLD} (sabit)")
    print("=" * 60)

    sequences = load_dataset(load_model()[1])

    print(f"\n{'=' * 60}")
    print(
        f"{'AMP':>4} | {'Init Loss':>10} | {'Final Loss':>10} | {'STDP Δ%':>8} | {'W_norm':>8}"
    )
    print(f"{'=' * 60}")

    results = []
    best_amp = None
    best_stdp = 0

    for amp in AMPLITUDES:
        model, ckpt = load_model(threshold=THRESHOLD, amplitude=amp)
        initial_loss = compute_loss(model, sequences)

        # STDP uygula
        for step in range(N_STEPS):
            x, y = sequences[step % len(sequences)]
            apply_stdp(model, x, y, LR)

        final_loss = compute_loss(model, sequences)
        stdp_change = (final_loss - initial_loss) / initial_loss * 100
        w_norm = model.lm_head.weight.norm().item()

        print(
            f"{amp:4.1f} | {initial_loss:10.4f} | {final_loss:10.4f} | {stdp_change:+7.1f}% | {w_norm:8.4f}"
        )

        results.append(
            {
                "amplitude": amp,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "stdp_change": stdp_change,
                "w_norm": w_norm,
            }
        )

        if abs(stdp_change) > abs(best_stdp):
            best_stdp = stdp_change
            best_amp = amp

    print(f"\n{'=' * 60}")
    print("SONUÇ")
    print(f"{'=' * 60}")
    print(f"  En iyi amplitude: {best_amp}")
    print(f"  STDP etkisi: {best_stdp:+.1f}%")

    # Binary baseline ile karşılaştır
    print(f"\n  Karşılaştırma:")
    print(f"    Binary (amp=1.0): -0.5%")
    print(f"    Ternary (amp=1.0): {results[1]['stdp_change']:+.1f}%")
    print(f"    Ternary (amp={best_amp}): {best_stdp:+.1f}%")

    if abs(best_stdp) > 1.7:
        print(
            f"\n  ✅ Amplitude tuning başarılı! Binary'den {abs(best_stdp) / 0.5:.1f}× daha etkili"
        )
    else:
        print(f"\n  ⚠️ Amplitude etkisi sınırlı")


if __name__ == "__main__":
    main()
