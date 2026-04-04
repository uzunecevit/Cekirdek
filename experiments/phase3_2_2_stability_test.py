#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 3.2.2: Long-Sequence Stability (1000 İterasyon Stres Testi)

Konfigürasyon: threshold=0.3, amplitude=2.0, decay=0.1
Hedef: 1000 iterasyonda W_norm ve loss stabilitesini doğrula.
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_STEPS = 1000
LR = 0.0001  # Daha düşük LR (0.0005 → 0.0001)
N_SAMPLES = 50
THRESHOLD = 0.3
AMPLITUDE = 2.0
DECAY = 0.1

# Kill-switch
MAX_W_NORM = 5.0
MAX_LOSS = 5.0  # Daha toleranslı


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
    print("VİCDAN_SPIKE — Phase 3.2.2: 1000 İterasyon Stres Testi")
    print(f"Config: threshold={THRESHOLD}, amplitude={AMPLITUDE}, decay={DECAY}")
    print(f"Kill-switch: W_norm > {MAX_W_NORM} veya loss > {MAX_LOSS}")
    print("=" * 60)

    model, ckpt = load_model()
    sequences = load_dataset(ckpt)
    initial_loss = compute_loss(model, sequences)

    print(f"\n  Başlangıç loss: {initial_loss:.4f}")
    print(f"\n{'=' * 60}")
    print(f"{'Iter':>6} | {'Loss':>10} | {'Δ%':>8} | {'W_norm':>8} | {'Status':>10}")
    print(f"{'=' * 60}")

    history = []
    killed = False
    kill_reason = None

    for step in range(1, N_STEPS + 1):
        x, y = sequences[step % len(sequences)]
        loss = apply_stdp(model, x, y, LR)

        w_norm = model.lm_head.weight.norm().item()

        # Kill-switch checks
        if w_norm > MAX_W_NORM:
            killed = True
            kill_reason = f"W_norm {w_norm:.2f} > {MAX_W_NORM}"
            break

        if loss > MAX_LOSS:
            killed = True
            kill_reason = f"Loss {loss:.4f} > {MAX_LOSS}"
            break

        # Raporlama her 100 adımda
        if step % 100 == 0:
            stdp_change = (loss - initial_loss) / initial_loss * 100
            print(
                f"{step:6d} | {loss:10.4f} | {stdp_change:+7.1f}% | {w_norm:8.4f} | {'✅':>10}"
            )
            history.append(
                {
                    "step": step,
                    "loss": loss,
                    "stdp_change": stdp_change,
                    "w_norm": w_norm,
                }
            )

    # Final rapor
    print(f"\n{'=' * 60}")
    print("SONUÇ")
    print(f"{'=' * 60}")

    if killed:
        print(f"  🚨 KILL-SWITCH: {kill_reason}")
        print(f"  Durduğu iterasyon: {step}")
    else:
        print(f"  ✅ 1000 iterasyon tamamlandı — sistem stabil!")

    # Trend analizi
    print(f"\n  Loss Trendi:")
    for h in history:
        bar = "█" * int(abs(h["stdp_change"]) * 5)
        print(f"    {h['step']:4d}: {h['loss']:.4f} ({h['stdp_change']:+.1f}%) {bar}")

    if history:
        first = history[0]
        last = history[-1]
        print(f"\n  İlk 100: {first['loss']:.4f} ({first['stdp_change']:+.1f}%)")
        print(f"  Son 100:  {last['loss']:.4f} ({last['stdp_change']:+.1f}%)")
        print(f"  W_norm:   {last['w_norm']:.4f}")

        if not killed and last["stdp_change"] < -2.0:
            print(
                f"\n  ✅ STABİL ÖĞRENME: {abs(last['stdp_change']):.1f}% düşüş, W_norm stabil"
            )
        elif not killed:
            print(f"\n  ⚠️ Stabil ama öğrenme zayıf: {last['stdp_change']:+.1f}%")


if __name__ == "__main__":
    main()
