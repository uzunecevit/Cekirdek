#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 3.2.1: Decay Sweep

Sabitler: threshold=0.3, amplitude=2.0
Değişken: decay = [0.3, 0.2, 0.1, 0.05]

Hedef: Optimal decay değerini bul.
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_STEPS = 200  # Daha uzun test
LR = 0.0005
N_SAMPLES = 50
DECAYS = [0.3, 0.2, 0.1, 0.05]
THRESHOLD = 0.3
AMPLITUDE = 2.0

# Kill-switch
MAX_W_NORM = 10.0


def load_model(threshold=THRESHOLD, amplitude=AMPLITUDE, decay=0.3):
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


def apply_stdp_with_monitoring(model, x, y, lr, step):
    """STDP + W_fast norm monitoring + kill-switch."""
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

        # W_fast norm check (kill-switch)
        w_norm = model.lm_head.weight.norm().item()
        if w_norm > MAX_W_NORM:
            print(
                f"\n  🚨 KILL-SWITCH: W_norm={w_norm:.2f} > {MAX_W_NORM} at step {step}"
            )
            return loss.item(), True

        return loss.item(), False


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 3.2.1: Decay Sweep")
    print(f"threshold={THRESHOLD}, amplitude={AMPLITUDE}")
    print("=" * 60)

    sequences = load_dataset(load_model()[1])

    print(f"\n{'=' * 70}")
    print(
        f"{'Decay':>6} | {'Init Loss':>10} | {'Final Loss':>10} | {'STDP Δ%':>8} | {'W_norm':>8} | {'Status':>10}"
    )
    print(f"{'=' * 70}")

    results = []
    best_decay = None
    best_stdp = 0

    for decay in DECAYS:
        model, ckpt = load_model(threshold=THRESHOLD, amplitude=AMPLITUDE, decay=decay)
        initial_loss = compute_loss(model, sequences)

        killed = False
        for step in range(N_STEPS):
            x, y = sequences[step % len(sequences)]
            loss, killed = apply_stdp_with_monitoring(model, x, y, LR, step)
            if killed:
                break

        if not killed:
            final_loss = compute_loss(model, sequences)
            stdp_change = (final_loss - initial_loss) / initial_loss * 100
        else:
            final_loss = loss
            stdp_change = (final_loss - initial_loss) / initial_loss * 100

        w_norm = model.lm_head.weight.norm().item()
        status = "💥 KILLED" if killed else "✅ OK"

        print(
            f"{decay:6.2f} | {initial_loss:10.4f} | {final_loss:10.4f} | {stdp_change:+7.1f}% | {w_norm:8.4f} | {status:>10}"
        )

        results.append(
            {
                "decay": decay,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "stdp_change": stdp_change,
                "w_norm": w_norm,
                "killed": killed,
            }
        )

        if not killed and abs(stdp_change) > abs(best_stdp):
            best_stdp = stdp_change
            best_decay = decay

    print(f"\n{'=' * 70}")
    print("SONUÇ")
    print(f"{'=' * 70}")

    if best_decay is not None:
        print(f"  Optimal decay: {best_decay}")
        print(f"  STDP etkisi: {best_stdp:+.1f}%")
    else:
        print("  ⚠️ Tüm testler kill-switch ile durduruldu")

    # Trend analizi
    print(f"\n  Trend:")
    for r in results:
        status = "💥" if r["killed"] else "✅"
        print(f"    decay={r['decay']:.2f}: {r['stdp_change']:+.1f}% {status}")


if __name__ == "__main__":
    main()
