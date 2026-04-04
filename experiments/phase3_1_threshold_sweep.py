#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 3.1.1: Threshold Sweep (Aşama 1)

Hedef: Spike rate'i %80'den %20-30'a çek.
Sabitler: amplitude=1.0, decay=0.3
Değişken: threshold = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
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
THRESHOLDS = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]


def load_model(threshold=0.3, amplitude=1.0):
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


def measure_spike_rate(model, sequences):
    """
    Tek bir uzun sequence ile spike rate ölç (block_size içinde).
    """
    # İlk sequence'i kullan, block_size'a kısalt
    x, y = sequences[0]
    seq_len = min(len(x), model.block_size)
    idx = torch.tensor([x[:seq_len]], device=DEVICE)
    target = torch.tensor([y[:seq_len]], device=DEVICE)

    with torch.no_grad():
        model(idx, targets=target)

    # Stats topla
    total_spikes = 0
    total_steps = 0
    total_pos = 0
    total_neg = 0

    for name, module in model.named_modules():
        if hasattr(module, "total_steps") and module.total_steps > 0:
            total_spikes += module.spike_count
            total_steps += module.total_steps
            total_pos += module.pos_count
            total_neg += module.neg_count

    rate = total_spikes / max(total_steps, 1)
    balance = total_pos / max(total_neg, 1)
    return rate, balance


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 3.1.1: Threshold Sweep")
    print("=" * 60)

    sequences = load_dataset(load_model()[1])

    print(f"\n{'=' * 60}")
    print(
        f"{'THR':>4} | {'Init Loss':>10} | {'Final Loss':>10} | {'STDP Δ%':>8} | {'Spike%':>7} | {'Balance':>8}"
    )
    print(f"{'=' * 60}")

    results = []

    for thr in THRESHOLDS:
        model, ckpt = load_model(threshold=thr, amplitude=1.0)
        initial_loss = compute_loss(model, sequences)

        # Spike rate ölç
        spike_rate, balance = measure_spike_rate(model, sequences)

        # STDP uygula
        for step in range(N_STEPS):
            x, y = sequences[step % len(sequences)]
            apply_stdp(model, x, y, LR)

        final_loss = compute_loss(model, sequences)
        stdp_change = (final_loss - initial_loss) / initial_loss * 100

        # Dead neuron check
        dead = 0
        for name, module in model.named_modules():
            if hasattr(module, "total_steps") and module.total_steps > 0:
                if module.spike_rate == 0:
                    dead += 1

        print(
            f"{thr:4.1f} | {initial_loss:10.4f} | {final_loss:10.4f} | {stdp_change:+7.1f}% | {spike_rate:6.1f}% | {balance:7.2f} | dead={dead}"
        )

        results.append(
            {
                "threshold": thr,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "stdp_change": stdp_change,
                "spike_rate": spike_rate,
                "balance": balance,
                "dead_neurons": dead,
            }
        )

        # Safety check
        if initial_loss > 4.0:
            print(f"\n  ⚠️ Loss {initial_loss:.2f} > 4.0 — durduruluyor.")
            break

    # Optimal threshold seçimi
    print(f"\n{'=' * 60}")
    print("OPTIMAL THRESHOLD ANALİZİ")
    print(f"{'=' * 60}")

    best = None
    for r in results:
        if 15 <= r["spike_rate"] <= 35:
            if best is None or abs(r["stdp_change"]) > abs(best["stdp_change"]):
                best = r

    if best:
        print(f"  Optimal threshold: {best['threshold']:.1f}")
        print(f"  Spike rate: {best['spike_rate']:.1f}%")
        print(f"  STDP etkisi: {best['stdp_change']:+.1f}%")
        print(f"  Balance: {best['balance']:.2f}")
        print(
            f"\n  → Aşama 2'ye geç: amplitude sweep (threshold={best['threshold']:.1f} sabit)"
        )
    else:
        print("  ⚠️ Hedef aralıkta (%15-35) sonuç yok.")
        if results:
            closest = min(results, key=lambda r: abs(r["spike_rate"] - 25))
            print(
                f"  → En yakın: threshold={closest['threshold']:.1f}, rate={closest['spike_rate']:.1f}%"
            )
            print(f"  → Aşama 2'ye geç: amplitude sweep ile rate'i artır")


if __name__ == "__main__":
    main()
