#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 3.2.3: Context Gating (Sürpriz Odaklı Öğrenme)

Mekanizma: Sadece yüksek entropi (sürpriz) anlarında STDP uygula.
Düşük entropi = model zaten biliyor = güncelleme yapma (gürültü önleme)
Yüksek entropi = model şaşırdı = öğrenme fırsatı = STDP uygula

Konfigürasyon: threshold=0.3, amplitude=2.0, decay=0.1
LR: 0.0005 (seyrek update sayesinde daha yüksek LR mümkün)
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
LR = 0.0005  # Seyrek update ile daha yüksek LR
N_SAMPLES = 50
THRESHOLD = 0.3
AMPLITUDE = 2.0
DECAY = 0.1

SURPRISE_THRESHOLDS = [0.3, 0.5, 0.7]

# Kill-switch
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
    """
    Entropi bazlı sürpriz ölçümü.
    surprise = H(probs) / H_max
    H_max = log(vocab_size)
    """
    eps = 1e-9
    H = -(probs * torch.log(probs + eps)).sum()
    H_max = math.log(probs.size(-1))
    return (H / H_max).item()


def apply_stdp_gated(model, x, y, lr, surprise_threshold):
    """
    Context Gating ile STDP.
    Sadece surprise > threshold ise update uygula.
    """
    with torch.no_grad():
        idx = torch.tensor([x], device=DEVICE)
        target = torch.tensor([y], device=DEVICE)
        logits, loss = model(idx, targets=target)
        vocab_size = logits.size(-1)

        # Son token'ın entropisini hesapla (sürpriz ölçümü)
        last_probs = F.softmax(logits[0, -1, :], dim=-1)
        surprise = compute_surprise(last_probs)

        updates_applied = 0

        if surprise > surprise_threshold:
            # Sürpriz var → STDP uygula (fast weight'e)
            for t in range(logits.size(1)):
                probs = F.softmax(logits[0, t, :], dim=-1)
                target_token = y[t]
                target_onehot = torch.zeros(vocab_size, device=DEVICE)
                target_onehot[target_token] = 1.0
                token_idx = x[min(t, len(x) - 1)]
                pre = model.wte.weight[token_idx]
                error = target_onehot - probs
                dw = lr * error.unsqueeze(1) * pre.unsqueeze(0)
                model.lm_head_fast.add_(dw)
                model.lm_head_fast.clamp_(-1.0, 1.0)
            updates_applied = 1

        return loss.item(), surprise, updates_applied


def test_with_threshold(surprise_threshold, sequences, initial_loss):
    """Belirli bir surprise threshold ile 1000 iterasyon test."""
    model, _ = load_model()

    killed = False
    kill_reason = None
    total_updates = 0
    total_surprise = 0

    for step in range(1, N_STEPS + 1):
        x, y = sequences[step % len(sequences)]
        loss, surprise, updates = apply_stdp_gated(model, x, y, LR, surprise_threshold)

        total_updates += updates
        total_surprise += surprise

        # Kill-switch
        w_norm = model.lm_head.weight.norm().item()
        if w_norm > MAX_W_NORM:
            killed = True
            kill_reason = f"W_norm {w_norm:.2f} > {MAX_W_NORM}"
            break
        if loss > MAX_LOSS:
            killed = True
            kill_reason = f"Loss {loss:.4f} > {MAX_LOSS}"
            break

    if not killed:
        final_loss = compute_loss(model, sequences)
        stdp_change = (final_loss - initial_loss) / initial_loss * 100
    else:
        final_loss = loss
        stdp_change = (final_loss - initial_loss) / initial_loss * 100

    w_norm = model.lm_head.weight.norm().item()
    update_rate = total_updates / N_STEPS * 100
    avg_surprise = total_surprise / N_STEPS

    return {
        "threshold": surprise_threshold,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "stdp_change": stdp_change,
        "w_norm": w_norm,
        "update_rate": update_rate,
        "avg_surprise": avg_surprise,
        "killed": killed,
        "kill_reason": kill_reason,
    }


def main():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 3.2.3: Context Gating (Sürpriz Odaklı Öğrenme)")
    print(
        f"Config: threshold={THRESHOLD}, amplitude={AMPLITUDE}, decay={DECAY}, LR={LR}"
    )
    print(f"Kill-switch: W_norm > {MAX_W_NORM} veya loss > {MAX_LOSS}")
    print("=" * 70)

    sequences = load_dataset(load_model()[1])
    initial_loss = compute_loss(load_model()[0], sequences)

    print(f"\n  Başlangıç loss: {initial_loss:.4f}")
    print(f"\n{'=' * 80}")
    print(
        f"{'Thr':>5} | {'Init':>8} | {'Final':>8} | {'Δ%':>7} | {'W_norm':>8} | {'Upd%':>6} | {'AvgSurp':>8} | {'Status':>8}"
    )
    print(f"{'=' * 80}")

    results = []
    best_thr = None
    best_stdp = 0

    for thr in SURPRISE_THRESHOLDS:
        result = test_with_threshold(thr, sequences, initial_loss)
        status = "💥 KILLED" if result["killed"] else "✅ OK"

        print(
            f"{result['threshold']:5.1f} | {result['initial_loss']:8.4f} | {result['final_loss']:8.4f} | {result['stdp_change']:+6.1f}% | {result['w_norm']:8.4f} | {result['update_rate']:5.1f}% | {result['avg_surprise']:7.3f} | {status:>8}"
        )

        results.append(result)

        if not result["killed"] and result["stdp_change"] < best_stdp:
            best_stdp = result["stdp_change"]
            best_thr = thr

    print(f"\n{'=' * 80}")
    print("SONUÇ")
    print(f"{'=' * 80}")

    if best_thr is not None:
        best = next(r for r in results if r["threshold"] == best_thr)
        print(f"  Optimal surprise threshold: {best_thr}")
        print(f"  STDP etkisi: {best['stdp_change']:+.1f}%")
        print(
            f"  Update oranı: {best['update_rate']:.1f}% (tasarruf: {100 - best['update_rate']:.1f}%)"
        )
        print(f"  W_norm: {best['w_norm']:.4f}")

        if best["stdp_change"] < -2.0:
            print(f"\n  ✅ BAŞARILI: Context Gating ile güçlü öğrenme + stabilite!")
        elif best["stdp_change"] < 0:
            print(f"\n  ⚠️ Pozitif etki var ama zayıf")
        else:
            print(f"\n  ❌ Öğrenme yok (loss artıyor)")
    else:
        print("  ⚠️ Tüm testler kill-switch ile durduruldu")

    # Karşılaştırma
    print(f"\n  Karşılaştırma:")
    print(f"    Binary (LR=0.0005): -0.5%")
    print(f"    Ternary amp=2.0:    -1.9%")
    print(f"    Decay=0.1:          -4.0% (200 iter)")
    for r in results:
        if not r["killed"]:
            print(
                f"    Gating {r['threshold']:.1f}:       {r['stdp_change']:+.1f}% ({r['update_rate']:.0f}% update)"
            )


if __name__ == "__main__":
    main()
