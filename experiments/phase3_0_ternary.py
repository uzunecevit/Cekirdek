#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 3.0: Ternary Spike vs Binary Karşılaştırması

Test:
1. Binary {0,1} vs Ternary {-1,0,+1}
2. Spike distribution
3. STDP etki karşılaştırması
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


def load_model(ternary=True, amplitude=1.0):
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
        threshold=0.3,
        decay=0.3,
        amplitude=amplitude,
        use_surrogate=False,
        ternary=ternary,
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


def get_spike_stats(model):
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, "total_steps") and module.total_steps > 0:
            stats[name] = {
                "rate": module.spike_rate,
                "balance": module.balance_ratio
                if hasattr(module, "balance_ratio")
                else None,
            }
    return stats


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 3.0: Ternary vs Binary")
    print("=" * 60)

    sequences = load_dataset(load_model()[1])

    # Binary model
    model_bin, ckpt = load_model(ternary=False)
    initial_loss_bin = compute_loss(model_bin, sequences)
    stats_bin = get_spike_stats(model_bin)
    print(f"\n--- Binary {0, 1} ---")
    print(f"  Initial loss: {initial_loss_bin:.4f}")
    print(
        f"  Spike rate (input_lif): {stats_bin.get('input_lif', {}).get('rate', 0):.2%}"
    )

    # Ternary model
    model_ter, _ = load_model(ternary=True, amplitude=1.0)
    initial_loss_ter = compute_loss(model_ter, sequences)
    stats_ter = get_spike_stats(model_ter)
    print(f"\n--- Ternary {-1, 0, +1} ---")
    print(f"  Initial loss: {initial_loss_ter:.4f}")
    print(
        f"  Spike rate (input_lif): {stats_ter.get('input_lif', {}).get('rate', 0):.2%}"
    )
    if stats_ter.get("input_lif", {}).get("balance"):
        print(f"  Balance ratio (input_lif): {stats_ter['input_lif']['balance']:.2f}")

    # STDP test - Binary
    print(f"\n--- STDP: Binary ---")
    for step in range(N_STEPS):
        x, y = sequences[step % len(sequences)]
        apply_stdp(model_bin, x, y, LR)
    final_loss_bin = compute_loss(model_bin, sequences)
    print(
        f"  Final loss: {final_loss_bin:.4f} ({(final_loss_bin - initial_loss_bin) / initial_loss_bin * 100:+.1f}%)"
    )

    # STDP test - Ternary
    print(f"\n--- STDP: Ternary ---")
    for step in range(N_STEPS):
        x, y = sequences[step % len(sequences)]
        apply_stdp(model_ter, x, y, LR)
    final_loss_ter = compute_loss(model_ter, sequences)
    print(
        f"  Final loss: {final_loss_ter:.4f} ({(final_loss_ter - initial_loss_ter) / initial_loss_ter * 100:+.1f}%)"
    )

    # W_fast norm karşılaştırması
    print(f"\n--- LM Head Weight Norm ---")
    print(f"  Binary: {model_bin.lm_head.weight.norm().item():.4f}")
    print(f"  Ternary: {model_ter.lm_head.weight.norm().item():.4f}")

    # Karar
    print(f"\n{'=' * 60}")
    print("SONUÇ")
    print(f"{'=' * 60}")
    bin_change = (final_loss_bin - initial_loss_bin) / initial_loss_bin * 100
    ter_change = (final_loss_ter - initial_loss_ter) / initial_loss_ter * 100
    print(f"  Binary STDP: {bin_change:+.1f}%")
    print(f"  Ternary STDP: {ter_change:+.1f}%")

    if abs(ter_change) > abs(bin_change):
        print(f"\n  ✅ Ternary daha etkili (daha fazla değişim)")
    else:
        print(f"\n  ⚠️ Ternary daha az etkili")


if __name__ == "__main__":
    main()
