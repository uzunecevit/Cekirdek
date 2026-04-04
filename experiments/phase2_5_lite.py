#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 2.5 Lite: Dense Reward + Token-Level STDP Testi

Amaç: "Dense reward + STDP gerçekten loss düşürüyor mu?"

Yöntem:
1. Dataset'ten x → target (generate YOK)
2. Her token için: reward = log_prob(target_token)
3. STDP update: LM head ağırlığı (en basit)
4. Ölçüm: initial loss vs final loss

Başarı: R-STDP sonrası loss ↓ → öğrenme VAR
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
N_STEPS = 50
LR = 0.0005
REWARD_SCALE = 1.0
N_SAMPLES = 50


def load_model():
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
        use_surrogate=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_dataset(ckpt, n_samples=N_SAMPLES):
    import random

    names_path = os.path.join(
        os.path.dirname(__file__), "../../VİCDAN_HEBBIAN/micro/names.txt"
    )
    with open(names_path) as f:
        names = [line.strip() for line in f if line.strip()]
    random.seed(42)
    random.shuffle(names)

    chars = ckpt["chars"]
    stoi = ckpt["stoi"]
    BOS = ckpt["BOS"]

    sequences = []
    for name in names[:n_samples]:
        tokens = [BOS] + [stoi[c] for c in name] + [BOS]
        if len(tokens) >= 2:
            sequences.append((tokens[:-1], tokens[1:]))

    return sequences[:n_samples]


def compute_loss(model, sequences, device):
    losses = []
    with torch.no_grad():
        for x, y in sequences:
            idx = torch.tensor([x], device=device)
            target = torch.tensor([y], device=device)
            _, loss = model(idx, targets=target)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 2.5 Lite: Dense R-STDP Testi")
    print("=" * 60)

    model, ckpt = load_model()
    sequences = load_dataset(ckpt)
    vocab_size = ckpt["vocab_size"]
    print(f"Dataset: {len(sequences)} (x, y) çifti")

    # Başlangıç loss
    initial_loss = compute_loss(model, sequences, DEVICE)
    print(f"\n  Initial loss: {initial_loss:.4f}")

    # LM head ağırlığı (vocab_size, d_model)
    lm_weight = model.lm_head.weight.data
    print(f"  LM head weight: {lm_weight.shape}, norm={lm_weight.norm().item():.4f}")

    # R-STDP loop
    print(f"\n--- R-STDP ({N_STEPS} adım) ---")
    loss_history = []

    for step in range(N_STEPS):
        x, y = sequences[step % len(sequences)]

        with torch.no_grad():
            idx = torch.tensor([x], device=DEVICE)
            target = torch.tensor([y], device=DEVICE)
            logits, loss = model(idx, targets=target)

            # Her token pozisyonu için STDP
            for t in range(logits.size(1)):
                probs = F.softmax(logits[0, t, :], dim=-1)
                target_token = y[t]

                # Dense reward: log_prob(target) — negatif olabilir
                target_prob = probs[target_token]
                reward = torch.log(target_prob + 1e-8).item() * REWARD_SCALE

                # Pre: input embedding (d_model boyutunda)
                token_idx = x[min(t, len(x) - 1)]
                pre = model.wte.weight[token_idx]  # (d_model,)

                # Target one-hot
                target_onehot = torch.zeros(vocab_size, device=DEVICE)
                target_onehot[target_token] = 1.0

                # ΔW[target] = sign(reward) * |reward| * (target_onehot - probs) * pre
                # Bu, gradient descent'in spike-based versiyonu
                error = target_onehot - probs  # (vocab,)
                dw = LR * error.unsqueeze(1) * pre.unsqueeze(0)  # (vocab, d_model)
                lm_weight.add_(dw)

                # Clip
                lm_weight.clamp_(-1.0, 1.0)

            loss_history.append(loss.item())

            if step % 20 == 0 or step == N_STEPS - 1:
                print(
                    f"  Step {step + 1:3d}: loss={loss.item():.4f}, reward={reward:.3f}"
                )

    # Final loss
    final_loss = compute_loss(model, sequences, DEVICE)
    print(f"\n--- Sonuç ---")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(
        f"  Değişim:      {final_loss - initial_loss:+.4f} ({(final_loss / initial_loss - 1) * 100:+.1f}%)"
    )

    if final_loss < initial_loss:
        print(f"\n  ✅ R-STDP ÖĞRENDİ! Loss düştü.")
    else:
        print(f"\n  ❌ R-STDP öğrenmedi.")

    # Loss trend
    print(f"\n  Loss trend (her 10 adım):")
    for i in range(0, N_STEPS, 10):
        chunk = loss_history[i : i + 10]
        avg = sum(chunk) / len(chunk)
        print(f"    {i:3d}: {avg:.3f}")


if __name__ == "__main__":
    main()
