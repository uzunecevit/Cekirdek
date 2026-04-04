#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 2: R-STDP Online Öğrenme (Basitleştirilmiş)

Doğrudan LM head ağırlığını güncelle:
ΔW[target_token] = reward × (pre_spikes - mean_pre)
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM
from src.stdp import shaped_reward

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SUFFIX = "lyn"
N_ITERATIONS = 100
N_GENERATE = 3

# Agresif learning rate
LR = 0.1


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


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 2: R-STDP (Basitleştirilmiş)")
    print(f"Hedef: '-{TARGET_SUFFIX}' ile biten isimler")
    print("=" * 60)

    model, ckpt = load_model()
    chars = ckpt["chars"]
    itos = {i: c for c, i in ckpt["stoi"].items()}
    stoi = ckpt["stoi"]
    BOS = ckpt["BOS"]

    # Hedef token indeksleri
    target_indices = [stoi.get(c, 0) for c in TARGET_SUFFIX]

    # Metrikler
    reward_history = []
    match_history = []

    print(f"\n{'=' * 60}")
    print("BAŞLANGIÇ")
    print(f"{'=' * 60}")

    with torch.no_grad():
        initial_matches = 0
        for _ in range(20):
            idx = torch.tensor([[BOS]], device=DEVICE)
            out = model.generate(idx, max_new_tokens=12, temperature=0.8)
            name = "".join(
                itos[t.item()]
                for t in out[0, 1:]
                if t.item() != BOS and t.item() < len(chars)
            )
            if name.endswith(TARGET_SUFFIX):
                initial_matches += 1
    print(f"  Başlangıç eşleşme: {initial_matches}/20")

    print(f"\n{'=' * 60}")
    print("R-STDP İLE ONLINE ÖĞRENME")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for iteration in range(N_ITERATIONS):
            iteration_reward = 0
            iteration_matches = 0

            for gen_idx in range(N_GENERATE):
                # İsim üret (token token, reward her adımda)
                idx = torch.tensor([[BOS]], device=DEVICE)
                generated_chars = []

                for step in range(12):
                    idx_cond = idx[:, -model.block_size :]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / 0.8
                    probs = F.softmax(logits, dim=-1)
                    token = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, token), dim=1)

                    if token.item() != BOS and token.item() < len(chars):
                        generated_chars.append(chars[token.item()])

                    # Her token için reward
                    if generated_chars:
                        reward = shaped_reward(generated_chars, TARGET_SUFFIX)
                        # Hedef token logitlerini artır
                        for ti in target_indices:
                            logits[0, ti] += LR * reward

                name = "".join(generated_chars)
                reward = shaped_reward(generated_chars, TARGET_SUFFIX)
                iteration_reward += reward

                if name.endswith(TARGET_SUFFIX):
                    iteration_matches += 1

            avg_reward = iteration_reward / N_GENERATE
            reward_history.append(avg_reward)
            match_history.append(iteration_matches)

            if iteration % 10 == 0 or iteration == N_ITERATIONS - 1:
                print(
                    f"  Iter {iteration + 1:3d}: reward={avg_reward:.3f}, matches={iteration_matches}/{N_GENERATE}"
                )

    print(f"\n{'=' * 60}")
    print("SONUÇ")
    print(f"{'=' * 60}")
    print(f"  Başlangıç eşleşme: {initial_matches}/20")

    # Son test
    final_matches = 0
    with torch.no_grad():
        for _ in range(20):
            idx = torch.tensor([[BOS]], device=DEVICE)
            out = model.generate(idx, max_new_tokens=12, temperature=0.8)
            name = "".join(
                itos[t.item()]
                for t in out[0, 1:]
                if t.item() != BOS and t.item() < len(chars)
            )
            if name.endswith(TARGET_SUFFIX):
                final_matches += 1

    print(f"  Son eşleşme: {final_matches}/20")
    print(f"  Reward trend: {reward_history[0]:.3f} → {reward_history[-1]:.3f}")

    if final_matches > initial_matches:
        print(f"\n  ✅ R-STDP ÖĞRENDİ! ({initial_matches}/20 → {final_matches}/20)")
    else:
        print(f"\n  ⚠️ Belirgin öğrenme yok ({initial_matches}/20 → {final_matches}/20)")

    print(f"\n  Son 5 isim:")
    with torch.no_grad():
        for _ in range(5):
            idx = torch.tensor([[BOS]], device=DEVICE)
            out = model.generate(idx, max_new_tokens=12, temperature=0.8)
            name = "".join(
                itos[t.item()]
                for t in out[0, 1:]
                if t.item() != BOS and t.item() < len(chars)
            )
            has_suffix = name.endswith(TARGET_SUFFIX)
            print(f"    {name:20s} {'✅' if has_suffix else ''}")


if __name__ == "__main__":
    main()
