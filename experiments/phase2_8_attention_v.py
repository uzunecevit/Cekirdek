#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 2.8: STDP in Attention V Projection

Kritik değişiklik: STDP artık LM head'de DEĞİL.
STDP → Attention V projeksiyonunda.

Neden V?
- V, "ne hatırlanacak" bilgisini taşır
- LM head bias injection yapıyordu
- V projeksiyonu feature learning için doğru yer
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TRAIN = 30
N_VAL = 20
N_UPDATES = 200
LR = 0.0002
WEIGHT_DECAY = 0.001
ERROR_SCALE = 0.1


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


def load_splits(ckpt):
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

    train_names = names[:100]
    val_names = names[100:200]

    def make_seqs(name_list):
        seqs = []
        for name in name_list:
            tokens = [BOS] + [stoi[c] for c in name] + [BOS]
            if len(tokens) >= 2:
                seqs.append((tokens[:-1], tokens[1:]))
        return seqs

    return make_seqs(train_names)[:N_TRAIN], make_seqs(val_names)[:N_VAL]


def compute_loss(model, sequences):
    losses = []
    with torch.no_grad():
        for x, y in sequences:
            idx = torch.tensor([x], device=DEVICE)
            target = torch.tensor([y], device=DEVICE)
            _, loss = model(idx, targets=target)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def apply_stdp_v_projection(model, x, y, lr, weight_decay, error_scale):
    """
    STDP update: Attention V projeksiyonunda.

    ΔW_v = error_scale × (target_onehot - probs) × pre^T
    W_v *= (1 - weight_decay)
    """
    with torch.no_grad():
        idx = torch.tensor([x], device=DEVICE)
        target = torch.tensor([y], device=DEVICE)
        logits, loss = model(idx, targets=target)
        vocab_size = logits.size(-1)
        d_model = model.d_model

        # V projeksiyon ağırlığı: (d_model, d_model)
        w_v = model.blocks[0].attn.wv.linear.weight

        for t in range(logits.size(1)):
            probs = F.softmax(logits[0, t, :], dim=-1)
            target_token = y[t]
            target_onehot = torch.zeros(vocab_size, device=DEVICE)
            target_onehot[target_token] = 1.0

            # Pre: input embedding (d_model)
            token_idx = x[min(t, len(x) - 1)]
            pre = model.wte.weight[token_idx]

            # Error sinyali (zayıflatılmış)
            error = (target_onehot - probs) * error_scale

            # V projeksiyonuna update:
            # error (vocab,) → d_model'e project et (LM head transpose ile)
            # ΔW_v = LM_head^T @ error × pre^T
            lm_head_t = model.lm_head.weight.T  # (d_model, vocab)
            error_projected = lm_head_t @ error  # (d_model,)

            # ΔW_v = error_projected × pre^T → (d_model, d_model)
            dw = lr * torch.outer(error_projected, pre)

            w_v.add_(dw)

            # Weight decay
            w_v.mul_(1 - weight_decay)

            # Clip
            w_v.clamp_(-1.0, 1.0)

        return loss.item()


def test_attention_v_stdp():
    """Test: STDP in Attention V projection."""
    print("\n" + "=" * 60)
    print("TEST: STDP in Attention V Projection")
    print("=" * 60)

    train_seqs, val_seqs = load_splits(load_model()[1])
    model, ckpt = load_model()

    # LM head'i kilitle
    for p in model.lm_head.parameters():
        p.requires_grad = False

    initial_train = compute_loss(model, train_seqs)
    initial_val = compute_loss(model, val_seqs)
    print(f"  Initial — Train: {initial_train:.4f}, Val: {initial_val:.4f}")
    print(f"  LR={LR}, Weight Decay={WEIGHT_DECAY}, Error Scale={ERROR_SCALE}")

    # STDP ile güncelle
    for step in range(N_UPDATES):
        x, y = train_seqs[step % len(train_seqs)]
        apply_stdp_v_projection(model, x, y, LR, WEIGHT_DECAY, ERROR_SCALE)

    final_train = compute_loss(model, train_seqs)
    final_val = compute_loss(model, val_seqs)

    train_change = (final_train - initial_train) / initial_train * 100
    val_change = (final_val - initial_val) / initial_val * 100

    print(f"  Final  — Train: {final_train:.4f} ({train_change:+.1f}%)")
    print(f"  Final  — Val:   {final_val:.4f} ({val_change:+.1f}%)")

    if val_change < 0:
        print(f"\n  ✅ Gerçek öğrenme: val loss da düştü!")
    elif val_change < 0.5:
        print(f"\n  ⚠️ Zayıf ama umut verici: val loss neredeyse aynı")
    else:
        print(f"\n  ❌ Hâlâ overfitting: val loss arttı")


def test_stability_v():
    """Test: Uzun vadeli stabilite."""
    print("\n" + "=" * 60)
    print("TEST: Stabilite (V Projection)")
    print("=" * 60)

    train_seqs, val_seqs = load_splits(load_model()[1])

    for n_updates in [100, 500, 1000]:
        model, ckpt = load_model()
        for p in model.lm_head.parameters():
            p.requires_grad = False

        for step in range(n_updates):
            x, y = train_seqs[step % len(train_seqs)]
            apply_stdp_v_projection(model, x, y, LR, WEIGHT_DECAY, ERROR_SCALE)

        train_loss = compute_loss(model, train_seqs)
        val_loss = compute_loss(model, val_seqs)
        print(f"  Updates={n_updates:4d}: train={train_loss:.4f}, val={val_loss:.4f}")


def test_error_scale():
    """Test: Error scale etkisi."""
    print("\n" + "=" * 60)
    print("TEST: Error Scale Karşılaştırma")
    print("=" * 60)

    train_seqs, val_seqs = load_splits(load_model()[1])

    for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        model, ckpt = load_model()
        for p in model.lm_head.parameters():
            p.requires_grad = False

        initial_val = compute_loss(model, val_seqs)

        for step in range(200):
            x, y = train_seqs[step % len(train_seqs)]
            apply_stdp_v_projection(model, x, y, LR, WEIGHT_DECAY, scale)

        final_val = compute_loss(model, val_seqs)
        change = (final_val - initial_val) / initial_val * 100
        print(
            f"  Error Scale={scale:.2f}: val={initial_val:.4f}→{final_val:.4f} ({change:+.1f}%)"
        )


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 2.8: STDP in Attention V")
    print("=" * 60)

    test_attention_v_stdp()
    test_stability_v()
    test_error_scale()

    print("\n" + "=" * 60)
    print("TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
