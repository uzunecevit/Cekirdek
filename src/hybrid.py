"""
VİCDAN — Hybrid v0.1: Feedback Köprüsü (LM Head = Backprop, Internal = STDP)

Mimari:
    "3+4=" → intent: ADD(3,4) → engine: 7 → target: "3+4=7"
    → SNN forward → loss hesaplama
    → LM head: backprop/Adam (interference-free mapping)
    → Internal: STDP (online, biyolojik öğrenme)
    → Consolidation → W_static
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.intent import extract_intent
from src.engine import run_engine


def encode_sequence(text: str, char2idx: dict) -> list[int]:
    """String'i token index listesine çevir."""
    return [char2idx.get(c, 0) for c in text if c in char2idx]


def build_targets(
    text: str, result: int, char2idx: dict
) -> tuple[list[int], list[int]]:
    """
    Input ve target token'larını oluştur.

    x = "3+4=" → [3, +, 4, =]
    y = [+, 4, =, 7]  (her pozisyonda bir sonraki token)
    """
    target_text = text + str(result)
    x_tokens = encode_sequence(text, char2idx)
    y_full = encode_sequence(target_text, char2idx)
    y_tokens = y_full[1:]  # next token alignment

    min_len = min(len(x_tokens), len(y_tokens))
    return x_tokens[:min_len], y_tokens[:min_len]


def hybrid_step(
    model,
    text: str,
    char2idx: dict,
    lr: float = 0.001,
    surprise_threshold: float = 0.0,
    ffn_lr: float = 0.0,
) -> dict:
    """
    Tek hybrid adım: intent → engine → backprop (LM head) + STDP (internal).

    Args:
        model: SpikingLM
        text: Input string (örn: "3+4=")
        char2idx: Vocab mapping
        lr: Learning rate for LM head backprop
        surprise_threshold: Context gating (0.0 = her zaman öğren)
        ffn_lr: FFN fast weight learning rate (0.0 = kapalı)

    Returns:
        {"intent": dict, "result": int, "target": str, "loss": float, "updated": bool}
    """
    intent = extract_intent(text)
    if intent is None:
        return {
            "intent": None,
            "result": None,
            "target": None,
            "loss": None,
            "updated": False,
        }

    result = run_engine(intent)
    if result is None:
        return {
            "intent": intent,
            "result": None,
            "target": None,
            "loss": None,
            "updated": False,
        }

    target_text = text + str(result)
    x_tokens, y_tokens = build_targets(text, result, char2idx)

    if len(x_tokens) < 2 or len(y_tokens) < 2:
        return {
            "intent": intent,
            "result": result,
            "target": target_text,
            "loss": None,
            "updated": False,
        }

    loss, updated = _hybrid_update(
        model, x_tokens, y_tokens, lr, surprise_threshold, ffn_lr
    )

    return {
        "intent": intent,
        "result": result,
        "target": target_text,
        "loss": loss,
        "updated": updated,
    }


def _hybrid_update(
    model,
    x: list[int],
    y: list[int],
    lr: float,
    surprise_threshold: float,
    ffn_lr: float,
) -> tuple[float, bool]:
    """
    Hibrit update: LM head = backprop, internal = STDP.

    LM head için backprop kullanıyoruz çünkü interference problemi var.
    STDP formülü (rank-1 update) birden fazla pattern'i aynı anda öğrenemiyor.
    """
    device = next(model.parameters()).device
    vocab_size = model.lm_head_static.weight.size(0)

    idx = torch.tensor([x], device=device)
    target = torch.tensor([y], device=device)

    # Forward pass
    logits, loss = model(idx, targets=target)
    probs = F.softmax(logits[0, :, :], dim=-1)
    last_probs = probs[-1]
    surprise = _compute_surprise(last_probs)

    if surprise <= surprise_threshold:
        return loss.item(), False

    # ── 1. LM Head: Backprop (Adam) ──
    # Sadece LM head parametrelerini optimize et
    optimizer = optim.Adam(
        [
            {"params": [model.lm_head_static.weight], "lr": lr},
            {"params": [model.lm_head_bias], "lr": lr},
        ]
    )
    optimizer.zero_grad()

    # Loss'u tekrar hesapla (gradient graph için)
    logits2, loss2 = model(idx, targets=target)
    loss2.backward()
    optimizer.step()

    # ── 2. Internal Layers: STDP (fast weights) ──
    if ffn_lr > 0:
        with torch.no_grad():
            t = logits.size(1) - 1
            target_token = y[-1]
            pre = model._last_hidden[0, t]

            target_onehot = torch.zeros(vocab_size, device=device)
            target_onehot[target_token] = 1.0
            error = target_onehot - probs[t]

            for block in model.blocks:
                fc2 = block.ffn.fc2
                if fc2.fast_weight and block.ffn._pre_activation is not None:
                    pre_act = block.ffn._pre_activation[0]
                    post_act = block.ffn._post_activation[0]
                    error_mag = error.abs().mean()
                    dw_ffn = ffn_lr * error_mag * torch.outer(post_act, pre_act)
                    fc2.weight_fast.add_(dw_ffn)
                    fc2.weight_fast.clamp_(-10.0, 10.0)

    return loss.item(), True


def _hybrid_batch_update(
    model,
    samples_data: list[tuple[list[int], list[int]]],
    lr: float,
    ffn_lr: float = 0.0,
) -> float:
    """
    Batched hibrit update: Tüm sample'lar için tek bir forward + backprop.
    Bu, mini-batch gradient descent'e eşdeğer - interference'ı çözer.
    """
    device = next(model.parameters()).device

    # Tüm sample'ları bir batch olarak topla
    all_x = []
    all_y = []
    max_len = 0

    for x, y in samples_data:
        all_x.append(x)
        all_y.append(y)
        max_len = max(max_len, len(x))

    # Padding
    B = len(all_x)
    padded_x = torch.zeros(B, max_len, dtype=torch.long, device=device)
    padded_y = torch.full((B, max_len), -100, dtype=torch.long, device=device)

    for i, (x, y) in enumerate(zip(all_x, all_y)):
        padded_x[i, : len(x)] = torch.tensor(x, device=device)
        padded_y[i, : len(y)] = torch.tensor(y, device=device)

    # Forward + Backprop (tüm sample'lar aynı anda)
    optimizer = optim.Adam(
        [
            {"params": [model.lm_head_static.weight], "lr": lr},
            {"params": [model.lm_head_bias], "lr": lr},
        ]
    )
    optimizer.zero_grad()

    logits, loss = model(padded_x, targets=padded_y)
    loss.backward()
    optimizer.step()

    return loss.item()


def _compute_surprise(probs: torch.Tensor) -> float:
    """Normalized entropy (0-1 arası)."""
    import math

    eps = 1e-9
    H = -(probs * torch.log(probs + eps)).sum()
    H_max = math.log(probs.size(-1))
    return (H / H_max).item()


def hybrid_train_loop(
    model,
    samples: list[str],
    char2idx: dict,
    epochs: int = 5,
    lr: float = 0.001,
    consolidation_alpha: float = 0.05,
    consolidation_threshold: float = 0.5,
    consolidation_every: int = 1,
    ffn_lr: float = 0.0,
    batched: bool = True,
) -> list[dict]:
    """
    Hybrid eğitim döngüsü.

    Args:
        batched: True = tüm sample'lar aynı anda (mini-batch), False = sequential
    """
    history = []

    for epoch in range(epochs):
        if batched:
            # Tüm sample'ları bir batch olarak işle
            samples_data = []
            for sample in samples:
                intent = extract_intent(sample)
                if intent is None:
                    continue
                result = run_engine(intent)
                if result is None:
                    continue
                x_tokens, y_tokens = build_targets(sample, result, char2idx)
                if len(x_tokens) >= 2 and len(y_tokens) >= 2:
                    samples_data.append((x_tokens, y_tokens))

            if samples_data:
                avg_loss = _hybrid_batch_update(model, samples_data, lr, ffn_lr)
            else:
                avg_loss = 0.0
        else:
            # Sequential (eski yöntem)
            epoch_loss = 0
            n_updates = 0
            for sample in samples:
                result = hybrid_step(model, sample, char2idx, lr=lr, ffn_lr=ffn_lr)
                if result["loss"] is not None:
                    epoch_loss += result["loss"]
                    n_updates += 1
            avg_loss = epoch_loss / max(n_updates, 1)

        if (epoch + 1) % consolidation_every == 0:
            report = model.consolidate(
                alpha=consolidation_alpha, threshold=consolidation_threshold
            )
            consolidated = report.get("total_consolidated", 0)
        else:
            consolidated = 0

        history.append(
            {"epoch": epoch + 1, "loss": avg_loss, "consolidated": consolidated}
        )

    return history
