#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 5.1: Trainable VSA Embedding

Hedef: VSA embedding vektörlerini backprop ile eğiterek
bind(3, bind(4, +)) → 7 benzerliğini öğrenmek.

Loss: 1 - cosine_similarity(result, target)
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.vsa import VSA

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
N_DIMS = 512
EPOCHS = 1000
LR = 0.01
BATCH_SIZE = 64


class TrainableVSA(nn.Module):
    """Eğitilebilir VSA Embedding."""

    def __init__(self, symbols, n_dims=512, device="cuda"):
        super().__init__()
        self.n_dims = n_dims
        self.device = device
        self.symbols = symbols
        self.n_symbols = len(symbols)

        # Eğitilebilir embedding vektörleri
        # Başlangıç: rastgele birim vektörler
        embeddings = torch.randn(self.n_symbols, n_dims, device=device)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        self.embeddings = nn.Parameter(embeddings)

        # Sembol → index mapping
        self.sym2idx = {s: i for i, s in enumerate(symbols)}

        # VSA operasyonları (sabittir, eğitilmez)
        self.vsa = VSA(n_dims=n_dims, device=device)

    def get_vector(self, symbol):
        """Sembolün vektörünü al."""
        idx = self.sym2idx[symbol]
        return self.embeddings[idx]

    def bind(self, a, b):
        """FHRR binding: element-wise çarpım."""
        return a * b

    def bundle(self, vectors):
        """Bundling: toplam + normalize."""
        if not vectors:
            return torch.zeros(self.n_dims, device=self.device)
        bundled = torch.stack(vectors).sum(dim=0)
        return bundled / bundled.norm()

    def similarity(self, a, b):
        """Kosinüs benzerliği."""
        return F.cosine_similarity(a, b, dim=0)

    def cleanup(self, query):
        """Cleanup: en yakın sembolü bul."""
        sims = torch.stack(
            [self.similarity(query, self.embeddings[i]) for i in range(self.n_symbols)]
        )
        idx = sims.argmax().item()
        return self.symbols[idx], sims[idx].item()

    def forward_math(self, a_sym, b_sym, op_sym, result_sym):
        """
        Matematik işlemi: a op b = result

        Loss: 1 - cosine_similarity(bind(a, bind(b, bind(op, eq))), result)
        """
        a = self.get_vector(a_sym)
        b = self.get_vector(b_sym)
        op = self.get_vector(op_sym)
        eq = self.get_vector("=")
        result = self.get_vector(result_sym)

        # Binding: a ⊛ b ⊛ op ⊛ =
        bound = self.bind(a, b)
        bound = self.bind(bound, op)
        bound = self.bind(bound, eq)

        # Loss: 1 - cosine similarity
        sim = self.similarity(bound, result)
        loss = 1.0 - sim

        return loss, sim.item()

    def forward_sequence(self, symbols):
        """Sekans temsili: positional encoding ile."""
        vectors = [self.get_vector(s) for s in symbols]
        positions = [self.vsa.random_vector() for _ in range(len(symbols))]
        encoded = [self.bind(v, p) for v, p in zip(vectors, positions)]
        return self.bundle(encoded)

    def decode_sequence(self, seq_vector, symbols, positions):
        """Sekansı decode et."""
        decoded = []
        for i, pos_vec in enumerate(positions):
            decoded_vec = self.bind(seq_vector, pos_vec)
            sym, sim = self.cleanup(decoded_vec)
            decoded.append((sym, sim))
        return decoded


def generate_math_dataset():
    """Matematik dataset'i oluştur."""
    dataset = []

    # Toplama
    for a in range(1, 10):
        for b in range(1, 10):
            if a + b < 10:
                dataset.append((str(a), str(b), "+", str(a + b)))

    # Çıkarma
    for a in range(2, 10):
        for b in range(1, a):
            dataset.append((str(a), str(b), "-", str(a - b)))

    return dataset


def train_vsa():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 5.1: Trainable VSA Embedding")
    print("=" * 70)

    # Semboller
    symbols = (
        [str(i) for i in range(10)]
        + ["+", "-", "*", "/", "=", "(", ")", " ", "\n"]
        + list("abcdefghijklmnopqrstuvwxyz")
        + ["ç", "ğ", "ı", "ö", "ş", "ü", "Ç", "Ğ", "İ", "Ö", "Ş", "Ü"]
    )

    model = TrainableVSA(symbols, n_dims=N_DIMS, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Dataset
    dataset = generate_math_dataset()
    random.shuffle(dataset)
    print(f"\n  Dataset: {len(dataset)} matematik işlemi")
    print(f"  Semboller: {len(symbols)}")
    print(f"  Vektör boyutu: {N_DIMS}")

    # Eğitim
    print(f"\n{'=' * 70}")
    print("EĞİTİM")
    print(f"{'=' * 70}")

    best_loss = float("inf")
    best_sim = 0

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(dataset)

        epoch_loss = 0
        epoch_sim = 0
        n_batch = 0

        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i : i + BATCH_SIZE]
            batch_loss = 0
            batch_sim = 0

            for a, b, op, result in batch:
                loss, sim = model.forward_math(a, b, op, result)
                batch_loss += loss
                batch_sim += sim

            batch_loss = batch_loss / len(batch)
            batch_sim = batch_sim / len(batch)

            # Repulsion loss: farklı sembollerin vektörleri birbirinden uzak olmalı
            # Rastgele sembol çiftleri seç, benzerliklerini minimize et
            repulsion_loss = 0
            n_repel = min(32, len(symbols))
            for _ in range(n_repel):
                i, j = random.sample(range(len(symbols)), 2)
                vi = model.embeddings[i]
                vj = model.embeddings[j]
                sim = F.cosine_similarity(vi, vj, dim=0)
                repulsion_loss += sim**2  # Benzerlik karesi: 0'a yakın olmalı
            repulsion_loss = repulsion_loss / n_repel

            # Toplam loss: binding loss + repulsion loss
            total_loss = batch_loss + 0.1 * repulsion_loss  # Repulsion weight: 0.1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Embedding'leri normalize et (birim vektör kalsın)
            with torch.no_grad():
                model.embeddings.data = (
                    model.embeddings.data
                    / model.embeddings.data.norm(dim=1, keepdim=True)
                )

            epoch_loss += batch_loss.item()
            epoch_sim += batch_sim
            n_batch += 1

        avg_loss = epoch_loss / n_batch
        avg_sim = epoch_sim / n_batch

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_sim = avg_sim

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(
                f"  Epoch {epoch:4d}/{EPOCHS}: loss={avg_loss:.4f}, sim={avg_sim:.4f}, best_sim={best_sim:.4f}"
            )

    # Test
    print(f"\n{'=' * 70}")
    print("TEST")
    print(f"{'=' * 70}")

    model.eval()

    # Eğitim setinden örnekler
    test_examples = [
        ("3", "4", "+", "7"),
        ("5", "2", "+", "7"),
        ("1", "6", "+", "7"),
        ("8", "3", "-", "5"),
        ("9", "4", "-", "5"),
        ("2", "5", "+", "7"),
    ]

    print("\n  Matematik işlemleri (a op b = result):")
    correct = 0
    for a, b, op, expected in test_examples:
        loss, sim = model.forward_math(a, b, op, expected)

        # Binding sonucu ile cleanup
        a_vec = model.get_vector(a)
        b_vec = model.get_vector(b)
        op_vec = model.get_vector(op)
        eq_vec = model.get_vector("=")
        bound = model.bind(a_vec, b_vec)
        bound = model.bind(bound, op_vec)
        bound = model.bind(bound, eq_vec)

        result, cleanup_sim = model.cleanup(bound)
        match = "✅" if result == expected else "❌"
        if result == expected:
            correct += 1

        print(
            f"    {a}{op}{b}={expected} → '{result}' {match} (sim={sim:.4f}, cleanup={cleanup_sim:.4f})"
        )

    print(
        f"\n  Doğruluk: {correct}/{len(test_examples)} ({correct / len(test_examples) * 100:.0f}%)"
    )

    # Sekans testi
    print(f"\n  Sekans temsili testi:")
    seq = ["3", "+", "4", "=", "7"]
    positions = [model.vsa.random_vector() for _ in range(len(seq))]
    seq_vec = model.forward_sequence(seq)
    decoded = model.decode_sequence(seq_vec, symbols, positions)

    for i, (sym, sim) in enumerate(decoded):
        expected = seq[i]
        match = "✅" if sym == expected else "❌"
        print(
            f"    Pozisyon {i}: '{sym}' (beklenen: '{expected}') {match} sim={sim:.4f}"
        )

    # Kaydet
    print(f"\n{'=' * 70}")
    print("KAYDETME")
    print(f"{'=' * 70}")

    # Eğitilmiş embedding'leri kaydet
    state = {
        "symbols": symbols,
        "embeddings": model.embeddings.cpu().detach().tolist(),
        "n_dims": N_DIMS,
        "best_loss": best_loss,
        "best_sim": best_sim,
        "accuracy": correct / len(test_examples),
    }

    save_path = os.path.join(DATA_DIR, "vsa_trained.json")
    with open(save_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"  Kaydedildi: {save_path}")
    print(f"  En iyi benzerlik: {best_sim:.4f}")
    print(f"  Test doğruluğu: {correct}/{len(test_examples)}")


if __name__ == "__main__":
    train_vsa()
