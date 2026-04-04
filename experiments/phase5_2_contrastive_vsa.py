#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 5.2: VSA Kontrastif Öğrenme

Sorun: Model matematiksel ilişkiyi değil, sonucu ezberliyor.
Çözüm: Negatif örnekler + kontrastif loss + dengeli dataset.

Kontrastif loss:
- Pozitif: bind(a, b, op, =) → result (yakın olmalı)
- Negatif: bind(a, b, op, =) → wrong_result (uzak olmalı)
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.vsa import VSA

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_DIMS = 512
EPOCHS = 2000
LR = 0.005
BATCH_SIZE = 32
TEMPERATURE = 0.1  # Kontrastif loss sıcaklığı


class TrainableVSA(nn.Module):
    """Eğitilebilir VSA Embedding + Kontrastif Öğrenme."""

    def __init__(self, symbols, n_dims=512, device="cuda"):
        super().__init__()
        self.n_dims = n_dims
        self.device = device
        self.symbols = symbols
        self.n_symbols = len(symbols)

        # Eğitilebilir embedding vektörleri
        embeddings = torch.randn(self.n_symbols, n_dims, device=device)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        self.embeddings = nn.Parameter(embeddings)

        # Eğitilebilir positional encoding vektörleri
        self.max_pos = 16
        self.pos_encodings = nn.Parameter(
            torch.randn(self.max_pos, n_dims, device=device)
        )
        self.pos_encodings.data = (
            self.pos_encodings.data / self.pos_encodings.data.norm(dim=1, keepdim=True)
        )

        self.sym2idx = {s: i for i, s in enumerate(symbols)}
        self.vsa = VSA(n_dims=n_dims, device=device)

    def get_vector(self, symbol):
        idx = self.sym2idx[symbol]
        return self.embeddings[idx]

    def get_pos_encoding(self, pos):
        return self.pos_encodings[pos % self.max_pos]

    def bind(self, a, b):
        return a * b

    def bundle(self, vectors):
        if not vectors:
            return torch.zeros(self.n_dims, device=self.device)
        bundled = torch.stack(vectors).sum(dim=0)
        return bundled / bundled.norm()

    def similarity(self, a, b):
        return F.cosine_similarity(a, b, dim=0)

    def cleanup(self, query):
        sims = torch.stack(
            [self.similarity(query, self.embeddings[i]) for i in range(self.n_symbols)]
        )
        idx = sims.argmax().item()
        return self.symbols[idx], sims[idx].item()

    def encode_sequence(self, symbols):
        """Sekans kodlama: semantik + pozisyonel encoding."""
        encoded = []
        for i, sym in enumerate(symbols):
            v = self.get_vector(sym)
            p = self.get_pos_encoding(i)
            encoded.append(self.bind(v, p))
        return self.bundle(encoded)

    def decode_sequence(self, seq_vector, length):
        """Sekans decode."""
        decoded = []
        for i in range(length):
            p = self.get_pos_encoding(i)
            decoded_vec = self.bind(seq_vector, p)
            sym, sim = self.cleanup(decoded_vec)
            decoded.append((sym, sim))
        return decoded

    def contrastive_loss(
        self, a_sym, b_sym, op_sym, result_sym, neg_results, temperature=TEMPERATURE
    ):
        """
        Kontrastif loss:
        - Pozitif örnek: bind(a, b, op, =) → result
        - Negatif örnekler: bind(a, b, op, =) → wrong_result_1, wrong_result_2, ...

        InfoNCE loss: -log(exp(sim_pos/τ) / Σexp(sim_i/τ))
        """
        a = self.get_vector(a_sym)
        b = self.get_vector(b_sym)
        op = self.get_vector(op_sym)
        eq = self.get_vector("=")

        # Binding: a ⊛ b ⊛ op ⊛ =
        bound = self.bind(a, b)
        bound = self.bind(bound, op)
        bound = self.bind(bound, eq)
        bound = bound / bound.norm()

        # Pozitif benzerlik
        result_vec = self.get_vector(result_sym)
        sim_pos = self.similarity(bound, result_vec)

        # Negatif benzerlikler
        neg_sims = []
        for neg_sym in neg_results:
            neg_vec = self.get_vector(neg_sym)
            neg_sim = self.similarity(bound, neg_vec)
            neg_sims.append(neg_sim)

        # InfoNCE loss
        all_sims = torch.tensor([sim_pos] + neg_sims, device=self.device) / temperature
        log_softmax = F.log_softmax(all_sims, dim=0)
        loss = -log_softmax[0]  # Pozitif örnek için negative log likelihood

        return loss, sim_pos.item()

    def sequence_loss(self, symbols):
        """Sekans öğrenme loss'u."""
        seq_vector = self.encode_sequence(symbols)

        loss = 0
        for i, sym in enumerate(symbols):
            p = self.get_pos_encoding(i)
            decoded_vec = self.bind(seq_vector, p)
            target_vec = self.get_vector(sym)
            sim = self.similarity(decoded_vec, target_vec)
            loss += 1.0 - sim

        return loss / len(symbols)

    def repulsion_loss(self, n_samples=64):
        """Farklı sembollerin vektörlerini birbirinden uzaklaştır."""
        loss = 0
        for _ in range(n_samples):
            i, j = random.sample(range(self.n_symbols), 2)
            vi = self.embeddings[i]
            vj = self.embeddings[j]
            sim = self.similarity(vi, vj)
            loss += sim**2
        return loss / n_samples

    def pos_repulsion_loss(self, n_samples=32):
        """Pozisyonel encoding'leri birbirinden uzaklaştır."""
        loss = 0
        for _ in range(n_samples):
            i, j = random.sample(range(self.max_pos), 2)
            pi = self.pos_encodings[i]
            pj = self.pos_encodings[j]
            sim = self.similarity(pi, pj)
            loss += sim**2
        return loss / n_samples


def generate_balanced_math_dataset():
    """Dengeli matematik dataset'i — her sonuç eşit sıklıkta."""
    dataset = []

    # Toplama: her sonuç (0-9) eşit sayıda
    for result in range(10):
        for a in range(10):
            for b in range(10):
                if a + b == result:
                    dataset.append((str(a), str(b), "+", str(result)))

    # Çıkarma: her sonuç (0-9) eşit sayıda
    for result in range(10):
        for a in range(10):
            for b in range(10):
                if a - b == result and a >= b:
                    dataset.append((str(a), str(b), "-", str(result)))

    return dataset


def generate_negative_examples(result_sym, dataset, n_neg=5):
    """Yanlış sonuçlar üret."""
    all_results = set()
    for a, b, op, r in dataset:
        all_results.add(r)
    all_results.discard(result_sym)

    neg_results = random.sample(list(all_results), min(n_neg, len(all_results)))
    return neg_results


def train_vsa_contrastive():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 5.2: VSA Kontrastif Öğrenme")
    print("=" * 70)

    symbols = (
        [str(i) for i in range(10)]
        + ["+", "-", "*", "/", "=", "(", ")", " ", "\n"]
        + list("abcdefghijklmnopqrstuvwxyz")
        + ["ç", "ğ", "ı", "ö", "ş", "ü", "Ç", "Ğ", "İ", "Ö", "Ş", "Ü"]
    )

    model = TrainableVSA(symbols, n_dims=N_DIMS, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = generate_balanced_math_dataset()
    print(f"\n  Dataset: {len(dataset)} matematik işlemi (dengeli)")
    print(f"  Semboller: {len(symbols)}")
    print(f"  Vektör boyutu: {N_DIMS}")

    # Sonuç dağılımını kontrol et
    result_counts = {}
    for a, b, op, r in dataset:
        result_counts[r] = result_counts.get(r, 0) + 1
    print(f"  Sonuç dağılımı: {dict(sorted(result_counts.items()))}")

    print(f"\n{'=' * 70}")
    print("EĞİTİM")
    print(f"{'=' * 70}")

    best_loss = float("inf")
    best_acc = 0

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
                # Negatif örnekler
                neg_results = generate_negative_examples(result, dataset, n_neg=5)

                # Kontrastif loss
                loss, sim = model.contrastive_loss(a, b, op, result, neg_results)
                batch_loss += loss
                batch_sim += sim

            # Repulsion loss
            rep_loss = model.repulsion_loss(n_samples=32)
            pos_rep_loss = model.pos_repulsion_loss(n_samples=16)

            # Toplam loss
            total_loss = batch_loss / len(batch) + 0.05 * rep_loss + 0.05 * pos_rep_loss
            avg_sim = batch_sim / len(batch)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Normalize
            with torch.no_grad():
                model.embeddings.data = (
                    model.embeddings.data
                    / model.embeddings.data.norm(dim=1, keepdim=True)
                )
                model.pos_encodings.data = (
                    model.pos_encodings.data
                    / model.pos_encodings.data.norm(dim=1, keepdim=True)
                )

            epoch_loss += total_loss.item()
            epoch_sim += avg_sim
            n_batch += 1

        avg_loss = epoch_loss / n_batch
        avg_sim = epoch_sim / n_batch

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            print(
                f"  Epoch {epoch:4d}/{EPOCHS}: loss={avg_loss:.4f}, sim={avg_sim:.4f}, rep={rep_loss.item():.4f}"
            )

    # Test
    print(f"\n{'=' * 70}")
    print("TEST")
    print(f"{'=' * 70}")

    model.eval()

    # Matematik test
    test_examples = [
        ("3", "4", "+", "7"),
        ("5", "2", "+", "7"),
        ("1", "6", "+", "7"),
        ("8", "3", "-", "5"),
        ("9", "4", "-", "5"),
        ("2", "5", "+", "7"),
        ("6", "3", "+", "9"),
        ("7", "2", "-", "5"),
    ]

    print("\n  Matematik işlemleri:")
    correct = 0
    for a, b, op, expected in test_examples:
        neg_results = generate_negative_examples(expected, dataset, n_neg=5)
        loss, sim = model.contrastive_loss(a, b, op, expected, neg_results)

        # Cleanup test
        a_vec = model.get_vector(a)
        b_vec = model.get_vector(b)
        op_vec = model.get_vector(op)
        eq_vec = model.get_vector("=")
        bound = model.bind(a_vec, b_vec)
        bound = model.bind(bound, op_vec)
        bound = model.bind(bound, eq_vec)
        bound = bound / bound.norm()

        result, cleanup_sim = model.cleanup(bound)
        match = "✅" if result == expected else "❌"
        if result == expected:
            correct += 1

        print(
            f"    {a}{op}{b}={expected} → '{result}' {match} (sim={sim:.4f}, cleanup={cleanup_sim:.4f})"
        )

    print(
        f"\n  Matematik doğruluğu: {correct}/{len(test_examples)} ({correct / len(test_examples) * 100:.0f}%)"
    )

    # Sekans testi
    print(f"\n  Sekans temsili testi:")
    seq = ["3", "+", "4", "=", "7"]
    seq_vec = model.encode_sequence(seq)
    decoded = model.decode_sequence(seq_vec, len(seq))

    seq_correct = 0
    for i, (sym, sim) in enumerate(decoded):
        expected = seq[i]
        match = "✅" if sym == expected else "❌"
        if sym == expected:
            seq_correct += 1
        print(
            f"    Pozisyon {i}: '{sym}' (beklenen: '{expected}') {match} sim={sim:.4f}"
        )

    print(
        f"\n  Sekans doğruluğu: {seq_correct}/{len(seq)} ({seq_correct / len(seq) * 100:.0f}%)"
    )

    # Genel doğruluk
    total_correct = correct + seq_correct
    total_tests = len(test_examples) + len(seq)
    print(
        f"\n  Toplam doğruluk: {total_correct}/{total_tests} ({total_correct / total_tests * 100:.0f}%)"
    )

    if best_loss < best_loss:
        pass  # Already tracked

    # Kaydet
    print(f"\n{'=' * 70}")
    print("KAYDETME")
    print(f"{'=' * 70}")

    state = {
        "symbols": symbols,
        "embeddings": model.embeddings.cpu().detach().tolist(),
        "pos_encodings": model.pos_encodings.cpu().detach().tolist(),
        "n_dims": N_DIMS,
        "best_loss": best_loss,
        "math_accuracy": correct / len(test_examples),
        "seq_accuracy": seq_correct / len(seq),
        "total_accuracy": total_correct / total_tests,
    }

    save_path = os.path.join(DATA_DIR, "vsa_trained_v2.json")
    with open(save_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"  Kaydedildi: {save_path}")
    print(f"  En iyi loss: {best_loss:.4f}")
    print(f"  Matematik doğruluğu: {correct}/{len(test_examples)}")
    print(f"  Sekans doğruluğu: {seq_correct}/{len(seq)}")


if __name__ == "__main__":
    train_vsa_contrastive()
