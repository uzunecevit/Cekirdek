"""
VİCDAN_SPIKE — Vector Symbolic Architecture (VSA)
FHRR (Fourier Holographic Reduced Representation) tabanlı sembolik temsil.

Operasyonlar:
- Binding (Bağlama): Element-wise çarpım (a * b)
- Bundling (Toplama): Element-wise toplam + normalize
- Cleanup Memory: Cosine similarity ile en yakın sembolü bul
- Similarity: Kosinüs benzerliği
"""

import torch
import torch.nn.functional as F
import json
import os
import numpy as np


class VSA:
    """Vektör Sembolik Mimari - FHRR"""

    def __init__(self, n_dims=512, device="cuda"):
        self.n_dims = n_dims
        self.device = device
        self.vocab = {}  # {'symbol': tensor}
        self._cleanup_memory = None

    def random_vector(self):
        """Rastgele birim VSA vektörü oluştur."""
        v = torch.randn(self.n_dims, device=self.device)
        return v / v.norm()

    def bind(self, a, b):
        """
        Bağlama (Binding): İki vektörü birleştir.
        FHRR'de element-wise çarpım kullanılır.
        """
        return a * b

    def bundle(self, vectors, weights=None):
        """
        Toplama (Bundling): Vektörleri süperpozisyon yap.
        """
        if not vectors:
            return torch.zeros(self.n_dims, device=self.device)
        stacked = torch.stack(vectors)
        if weights is not None:
            w = torch.tensor(weights, device=self.device).view(-1, 1)
            bundled = (stacked * w).sum(dim=0)
        else:
            bundled = stacked.sum(dim=0)
        return bundled / bundled.norm()

    def similarity(self, a, b):
        """Kosinüs benzerliği."""
        return F.cosine_similarity(a, b, dim=0).item()

    def cleanup(self, query, top_k=1):
        """
        Cleanup Memory: Query'e en yakın sembolü bul.
        Cosine similarity + softmax ağırlıkları.
        """
        if not self.vocab:
            return None, -1.0

        similarities = {}
        for symbol, vector in self.vocab.items():
            similarities[symbol] = self.similarity(query, vector)

        # Softmax over similarities
        symbols = list(similarities.keys())
        sims = torch.tensor([similarities[s] for s in symbols], device=self.device)
        probs = F.softmax(sims * 10.0, dim=0)  # Temperature scaling

        # Top-k
        top_probs, top_indices = torch.topk(probs, min(top_k, len(symbols)))

        results = []
        for prob, idx in zip(top_probs, top_indices):
            sym = symbols[idx.item()]
            results.append((sym, similarities[sym], prob.item()))

        return results[0] if top_k == 1 else results

    def add_symbol(self, symbol, vector=None):
        """Sözlüğe yeni sembol ekle."""
        if vector is None:
            vector = self.random_vector()
        self.vocab[symbol] = vector.to(self.device)
        return vector

    def save_vocab(self, path):
        """Vektör sözlüğünü JSON olarak kaydet."""
        data = {
            "n_dims": self.n_dims,
            "symbols": list(self.vocab.keys()),
            "vectors": {k: v.cpu().tolist() for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_vocab(self, path):
        """Vektör sözlüğünü JSON'dan yükle."""
        with open(path) as f:
            data = json.load(f)
        self.n_dims = data["n_dims"]
        self.vocab = {
            k: torch.tensor(v, device=self.device) for k, v in data["vectors"].items()
        }

    def generate_math_vocab(self):
        """Matematik sembolleri için VSA sözlüğü oluştur."""
        symbols = (
            [str(i) for i in range(10)]  # 0-9
            + ["+", "-", "*", "/", "=", "(", ")", " ", "\n"]
            + list("abcdefghijklmnopqrstuvwxyz")
            + ["ç", "ğ", "ı", "ö", "ş", "ü", "Ç", "Ğ", "İ", "Ö", "Ş", "Ü"]
        )
        for sym in symbols:
            self.add_symbol(sym)
        return self.vocab


if __name__ == "__main__":
    # Test
    vsa = VSA(n_dims=512, device="cuda" if torch.cuda.is_available() else "cpu")
    vsa.generate_math_vocab()

    # Binding test
    three = vsa.vocab["3"]
    four = vsa.vocab["4"]
    add = vsa.vocab["+"]

    # 3 + 4 binding
    bound = vsa.bind(three, four)
    bound = vsa.bind(bound, add)

    print(f"VSA Test:")
    print(f"  Vocab size: {len(vsa.vocab)}")
    print(f"  3+4 bound norm: {bound.norm().item():.4f}")

    # Cleanup test
    result, sim, prob = vsa.cleanup(bound)
    print(f"  Cleanup result: '{result}' (sim={sim:.4f}, prob={prob:.4f})")

    # Save
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    vsa.save_vocab(os.path.join(data_dir, "vsa_vocab.json"))
    print(f"  Saved: vsa_vocab.json")
