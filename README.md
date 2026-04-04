# Çekirdek

> **Pattern-triggered, outcome-reinforced, multi-action neuro-symbolic task router.**

76K parametreli Spiking Neural Network (SNN) ile inşa edilmiş, online öğrenen, catastrophic forgetting'ı sleep consolidation ile önleyen ve neuro-symbolic routing (compute, verify, generate) yapabilen bir bilişsel mimari araştırma projesi.

---

## 🎯 Ne Yapar?

Girdiyi analiz eder, hangi aracı kullanması gerektiğine karar verir ve sonucu üretir:

| Girdi | Aksiyon | Çıktı |
|-------|---------|-------|
| `"3+4="` | compute | `"7"` |
| `"5-3=2 doğru"` | verify | `"Evet, 2 (beklenen: 2)"` |
| `"merhaba"` | generate | `"merhaba nasılsın..."` |

---

## 🏗️ Mimari

```
Input → SNN + Saliency (op_boost=10, verify_boost=5)
  ↓
ActionHead (3 classes: generate, compute, verify)
  ├─ generate → SNN text generation
  ├─ compute  → Symbolic Engine → result
  └─ verify   → Symbolic Engine → compare → True/False
```

### Bileşenler

| Katman | Rol | Parametre |
|--------|-----|-----------|
| **SpikingLM** | Pattern tanıma, hidden state | 76,480 |
| **ActionHead** | Aksiyon seçimi (STDP ile öğrenir) | 195 |
| **Intent Extractor** | Deterministik parsing | — |
| **Symbolic Engine** | Deterministik hesaplama | — |
| **Saliency** | Operatör/keyword boost | — |

---

## 📊 Yolculuk: Phase 0 → Phase 6.8

Bu proje sadece bir "çalışan sistem" değil, **SNN'lerin sınırlarını ve potansiyelini haritalayan bir araştırma yolculuğu**. 8 ana faz, 20+ deney, 42+ öğrenilen ders.

| Faz | Yöntem | Sonuç | Not |
|-----|--------|-------|-----|
| Phase 0-1 | SNN sıfırdan yazıldı | ✅ 76K param, %16 spike rate | Kendi LIF, attention, FFN |
| Phase 2 | R-STDP online öğrenme | ✅ 21× öğrenme artışı | Context Gating ile |
| Phase 3 | Ternary spike + amplitude | ✅ threshold=0.3, amp=2.0 | Binary'den 3.4× etkili |
| Phase 4 | Sleep Consolidation | ✅ Forgetting +12.1% → +1.5% | Catastrophic forgetting önlendi |
| Phase 5 | VSA ile sembolik akıl yürütme | ❌ Başarısız | Binding commutative, işlem ayrımı yok |
| Phase 6.0 | Hybrid Backprop | ❌ 0% generalization | LM head lookup table oldu |
| Phase 6.1 | Intent classification | ❌ 40% unseen | Soyutlama yetersiz |
| Phase 6.2 | Task Gating (supervised) | ✅ 100% | Pattern tanıma SNN'in işi |
| Phase 6.3 | STDP decision (supervised) | ✅ 100% | Decision reinforcement çalıştı |
| Phase 6.4 | Outcome-Based STDP | ✅ 100% | Label-free, sonuçtan öğrenme |
| Phase 6.5 | Ambiguity injection | ⚠️ 50% | Yüzey-level representation sınırı |
| Phase 6.6 | Saliency-Guided | ✅ 100% | Op boost=10, hidden state modulation |
| Phase 6.7 | 3-Action Pipeline | ⚠️ 80% | Verify öğrenilemedi |
| **Phase 6.8** | **Keyword-Enhanced 3-Action** | **✅ 100%** | Verify boost=5, 20 verify örneği |

---

## 🔑 Öğrenilen Dersler

### SNN Hakkında
1. **STDP istatistiksel pattern öğrenir, sembolik mantık değil** → `3+4=7` bir kural, olasılık değil
2. **SNN reasoning yapmaz, karar verir** → Doğru aracı seçme, hesaplama yapma
3. **Ternary spike {-1,0,+1}** → Binary'den 3.4× daha etkili
4. **Context Gating** → 21× öğrenme artışı (surprise_threshold=0.3)
5. **Sleep Consolidation** → Catastrophic forgetting +12.1% → +1.5%

### Hybrid Mimari Hakkında
6. **Batched backprop interference'ı çözer** → Sequential update catastrophic forgetting'e yol açar
7. **Keyword boost > operatör boost** → "doğru", "yanlış" gibi kelimeler verify sinyali olarak kritik
8. **Saliency tek başına yetersiz** → ActionHead'in saliency-modulated state'lerle yeniden eğitilmesi gerekir
9. **Outcome-based STDP çalışır** → Label olmadan, sonuçtan öğrenme mümkün
10. **3-action decision boundary** → 2-action'dan çok daha karmaşık, daha fazla veri gerekli

### Başarısızlıklardan
11. **VSA binding matematiksel ilişki KODLAMAZ** → Commutative, işlem ayrımı yok
12. **Intent classification generalize etmez** → SNN yüzey-level pattern tanır, semantik anlamaz
13. **LM head lookup table olur** → Backprop ile mapping öğrenir, reasoning yapmaz
14. **Başarısız deneyler, başarılılar kadar değerli** → Sınırları netleştirirler

---

## 🚀 Kurulum

```bash
# Python 3.11+ venv
cd Cekirdek
python -m venv .venv
source .venv/bin/activate
pip install torch
```

> **Not:** Model checkpoint'leri (`.pt` dosyaları) repo boyutunu küçük tutmak için `.gitignore`'dadır.
> Kendi modelinizi eğitmek için `experiments/phase1_train.py` ile başlayın.

---

## ⚡ Hızlı Başlangıç

```bash
# Phase 6.8: Final pipeline test (kendi checkpoint'lerinizle)
python experiments/phase6_8_keyword.py
```

Beklenen çıktı:
```
Compute: 100%
Verify:  100%
Generate: 100%
Pipeline: 100%
```

---

## 📁 Proje Yapısı

```
Cekirdek/
├── README.md                 ← Bu dosya
├── docs/
│   └── DETAILED_PLAN.md      ← Tüm phase'lerin detaylı planı + 42+ ders
├── src/
│   ├── model.py              ← SpikingLM, TernaryLIF, TernarySpikeFunction
│   ├── stdp.py               ← R-STDP, SimpleRSTDP, shaped_reward
│   ├── intent.py             ← Intent extractor (expression + statement)
│   ├── engine.py             ← Symbolic math engine (ADD, SUB, MUL)
│   ├── action_head.py        ← ActionHead + STDP decision update
│   ├── task_gating.py        ← TaskClassifier (Phase 6.2)
│   ├── hybrid.py             ← Hybrid feedback köprüsü
│   └── vsa.py                ← VSA operations (FHRR binding, bundling)
├── experiments/
│   ├── phase1_train.py       ← Temel SNN eğitimi
│   ├── phase2_5_lite.py      ← Dense reward kanıtı
│   ├── phase3_2_3_context_gating.py ← Context Gating (21× artış)
│   ├── phase4_1_multi_task.py ← Multi-Task Stability
│   ├── phase4_3_sleep_consolidation.py ← Sleep-Mediated Consolidation
│   ├── phase6_2_gating.py    ← Task Gating
│   ├── phase6_4_outcome.py   ← Outcome-Based STDP (label-free)
│   ├── phase6_6_saliency_training.py ← Saliency-Guided Learning
│   ├── phase6_8_keyword.py   ← Final: Keyword-Enhanced 3-Action
│   └── ...                   ← 20+ deney script'i
├── data/
│   ├── vocab.json            ← 64 karakter vocab (Türkçe + rakamlar + operatörler)
│   ├── pretrain_samples.txt  ← Karışık dataset
│   ├── turkish.txt           ← Türkçe dataset (çekim ekleri + atasözleri)
│   └── math.txt              ← Matematik dataset (toplama, çıkarma, çarpma)
├── checkpoints/              ← Model checkpoint'leri (.gitignore'da)
│   └── .gitkeep
└── .gitignore
```

---

## 📖 Detaylı Plan

Tüm phase'lerin detaylı açıklamaları, deney sonuçları, ablation çalışmaları ve mimari kararlar için:

→ [docs/DETAILED_PLAN.md](docs/DETAILED_PLAN.md)

---

## 🔮 Gelecek Vizyon

- **Phase 6.9:** Online user feedback ile sürekli öğrenme
- **Phase 7:** Yeni action'lar (search, memory, code_execution)
- **Phase 8:** Daha büyük SNN (d_model=128, n_layer=4) ile ölçeklendirme
- **Edge Deployment:** Neuromorphic donanımda 21W hedefi

---

## 📝 Lisans

Araştırma amaçlı. Ticari kullanım için iletişime geçin.

---

> *"Öğrenme, her şeyi ezberlemek değil; belirsizliğin (entropi) içinde anlam aramaktır."*
