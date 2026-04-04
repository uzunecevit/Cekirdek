# Çekirdek

> **Pattern-triggered, outcome-reinforced, multi-action neuro-symbolic task router.**

76K parametreli Spiking Neural Network (SNN) ile inşa edilmiş, online öğrenen, catastrophic forgetting'ı sleep consolidation ile önleyen ve neuro-symbolic routing (compute, verify, generate) yapabilen bir bilişsel mimari prototipi.

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

## 📊 Sonuçlar

| Faz | Yöntem | Sonuç |
|-----|--------|-------|
| Phase 0-1 | SNN Temelleri | ✅ 76K param, %16 spike rate |
| Phase 2 | R-STDP | ✅ 21× öğrenme artışı |
| Phase 3 | Ternary + Context Gating | ✅ threshold=0.3, amplitude=2.0 |
| Phase 4 | Sleep Consolidation | ✅ Forgetting +12.1% → +1.5% |
| Phase 5 | VSA Denemeleri | ❌ Math reasoning yetersiz |
| Phase 6.0-6.4 | Hybrid + Outcome STDP | ✅ 100% (2-action) |
| Phase 6.5 | Ambiguity Injection | ⚠️ 50% ambiguous |
| Phase 6.6 | Saliency-Guided | ✅ 100% unseen ambiguous |
| Phase 6.7 | 3-Action Pipeline | ⚠️ 80% (verify 0%) |
| **Phase 6.8** | **Keyword-Enhanced 3-Action** | **✅ 100% Pipeline** |

---

## 🚀 Kurulum

```bash
# Python 3.11+ venv
cd Cekirdek
python -m venv .venv
source .venv/bin/activate
pip install torch
```

---

## ⚡ Hızlı Başlangıç

```bash
# Phase 6.8: Final pipeline test
python experiments/phase6_8_keyword.py
```

Çıktı:
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
│   └── DETAILED_PLAN.md      ← Tüm phase'lerin detaylı planı
├── src/
│   ├── model.py              ← SpikingLM, TernaryLIF, TernarySpikeFunction
│   ├── stdp.py               ← R-STDP, SimpleRSTDP
│   ├── intent.py             ← Intent extractor (expression + statement)
│   ├── engine.py             ← Symbolic math engine
│   ├── action_head.py        ← ActionHead + STDP update
│   ├── task_gating.py        ← TaskClassifier (Phase 6.2)
│   └── vsa.py                ← VSA operations (FHRR binding, bundling)
├── experiments/
│   ├── phase1_train.py       ← Temel SNN eğitimi
│   ├── phase2_5_lite.py      ← Dense reward kanıtı
│   ├── phase3_2_3_context_gating.py ← Context Gating (21× artış)
│   ├── phase4_1_multi_task.py ← Multi-Task Stability
│   ├── phase4_3_sleep_consolidation.py ← Sleep-Mediated Consolidation
│   ├── phase6_2_gating.py    ← Task Gating
│   ├── phase6_4_outcome.py   ← Outcome-Based STDP
│   ├── phase6_6_saliency_training.py ← Saliency-Guided Learning
│   └── phase6_8_keyword.py   ← Final: Keyword-Enhanced 3-Action
├── checkpoints/
│   ├── spiking_lm_v2.pt      ← Eğitilmiş SNN (76K param)
│   └── action_head_v6_keyword.pt ← Eğitilmiş ActionHead
├── data/
│   ├── vocab.json            ← 64 karakter vocab
│   ├── pretrain_samples.txt  ← Karışık dataset
│   ├── turkish.txt           ← Türkçe dataset
│   └── math.txt              ← Matematik dataset
└── .gitignore
```

---

## 🔑 Öğrenilen Dersler

1. **STDP istatistiksel pattern öğrenir, sembolik mantık değil** → `3+4=7` bir kural, olasılık değil
2. **SNN reasoning yapmaz, karar verir** → Doğru aracı seçme, hesaplama yapma
3. **Batched backprop interference'ı çözer** → Sequential update catastrophic forgetting'e yol açar
4. **Keyword boost > operatör boost** → "doğru", "yanlış" gibi kelimeler verify sinyali olarak kritik
5. **Saliency tek başına yetersiz** → ActionHead'in saliency-modulated state'lerle yeniden eğitilmesi gerekir
6. **Başarısız deneyler, başarılılar kadar değerli** → VSA, intent classification, ambiguity injection sınırları netleştirdi

---

## 📖 Detaylı Plan

Tüm phase'lerin detaylı açıklamaları, deney sonuçları ve mimari kararlar için:

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
