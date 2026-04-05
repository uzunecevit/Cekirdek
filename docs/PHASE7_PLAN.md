# Çekirdek — Faz 7: Öğretmen Model Doğrulama ve Bilgi Damıtma

> *"Öğretmeni seç, bilgiyi damıt, ölçeklendir."*

**Oluşturulma:** 2026-04-05
**Son güncelleme:** 2026-04-05
**Durum:** Faz 7.0.5 (Qwen Test) devam ediyor

---

## 🎯 Hedef

Mevcut 76K parametreli Çekirdek modelinin **generalization problemini** (unseen task'larda %0) çözmek için:
1. Büyük bir öğretmen modelden bilgi damıtmak
2. Çekirdek'i ölçeklendirmek (76K → 300-500K)
3. Sentetik veri ile eğitim dataset'ini genişletmek (300 → 5K-10K)

---

## 📋 Yol Haritası

```
Faz 7.0:   Mamba Test              → ❌ Elendi (Türkçe yok)
Faz 7.0.5: Qwen3.5-9B Test        → ⏳ Devam ediyor
Faz 7.1:   Scaling (76K → 300-500K) → ⏳ Planlandı
Faz 7.2:   Sentetik Veri (TURBO)   → ⏳ Planlandı
Faz 7.3:   Bilgi Damıtma           → ⏳ Planlandı
Faz 7.4:   Birleşmiş Pipeline      → ⏳ Planlandı
```

---

## 🔬 Faz 7.0: Mamba-Codestral-7B Test (Tamamlandı — ❌ Elendi)

### Test Edilen Model
- **Mamba-Codestral-7B-v0.1** (`/home/ayandon/KAPTAN/modeller/`)
- 7B parametre, SSM mimarisi, kod odaklı

### Test Seti (21 örnek)

| Kategori | Örnekler | Sayı |
|----------|----------|------|
| Compute | `3+4=`, `5-2=`, `6*7=`, `12+34=`, `100-50=`, `9*9=`, `7*8=`, `15-7=` | 8 |
| Verify | `5-3=2 doğru`, `4*4=16 doğru mu?`, `3+7=10 doğru`, `8-5=2 doğru mu?`, `6*6=35 doğru mu?` | 5 |
| Generate | `merhaba`, `istanbul`, `bir gün`, `ahmet`, `nasılsın`, `bugün hava`, `kedi`, `araba` | 8 |

### Sonuçlar

| Test | Sonuç | Not |
|------|-------|-----|
| Compute | ✅ 100% (8/8) | Matematik evrensel |
| Verify | ❌ 0% (0/5) | Türkçe anlamıyor |
| Generate | ❌ 0% (0/8) | Türkçe üretemiyor |
| Action Selection | ❌ 0% (0/21) | Türkçe prompt anlamıyor |

### Karar
Mamba-Codestral bir **kod modeli** — Türkçe bilmiyor. Distillation için **uygun değil**.

---

## 🔬 Faz 7.0.5: Qwen3.5-9B Test (Tamamlandı — ✅ PASS)

### Test Edilen Model
- **Qwen3.5-9B-Q4_K_M.gguf** (`/home/ayandon/KAPTAN/modeller/`)
- 9B parametre, Hybrid SSM+Attention, 4-bit quantized

### Sonuçlar

| Test | Sonuç | Not |
|------|-------|-----|
| Overall | ✅ 76% (16/21) | Eşik %70 → PASS |
| Compute | ✅ 88% (7/8) | 1 hata (12+34= boş çıktı) |
| Verify | ❌ 20% (1/5) | "doğru mu?" soruları boş |
| Generate | ✅ 100% (8/8) | Türkçe mükemmel |

### Yanlışlar
| Input | Beklenen | Çıktı | Sorun |
|-------|----------|-------|-------|
| `12+34=` | compute | (boş) | n_ctx limiti? |
| `5-3=2 doğru` | verify | "mu" | Soru eki olarak algıladı |
| `4*4=16 doğru mu?` | verify | (boş) | Soru işareti sonrası boş |
| `8-5=2 doğru mu?` | verify | (boş) | Aynı sorun |
| `6*6=35 doğru mu?` | verify | (boş) | Aynı sorun |

### Karar
**Qwen öğretmen olarak UYGUN** (overall 76% > %70). Verify zayıf ama sentetik veri ile iyileştirilebilir.

---

## 📐 Faz 7.1: Scaling (Planlandı)

### Hedef
76K → 300-500K parametre (4-6× artış)

### Konfigürasyon Değişiklikleri

| Parametre | Mevcut | Hedef | Artış |
|-----------|--------|-------|-------|
| `d_model` | 64 | 128 | 2× |
| `n_layer` | 2 | 4 | 2× |
| `d_ff` | 128 | 256 | 2× |
| `n_head` | 4 | 8 | 2× |
| **Toplam** | **76K** | **~300-500K** | **~4-6×** |

### Bilimsel Dayanak
- **Power-Law Scaling:** SNN'lerde parametre sayısı ile performans artışı
- **Chinchilla Scaling Laws:** Model boyutu + data boyutu birlikte optimize edilmeli
- **SNN-specific:** Spike sparsity arttıkça capacity artar ama credit assignment zorlaşır

### Riskler

| Risk | Olasılık | Azaltma |
|------|----------|---------|
| Overfitting | Orta | Dropout, weight decay, early stopping |
| VRAM yetersiz | Düşük | Gradient checkpointing, mixed precision |
| Credit assignment bozulur | Orta | Context Gating'i yeniden tune et |

### Script
- `experiments/phase7_1_scaling.py` — Yeni konfigürasyon ile training

---

## 📊 Faz 7.2: Sentetik Veri Üretimi (Planlandı)

### Hedef
300 → 5.000-10.000 örnek (16-33× artış)

### Altyapı: PROJECT-TURBO
- `/home/ayandon/PROJECT-TURBO/` — TurboQuant KV Cache Compression
- 128K context, ~4GB VRAM (RTX 3060'da çalışır)
- Qwen3.5-9B ile EXACT MATCH doğrulandı
- 50 tok/s decode hızı

### Neden TURBO Gerekli?
- Sentetik veri üretimi uzun prompt'lar gerektirir (reasoning zincirleri)
- Standart Qwen: 4K-8K context, ~16GB VRAM (OOM riski)
- TURBO: 128K context, ~4GB VRAM (güvenli)
- 10K örnek üretimi: ~3 saat (50 tok/s)

### Veri Kategorileri

| Kategori | Hedef Sayı | Kaynak |
|----------|-----------|--------|
| Compute | 2.000 | Qwen + math generator |
| Verify | 2.000 | Qwen + math generator |
| Generate | 2.000 | Qwen + Türkçe corpus |
| Ambiguous | 2.000 | Qwen + manuel |

### Sentetik Veri Üretim Stratejisi

```python
# Qwen'e prompt ver, sentetik örnek üret
prompt = """Matematik ifadesi üret. Format: "a+b=", "a-b=c doğru", "metin"
Kategori: compute"""
response = qwen.generate(prompt)
# response: "23+45="
```

### Kalite Kontrol
- Qwen output filtering (geçersiz formatları ele)
- Manuel review (random 100 örnek)
- Ground truth labeling (her örneğin doğru action'ı belli olmalı)

### Script
- `experiments/phase7_2_synthetic_data.py` — TURBO + Qwen ile sentetik veri üretimi
- `data/synthetic/compute.txt`, `verify.txt`, `generate.txt`, `ambiguous.txt`

---

## 🧬 Faz 7.3: Bilgi Damıtma (Planlandı)

### Hedef
Qwen'in action selection yeteneğini SNN ActionHead'e aktarmak

### Yöntem: Logits-Level Distillation

```
Qwen (öğretmen) → input → logits_teacher (3 class: compute/verify/generate)
SNN (öğrenci)   → input → logits_student (3 class)

Loss = α × CE(ground_truth, logits_student) + (1-α) × KL(teacher || student)
```

### Neden Logits-Level?
- Layer-level distillation zor (Transformer → SNN mimari farkı)
- Logits-level: sadece output distribution'ı eşle
- SpikeBERT, SpikingMamba benzeri çalışmalar bunu kullanıyor

### Distillation Pipeline

```
1. Qwen ile her input için action logits üret
2. SNN ile aynı input için action logits üret
3. KL divergence loss hesapla
4. Backprop ile SNN ActionHead'i güncelle
5. Consolidation ile kalıcı yap
```

### Mimari

```
Input → Qwen (teacher) → action logits (3 class)
  ↓
Input → SNN (student)  → action logits (3 class)
  ↓
Loss = α × CE + (1-α) × KL(teacher || student)
  ↓
Backprop → SNN ActionHead güncelle
  ↓
Consolidation → W_static
```

### Riskler

| Risk | Olasılık | Azaltma |
|------|----------|---------|
| Mimari gap çok büyük | Orta | Logits-level ile minimize |
| Distillation transfer başarısız | Düşük | SpikingMamba metodolojisini takip et |
| Temperature scaling zor | Düşük | Grid search ile optimal T bul |

### Script
- `experiments/phase7_3_distillation.py` — Logits-level distillation

---

## 🚀 Faz 7.4: Birleşmiş Pipeline (Planlandı)

### Hedef
Cekircek-H = Cekircek-M (scaled) + Distillation + Hybrid

### Mimari

```
Input → SNN-M (300-500K) + Saliency
  ↓
ActionHead (distilled from Qwen)
  ├─ generate → SNN text generation
  ├─ compute  → Symbolic Engine → result
  └─ verify   → Symbolic Engine → compare
```

### Başarı Kriterleri

| Metrik | Mevcut (Faz 6.8) | Hedef (Faz 7.4) |
|--------|------------------|-----------------|
| Clean accuracy | 100% | 100% |
| Ambiguous accuracy | 60% | >%80 |
| Unseen generalization | 0% | >%50 |
| VRAM (inference) | <1GB | <2GB |
| VRAM (training) | <4GB | <8GB |

---

## ⚠️ Riskler ve Azaltma

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| Qwen de başarısız | Düşük | Yüksek | Farklı öğretmen ara (Llama-3, Mistral) |
| Scaling overfitting | Orta | Orta | Dropout, weight decay, early stopping |
| Distillation transfer | Düşük | Yüksek | Logits-level, temperature scaling |
| VRAM yetersiz | Düşük | Orta | Gradient checkpointing, mixed precision |
| Dataset kalitesi | Orta | Orta | Qwen filtering + human review |

---

## 📁 Dosya Yapısı (Faz 7)

```
Cekirdek/
├── experiments/
│   ├── phase7_0_mamba_test.py      ✅ Tamamlandı (elendi)
│   ├── phase7_0_5_qwen_test.py     ⏳ Devam ediyor
│   ├── phase7_0_5_qwen_results.json ⏳ Sonuçlar
│   ├── phase7_1_scaling.py         ⏳ Planlandı
│   ├── phase7_2_synthetic_data.py  ⏳ Planlandı
│   ├── phase7_3_distillation.py    ⏳ Planlandı
│   └── phase7_4_unified.py         ⏳ Planlandı
├── data/
│   └── synthetic/                  🆕 Sentetik veriler
│       ├── compute.txt
│       ├── verify.txt
│       ├── generate.txt
│       └── ambiguous.txt
└── docs/
    ├── PHASE7_PLAN.md              📄 Bu dosya
    └── DETAILED_PLAN.md            📄 Tüm proje planı
```

---

## 🔮 Gelecek Vizyon

Faz 7 tamamlandığında:
- **Cekircek-H:** 300-500K parametre, >%50 unseen generalization
- **KAPTAN entegrasyonu:** Vicdan modülü olarak
- **Edge deployment:** Neuromorphic donanımda 21W hedefi
- **Araştırma yayını:** "Scaling SNNs with LLM Distillation"

---

> *"Öğretmenini akıllı seç, ama kendi yolunu çiz."*
