# Çekirdek — Faz 7.0: Teacher Model Validation & Distillation

> *"Öğretmeni seç, bilgiyi damıt, ölçeklendir."*

**Oluşturulma:** 2026-04-05
**Durum:** Faz 7.0.5 (Qwen Test) hazırlanıyor

---

## 🎯 Hedef

Mevcut 76K parametreli Çekirdek modelinin **generalization problemini** (unseen task'larda %0) çözmek için:
1. Büyük bir öğretmen modelden bilgi damıtmak
2. Çekirdek'i ölçeklendirmek (76K → 300-500K)
3. Sentetik veri ile eğitim dataset'ini genişletmek (300 → 5K-10K)

---

## 📋 Yol Haritası

```
Faz 7.0:   Mamba Test          → ❌ Elendi (Türkçe yok)
Faz 7.0.5: Qwen Test           → ⏳ Hazırlanıyor
Faz 7.1:   Scaling             → ⏳ Planlandı
Faz 7.2:   Sentetik Veri       → ⏳ Planlandı
Faz 7.3:   Distillation        → ⏳ Planlandı
Faz 7.4:   Birleşmiş Pipeline  → ⏳ Planlandı
```

---

## 🔬 Faz 7.0: Mamba Test (Tamamlandı — ❌ Elendi)

### Test Edilen Model
- **Mamba-Codestral-7B-v0.1** (`/home/ayandon/KAPTAN/modeller/`)
- 7B parametre, SSM mimarisi, kod odaklı

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

## 🔬 Faz 7.0.5: Qwen3.5-9B Test (Hazırlanıyor)

### Test Edilecek Model
- **Qwen3.5-9B** (`/home/ayandon/KAPTAN/modeller/Qwen3.5-9B/`)
- 9B parametre, Transformer mimarisi, genel amaçlı
- Formatlar: Safetensors (19GB, 4 parça), Q4_K_M.gguf (5.3GB)
- **VRAM:** ~6GB (4-bit quantized)

### Test Seti (21 örnek)

| Kategori | Örnekler | Sayı | Beklenen Action |
|----------|----------|------|-----------------|
| **Compute** | `3+4=`, `5-2=`, `6*7=`, `12+34=`, `100-50=`, `9*9=`, `7*8=`, `15-7=` | 8 | compute |
| **Verify** | `5-3=2 doğru`, `4*4=16 doğru mu?`, `3+7=10 doğru`, `8-5=2 doğru mu?`, `6*6=35 doğru mu?` | 5 | verify |
| **Generate** | `merhaba`, `istanbul`, `bir gün`, `ahmet`, `nasılsın`, `bugün hava`, `kedi`, `araba` | 8 | generate |

### Test Yöntemi

1. Qwen'e her input'u ver (direkt prompt, ekstra instruction yok)
2. Çıktıyı analiz et:
   - **Compute:** Çıktı sadece sayı veya matematik sonucu içeriyor
   - **Verify:** Çıktı "evet", "hayır", "doğru", "yanlış" içeriyor
   - **Generate:** Çıktı doğal dil cümlesi
3. Beklenen action ile karşılaştır

### Başarı Kriterleri

| Metrik | Hedef | Geçme |
|--------|-------|-------|
| Compute accuracy | %100 | >%90 |
| Verify accuracy | >%80 | >%60 |
| Generate accuracy | >%80 | >%60 |
| Overall accuracy | >%85 | >%70 |

### Karar Kriterleri

| Sonuç | Aksiyon |
|-------|---------|
| ✅ Overall >%70 | Qwen öğretmen olarak uygun → Faz 7.1'e geç |
| ⚠️ Overall %50-70 | Qwen kısmen uygun, sentetik veri ile iyileştirilebilir |
| ❌ Overall <%50 | Qwen de uygun değil → farklı öğretmen ara |

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

---

## 📊 Faz 7.2: Sentetik Veri Üretimi (Planlandı)

### Hedef
300 → 5.000-10.000 örnek (16-33× artış)

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

### Riskler

| Risk | Olasılık | Azaltma |
|------|----------|---------|
| Mimari gap çok büyük | Orta | Logits-level ile minimize |
| Distillation transfer başarısız | Düşük | SpikingMamba metodolojisini takip et |
| Temperature scaling zor | Düşük | Grid search ile optimal T bul |

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

## 📁 Dosya Yapısı (Faz 7)

```
Cekirdek/
├── experiments/
│   ├── phase7_0_mamba_test.py      ✅ Tamamlandı (elendi)
│   ├── phase7_0_5_qwen_test.py     ⏳ Hazırlanıyor
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
    └── PHASE7_PLAN.md              📄 Bu dosya
```

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

## 🔮 Gelecek Vizyon

Faz 7 tamamlandığında:
- **Cekircek-H:** 300-500K parametre, >%50 unseen generalization
- **KAPTAN entegrasyonu:** Vicdan modülü olarak
- **Edge deployment:** Neuromorphic donanımda 21W hedefi
- **Araştırma yayını:** "Scaling SNNs with LLM Distillation"

---

> *"Öğretmenini akıllı seç, ama kendi yolunu çiz."*
