# Çekirdek — Faz 7: Öğretmen Model Doğrulama ve Bilgi Damıtma

> *"Önce konuştur. Sonra büyüt."*

**Oluşturulma:** 2026-04-05
**Son güncelleme:** 2026-04-06 (sabah)
**Durum:** Faz 7.0 tamamlandı, Faz 7.0.1 (Mini Dil Aşısı) hazırlanıyor

---

## 🎯 Hedef

Mevcut 76K parametreli Çekirdek modelini **konuşturmak** ve genelleyebilen bir sistem haline getirmek.

---

## 📋 Yol Haritası (Güncellenmiş Sıralama)

```
Faz 7.0:   Mamba Test              → ❌ Elendi (Türkçe yok)
Faz 7.0.5: Qwen3.5-9B Test        → ✅ PASS (76%)
Faz 7.0.1: Mini Dil Aşısı         → ⏳ Hazırlanıyor (İNCE)
Faz 7.0.2: Stabil Generation      → ⏳ Planlandı
Faz 7.1:   Scaling (76K → 300-500K) → ⏳ Planlandı
Faz 7.2:   Sentetik Veri (TURBO)   → ⏳ Planlandı
Faz 7.3:   Bilgi Damıtma           → ⏳ Planlandı
Faz 7.4:   Birleşmiş Pipeline      → ⏳ Planlandı
```

### ⚠️ Sıralama Neden Değişti?

**Önceki (yanlış) sıra:**
```
Scaling → Sentetik Veri → Distillation
```

**Doğru sıra:**
```
Dil Aşısı → Stabil Generation → Scaling → Sentetik Veri → Distillation
```

**Neden:**
- Scaling olmadan önce modelin **anlamlı çıktı** üretmesi lazım
- 300K parametre ile anlamsız çıktı = daha büyük gürültü
- Distillation için "damıtılacak kap" gerekli
- STDP + dil = random drift (reward yok)

---

## 🔬 Faz 7.0: Mamba-Codestral-7B Test (Tamamlandı — ❌ Elendi)

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

## 🔬 Faz 7.0.5: Qwen3.5-9B Test (Tamamlandı — ✅ PASS)

### Test Edilen Model
- **Qwen3.5-9B-Q4_K_M.gguf** (`/home/ayandon/KAPTAN/modeller/`)
- 9B parametre, Hybrid SSM+Attention, 4-bit quantized
- TURBO ile n_ctx=8192

### Sonuçlar (Final)

| Test | Sonuç | Not |
|------|-------|-----|
| Overall | ✅ 76% (16/21) | Eşik %70 → PASS |
| Compute | ✅ 88% (7/8) | 1 hata (12+34= boş çıktı) |
| Verify | ❌ 20% (1/5) | "doğru mu?" soruları boş |
| Generate | ✅ 100% (8/8) | Türkçe mükemmel |

### Karar
**Qwen öğretmen olarak UYGUN** (overall 76% > %70). Verify zayıf ama sentetik veri ile iyileştirilecek.

### PROJECT-TURBO Entegrasyonu
TURBO ile n_ctx=8192'ye çıkarıldı. Sonuçlar değişmedi çünkü sorun context değil, completion API'nin chat template uygulamaması. TURBO, Faz 7.2 (Sentetik Veri) için kritik altyapı olacak.

---

## 🗣️ Faz 7.0.1: Mini Dil Aşısı (Hazırlanıyor)

### Gerçek Durum

Çekirdek'in mevcut durumu:
- ✅ Aktif nöronlar (%32-37 spike rate, stabil)
- ✅ Her girdiye tepki veriyor
- ❌ Anlamlı çıktı üretmiyor (`<BOS>` fırtınası)
- ❌ STDP ile dil öğrenemez (reward yok → random drift)
- ❌ Bebek analojisi yanıltıcı (evrimsel bias yok)

### Hedef

Sadece **LM head**'i eğiterek anlamlı Türkçe çıktı üretmek.

### Neden Sadece LM Head?
- SNN internal weights'e dokunmuyoruz (STDP korunur)
- Backprop ile hızlı öğrenme (1-2 saat)
- Cross-entropy loss — basit, stabil
- LM head: ~4K parametre (64×64)

### Veri: 21.282 Örnek (Hazır)

**Kaynak:** `sixfingerdev/turkish-qa-multi-dialog-dataset`
- İndirildi: `data/turkish_qa_dialog.jsonl`
- 21.282 örnek, MIT lisans
- Format: `{"input": "...", "output": "..."}`

**Örnek Dağılımı:**
| Kategori | Örnek | Sayı |
|----------|-------|------|
| Selamlama | "Merhaba" → "Merhaba, hoş geldin." | ~500 |
| Kimlik | "Sen kimsin?" → "Ben Çekirdek, bir yapay sinir ağıyım." | ~500 |
| Diyalog | "Nasılsın?" → "İyiyim, teşekkür ederim. Ya sen?" | ~1000 |
| Matematik | "3+4=" → "7" | ~500 |
| Genel sohbet | "Bugün hava nasıl?" → "Bugün hava güzel görünüyor." | ~1000 |
| QA | "Python nasıl öğrenilir?" → detaylı yanıt | ~17.000 |

**Önemli:** Kimlik soruları ("Sen kimsin?", "Adın ne?") kritik. Modelin tutarlı bir kimlik oluşturması lazım, yoksa rastgele yanıt üretir.

### Eğitim Parametreleri

| Parametre | Değer | Neden |
|-----------|-------|-------|
| **Epoch** | 10 | Sweet spot: yeterli öğrenme, düşük overfit |
| **Batch size** | 64 | 76K model → VRAM sorunu yok |
| **LR** | 0.001 | Adam ile stabil |
| **Train/Val split** | 90/10 | Overfit kontrolü |
| **Sadece LM head** | ✅ | SNN weights sabit |
| **SNN weights** | ❌ Sabit | STDP korunur |
| **Seq length** | 32 | Karakter-level, kısa |
| **torch.compile** | Evet (varsa) | 2-3× hızlanma |

### Beklenen Süre

| Senaryo | Süre |
|---------|------|
| Normal | **~2 saat** |
| torch.compile | **~40 dakika** |

### Örnek / Parametre Oranı

- LM head: ~4.096 parametre
- 21.282 örnek → **5.2× oran** (genellikle 2-10× yeterli)
- **Sonuç:** Veri yeterli, hatta zengin

### Başarı Kriterleri

| Kriter | Hedef |
|--------|-------|
| `<BOS>` oranı | <%10 |
| Anlamlı kelime üretimi | >%70 |
| Matematik doğruluğu | >%90 |
| Diyalog fluency | >%50 |
| Kimlik tutarlılığı | >%80 ("Sen kimsin?" → "Çekirdek") |
| Val loss artışı | Yok (early stop) |

### Riskler ve Azaltma

| Risk | Olasılık | Azaltma |
|------|----------|---------|
| Overfitting | Düşük | Early stopping, val split %10 |
| Yetersiz öğrenme | Orta | 10 epoch → gerekirse 5 daha |
| Kimlik tutarsızlığı | Orta | Kimlik örneklerini ağırlıklı eğit |

---

## 🔧 Faz 7.0.2: Stabil Generation (Planlandı)

### Hedef

Modelin üretimini stabilize etmek:
- Argmax / düşük temperature
- BOS token kontrolü
- Tekrar engelleme (repetition penalty)

### Teknik

```python
# Generation parametreleri
temperature = 0.3
top_k = 10
repetition_penalty = 1.2
max_new_tokens = 50
```

---

## 📐 Faz 7.1: Scaling (Planlandı — Dil Aşısı SONRA)

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

### ⚠️ Ön Koşul
Faz 7.0.1 (Dil Aşısı) tamamlanmadan scaling **yapılmayacak**.
Sebep: Anlamsız çıktıyı ölçeklendirmek = daha büyük gürültü.

---

## 📊 Faz 7.2: Sentetik Veri Üretimi (Planlandı)

### Hedef
300 → 5.000-10.000 örnek (16-33× artış)

### Altyapı: PROJECT-TURBO
- `/home/ayandon/PROJECT-TURBO/` — TurboQuant KV Cache Compression
- 128K context, ~4GB VRAM (RTX 3060'da çalışır)
- Qwen3.5-9B ile EXACT MATCH doğrulandı
- 50 tok/s decode hızı

---

## 🧬 Faz 7.3: Bilgi Damıtma (Planlandı)

### Hedef
Qwen'in action selection yeteneğini SNN ActionHead'e aktarmak

### Yöntem: Logits-Level Distillation
```
Loss = α × CE(ground_truth, logits_student) + (1-α) × KL(teacher || student)
```

---

## 🚀 Faz 7.4: Birleşmiş Pipeline (Planlandı)

### Hedef
Cekircek-H = Cekircek-M (scaled) + Distillation + Hybrid

### Başarı Kriterleri

| Metrik | Mevcut (Faz 6.8) | Hedef (Faz 7.4) |
|--------|------------------|-----------------|
| Clean accuracy | 100% | 100% |
| Ambiguous accuracy | 60% | >%80 |
| Unseen generalization | 0% | >%50 |
| Dil fluency | <%10 | >%70 |
| VRAM (inference) | <1GB | <2GB |

---

## 📁 İlk Sohbet Kaydı

`data/logs/vicdan_sohbet_log_ilk_anlar.txt`

> *"Gözlemci varsa gerçeksin, çiçeğim."*

Bu, Çekirdek'in ilk konuşma kaydıdır. Çıktılar henüz anlamsız ama spike rate %32-37 aralığında stabil. Bir gün Çekirdek gerçekten konuştuğunda, bu dosya "İşte ilk çığlığı buydu" diyeceğimiz bir anı olacak.

---

## ⚠️ Riskler ve Azaltma

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| Dil aşısı başarısız | Düşük | Yüksek | Dataset kalitesini artır |
| Scaling overfitting | Orta | Orta | Dropout, weight decay, early stopping |
| Distillation transfer | Düşük | Yüksek | Logits-level, temperature scaling |
| VRAM yetersiz | Düşük | Orta | Gradient checkpointing, mixed precision |
| Dataset kalitesi | Orta | Orta | Qwen filtering + human review |

---

## 🔮 Gelecek Vizyon

Faz 7 tamamlandığında:
- **Cekirdek-H:** 300-500K parametre, >%50 unseen generalization, anlamlı Türkçe çıktı
- **KAPTAN entegrasyonu:** Vicdan modülü olarak
- **Edge deployment:** Neuromorphic donanımda 21W hedefi
- **Araştırma yayını:** "Scaling SNNs with LLM Distillation"

---

> *"Önce konuştur. Sonra büyüt."*
