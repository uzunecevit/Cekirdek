# Çekirdek — Faz 7: Hybrid Cognitive Router

> *"SNN üretmez — ama neyin üretileceğine karar verir."*

**Oluşturulma:** 2026-04-05
**Son güncelleme:** 2026-04-06 (öğleden sonra)
**Durum:** Faz 7.0.1 kapatıldı (SNN dil üretimi yetersiz). Yeni yön: Multi-Action RL (Phase 6.7 genişletilmiş)

---

## 🎯 Hedef

SNN'i bir **bilişsel yönlendirici (cognitive router)** olarak mükemmelleştirmek:
- Input'u analiz et → doğru action'ı seç → doğru aracı çağır

---

## 📋 Yol Haritası

```
Faz 7.0:   Mamba Test              → ❌ Elendi (Türkçe yok)
Faz 7.0.5: Qwen3.5-9B Test        → ✅ PASS (76%)
Faz 7.0.1: Mini Dil Aşısı         → ❌ KAPATILDI (SNN dil üretemez)
Faz 6.7.2: Multi-Action RL (4)    → ⏳ Hazırlanıyor
Faz 7.1:   Hybrid Pipeline        → ⏳ Planlandı
Faz 7.2:   LLM Entegrasyonu       → ⏳ Planlandı
Faz 7.3:   Memory Module          → ⏳ Uzun vadeli
```

---

## 🔬 Faz 7.0.1: Mini Dil Aşısı (Kapatıldı — ❌)

### Ne Yaptık?
- Sadece LM head eğitimi (4.160 parametre)
- 6.504 temiz Türkçe çift ile eğitim
- 10 epoch, ~7 saniye/epoch

### Sonuçlar

| Metrik | Başlangıç | Son |
|--------|-----------|-----|
| Train loss | 3.57 | 3.12 (-12.6%) |
| Val loss | 3.38 | 3.14 (-7.1%) |
| Çıktı kalitesi | ❌ Çöp | ❌ Çöp |

### Neden Başarısız Oldu?

1. **Hidden state'ler dil için optimize değil** — SNN pattern/spike dynamics için eğitildi
2. **LM head sadece 4.160 parametre** — bu kadar az parametre ile dil üretilemez
3. **Loss hâlâ 3.14** — model hâlâ rastgele tahmin yapıyor (random ≈ 4.16)

### Öğrenilen Ders

> **SNN dil üretmez — ama neyin üretileceğine karar verir.**

Bu, Phase 5'te VSA ile matematik yapmaya çalışmakla **aynı hata**:
- Phase 5: "SNN'e matematik yaptıralım" → ❌ Başarısız
- Phase 7: "SNN'e dil ürettirelim" → ❌ Başarısız

Her iki seferde de **mimarinin doğasını** görmezden geldik.

---

## 🧠 SNN'in Gerçek Gücü

| Görev | SNN Yapabilir mi? | Literatür | Bizim Deney |
|-------|-------------------|-----------|-------------|
| Pattern tanıma | ✅ Evet | ✅ | ✅ Phase 3-4: 21× artış |
| Tool selection | ✅ Evet | ✅ (ALoRS, DDM) | ✅ Phase 6.8: %100 |
| Task switching | ✅ Evet | ✅ (Lateral inhibisyon) | ✅ Phase 6.2: %100 |
| Dil üretimi | ❌ Hayır | ❌ (k² scaling) | ❌ Phase 7.0.1: Loss 3.14 |
| Math reasoning | ❌ Hayır | ❌ (sembolik değil) | ❌ Phase 5: VSA başarısız |

---

## 🏗️ Faz 6.7.2: Multi-Action RL (4 Action) — Hazırlanıyor

### Mimari

```
Input → SNN + Saliency → ActionHead (4 classes)
  ├─ compute  → Symbolic Engine → result
  ├─ verify   → Symbolic Engine → True/False
  ├─ generate → LLM (Qwen) → text
  └─ explain  → LLM (Qwen) → açıklama
```

### Action Set

| Action | Tool | Örnek Input |
|--------|------|-------------|
| `compute` | Symbolic Engine | `"3+4="`, `"12*5="` |
| `verify` | Symbolic Engine | `"2+2=5 doğru mu?"`, `"10-3=7 doğru"` |
| `generate` | LLM | `"merhaba"`, `"nasılsın?"` |
| `explain` | LLM | `"3+4 neden 7 eder?"`, `"Nasıl çalışıyorsun?"` |

### Priority-Based Reward

Aynı input birden fazla kategoriye girebilir. Priority sistemi:

```python
PRIORITY = ["compute", "verify", "explain", "generate"]

def get_reward(action, input):
    if is_math_expression(input):
        correct = "compute"
    elif is_verification(input):
        correct = "verify"
    elif is_why_question(input):
        correct = "explain"
    else:
        correct = "generate"
    
    return +1 if action == correct else -1
```

### Training Parametreleri

| Parametre | Değer | Neden |
|-----------|-------|-------|
| LR | 0.02-0.03 | 4 action → daha hassas |
| Temperature | 1.2 | Erken aşamada daha iyi exploration |
| Epochs | 50-100 | 2 action'dan daha uzun |
| Consolidation | Her 5 epoch | Fast weights → static |

### Beklenen Davranış

- İlk epoch'lar: kaotik seçim, reward düşük
- Sonra: pattern separation, hızlı convergence
- Hedef: >%80 accuracy (unseen)

### Başarı Kriterleri

| Kriter | Hedef |
|--------|-------|
| Compute accuracy | >%95 |
| Verify accuracy | >%80 |
| Generate accuracy | >%80 |
| Explain accuracy | >%70 |
| Overall | >%80 |

---

## 🚀 Faz 7.1: Hybrid Pipeline (Planlandı)

### Mimari

```
Input → SNN → Action Selection
  ├─ compute  → Engine → result
  ├─ verify   → Engine → bool
  ├─ generate → LLM (Qwen) → text
  └─ explain  → LLM (Qwen) → açıklama
```

### LLM Routing

```python
if action == "generate":
    prompt = f"Kullanıcı: {input}\nCevap:"
elif action == "explain":
    prompt = f"Aşağıdaki soruyu açıkla:\n{input}\nAçıklama:"
```

---

## 🧬 Faz 7.2: LLM Entegrasyonu (Planlandı)

### Hedef
- Qwen3.5-9B'yi `generate` ve `explain` action'ları için entegre et
- TURBO ile KV cache optimizasyonu
- Response kalitesi ve hız ölçümü

---

## 📁 Faz 7.3: Memory Module (Uzun Vadeli)

### Hedef
- Hafıza modülü hazır olunca `memory` action'ı ekle
- SNN zaten 4 action'ı öğrenmiş olacak, 5'inci kolay eklenebilir

### Neden Şimdi Değil?
- Memory action'ı şimdi eklemek = ölü sinaps
- SNN "memory" seçerse gidecek yer yok
- Training sırasında kafa karışıklığı yaratır

---

## ⚠️ Riskler ve Azaltma

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| 4 action öğrenilemez | Orta | Yüksek | LR düşür (0.02), temperature artır (1.2) |
| Reward çelişkisi | Orta | Orta | Priority-based reward |
| LLM entegrasyon gecikmesi | Düşük | Orta | Önce SNN action selection'ı tamamla |
| Memory modülü gecikmesi | Yüksek | Düşük | Şimdi ekleme, sonra ekle |

---

## 🔮 Gelecek Vizyon

Faz 7 tamamlandığında:
- **Çekirdek-H:** 4-action cognitive router, >%80 accuracy
- **Hybrid Pipeline:** SNN → action → engine/LLM routing
- **KAPTAN entegrasyonu:** Vicdan modülü olarak
- **Edge deployment:** Neuromorphic donanımda 21W hedefi

---

> *"SNN üretmez — ama neyin üretileceğine karar verir."*
