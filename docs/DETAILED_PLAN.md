# Çekirdek — Detaylı Plan

> Tüm phase'lerin detaylı açıklamaları, deney sonuçları ve mimari kararlar.

---

## Bölüm 1: SNN Temelleri (Phase 0-5)

### 🎯 Hedef

Transformer'ların statik ağırlık sorununu çözen, **online öğrenen**, **spike-based** bir dil modeli.

| Özellik | Hedef |
|---------|-------|
| Parametre | 70K → 10M (kademeli) |
| Aktif nöron oranı | %10-20 |
| Online öğrenme | Context-Gated R-STDP ile |
| VRAM | < 4GB (RTX 3060) |
| Tokenizasyon | Karakter-level (~64) |

---

### Phase 0: Ortam Kurulumu ✅

- Python 3.11 venv oluşturuldu
- Karar: SpikingJelly KULLANMAYACAĞIZ — kendi minimal SNN'imizi yazıyoruz
- PyTorch GPU desteği doğrulandı

---

### Phase 1: Temel SNN Eğitimi ✅

- LIFNeuron implementasyonu (surrogate gradient)
- SpikingLinear, SpikingSelfAttention, SpikingFFN
- SpikingLM modeli: 76,480 parametre
- Spike oranı: %16.83
- Eğitim: loss 2.33 → 2.09 (5 epoch)

---

### Phase 2: Reward-Modulated STDP ✅

| Deney | Sonuç |
|-------|-------|
| Dense reward | ✅ loss -0.2% |
| LR Scaling (0.0005→0.01) | ✅ 12× artış |
| Multi-Step (1→10 tekrar) | ✅ 5× artış |
| Generalization | ❌ Overfitting |
| Attention V'de STDP | ❌ Etkisiz |

**Önemli Bulgu:** STDP Backprop'tan daha az overfit yapar ama hâlâ overfit.

---

### Phase 3: Ternary Spike + Context Gating ✅

| Deney | Sonuç |
|-------|-------|
| Ternary {-1,0,+1} | ✅ Binary'den 3.4× etkili |
| Amplitude=2.0 | ✅ -1.9% STDP |
| Decay=0.1 | ✅ -4.0% STDP |
| Context Gating (surprise=0.3) | ✅ **21× artış** |

**Altın Kural:**
> Yüksek seyreklik, zayıf credit assignment altında öğrenmeyi boğar.
> Optimal öğrenme; düşük eşikli ternary sinyallerin, yüksek genlikli darbelerle
> sinapsları dövmesiyle gerçekleşir.
> **Optimal: threshold=0.3, amplitude=2.0, decay=0.1, surprise_threshold=0.5**

---

### Phase 4: Consolidation & Multi-Task ✅

| Deney | Sonuç |
|-------|-------|
| Consolidation α=0.05 | ✅ -3.3% loss |
| Multi-Task Türkçe | ✅ -3.4% |
| Multi-Task Matematik | ❌ +11.1% (öğrenilemedi) |
| Sleep Consolidation | ✅ Forgetting +12.1% → +1.5% |

**Önemli Bulgu:** Sleep-Mediated Consolidation catastrophic forgetting'ı önler.

---

### Phase 5: VSA Denemeleri ❌

| Deney | Sonuç |
|-------|-------|
| VSA Binding | ❌ Commutative, işlem ayrımı yok |
| VSA + Contrastive | ❌ 0% math accuracy |
| VSA Sekans | ✅ 60% decode accuracy |

**Sonuç:** VSA binding matematiksel ilişki KODLAMAZ. Harici sembolik motor gerekli.

---

## Bölüm 2: Hybrid Neuro-Symbolic (Phase 6)

### 🎯 Hedef

SNN'e matematik yaptırmaya çalışmak yerine:
- SNN → **aksiyon seçimi** (hangi araç?)
- Symbolic Engine → **deterministik hesaplama**
- STDP Feedback → **tool selection öğrenme**

---

### Phase 6.0: Hybrid Backprop LM Head

- LM head için backprop, internal için STDP
- **Sonuç:** 100% train, 0% generalization
- **Neden:** LM head lookup table oluyor, reasoning yapmıyor

---

### Phase 6.1: Intent Classification

- LM head'i intent classifier'a çevirdik
- **Seen:** 83%, **Unseen:** 40%
- **Karar:** Intent extraction'ı SNN'e YAPTIRMAYACAĞIZ — string parsing zaten %100

---

### Phase 6.2: Task Gating + Confidence Routing ✅

- TaskClassifier: 64→2 (130 param)
- **Seen:** 100%, **Unseen:** 100%, **Pipeline:** 100%
- **Neden çalıştı:** Kategorizasyon, genelleme değil. SNN pattern tanıma için optimize.

---

### Phase 6.3: STDP Tool-Use Reinforcement ✅

- ActionHead ile STDP decision reinforcement
- **Seen:** 100%, **Unseen:** 100%, **Pipeline:** 100%
- **Önemli:** Supervised STDP + reward scaling formülü bulundu

---

### Phase 6.4: Outcome-Based STDP (Label-Free) ✅

- Label kaldırıldı, reward sonuçtan üretiliyor
- **Seen:** 100%, **Unseen:** 100%, **Pipeline:** 100%
- **Formül:** `ΔW = lr × reward × (chosen_onehot - probs) × pre^T`

---

### Phase 6.5: Ambiguity Injection ⚠️

- Belirsiz input'lar eklendi
- **Clean:** 100%, **Ambiguous:** 50%
- **Neden:** SNN representation'ı yüzey-level, semantik anlayış yok

---

### Phase 6.6: Saliency-Guided Learning ✅

- Op boost=10.0 + outcome-based STDP
- **Clean:** 100%, **Unseen ambiguous:** 100%, **Training ambiguous:** 60%
- **Bulgu:** Keyword boost performansı düşürüyor, sadece operatör boost güvenilir

---

### Phase 6.7: Multi-Step Reasoning (3 Action) ⚠️

- 3 action: generate, compute, verify
- **Compute:** 100%, **Verify:** 0%, **Generate:** 100%, **Pipeline:** 80%
- **Neden verify başarısız?** Training data az, pattern overlap, keyword boost yok

---

### Phase 6.8: Keyword-Enhanced Action Selection ✅

- Verify keyword boost (5.0): "doğru", "yanlış", "mı/mi"
- Verify training data: 4 → 20 örnek
- **Compute:** 100%, **Verify:** 100%, **Generate:** 100%, **Pipeline:** 100%

**Final Mimari:**
```
Input → SNN + Saliency (op_boost=10, verify_boost=5)
  ↓
ActionHead (3 classes: generate, compute, verify)
  ├─ generate → SNN text generation
  ├─ compute  → Symbolic Engine → result
  └─ verify   → Symbolic Engine → compare → True/False response
```

---

## 📊 Tüm Deney Sonuçları

| Phase | Yöntem | Sonuç |
|-------|--------|-------|
| 2.5 | Dense reward | ✅ -0.2% |
| 2.6 | LR Scaling | ✅ 12× artış |
| 3.0 | Ternary spike | ✅ 3.4× |
| 3.2 | Context Gating | ✅ **21×** |
| 4.0 | Consolidation α=0.05 | ✅ -3.3% |
| 4.3 | Sleep Consolidation | ✅ Forgetting +1.5% |
| 5.0 | VSA Binding | ❌ |
| 6.0 | Hybrid Backprop | ❌ 0% generalization |
| 6.2 | Task Gating | ✅ 100% |
| 6.4 | Outcome STDP | ✅ 100% |
| 6.5 | Ambiguity | ⚠️ 50% |
| 6.6 | Saliency-Guided | ✅ 100% |
| 6.7 | 3-Action | ⚠️ 80% |
| **6.8** | **Keyword-Enhanced** | **✅ 100%** |

---

## 🔑 Öğrenilen Dersler (Tümü)

### VİCDAN_HEBBIAN'dan:
1. Pretrained modele sonradan ekleme çalışmaz
2. STDP tek başına gürültü → Reward-modulated olmalı
3. Küçük başla → 70K parametre ile başla
4. Sıralama kritik

### VİCDAN_SPIKE'dan:
5. Sorun kapasite değil, sinyal
6. Seyrek reward = öğrenme yok
7. Dense reward = öğrenme var
8. Ternary spike {-1,0,+1} → Binary'den 3.4× etkili
9. Amplitude=2.0 → Düşük spike rate'i yüksek güçle kompanse et
10. Decay=0.1 → Hızlı unutma, taze başlangıç
11. Context Gating → 21× öğrenme artışı
12. Consolidation α=0.05 → Kalıcı öğrenme
13. Sleep consolidation → Forgetting +1.5%
14. STDP istatistiksel pattern öğrenir, sembolik mantık değil
15. VSA binding matematiksel ilişki KODLAMAZ
16. Batched backprop interference'ı çözer
17. LM head lookup table olur, reasoning yapmaz
18. Keyword boost > operatör boost (verify için)
19. Saliency tek başına yetersiz → ActionHead yeniden eğitilmeli
20. SNN reasoning yapmaz, karar verir

---

## 🔮 Gelecek Vizyon

- **Phase 6.9:** Online user feedback ile sürekli öğrenme
- **Phase 7:** Yeni action'lar (search, memory, code_execution)
- **Phase 8:** Daha büyük SNN (d_model=128, n_layer=4)
- **Edge Deployment:** Neuromorphic donanımda 21W hedefi

---

> *"Öğrenme, her şeyi ezberlemek değil; belirsizliğin (entropi) içinde anlam aramaktır."*
