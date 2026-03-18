# TMDB Veri Gönderim ve Doğrulama Süreci - Tam Dokümantasyon

## Özet

OCR'dan gelen isim ve film bilgileri TMDB'ye gönderiliyor, ters doğrulama ile check ediliyor, gerekirse Gemini'ye gidiyor, sonuçlar işlenip rapor oluşturuluyor.

---

## 1. BAŞLANGIÇ: OCR'DEN GELEN VERİ

OCR'den frameler geldi, OneOCR ile okuduk, düzelttik, elimizde:
- Doğrusu ile yanlışı ile bir grup **isim** (oyuncular, yönetmenler)
- **Sıfatlar** (teknik ekip)
- **Film adı** (genellikle dosya adından)
- **OCR satırları** (her satırda metin + güven skoru)

Bu veri `pipeline_runner.py` içinde `[5/6] CREDITS_PARSE` aşamasından sonra elimizde:
```python
cdata = {
    "film_title": "KADIN VE DENİZCİ",
    "cast": [{"name": "Tommy Rettig", "confidence": 0.92}, ...],
    "directors": ["William Wellman"],
    "crew": [...],
    "total_actors": 15,
    "total_crew": 8
}
```

---

## 2. TMDB'YE GÖNDERİM ÖNCESİ: GEMİNİ CAST EXTRACT (HER ZAMAN ÇALIŞIR)

**Dosya:** `core/gemini_cast_extractor.py`
**Tetiklenme:** TMDB'den önce **HER ZAMAN** çalışır (satır 445-455, `pipeline_runner.py`)

### Ne Yapar?
OCR satırlarını Gemini'ye gönderip oyuncu/yönetmen listesini temizletir:
- Multi-column layout'ları doğru parse eder
- Stunt/dublör rollerini atar
- Film başlığını anchor kullanarak OCR hatalarını düzeltir

### Örnek:
```
OCR: "Tommy Murvyn Douglas Rettig Vye Spencer"
Gemini: ["Tommy Rettig", "Murvyn Vye", "Douglas Spencer"]
```

### Anchor Koruması:
Dosya adından gelen film başlığı **kutsal**dır. Gemini film başlığını değiştirirse:
```python
if cdata["film_title"] != film_title_from_filename:
    cdata["_gemini_suggested_title"] = cdata["film_title"]
    cdata["film_title"] = film_title_from_filename  # Dosya adı geri yüklenir
```

**Sonuç:** Gemini'nin temizlediği cast/crew listesi `cdata`'ya yazılır, ama dosya adı anchor'ı korunur.

---

## 3. İSİM DOĞRULAMA: 3 KATMANLI SİSTEM

**Dosya:** `core/name_verify.py`
**Aşama:** `[6/6] NAME_VERIFY` (satır 457-517, `pipeline_runner.py`)

### Katman 1: Blacklist + Yapısal Kontroller
- Şirket isimleri atılır: "Productions", "Ltd", "Inc", "Canal", "MK2" vs.
- Anormal cast sayısı kontrolü: >500 oyuncu → uyarı
- Sesli harf oranı kontrolü: %15-80 arasında olmalı
- Büyük harf dizileri: "AABBCCDD" gibi → reddedilir

### Katman 2: NameDB (Türk İsim Veritabanı)
- Türk isimlerini doğrular
- Türk karakterleri düzeltir (ç, ğ, ı, ö, ş, ü)
- Yaygın isimlere güven puanı boost verir

### Katman 3: TMDB Person Search
- TMDB'de kişi araması yapar (`search/person`)
- Kişinin filmografisini kontrol eder
- Bulunan isimler için canonical (resmi) isim döndürür

**Sonuç:** Her isim için `verification_log` oluşturulur:
```python
cdata["_verification_log"] = {
    "Tommy Rettig": {"method": "tmdb_person", "confidence": 1.0},
    "Murvyn Vye": {"method": "namedb", "confidence": 0.85},
    ...
}
```

---

## 4. TMDB FİLM ARAMASI: 2 STRATEJİ

**Dosya:** `core/tmdb_verify.py` → `_find_tmdb_entry` (satır 554-757)
**Tetiklenme:** Sadece profile'da `tmdb_enabled=true` ise (satır 519-586, `pipeline_runner.py`)

### Strateji 1: Başlık + Oyuncularla Ara
1. Film başlığıyla `search/multi` API'sini çağır
2. Top 10 sonuçta `movie/{id}/credits` veya `tv/{id}/credits` çek
3. OCR'daki oyuncuları TMDB cast ile karşılaştır

**Kabul Kriterleri:**
- Film adı + 2 oyuncu eşleşmesi → **%100 güven**
- Film adı + 1 oyuncu + yönetmen → **%100 güven**
- Film adı + yönetmen (oyuncu yok) → **kabul**

**Başlık Varyantları:**
```python
# "Madam" → ["Madam", "Madame"]
# "Dövüş Kulübü" → ["Dövüş Kulübü", "Fight Club"]
```

### Strateji 1b: Orijinal Başlıkla Ara (Fallback)
Eğer Strateji 1 başarısız olursa VE XML sidecar'dan `original_title` varsa:
```python
# Türkçe başlık: "Dövüş Kulübü"
# Orijinal başlık: "Fight Club"
→ TMDB'ye "Fight Club" ile tekrar ara
```

**Tetiklenme:** Sadece `original_title` doluysa ve normalize edilmiş hali film başlığından farklıysa (satır 622-666)

**Log:** `[TMDB] Orijinal başlık TMDB'ye iletiliyor: 'Fight Club'`

### Strateji 2: Sadece Oyuncularla Ara (Başlık Yoksa)
1. Her oyuncu/yönetmen için `search/person` çağır
2. Bulunan kişi için `person/{id}/combined_credits` çek
3. Aynı filmde çalışan 3+ kişi bul

**Kabul Kriteri:**
- 3+ kişi (oyuncu + yönetmen karışık) aynı filmde → **%100 güven**

**Not:** Yönetmenler `combined_credits` içinde **crew** bölümünde aranır.

### Fuzzy Matching
```python
# RapidFuzz kütüphanesi kullanılır
# Oyuncu matching: %82 eşik
# Başlık varyantları: %85 eşik
# Fallback: normalize edilmiş exact matching
```

**Sonuç:** TMDB'de film bulunursa:
```python
tmdb_result = {
    "matched_id": 550,
    "matched_title": "Fight Club",
    "matched_via": "title",  # veya "cast_only"
    "hits": 8,  # kaç oyuncu eşleşti
    "misses": 2,
    "cast": [...],
    "crew": [...]
}
```

---

## 5. TERS DOĞRULAMA (REVERSE VALIDATION): SKORLAMA SİSTEMİ

**Dosya:** `core/tmdb_verify.py` → `_reverse_validate` (satır 836-1111)
**Amaç:** "Bu TMDB filmi gerçekten bizim filmimiz mi?" kontrolü

### Skorlama Kategorileri

#### A) Başlık Eşleşmesi (max +2.5 / max -4.0)
**Dil Kontrolü:**
```python
# OCR başlığı Türkçe mi? (ç, ğ, ı, ö, ş, ü var mı?)
# TMDB başlığı Türkçe mi?
# TMDB original_title Türkçe mi?
→ Aynı dildeki başlığı karşılaştır
```

**Puanlama:**
- Tam eşleşme (normalize) → **+2.5**
- Fuzzy %90+ → **+2.0**
- Fuzzy %80-89 → **+1.0**
- Kısmi eşleşme (substring) → **+0.5**
- Farklı dil → **0.0/0.0** (atlanır)
- Aynı dil, fuzzy <60% → **-4.0** (ağır ceza)
- Aynı dil, fuzzy <70% → **-1.0**

#### B) Yönetmen Eşleşmesi (max +2.5 / max -2.5)
- Fuzzy %95+ → **+2.5**
- Fuzzy %80+ → **+1.5**
- Eşleşme yok → **-2.5**

#### C) Oyuncu Eşleşmesi (max +6.0 / max -3.0)
**Oran Bazlı:**
- %50+ eşleşme → **+3.0**
- %30-49 → **+2.0**
- %15-29 → **+1.0**
- %5-14 → **+0.5**

**Mutlak Sayı Bonusu:**
- 8+ oyuncu hit → **+3.0**
- 5-7 hit → **+2.0**
- 3-4 hit → **+1.0**

**Ceza (sadece hit <3 ise):**
- <5% oran → **-3.0**
- <10% → **-2.0**
- <15% → **-1.0**

#### D) Yıl Eşleşmesi (max +2.0 / max -3.0)
- 0-2 yıl fark → **+2.0**
- 3-5 yıl → **+1.0**
- 6-10 yıl → **+0.5**
- 10-20 yıl → **-1.0 ile -2.0 arası**
- 20+ yıl → **-3.0**

### Dinamik Eşik
```python
# Her kategori için max pozitif skor topla
# Sadece aktif kategoriler sayılır (örn: yıl bilinmiyorsa atlanır)
max_positive = sum([title_max, director_max, cast_max, year_max])
threshold = max_positive * 0.40  # %40 eşiği

# Örnek: Tüm kategoriler aktif
# max_positive = 2.5 + 2.5 + 6.0 + 2.0 = 13.0
# threshold = 13.0 * 0.40 = 5.2

if score >= threshold:
    return True  # KABUL
else:
    return False  # RED
```

### Test Senaryoları
**Test dosyası:** `tests/test_reverse_validate.py`

**REVVAL-01: Yanlış Film**
```python
OCR: "KULÜBE" + ["Brad Pitt", "Edward Norton"]
TMDB: "Dövüş Kulübü" + ["Brad Pitt", "Edward Norton", ...]
→ Başlık fuzzy <60% → -4.0 ceza
→ Oyuncular eşleşse de başlık cezası ağır
→ RET
```

**REVVAL-02: Doğru Film**
```python
OCR: "FIGHT CLUB" + ["Brad Pitt", "Edward Norton"]
TMDB: "Fight Club" + ["Brad Pitt", "Edward Norton", ...]
→ Başlık %90+ → +2.0
→ Oyuncular 50%+ → +3.0
→ KABUL
```

**Sonuç:** Ters doğrulama başarılıysa `tmdb_result` döndürülür, başarısızsa `None`.

---

## 6. TMDB SONUÇ İŞLEME: LOCK vs REFERANS MODU

**Dosya:** `pipeline_runner.py` (satır 536-581)

### LOCK Modu (matched_via="title")
**Şart:** Profile `FilmDizi-Hybrid` VE film başlığıyla eşleşme

**Ne Yapar:**
1. OCR verisini **SİLER**
2. TMDB cast/crew'u **kanonik veri** olarak yazar
3. Max 15 oyuncu alır
4. "uncredited" ve stunt rollerini atar

**QA Kontrolü:**
```python
# OCR'da olup TMDB'de olmayan oyuncular var mı?
from core.credits_qa import check_missing_actors
qa = check_missing_actors(ocr_lines, tmdb_cast)
if qa.missing_actors:
    cdata["cast_qa"] = qa.to_dict()
    # Log: "OCR'da 3 oyuncu TMDB'de yok: ['Ahmet Yılmaz', ...]"
```

**Sonuç:** TMDB verisi `cdata`'ya yazılır, OCR verisi kaybolur.

### Referans Modu (matched_via="cast_only" veya başlık eşleşmedi)
**Ne Yapar:**
1. OCR verisi **KORUNUR**
2. TMDB verisi **referans** olarak saklanır

```python
cdata["_tmdb_film_match"] = {
    "title": "Fight Club",
    "id": 550,
    "hits": 8,
    "misses": 2
}
cdata["_tmdb_cast_ref"] = [...]  # referans cast
cdata["_tmdb_crew_ref"] = [...]  # referans crew
```

**Sonuç:** Her iki veri de saklanır, OCR verisi esas alınır.

---

## 7. TMDB BAŞARISIZ OLURSA: BLOK2 + LLM FİLTRE + GOOGLE VI

### BLOK2: VLM Derin Okuma
**Dosya:** `pipeline_runner.py` (satır 587-622)
**Tetiklenme:** TMDB eşleşmedi VE `blok2_enabled=true` VE VLM mevcut

**Ne Yapar:**
- Tüm frameleri VLM (Qwen 2.5) ile **derin okuma**
- OCR'ın kaçırdığı isimleri yakalar
- Sonuçları OCR ile fuzzy merge eder

**Sonuç:** `ocr_lines` güncellenip `CreditsParser` tekrar çalışır.

### LLM Cast Filtresi
**Dosya:** `core/llm_cast_filter.py`
**Tetiklenme:** TMDB eşleşmedi VE `llm_filter_enabled=true` (satır 623-648)

**Ne Yapar:**
- Sahte isimleri Gemini/Ollama ile filtreler
- Batch işleme (50 item)
- Kabul edilen isimlere confidence boost verir

**Örnek:**
```python
# OCR: ["Tommy Rettig", "HOLLYWOOD PICTURES", "Best Boy Grip"]
# LLM: ["Tommy Rettig"]  # Şirket ve sıfatlar atıldı
```

### Google Video Intelligence
**Dosya:** `core/google_video_intelligence.py`
**Tetiklenme:** Akıllı tetik (satır 650-679, `pipeline_runner.py`)

**Şartlar:**
- TMDB eşleşmedi VE
- (Düşük çözünürlük VEYA non-standard font VEYA uzun segment süresi)

**Ne Yapar:**
- Entry/exit segmentlerini Google VI API'ye gönderir
- OCR sonuçlarıyla merge eder

**Sonuç:** Ekstra OCR satırları `ocr_lines`'a eklenir.

---

## 8. GEMİNİ ÖZETLEYİCİ: YABANCI DİL İKİ ADIMLI İŞLEM

**Dosya:** `core/gemini_summarizer.py`
**Tetiklenme:** Export aşamasında, TMDB'den özet gelmişse

### Türkçe İçerik
- `gemini-2.5-flash` ile direkt Türkçe özet (80-100 kelime)

### Yabancı Dil (2 Adım)
**Adım 1: Orijinal Dilde Özet**
- `gemini-2.5-pro` kullanılır
- Filmin orijinal dilinde özet üretilir

**Adım 2: Türkçe Çeviri**
- `gemini-2.5-flash` kullanılır
- TMDB cast referansı ile isimler korunur
- Türkification yapılmaz (örn: "Tom Hanks" → "Tom Hanks", "Tom Hankis" değil)

### Özellikler
- 80-100 kelime sınırı
- **Zero suspense:** Sonu/spoiler dahil
- Yabancı isimlerin orijinal yazımı korunur

**Sonuç:** Özet `cdata["summary"]` olarak kaydedilir.

---

## 9. EXPORT VE RAPOR OLUŞTURMA

**Dosya:** `core/export_engine.py`
**Aşama:** Pipeline sonunda (satır 680-750, `pipeline_runner.py`)

### Üretilen Dosyalar

#### A) JSON Rapor
**Dosya:** `{film_id} {film_adı}.json`
```json
{
  "file_info": {
    "original_filename": "...",
    "output_name": "1949-0039-1 KADIN VE DENİZCİ",
    "film_id": "1949-0039-1"
  },
  "credits": {
    "cast": [
      {"name": "Tommy Rettig", "role": "Oyuncu", "confidence": 0.92}
    ],
    "crew": [
      {"name": "William Wellman", "role": "Yönetmen"}
    ]
  },
  "tmdb_match": {
    "matched_id": 12345,
    "matched_title": "...",
    "confidence": 100
  },
  "verification_log": {...},
  "summary": "..."
}
```

#### B) TXT Rapor (Türk Dilbilgisi Kurallarıyla)
**Dosya:** `{film_id} {film_adı}.txt`
```
FİLM ADI: KADIN VE DENİZCİ (1949)

OYUNCULAR:
  1. Tommy Rettig
  2. Murvyn Vye
  ...

YÖNETMEN: William Wellman

TEKNİK EKİP:
  Görüntü Yönetmeni: John Seitz
  ...

ÖZET:
Bir kadın ve bir denizci... (80-100 kelime)
```

**Türkçe Büyük Harf Kuralları:**
```python
# "istanbul" → "İstanbul" (İ ile başlar)
# "KADIN VE DENİZCİ" → doğru (İ değil I)
```

#### C) User-Friendly Rapor
**Dosya:** `{film_id} {film_adı}_friendly.txt`
Daha okunabilir format, less teknik detay.

### Çıktı Dizini
```
output/
├── 1949-0039-1 KADIN VE DENİZCİ.json
├── 1949-0039-1 KADIN VE DENİZCİ.txt
└── 1949-0039-1 KADIN VE DENİZCİ_friendly.txt
```

---

## 10. DATABASE KOPYALAMA

**Dosya:** `pipeline_runner.py` → `_write_database` (satır 529-640)
**Tetiklenme:** Her pipeline sonunda (config: `database_enabled=true`)

### Hedef Dizin
```
D:\DATABASE\
└── FilmDizi\  (veya profile adı)
    └── 1949-0039-1\
        └── 18032026-1904\  (DDMMYY-HHmm timestamp)
            ├── credits_raw.json
            ├── 1949-0039-1 KADIN VE DENİZCİ.json
            ├── 1949-0039-1 KADIN VE DENİZCİ.txt
            ├── transcript.json
            ├── debug.log
            └── ocr_scores.json
```

### Kopyalanan Dosyalar
- `credits_raw.json` → Filtrelerden önce snapshot
- JSON/TXT raporlar → Final veriler
- `transcript.json` → Ses transkripti (eğer varsa)
- `debug.log` → Tüm pipeline logları
- `ocr_scores.json` → Dual confidence skorları

**Config:**
```python
database_root = "D:\DATABASE"  # veya env: VITOS_DATABASE_ROOT
database_enabled = True  # default
```

---

## 11. AKIŞ ŞEMASI (ÖZETİ)

```
┌────────────────────────────────────────────────────────────┐
│ 1. OCR'dan Veri Geldi                                      │
│    → isimler, sıfatlar, film adı, OCR satırları           │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ 2. Gemini Cast Extract (HER ZAMAN)                         │
│    → OCR satırları temizlenir, multi-column parse          │
│    → Dosya adı anchor'ı KORUNUR                            │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ 3. İsim Doğrulama (3 Katman)                               │
│    → Blacklist + NameDB + TMDB Person Search               │
│    → verification_log oluşturulur                          │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ 4. TMDB Film Araması (Opsiyonel, Profile'a Bağlı)         │
│    Strateji 1: Başlık + Oyuncular                          │
│    Strateji 1b: Orijinal Başlık (Fallback)                │
│    Strateji 2: Sadece Oyuncular                            │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
            TMDB Bulundu mu?
                 │
        ┌────────┴────────┐
        │                 │
       EVET              HAYIR
        │                 │
        ▼                 ▼
┌──────────────┐   ┌─────────────────┐
│ 5. TERS      │   │ 7. BLOK2        │
│ DOĞRULAMA    │   │    VLM okuma    │
│ (Skorlama)   │   └────────┬────────┘
└──────┬───────┘            │
       │                    ▼
  KABUL mı?          ┌─────────────────┐
       │             │ 8. LLM Filtre   │
  ┌────┴────┐        │    Sahte isim   │
 EVET      HAYIR     │    temizleme    │
  │          │       └────────┬────────┘
  ▼          ▼                │
┌────────────────┐            ▼
│ 6. LOCK/REFERANS│   ┌─────────────────┐
│    Modu         │   │ 9. Google VI    │
│                 │   │    (Akıllı tetik)│
│ LOCK: TMDB      │   └────────┬────────┘
│  verisi kanonik │            │
│                 │   ◄────────┘
│ REFERANS: OCR   │
│  korunur        │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ 10. Gemini Özetleyici                                      │
│     Türkçe: 1 adım                                         │
│     Yabancı: 2 adım (önce özet, sonra çeviri)             │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ 11. Export ve Rapor Oluşturma                              │
│     → JSON (schema uyumlu)                                 │
│     → TXT (Türk dilbilgisi kurallarıyla)                   │
│     → User-friendly rapor                                  │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ 12. Database Kopyalama                                     │
│     → D:\DATABASE\{profile}\{film_id}\{timestamp}\         │
│     → Tüm çıktılar + debug.log + credits_raw.json          │
└────────────────────────────────────────────────────────────┘
```

---

## 12. KRİTİK KARARLAR VE TASARIM

### Dosya Adı Anchor'ı
Film başlığı **her zaman** dosya adından gelir. Gemini/OCR başlığı ezerse geri yüklenir.

**Sebep:** Dosya adı en güvenilir kaynak, OCR/Gemini hata yapabilir.

### Katmanlı Doğrulama
Blacklist → NameDB → TMDB Person → Gemini sırası ile fallback.

**Sebep:** Her katman farklı güçlü yönü var; birleşince doğruluk artar.

### Ters Doğrulama Eşiği: %40
Max pozitif skorun %40'ı eşik olarak kullanılır.

**Sebep:** Çok düşük → false positive, çok yüksek → false negative. %40 dengeli.

### LOCK vs Referans Modu
Sadece **film başlığıyla** eşleşmede LOCK aktif.

**Sebep:** Oyuncularla eşleşme daha zayıf; yanlış film riski var.

### BLOK2/LLM/Google VI Tetik
Sadece **TMDB başarısız** olunca çalışır.

**Sebep:** Maliyetli/yavaş API'ler; TMDB başarılıysa gereksiz.

### Gemini Cast Extract: TMDB'den Önce
Her zaman önce çalışır, TMDB'ye temiz veri gönderir.

**Sebep:** TMDB araması OCR hatalarına hassas; temiz input → daha iyi sonuç.

### Orijinal Başlık Fallback
XML sidecar'dan gelen orijinal başlık TMDB Strateji 1b'de kullanılır.

**Sebep:** Türkçe başlık TMDB'de bulunamıyorsa orijinal başlık (örn: "Fight Club") daha başarılı.

---

## 13. SON KARAR: NE YAPTIK?

### ✅ Yaptıklarımız:
1. **Gemini Cast Extract** TMDB'den önce her zaman çalıştırıldı
2. **Dosya adı anchor'ı** korunması eklendi
3. **Orijinal başlık** XML'den çekilip TMDB'ye gönderildi (Strateji 1b)
4. **Ters doğrulama** skorlama sistemi kuruldu (%40 eşik)
5. **LOCK/Referans** modu ayrımı yapıldı (matched_via="title" kontrolü)
6. **3 katmanlı isim doğrulama** (Blacklist + NameDB + TMDB Person)
7. **BLOK2/LLM/Google VI** akıllı tetik ile optimize edildi
8. **Gemini özetleyici** yabancı dil için 2 adımlı işlem yapıyor
9. **Test coverage** eklendi (test_reverse_validate.py)
10. **Database kopyalama** timestamp ile organize edildi

### 🎯 Kritik Noktalar:
- **TMDB artık kutsal kaynak değil**, doğrulama katmanlarından biri
- **Ters doğrulama** false positive'leri engelliyor
- **Orijinal başlık** Türkçe başlık başarısız olunca devreye giriyor
- **LOCK modu** sadece film başlığıyla eşleşmede aktif (güvenlik)
- **OCR verisi** LOCK modunda silinir, Referans modunda korunur

### 📝 Dosya Referansları:
- **Pipeline:** `/Project/core/pipeline_runner.py` (satır 400-750)
- **TMDB:** `/Project/core/tmdb_verify.py` (satır 150-1111)
- **Gemini Cast:** `/Project/core/gemini_cast_extractor.py`
- **Gemini Özet:** `/Project/core/gemini_summarizer.py`
- **Name Verify:** `/Project/core/name_verify.py`
- **Export:** `/Project/core/export_engine.py`
- **Testler:** `/Project/tests/test_reverse_validate.py`

---

## SON SÖZ

Bu sistem son 2 günde çok değişti. Artık:
- Gemini her zaman önce temizlik yapıyor
- TMDB'ye temiz + orijinal başlık gidiyor
- Ters doğrulama yanlış eşleşmeleri engelliyor
- LOCK/Referans ayrımı riskleri minimize ediyor
- Katmanlı doğrulama her ismi multiple check'ten geçiriyor

**Kafan karışmasın:** Bu dokümantasyon "son karar" durumunu yansıtıyor. Kod değişirse burası güncellenecek.
