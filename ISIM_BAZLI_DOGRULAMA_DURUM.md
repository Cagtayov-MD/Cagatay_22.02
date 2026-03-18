# İsim Bazlı Doğrulama Sistem Durumu

## ✅ EVET, İSİM BAZLI KONTROL AKTİF ÇALIŞIYOR

Film bazlı değil, **kişi merkezli doğrulama** sistemi aktif ve çalışıyor.

---

## 📍 NEREDE ÇALIŞIYOR?

### Pipeline Stage: [6/6] NAME_VERIFY
**Dosya:** `Project/core/pipeline_runner.py` (satır 457-517)

```python
# ══ [6/6] NAME_VERIFY — Katmanlı İsim Doğrulama ══════
# TMDB artık kutsal kaynak değil, doğrulama katmanlarından biri.
# Film merkezli değil, kişi merkezli doğrulama.
# Katmanlar: Blacklist → NameDB → TMDB Person Search
```

### Çalışma Sırası:
```
1. Gemini Cast Extract (satır 445-455)
2. [6/6] NAME_VERIFY (satır 457-517)  ← İSİM BAZLI DOĞRULAMA BURADA
3. TMDB Film Search (satır 519-586)   ← Film bazlı (opsiyonel)
```

---

## 🔍 NASIL ÇALIŞIYOR?

### 3 Katmanlı İsim Doğrulama Sistemi

**Dosya:** `Project/core/name_verify.py`

#### Katman 1: Blacklist + Yapısal Kontrol (Maliyet: 0)
**İşlevi:** Teknik terimler, şirketler, noise temizleme
- Blacklist: 200+ teknik terim/şirket ismi (technicolor, best boy grip, vs.)
- Yapısal kontrol: Sesli harf oranı, büyük harf dizileri, vs.

**Kod:**
```python
is_bl, bl_reason = _blacklist_check(name)
if is_bl:
    self._add_log("BLACKLIST", role, name, "", "dropped", bl_reason)
    continue

passed, struct_reason = _structural_check(name)
if not passed:
    self._add_log("STRUCTURAL", role, name, "", "dropped", struct_reason)
    continue
```

#### Katman 2: NameDB (Türk İsim Veritabanı) (Maliyet: 0, lokal)
**İşlevi:** Türk isimlerini doğrulama ve düzeltme
- Exact match check: `name_db.is_name(name)`
- Fuzzy match: Score ≥ 0.80 → düzeltme + doğrulama
- Türk karakterleri (ç, ğ, ı, ö, ş, ü) düzeltme

**Kod:**
```python
if self._name_db.is_name(name):
    namedb_verified = True
    self._add_log("NAMEDB", role, name, name, "kept", "exact_match")
else:
    result_name, score, method = self._name_db.find_with_method(name)
    if result_name and score >= 0.80:
        namedb_verified = True
        corrected_name = result_name
        self._add_log("NAMEDB", role, name, corrected_name, "corrected", ...)
```

#### Katman 3: TMDB Person Search (Maliyet: düşük, HTTP)
**İşlevi:** Uluslararası isim doğrulama
- API: `search/person` endpoint
- Meslek kontrolü: known_for_department check (acting, directing, vs.)
- Canonical name döndürme

**Kod:**
```python
if self._tmdb_client:
    tmdb_result = _tmdb_person_verify(
        search_name, role, self._tmdb_client, self._log)

    if tmdb_result["verified"]:
        tmdb_verified = True
        # TMDB ismi canonical olarak kullan
        corrected_name = tmdb_result["tmdb_name"]
```

**Fonksiyon:** `_tmdb_person_verify()` (satır 219-270)
```python
def _tmdb_person_verify(name: str, expected_role: str, tmdb_client, log_cb=None):
    """TMDB person search ile isim doğrulama.

    Film aramıyor, kişi arıyor. Nisa Serezli gibi TMDB'de filmi olmayan
    ama kişi olarak kayıtlı olan isimler doğrulanır.
    """
    results = tmdb_client.search_person(name.strip())

    if not results:
        return {"verified": False, "reason": "not_found"}

    # En yüksek popularity sonucu al
    best = results[0]
    return {
        "verified": True,
        "tmdb_name": best.get("name"),
        "tmdb_department": best.get("known_for_department"),
        "tmdb_id": best.get("id"),
    }
```

---

## 🎯 KARAR MEKANİZMASI

### Crew (Yapım Ekibi) Doğrulama
**Dosya:** `name_verify.py` → `verify_crew()` (satır 356-470)

**2 Geçişli Sistem:**

**Geçiş 1: Verified vs Unverified Ayırma**
```python
for name in names:
    # Katman 1, 2, 3 kontroller

    if namedb_verified or tmdb_verified:
        verified_names.append(corrected_name)  # ✓ Doğrulandı
    else:
        unverified_candidates.append((name, corrected_name))  # ? Beklemede
```

**Geçiş 2: Unverified Adayları Değerlendirme**
```python
for orig_name, cand_name in unverified_candidates:
    if verified_names:
        # Aynı rol için verified isim var → unverified'ı düşür
        self._add_log("FINAL", role, orig_name, "", "dropped",
                      "unverified_has_alternative")
    else:
        # Bu rol için hiç verified isim yok → flag ile ekle
        self._add_log("FINAL", role, orig_name, orig_name, "flagged",
                      "unverified_no_alternative")
        verified_names.append(orig_name)
```

**Mantık:**
- Verified isim varsa → unverified atılır
- Verified isim yoksa → unverified flag ile eklenir (veri kaybı önleme)

### Cast (Oyuncu) Doğrulama
**Dosya:** `name_verify.py` → `verify_cast()` (satır 476-570)

**Daha basit mantık:**
```python
for entry in cast_list:
    # Katman 1, 2, 3 kontroller

    if namedb_verified or tmdb_verified:
        entry["is_verified_name"] = True
        verified_cast.append(entry)
    else:
        # Doğrulanamadı ama ver (cast için daha toleranslı)
        entry["is_verified_name"] = False
        verified_cast.append(entry)
```

**Not:** Oyuncularda daha toleranslı → veri kaybı riski düşük

---

## ⚙️ CONFIG VE FLAG'LER

### 1. tmdb_enabled
**Konum:** `content_profiles.json` ve `pipeline_runner.py` satır 467

```json
"tmdb_enabled": true  // TMDB Person Search aktif/pasif
```

```python
_tmdb_enabled = bool((content_profile or {}).get("tmdb_enabled", True))
if _tmdb_enabled:
    _tmdb_client = TMDBClient(...)
```

**Etki:**
- `true` → TMDB Person Search Katman 3 çalışır
- `false` → Sadece Blacklist + NameDB (Katman 1, 2)

### 2. tmdb_person_verify
**Konum:** `content_profiles.json` satır 19

```json
"tmdb_person_verify": true  // Explicit flag (dokümantasyon için)
```

**Not:** Bu flag şu anda kullanılmıyor, `tmdb_enabled` yeterli. Gelecekte ayrıştırılabilir:
- `tmdb_enabled`: Film search için
- `tmdb_person_verify`: Person search için

---

## 📊 VERİFICATION LOG

### Log Formatı
Her isim için detaylı log tutuluyor:

```python
cdata["_verification_log"] = [
    {
        "layer": "BLACKLIST",
        "role": "YÖNETMEN",
        "name_in": "TECHNICOLOR",
        "name_out": "",
        "action": "dropped",
        "reason": "company_keyword"
    },
    {
        "layer": "NAMEDB",
        "role": "OYUNCU",
        "name_in": "Turhan Bey",
        "name_out": "Turhan Bey",
        "action": "kept",
        "reason": "exact_match",
        "namedb_score": 1.0
    },
    {
        "layer": "TMDB_PERSON",
        "role": "OYUNCU",
        "name_in": "Tommy Rettig",
        "name_out": "Tommy Rettig",
        "action": "kept",
        "reason": "person_found",
        "tmdb_name": "Tommy Rettig",
        "tmdb_department": "acting",
        "tmdb_id": 12345
    },
    ...
]
```

### Log Kullanımı
**Pipeline:** `cdata["_verification_log"]` olarak saklanıyor (satır 511)

**Database:** `D:\DATABASE\{profile}\{film_id}\{timestamp}\` altına yazılıyor

**Text Export:** `get_log_text()` ile okunabilir format:
```
────────────────────────────────────────────────────────
  BLACKLIST
────────────────────────────────────────────────────────
  [YÖNETMEN] ✗ TECHNICOLOR  →  ÇIKARILDI (company_keyword)

────────────────────────────────────────────────────────
  NAMEDB
────────────────────────────────────────────────────────
  [OYUNCU] ✓ Turhan Bey (exact_match)
          NameDB: skor=1.00

────────────────────────────────────────────────────────
  TMDB_PERSON
────────────────────────────────────────────────────────
  [OYUNCU] ✓ Tommy Rettig (person_found)
          TMDB: Tommy Rettig [acting] id=12345
```

---

## 🔄 TMDB PERSON vs TMDB FILM FARKI

### TMDB Person Search (İsim Bazlı)
**Aşama:** [6/6] NAME_VERIFY
**API:** `search/person`
**Amaç:** Kişileri doğrulama (film bağımsız)
**Çalışma:** HER ZAMAN (tmdb_enabled=true ise)

**Örnek:**
```python
# Nisa Serezli'yi arıyor
tmdb_client.search_person("Nisa Serezli")
→ Sonuç: {"name": "Nisa Serezli", "department": "acting", "id": 123}
→ Doğrulandı ✓ (Film bilgisi gereksiz)
```

### TMDB Film Search (Film Bazlı)
**Aşama:** TMDB Film araması (NAME_VERIFY'dan sonra)
**API:** `search/multi`, `movie/{id}/credits`, `tv/{id}/credits`
**Amaç:** Filmi eşleştirme (LOCK/REFERANS modu)
**Çalışma:** OPSIYONEL (profile'a bağlı)

**Örnek:**
```python
# "Fight Club" filmini arıyor
tmdb_client.search_multi("Fight Club")
→ Film bulundu → credits çek
→ LOCK modu: OCR verisi silinir, TMDB cast yazılır
```

---

## ✅ AKTİF PROFILE'LER

### FilmDizi-Hybrid (satır 2-24, content_profiles.json)
```json
{
  "tmdb_enabled": true,           // ✓ Person Search AKTİF
  "tmdb_person_verify": true,     // ✓ Explicit flag
  "gemini_enabled": false,
  "llm_cast_filter": false,
  "blok2_enabled": false
}
```

**Durum:** İsim bazlı doğrulama **ÇALIŞIYOR** (3 katman)

### Spor Profile (satır 25-40)
```json
{
  "match_parse_enabled": true
  // tmdb_enabled belirtilmemiş → default: true
}
```

**Durum:** İsim bazlı doğrulama **ÇALIŞIYOR** (default true)

---

## 🧪 TEST DURUMU

### Mevcut Testler
**Arama sonucu:** `name_verify` için test dosyası bulunamadı

```bash
$ find tests -name "*.py" -exec grep -l "name_verify\|NameVerifier" {} \;
# Sonuç: Boş
```

**Yorum:** Unit test eksik → manuel test gerekli

### Manuel Test Önerisi
```python
# Test script
from core.name_verify import NameVerifier
from core.tmdb_verify import TMDBClient

# Setup
tmdb_client = TMDBClient(api_key="...", language="tr-TR")
verifier = NameVerifier(name_db=name_db, tmdb_client=tmdb_client)

# Test 1: Blacklist
crew = {"YÖNETMEN": ["TECHNICOLOR", "Charles Walters"]}
result = verifier.verify_crew(crew)
# Beklenen: {"YÖNETMEN": ["Charles Walters"]}

# Test 2: TMDB Person
crew = {"OYUNCU": ["Tommy Rettig", "FAKE NAME XYZ"]}
result = verifier.verify_crew(crew)
# Beklenen: Tommy Rettig doğrulanmış, FAKE NAME atılmış

# Test 3: Log
log = verifier.get_log()
# Beklenen: Her adım loglanmış
```

---

## 📈 PERFORMANS

### Katman Maliyetleri
1. **Blacklist:** 0ms (local, O(1) lookup)
2. **NameDB:** ~1ms (local, fuzzy match)
3. **TMDB Person:** ~100-200ms (HTTP request)

### Optimizasyonlar
- **Cache:** TMDB Person Search sonuçları cache'leniyor
- **Rate limiting:** 150ms throttle (TMDB API limits)
- **Fallback chain:** Ucuz katmanlar önce → pahalı son

### Örnek İstatistik
```
15 oyuncu + 8 crew = 23 isim

Katman 1 (Blacklist): 5 atıldı (0ms)
Katman 2 (NameDB):    10 doğrulandı (~10ms)
Katman 3 (TMDB):      8 doğrulandı (~1.5s, 8 * 200ms)

Toplam: ~1.5s (TMDB yavaş ama gerekli)
```

---

## 🎯 SONUÇ

### ✅ EVET, İSİM BAZLI DOĞRULAMA AKTİF

1. **Stage:** [6/6] NAME_VERIFY
2. **Katmanlar:** Blacklist → NameDB → TMDB Person (3 katman)
3. **API:** `search/person` (film bağımsız)
4. **Config:** `tmdb_enabled=true` (default)
5. **Profile:** FilmDizi-Hybrid ✓ AKTİF
6. **Log:** Detaylı verification log tutulıyor
7. **Veri:** `cdata["_verification_log"]` ve DATABASE export

### Film Bazlı vs İsim Bazlı

| Özellik | İsim Bazlı (NAME_VERIFY) | Film Bazlı (TMDB Film) |
|---------|--------------------------|------------------------|
| **API** | search/person | search/multi + credits |
| **Amaç** | Kişi doğrulama | Film eşleştirme |
| **Çalışma** | HER ZAMAN | Opsiyonel (profile) |
| **Bağımsız** | Film gereksiz | Film başlığı gerekli |
| **Örnek** | "Nisa Serezli" ✓ | "Fight Club" → LOCK |

### Güçlü Yönler
- ✅ 3 katmanlı fallback
- ✅ Maliyet optimize (ucuzdan pahalıya)
- ✅ Detaylı log (debugging kolay)
- ✅ Film bağımsız (kişi odaklı)
- ✅ Canonical name (TMDB'den)

### Zayıf Yönler
- ⚠️ Unit test yok
- ⚠️ TMDB Person yavaş (200ms/isim)
- ⚠️ `tmdb_person_verify` flag kullanılmıyor

### Öneriler
1. Unit test ekle (`tests/test_name_verify.py`)
2. TMDB Person cache optimize et
3. `tmdb_person_verify` flag'i ayrıştır (film vs person)
4. Batch TMDB Person Search (N isim → 1 request)

---

## 📝 KAYNAKLAR

- **Implementation:** `Project/core/name_verify.py`
- **Pipeline:** `Project/core/pipeline_runner.py` (satır 457-517)
- **Config:** `Project/config/content_profiles.json`
- **Dokümantasyon:** `TMDB_SUREC_DOKUMANTASYONU.md` (bölüm 3)
