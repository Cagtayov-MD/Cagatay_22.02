# Test Otomasyonu Kılavuzu

## 📋 İçerik

Bu klasörde 3 farklı test yöntemi var:

1. **`test_runner.py`** - Manuel CLI test runner
2. **`run_test.bat`** - Tek tıkla Windows batch tester
3. **`watch_folder.py`** - Otomatik klasör izleyici

---

## 🚀 Kullanım

### 1️⃣ Manuel Test (CLI)

```bash
# Tek video
python test_runner.py --config test_config.json <YOUR_TEST_DIR>\test.mp4

# Klasördeki tüm videolar
python test_runner.py --config test_config.json --folder <YOUR_TEST_DIR>
```

### 2️⃣ Batch Script (Tek Tık)

Windows'ta çift tıkla: `run_test.bat`

**Menü:**
- [1] Tek video test (<YOUR_TEST_DIR>\test.mp4)
- [2] Tüm videolar (<YOUR_TEST_DIR> klasörü)
- [3] Custom video seç
- [4] Custom klasör seç

### 3️⃣ Watch Script (Otomatik İzleme)

```bash
# Sürekli izle (30 saniyede bir kontrol)
python watch_folder.py <YOUR_TEST_DIR>

# Custom interval
python watch_folder.py <YOUR_TEST_DIR> --interval 10

# Bir kez kontrol et ve çık
python watch_folder.py <YOUR_TEST_DIR> --once

# Mevcut videoları da test et
python watch_folder.py <YOUR_TEST_DIR> --test-existing

# Custom config
python watch_folder.py <YOUR_TEST_DIR> --config test_config.json
```

**Özellikler:**
- ✅ Yeni `.mp4` dosyaları otomatik tespit
- ✅ Zaten test edilenleri tekrar test etmez (`.processed_videos.json`)
- ✅ Renkli konsol çıktısı
- ✅ Ctrl+C ile durdur

---

## 📁 Çıktı Dosyaları

```
<YOUR_TEST_DIR>\
├── test.mp4                           # Input video
├── arsiv_test_{timestamp}\            # Test sonuçları
│   ├── report.json
│   ├── report.txt
│   └── transcript.txt
├── test_results_{timestamp}.json      # Batch test log
└── .processed_videos.json             # Watch script log
```

---

## 🔧 Konfigürasyon

**`test_config.json`:**
```json
{
  "scope": "video+audio",
  "first_min": 2.0,
  "last_min": 6.0,
  "difficulty": "heavy",
  "use_gpu": true,
  "program_type": "film_dizi",
  "output_root": "<YOUR_TEST_DIR>"
}
```

---

## 🤖 VLM Model Yönetimi (GLM-4.6V-Flash)

### GLM Modeli Kurulum

```bash
ollama pull glm4.6v-flash:q4_K_M
```

### Qwen'den GLM'e Geçiş

1. `.env` dosyasında `VLM_MODEL` anahtarını ayarlayın:
   ```
   VLM_MODEL=glm4.6v-flash:q4_K_M
   VLM_THRESHOLD=0.80
   ```
2. Pipeline otomatik olarak GLM kullanacaktır.

### Geri Alma (Rollback) — Qwen'e Dönüş

`.env` dosyasında `VLM_MODEL` değerini Qwen olarak ayarlayın:
```
VLM_MODEL=qwen3-vl:8b
```
veya eski `QWEN_MODEL` anahtarını kullanın:
```
QWEN_MODEL=qwen3-vl:8b
```

### Config Öncelik Sırası (Model Seçimi)

1. `vlm_model` (JSON config)
2. `VLM_MODEL` (çevre değişkeni)
3. `qwen_model` (JSON config, legacy)
4. `QWEN_MODEL` (çevre değişkeni, legacy)
5. Varsayılan: `glm4.6v-flash:q4_K_M`

### VLM-as-OCR (Opsiyonel, Varsayılan Kapalı)

PaddleOCR'ın hiç metin bulamadığı frame'lerde VLM'i OCR yedeği olarak kullanmak için:

```json
{
  "vlm_ocr_enabled": true
}
```

---

## 💡 İpuçları

**Watch Script için Windows Task Scheduler:**
```cmd
Program: python.exe
Arguments: <PATH_TO_PROJECT>\watch_folder.py <YOUR_TEST_DIR>
Start in: <PATH_TO_PROJECT>
```

**Batch Script için Kısayol:**
1. `run_test.bat` üzerine sağ tık
2. "Kısayol oluştur"
3. Masaüstüne taşı
4. İstenirse icon değiştir

---

## 🐛 Sorun Giderme

**UTF-8 hataları:**
- Batch script otomatik `chcp 65001` yapar
- Python scriptlerde `io.TextIOWrapper` fix mevcut

**Video bulunamadı:**
- Tam path kullan: `<YOUR_TEST_DIR>\test.mp4` ✅
- Relative path: `.\test.mp4` ❌

**Watch script çalışmıyor:**
- Klasör var mı kontrol et
- `.processed_videos.json` varsa silebilirsin (reset için)
