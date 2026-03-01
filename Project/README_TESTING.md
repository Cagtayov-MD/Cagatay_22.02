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
python test_runner.py --config test_config.json F:\test\test.mp4

# Klasördeki tüm videolar
python test_runner.py --config test_config.json --folder F:\test
```

### 2️⃣ Batch Script (Tek Tık)

Windows'ta çift tıkla: `run_test.bat`

**Menü:**
- [1] Tek video test (F:\test\test.mp4)
- [2] Tüm videolar (F:\test klasörü)
- [3] Custom video seç
- [4] Custom klasör seç

### 3️⃣ Watch Script (Otomatik İzleme)

```bash
# Sürekli izle (30 saniyede bir kontrol)
python watch_folder.py F:\test

# Custom interval
python watch_folder.py F:\test --interval 10

# Bir kez kontrol et ve çık
python watch_folder.py F:\test --once

# Mevcut videoları da test et
python watch_folder.py F:\test --test-existing

# Custom config
python watch_folder.py F:\test --config test_config.json
```

**Özellikler:**
- ✅ Yeni `.mp4` dosyaları otomatik tespit
- ✅ Zaten test edilenleri tekrar test etmez (`.processed_videos.json`)
- ✅ Renkli konsol çıktısı
- ✅ Ctrl+C ile durdur

---

## 📁 Çıktı Dosyaları

```
F:\test\
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
  "output_root": "F:/test"
}
```

---

## 💡 İpuçları

**Watch Script için Windows Task Scheduler:**
```cmd
Program: python.exe
Arguments: F:\REPO_GitHub\Cagatay_22.02\Project\watch_folder.py F:\test
Start in: F:\REPO_GitHub\Cagatay_22.02\Project
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
- Tam path kullan: `F:\test\test.mp4` ✅
- Relative path: `.\test.mp4` ❌

**Watch script çalışmıyor:**
- Klasör var mı kontrol et
- `.processed_videos.json` varsa silebilirsin (reset için)
