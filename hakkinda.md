## 📋 PR Tavsiye Raporu

### ① Yapısal Kontrol — "Bu iş çalışıyor mu?"
✅ Değişen 3 Python dosyasının hepsi syntax açısından temiz. Ana modüller sorunsuz import ediliyor.

### ② Bağlam Kontrolü — "Amacımızı destekliyor mu?"
Bu PR toplamda 5 dosyayı etkiliyor. Ağırlıklı olarak 📸 OCR dalında 3 dosya üzerinde çalışma yapılmış. Ek olarak ⚙️ Config/DAG katmanında 2 dosya değişiklik var.

### ③ Çakışma Kontrolü — "Başka mekanizmaları bozuyor mu?"
✅ Mevcut 4 test dosyasındaki testlerin tamamı başarıyla geçti.



## 🎯 Parametre Skorları

| # | Parametre | Ağırlık | Mantık |
|---|-----------|---------|--------|
| 1 | Film adı tam eşleşme | +3.0 | OCR başlığı = TMDB başlığı (normalized) |
| 2 | Film adı kısmi/fuzzy | +1.5 | "Kulübe" vs "Dövüş Kulübü" — substring var ama tam değil |
| 3 | Film adı hiç eşleşmez | -2.0 | Başlıkta ciddi tutarsızlık |
| 4 | Yönetmen eşleşme | +3.5 | En ağır pozitif — yönetmen filmin kimliğidir |
| 5 | Yönetmen eşleşmez | -2.5 | OCR'dan yönetmen çıktıysa ama TMDB'dekiyle uyuşmuyorsa çok kötü |
| 6 | Yönetmen verisi yok | 0 | OCR yönetmen okuyamadıysa cezalandırma |
| 7 | Yıl tam eşleşme (±2) | +2.0 | Film yılı çok güçlü sinyal |
| 8 | Yıl yakın (±5) | +1.0 | Remake/restorasyon olabilir |
| 9 | Yıl uzak (±6-15) | -1.0 | Şüpheli |
| 10 | Yıl çok uzak (>15) | -3.0 | 1952 vs 1999 = 47 yıl → kesin yanlış film |
| 11 | Yıl verisi yok | 0 | OCR'dan yıl çıkmadıysa ceza yok |
| 12 | Cast oran ≥ %25 | +3.0 | Güçlü eşleşme |
| 13 | Cast oran %10-%25 | +1.0 | Makul |
| 14 | Cast oran %5-%10 | -1.0 | Şüpheli |
| 15 | Cast oran < %5 | -3.0 | 3/96 = %3.1 → büyük alarm |
| 16 | Crew eşleşme (≥1) | +1.5 | Ekstra güven |
| 17 | Crew eşleşmez | -0.5 | Hafif ceza (crew OCR'da zor okunur) |

---

## 🚦 Eşik Değerleri

| Toplam Skor | Karar | Açıklama |
|-------------|-------|----------|
| ≥ +4.0 | ✅ ACCEPT | Film doğrulandı, LOCK açılabilir |
| +1.0 — +3.9 | ⚠️ SOFT ACCEPT | Film kabul ama confidence "medium", LOCK açılmaz |
| -∞ — +0.9 | ❌ REJECT | Film reddedildi, TMDB eşleşmesi geçersiz sayılır |
