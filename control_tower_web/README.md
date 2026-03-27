# Control Tower Web

Bu klasor, aktif `Project/` yapisina dokunmadan hazirlanan bagimsiz web iskeletidir.

Amac:
- `FilmDizi-Hybrid` ve `Spor` profilleri icin web tabanli bir control tower hazirlamak
- mevcut masaustu UI'yi bozmadan yeni web gorunusunu ayri bir calisma alani olarak gelistirmek
- ilk asamada sadece mock/read-only veriyle calismak

Bu klasorun bilerek yapmadigi seyler:
- `Project/` altindaki aktif dosyalari degistirmek
- mevcut masaustu UI'yi kapatmak
- pipeline davranisini degistirmek

## Icerik

- `server.py`: bagimsiz, kutuphane istemeyen kucuk HTTP sunucusu
- `start_control_tower.bat`: web arayuzunu baslatmak icin Windows kisayolu
- `data/control_tower_state.json`: iki profil icin mock state verisi
- `static/index.html`: web arayuzu
- `static/app.css`: stil dosyasi
- `static/app.js`: istemci tarafi mantik

## Baslatma

PowerShell:

```powershell
python .\control_tower_web\server.py
```

veya:

```powershell
.\control_tower_web\start_control_tower.bat
```

Ardindan tarayicida:

`http://127.0.0.1:8011`

## Not

Bu ilk surum:
- read-only
- mock state kullaniyor
- iki profil gosteriyor: `FilmDizi-Hybrid`, `Spor`

Sonraki adimda bu sunucu:
- mevcut artifact klasorlerini
- stats loglarini
- queue durumunu
- profile/rules bilgisini

okuyan gercek bir adapter katmanina donusturulebilir.
