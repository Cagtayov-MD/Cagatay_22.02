"""
email_notifier.py — Pipeline çıktı TXT'sini e-posta olarak gönderir.

- TXT dosyasının içeriği mail GÖVDESINE yazılır (ek olarak DEĞİL)
- Konu: "Vitos Otomasyon <N>"  → N her gönderimde +1 artar
- Sayaç Project/data/email_counter.txt dosyasında kalıcı olarak saklanır
- Gönderici: vitosotomasyon@gmail.com  (Gmail App Password ile)
- Alıcı:     cagtayovsky@gmail.com
- Şifre:     ENV → EMAIL_PASSWORD  (kod içine yazılmaz)
"""

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

SENDER_EMAIL    = "vitosotomasyon@gmail.com"
RECIPIENT_EMAIL = "cagtayovsky@gmail.com"

_COUNTER_FILE = Path(__file__).resolve().parent.parent / "data" / "email_counter.txt"


def _read_counter() -> int:
    """Sayaç dosyasını okur; yoksa veya bozuksa 1 döner."""
    try:
        return int(_COUNTER_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return 1


def _write_counter(value: int) -> None:
    """Sayacı dosyaya yazar; klasör yoksa oluşturur."""
    _COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _COUNTER_FILE.write_text(str(value), encoding="utf-8")


def send_result_email(txt_path: str, log_cb=None) -> bool:
    """
    txt_path içindeki TXT dosyasını e-posta gövdesi olarak gönderir.

    Parametreler
    ------------
    txt_path : str
        Gönderilecek TXT raporunun tam yolu.
    log_cb : callable, optional
        Loglama callback'i; ``log_cb(mesaj)`` şeklinde çağrılır.

    Döner
    -----
    bool
        Gönderim başarılıysa True, aksi hâlde False.
    """
    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    password = os.environ.get("EMAIL_PASSWORD")
    if not password:
        _log("  [Email] EMAIL_PASSWORD env değişkeni bulunamadı — gönderim atlandı")
        return False

    try:
        with open(txt_path, encoding="utf-8-sig") as fh:
            body = fh.read()
    except Exception as exc:
        _log(f"  [Email] TXT okunamadı ({txt_path}): {exc}")
        return False

    counter = _read_counter()
    subject = f"Vitos Otomasyon <{counter}>"

    msg = EmailMessage()
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(SENDER_EMAIL, password)
            smtp.send_message(msg)
    except Exception as exc:
        _log(f"  [Email] Gönderim hatası: {exc}")
        return False

    _log(f"  [Email] Gönderildi → {RECIPIENT_EMAIL}  (konu: {subject})")
    try:
        _write_counter(counter + 1)
    except Exception as exc:
        _log(f"  [Email] Sayaç güncellenemedi: {exc}")
    return True
