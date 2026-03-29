"""
test_gemini_summarizer.py — A3: _is_turkish_text() Latin oranı bazlı dil tespiti.
Arapça/Kiril metinler artık "Türkçe değil" olarak tanınır.
"""
import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_turkish_text_is_turkish():
    """Normal Türkçe özet → True."""
    from core.gemini_summarizer import _is_turkish_text
    text = (
        "Film, İstanbul'da geçen bir aşk hikâyesini anlatır. "
        "Genç bir mühendis olan Ahmet, şehrin kalabalığında kaybolmuş "
        "gibi hissederken beklenmedik bir tanışma hayatını değiştirir."
    )
    assert _is_turkish_text(text)


def test_arabic_text_is_not_turkish():
    """Arapça metin → False (Latin oranı çok düşük)."""
    from core.gemini_summarizer import _is_turkish_text
    arabic = (
        "تدور أحداث الفيلم في مدينة إسطنبول، حيث يلتقي شاب مهندس "
        "بفتاة جميلة في أحد الأحياء العتيقة للمدينة، فتتغير حياته "
        "تغيرًا جذريًا بعد هذا اللقاء غير المتوقع."
    )
    assert not _is_turkish_text(arabic)


def test_arabic_with_single_turkish_char_is_not_turkish():
    """Arapça metin + tek ç harfi → False (Latin oranı hâlâ düşük)."""
    from core.gemini_summarizer import _is_turkish_text
    arabic_plus_c = (
        "تدور أحداث الفيلم في مدينة إسطنبول ç "
        "حيث يلتقي شاب مهندس بفتاة جميلة."
    )
    assert not _is_turkish_text(arabic_plus_c)


def test_cyrillic_text_is_not_turkish():
    """Kiril metni → False."""
    from core.gemini_summarizer import _is_turkish_text
    cyrillic = (
        "Фильм рассказывает историю молодого инженера в Стамбуле, "
        "который неожиданно встречает девушку своей мечты в старом "
        "районе города и его жизнь меняется навсегда."
    )
    assert not _is_turkish_text(cyrillic)


def test_empty_text_is_not_turkish():
    """Boş metin → False."""
    from core.gemini_summarizer import _is_turkish_text
    assert not _is_turkish_text("")
    assert not _is_turkish_text(None)


def test_short_text_is_not_turkish():
    """20 karakterden kısa metin → False (yetersiz örnek)."""
    from core.gemini_summarizer import _is_turkish_text
    assert not _is_turkish_text("kısa")
    assert not _is_turkish_text("Merhaba dünya!")


def test_english_text_without_tr_chars_is_not_turkish():
    """İngilizce metin (Türkçe karakter yok) → False."""
    from core.gemini_summarizer import _is_turkish_text
    english = (
        "The film tells the story of a young engineer in Istanbul "
        "who unexpectedly meets the woman of his dreams and his life "
        "changes forever in this romantic drama."
    )
    assert not _is_turkish_text(english)
