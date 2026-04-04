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


def test_ascii_only_turkish_text_is_still_turkish():
    """Türkçe karakter olmasa bile belirgin Türkçe söz dizimi kabul edilmeli."""
    from core.gemini_summarizer import _is_turkish_text
    text = (
        "Film bir adamin eve donup babasiyla hesaplasmasini anlatir "
        "ve sonunda kendi hayatina yeniden baslamasiyla biter."
    )
    assert _is_turkish_text(text)


def test_final_summary_validator_rejects_non_latin_script():
    """Final özet Latin dışı script sızıntısı içeriyorsa reddedilmeli."""
    from core.gemini_summarizer import is_valid_final_summary
    text = "Ahmed şehre döner ama أحمد adını unutamaz."
    assert not is_valid_final_summary(text)


def test_final_summary_validator_accepts_turkish_with_latin_foreign_names():
    """Türkçe özet + Latin yabancı isimler kabul edilmeli."""
    from core.gemini_summarizer import is_valid_final_summary
    text = (
        "İrfan eve döner, Ivan ile yüzleşir ve Jérôme'un gerçeği sakladığını öğrenir. "
        "Sonunda hepsi yollarını ayırır."
    )
    assert is_valid_final_summary(text)


def test_normalize_summary_text_preserves_paragraph_breaks():
    """İki paragraf korunmalı; paragraf içi satırlar tek satıra inmelidir."""
    from core.gemini_summarizer import normalize_final_summary_text

    text = (
        "Bir kadın eve döner.\n"
        "Kayıplarıyla yüzleşir.\n\n"
        "Chris ile vedalaşıp Washington'a gider.\n"
        "Yeni bir başlangıç yapar."
    )

    assert normalize_final_summary_text(text) == (
        "Bir kadın eve döner. Kayıplarıyla yüzleşir.\n\n"
        "Chris ile vedalaşıp Washington'a gider. Yeni bir başlangıç yapar."
    )


def test_final_summary_validator_accepts_two_paragraphs():
    """Geçerli Türkçe özet iki paragraf olduğunda da kabul edilmeli."""
    from core.gemini_summarizer import is_valid_final_summary

    text = (
        "Bir kadın eve döner ve oğlunu korumaya çalışır.\n\n"
        "Sonunda Chris ile vedalaşıp Washington'a gider."
    )

    assert is_valid_final_summary(text)
