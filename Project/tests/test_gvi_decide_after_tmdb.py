"""
test_gvi_decide_after_tmdb.py — decide_after_tmdb() unit tests.

Senaryolar:
  GVI-01: TMDB eşleşti → should_run=False
  GVI-02: TMDB miss + 480p → should_run=True
  GVI-03: TMDB miss + 1080p + standard font → should_run=False
  GVI-04: TMDB miss + 1080p + handwriting font → should_run=True
  GVI-05: TMDB miss + 720p + decorative font → should_run=True
  GVI-06: TMDB miss + 720p + standard font → should_run=False (720 >= 720)
  GVI-07: TMDB miss + parse edilemeyen çözünürlük → should_run=True (güvenli taraf)
  GVI-08: Kota yetersiz → should_run=False
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def _make_engine(monthly_used=0.0, available=True):
    """Test için GoogleVITextEngine örneği oluştur (is_available ve _get_monthly_usage yamalanmış)."""
    from core.google_video_intelligence import GoogleVITextEngine
    engine = GoogleVITextEngine(config={}, log_cb=lambda m: None)
    engine.is_available = lambda: available
    engine._get_monthly_usage = lambda: monthly_used
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# GVI-01: TMDB eşleşti
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_matched_no_vi():
    """GVI-01: TMDB eşleşince should_run=False."""
    engine = _make_engine()
    decision = engine.decide_after_tmdb(
        tmdb_matched=True,
        resolution="1920x1080",
        font_type="standard",
        segment_duration_min=5.0,
    )
    assert decision.should_run is False
    assert "TMDB" in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# GVI-02: TMDB miss + düşük çözünürlük (480p)
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_miss_low_resolution_sends_vi():
    """GVI-02: TMDB miss + 480p → should_run=True."""
    engine = _make_engine()
    decision = engine.decide_after_tmdb(
        tmdb_matched=False,
        resolution="854x480",
        font_type="standard",
        segment_duration_min=5.0,
    )
    assert decision.should_run is True
    assert "düşük" in decision.reason.lower() or "çözünürlük" in decision.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# GVI-03: TMDB miss + 1080p + standard font
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_miss_1080p_standard_font_no_vi():
    """GVI-03: TMDB miss + 1080p + standard font → should_run=False (lokal yeter)."""
    engine = _make_engine()
    decision = engine.decide_after_tmdb(
        tmdb_matched=False,
        resolution="1920x1080",
        font_type="standard",
        segment_duration_min=5.0,
    )
    assert decision.should_run is False
    assert "lokal" in decision.reason.lower() or "720p" in decision.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# GVI-04: TMDB miss + 1080p + handwriting font
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_miss_1080p_handwriting_sends_vi():
    """GVI-04: TMDB miss + 1080p + handwriting → should_run=True."""
    engine = _make_engine()
    decision = engine.decide_after_tmdb(
        tmdb_matched=False,
        resolution="1920x1080",
        font_type="handwriting",
        segment_duration_min=5.0,
    )
    assert decision.should_run is True
    assert "font" in decision.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# GVI-05: TMDB miss + 720p + decorative font
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_miss_720p_decorative_sends_vi():
    """GVI-05: TMDB miss + 720p + decorative → should_run=True (height=720 >= 720, non-standard)."""
    engine = _make_engine()
    decision = engine.decide_after_tmdb(
        tmdb_matched=False,
        resolution="1280x720",
        font_type="decorative",
        segment_duration_min=5.0,
    )
    assert decision.should_run is True


# ─────────────────────────────────────────────────────────────────────────────
# GVI-06: TMDB miss + 720p + standard font
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_miss_720p_standard_no_vi():
    """GVI-06: TMDB miss + 720p + standard font → should_run=False."""
    engine = _make_engine()
    decision = engine.decide_after_tmdb(
        tmdb_matched=False,
        resolution="1280x720",
        font_type="standard",
        segment_duration_min=5.0,
    )
    assert decision.should_run is False


# ─────────────────────────────────────────────────────────────────────────────
# GVI-07: Parse edilemeyen çözünürlük → güvenli taraf (VI'ya gönder)
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_miss_unparseable_resolution_sends_vi():
    """GVI-07: Çözünürlük parse edilemezse height=0 → should_run=True."""
    engine = _make_engine()
    decision = engine.decide_after_tmdb(
        tmdb_matched=False,
        resolution="unknown",
        font_type="standard",
        segment_duration_min=5.0,
    )
    assert decision.should_run is True


# ─────────────────────────────────────────────────────────────────────────────
# GVI-08: Kota yetersiz
# ─────────────────────────────────────────────────────────────────────────────

def test_quota_insufficient_no_vi():
    """GVI-08: Kota yetersizse should_run=False."""
    engine = _make_engine(monthly_used=999.0)  # limit=1000, remaining=1 < 5
    decision = engine.decide_after_tmdb(
        tmdb_matched=False,
        resolution="854x480",
        font_type="handwriting",
        segment_duration_min=5.0,
    )
    assert decision.should_run is False
    assert "kota" in decision.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# _parse_height yardımcı metodu
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_height_standard():
    """_parse_height: standart WxH formatı."""
    from core.google_video_intelligence import GoogleVITextEngine
    assert GoogleVITextEngine._parse_height("1920x1080") == 1080
    assert GoogleVITextEngine._parse_height("1280x720") == 720
    assert GoogleVITextEngine._parse_height("854x480") == 480


def test_parse_height_invalid_returns_zero():
    """_parse_height: geçersiz giriş → 0 (güvenli taraf)."""
    from core.google_video_intelligence import GoogleVITextEngine
    assert GoogleVITextEngine._parse_height("unknown") == 0
    assert GoogleVITextEngine._parse_height("") == 0
    assert GoogleVITextEngine._parse_height("1080") == 0
