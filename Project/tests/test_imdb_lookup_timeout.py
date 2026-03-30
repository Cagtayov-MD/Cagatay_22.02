"""test_imdb_lookup_timeout.py

IMDb DuckDB aramasının 120 saniyelik timeout mekanizmasını test eder.

Senaryolar:
  TO-01: DB yokken lookup hızlı başarısız döner (hata yakalanır)
  TO-02: _IMDB_LOOKUP_TIMEOUT_SEC sabiti 0 < değer <= 300 arasında
  TO-03: Timeout tetiklendiğinde IMDBLookupResult(matched=False, reason="timeout") döner
"""

import os
import sys
import concurrent.futures
from unittest.mock import patch, MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_to01_missing_db_returns_fast():
    """DB dosyası yoksa lookup hata yakalanıp hızlıca döner."""
    from core.imdb_lookup import IMDBLookup

    lookup = IMDBLookup()
    lookup._db_path = "/nonexistent/path/imdb.duckdb"
    cdata = {"film_title": "MAGYAROK", "cast": [], "directors": []}
    result = lookup.lookup(cdata)
    assert result.matched is False


def test_to02_timeout_constant_is_reasonable():
    """Timeout sabiti makul aralıkta (0-300 saniye)."""
    from core.imdb_lookup import _IMDB_LOOKUP_TIMEOUT_SEC

    assert 0 < _IMDB_LOOKUP_TIMEOUT_SEC <= 300


def test_to03_timeout_returns_not_matched():
    """Future.result TimeoutError fırlatırsa IMDBLookupResult(matched=False, reason='timeout') döner."""
    from core.imdb_lookup import IMDBLookup, IMDBLookupResult

    lookup = IMDBLookup()
    lookup._db_path = "/fake.duckdb"

    # Future.result'ın TimeoutError fırlatmasını simüle et — gerçekten uyumadan
    mock_future = MagicMock()
    mock_future.result.side_effect = concurrent.futures.TimeoutError()

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future
    mock_executor.shutdown = MagicMock()

    with patch("concurrent.futures.ThreadPoolExecutor", return_value=mock_executor):
        cdata = {"film_title": "TEST", "cast": [], "directors": []}
        result = lookup.lookup(cdata)

    assert result.matched is False
    assert result.reason == "timeout"
