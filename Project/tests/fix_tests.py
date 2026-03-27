"""
fix_tests.py
============
tests/test_runtime_hardening.py dosyasındaki iki sorunu otomatik düzeltir:

1. monkeypatch hedef hatası:
   - YANLIŞ:  monkeypatch.setattr(subprocess, "run", fake_run)
   - DOĞRU:   monkeypatch.setattr("sport_pipeline.subprocess.run", fake_run)

2. test_high_confidence_bypass:
   - bypass davranışı kasıtlı olarak kaldırıldığı için
     testi yeni davranışa göre günceller.

Kullanım:
    python fix_tests.py                          # dry-run (değişiklikleri gösterir)
    python fix_tests.py --apply                  # dosyayı yerinde değiştirir
    python fix_tests.py --path baska/yol/test.py # farklı dosya yolu
"""

import argparse
import re
import sys
from pathlib import Path

DEFAULT_TEST_PATH = Path("tests/test_runtime_hardening.py")

# ── 1. Monkeypatch hedef düzeltmeleri ──────────────────────────────────────────
# subprocess modülünü doğrudan patch'lemek yerine,
# onu kullanan modül üzerinden patch'le

MONKEYPATCH_PATTERNS = [
    # monkeypatch.setattr(subprocess, "run", fake_run)
    (
        re.compile(
            r'monkeypatch\.setattr\(\s*subprocess\s*,\s*["\']run["\']\s*,\s*(\w+)\s*\)'
        ),
        lambda m: f'monkeypatch.setattr("sport_pipeline.subprocess.run", {m.group(1)})',
    ),
    # monkeypatch.setattr("subprocess.run", fake_run)
    (
        re.compile(
            r'monkeypatch\.setattr\(\s*["\']subprocess\.run["\']\s*,\s*(\w+)\s*\)'
        ),
        lambda m: f'monkeypatch.setattr("sport_pipeline.subprocess.run", {m.group(1)})',
    ),
]


# ── 2. Bypass test düzeltmesi ──────────────────────────────────────────────────
# Bypass davranışı kaldırıldı → test artık bypass'ın OLMADIĞINI doğrulamalı

BYPASS_OLD_PATTERN = re.compile(
    r'(def test_high_confidence_bypass\b.*?)(?=\ndef |\nclass |\Z)',
    re.DOTALL,
)

BYPASS_NEW_BODY = '''\
def test_high_confidence_bypass_removed(tmp_path):
    """Yüksek güven skoru artık doğrulama adımını atlamıyor.

    Bypass davranışı kasıtlı olarak kaldırıldı; bu test
    yüksek güven skorunda bile doğrulamanın çalıştığını doğrular.
    """
    # --- arrange ---
    from sport_pipeline import validate_prediction  # kendi import yolunuza göre ayarlayın

    high_confidence_result = {
        "label": "goal",
        "confidence": 0.99,
    }

    # --- act ---
    result = validate_prediction(high_confidence_result)

    # --- assert: doğrulama ATLANIYOR OLMAMALI ---
    assert result.get("validated") is True, (
        "Yüksek güven skoru bile olsa doğrulama adımı çalışmalı"
    )
    # Eski davranış (bypass) artık geçerli değil:
    #   assert result.get("skipped_validation") is True   ← KALDIRILDI
'''


def fix_content(content: str) -> tuple[str, list[str]]:
    """İçeriği düzeltir, yapılan değişikliklerin listesini döndürür."""
    changes: list[str] = []

    # 1) monkeypatch hedeflerini düzelt
    for pattern, replacement in MONKEYPATCH_PATTERNS:
        new_content, n = pattern.subn(replacement, content)
        if n:
            changes.append(
                f"monkeypatch hedefi düzeltildi ({n} yer) → sport_pipeline.subprocess.run"
            )
            content = new_content

    # 2) bypass testini güncelle
    if BYPASS_OLD_PATTERN.search(content):
        content = BYPASS_OLD_PATTERN.sub(BYPASS_NEW_BODY, content)
        changes.append(
            "test_high_confidence_bypass → test_high_confidence_bypass_removed olarak güncellendi"
        )

    # 3) fake_run içine debug stack trace ekle (opsiyonel ama faydalı)
    if "fake_run" in content and "traceback.format_stack" not in content:
        # fake_run tanımının hemen içine stack trace ekleme
        fake_run_pattern = re.compile(
            r'(def fake_run\(\*args,\s*\*\*kwargs\):)\n'
        )
        debug_addition = (
            r'\1\n'
            r'        import traceback\n'
            r'        captured["_caller"] = "".join(traceback.format_stack())\n'
        )
        new_content = fake_run_pattern.sub(debug_addition, content)
        if new_content != content:
            changes.append("fake_run'a debug stack trace eklendi (captured['_caller'])")
            content = new_content

    return content, changes


def main():
    parser = argparse.ArgumentParser(description="test_runtime_hardening.py düzeltici")
    parser.add_argument("--apply", action="store_true", help="Değişiklikleri dosyaya yaz")
    parser.add_argument("--path", type=Path, default=DEFAULT_TEST_PATH, help="Test dosyası yolu")
    args = parser.parse_args()

    test_file: Path = args.path

    if not test_file.exists():
        print(f"HATA: {test_file} bulunamadı.", file=sys.stderr)
        print("  --path ile doğru yolu belirtin.", file=sys.stderr)
        sys.exit(1)

    original = test_file.read_text(encoding="utf-8")
    fixed, changes = fix_content(original)

    if not changes:
        print("Değişiklik gerekmiyor — dosya zaten düzgün görünüyor.")
        return

    print(f"{'UYGULANACAK' if args.apply else 'BULUNAN'} değişiklikler ({test_file}):\n")
    for i, c in enumerate(changes, 1):
        print(f"  {i}. {c}")

    if args.apply:
        # yedek al
        backup = test_file.with_suffix(".py.bak")
        backup.write_text(original, encoding="utf-8")
        print(f"\n  Yedek: {backup}")

        test_file.write_text(fixed, encoding="utf-8")
        print(f"  Dosya güncellendi: {test_file}")
        print("\n  Şimdi testleri çalıştırın:")
        print(f"    pytest {test_file} -v")
    else:
        print("\n  Uygulamak için:  python fix_tests.py --apply")
        print("\n--- Önizleme (diff) ---\n")
        try:
            import difflib
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                fixed.splitlines(keepends=True),
                fromfile=f"{test_file} (eski)",
                tofile=f"{test_file} (yeni)",
                n=3,
            )
            sys.stdout.writelines(diff)
        except Exception:
            print(fixed)


if __name__ == "__main__":
    main()
