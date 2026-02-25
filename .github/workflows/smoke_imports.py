import importlib
import sys

# Bu listeyi projendeki gerçek modül adlarına göre genişletebilirsin.
CANDIDATES = [
    "core.pipeline_runner",
    "core.asr_engine",
    "core.audio_bridge",
    "ocr_engine",
    "credits_parser",
    "tmdb_verify",
]

failed = []
for m in CANDIDATES:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, repr(e)))

if failed:
    print("IMPORT FAIL:")
    for m, e in failed:
        print(f"  - {m}: {e}")
    sys.exit(1)

print("IMPORT OK")