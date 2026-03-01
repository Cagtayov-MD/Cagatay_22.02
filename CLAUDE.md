# CLAUDE.md — Arsiv Decode / VİTOS

Comprehensive guide for AI assistants working on this codebase.

---

## Project Overview

**Arsiv Decode** (internal engine name: **VİTOS v7** — Video Intelligence & Transcription Orchestration System) is a Python application that automatically extracts cast/crew credits and transcripts from broadcast video files.

Core capabilities:
- OCR-based credits extraction from opening/closing segments (PaddleOCR GPU)
- Turkish name recognition and correction (356k-entry SQLite database)
- Audio transcription and speaker diarization (WhisperX + pyannote)
- Cast verification via TMDB API
- LLM-based noise filtering (Ollama: llama3.1:8b, qwen3-vl:8b)
- PySide6 dark-theme GUI + headless CLI mode

**Primary language**: Python 3.10. Code comments and UI strings are predominantly Turkish; module docstrings are Turkish. English is used in test files and CI configs.

---

## Repository Layout

```
Cagatay_22.02/
├── .github/workflows/
│   ├── ci.yml              # Quick CI: syntax + smoke test (ubuntu-latest)
│   ├── fulltest.yml        # Full GPU test (self-hosted Windows, manual dispatch)
│   └── ocr-test.yml        # OCR-specific test workflow
├── test/
│   ├── test.mp4            # Required CI asset
│   └── test.wav            # Required CI asset
└── Project/
    ├── main.py             # Entry point (GUI + headless CLI)
    ├── requirements.txt    # Main venv (PaddleOCR, PySide6, etc.)
    ├── requirements_audio.txt  # Audio venv (PyTorch, WhisperX, pyannote)
    ├── .env.template       # Environment variable template
    ├── test_config.json    # Default test pipeline config
    ├── test_config_ocr_only.json
    ├── test_runner.py      # CLI batch test runner
    ├── run_ocr_test.py     # OCR-specific test runner
    ├── watch_folder.py     # Auto-folder watcher
    ├── run_test.bat        # Windows batch test launcher
    ├── run_repo_only.ps1   # PowerShell runner
    ├── audio/              # Audio pipeline package (re-exports from core/)
    │   ├── audio_pipeline.py
    │   ├── stages/         # denoise, diarize, extract, post_process, transcribe
    │   └── utils/          # vram_manager
    ├── config/
    │   ├── runtime_paths.py        # Central path + API key resolution
    │   ├── profile_loader.py       # Content profile loader
    │   ├── content_profiles.json   # Profile definitions (FilmDizi, Spor, etc.)
    │   ├── profiles.json
    │   └── api_keys.json           # (gitignored in practice, template provided)
    ├── core/               # Primary processing engines
    │   ├── pipeline_runner.py      # Main DAG orchestrator (PipelineRunner)
    │   ├── frame_extractor.py      # ffmpeg-based frame extraction
    │   ├── text_filter.py          # MSER text region detection
    │   ├── ocr_engine.py           # PaddleOCR + 8-stage filtering pipeline
    │   ├── layout_analyzer.py      # OCR bbox → character↔actor pair detection
    │   ├── credits_parser.py       # OCR lines → structured cast/crew data
    │   ├── turkish_name_db.py      # 356k Turkish name database (SQLite + DP split)
    │   ├── qwen_verifier.py        # Qwen VL model OCR line verification
    │   ├── llm_cast_filter.py      # Ollama LLM cast noise rejection
    │   ├── tmdb_verify.py          # TMDB API cast verification
    │   ├── export_engine.py        # JSON + TXT report generation
    │   ├── queue_manager.py        # Video processing queue
    │   ├── audio_bridge.py         # Bridge: PipelineRunner → audio pipeline
    │   ├── audio_pipeline.py       # Audio pipeline orchestrator
    │   ├── audio_worker.py         # Audio worker thread
    │   ├── asr_pipeline.py         # ASR pipeline
    │   ├── audio_io.py             # Audio I/O utilities
    │   ├── google_ocr_engine.py    # Google Vision OCR (optional fallback)
    │   ├── google_video_intelligence.py
    │   ├── match_parser.py         # Sports match result parser
    │   ├── post_process.py         # Post-processing stage
    │   ├── transcribe.py           # Transcription logic
    │   ├── denoise.py              # Audio denoising
    │   ├── diarize.py              # Speaker diarization
    │   ├── extract.py              # Audio extraction
    │   ├── vram_manager.py         # GPU VRAM management
    │   └── text_filter.py          # Text region filtering
    ├── enrichment/         # Placeholder (not yet implemented)
    ├── logs/
    │   └── stats_log.jsonl         # Pipeline run statistics
    ├── schemas/
    │   └── audio_result.schema.json
    ├── tests/              # pytest unit tests
    │   ├── video_smoke.py          # Smoke: OCR pipeline input validation
    │   ├── asr_smoke.py            # Smoke: ASR pipeline
    │   ├── test_hardening.py       # Validation & error handling tests
    │   ├── test_asr_ocr_fixes.py
    │   ├── test_audio_result_schema.py
    │   ├── test_audio_worker_io.py
    │   ├── test_database_writer.py
    │   ├── test_ocr_fuzzy_dedup.py
    │   ├── test_qwen_verifier_name_guard.py
    │   └── test_transcribe_compute_type.py
    └── ui/
        ├── main_window.py          # PySide6 main window (dark theme)
        ├── dag_widget.py           # DAG pipeline visualization widget
        ├── dag_definitions.py      # DAG node/edge definitions per profile
        └── queue_tab.py            # Video queue management tab
```

---

## Application Entry Points

### GUI Mode (default)
```bash
cd Project
python main.py
# Optional: pass a video path as first argument to auto-load it
python main.py /path/to/video.mp4
```

### Headless CLI Mode
```bash
HEADLESS=1 python main.py /path/to/video.mp4

# With options
HEADLESS=1 SCOPE=video_only FIRST_MIN=2.0 LAST_MIN=6.0 python main.py video.mp4
HEADLESS=1 CONTENT_PROFILE=FilmDizi python main.py video.mp4
```

Environment variables for headless mode:
- `SCOPE`: `video_only` | `audio_only` | `video+audio` (default: `video+audio`)
- `FIRST_MIN`: opening segment duration in minutes (default: `1.0`)
- `LAST_MIN`: closing segment duration in minutes (default: `1.0`)
- `CONTENT_PROFILE`: profile name from `content_profiles.json`

---

## Pipeline Architecture

`PipelineRunner.run()` executes stages sequentially. Each stage is logged and timed. Stage stats are recorded in `StatsLogger`.

```
[1/6] INGEST          — ffprobe video metadata
[2/6] FRAME_EXTRACT   — Extract frames from first/last N minutes (ffmpeg)
[3/6] TEXT_FILTER     — MSER text region detection → candidate frames
[4/6] OCR_CREDITS     — PaddleOCR (GPU) → 8-stage filter → QwenVerifier
[5/6] CREDITS_PARSE   — OCR lines → cast/crew/companies structure
[LLM] LLM_CAST_FILTER — Ollama LLM noise rejection (llama3.1:8b)
[6/6] TMDB_VERIFY     — TMDB API cast match + confidence scoring
[AUDIO]               — Parallel: extract→(denoise)→(diarize)→transcribe
[EXPORT]              — JSON report + TXT report + transcript
[DATABASE]            — Archive copy with dual OCR/pipeline scores
```

### Scope Modes
- `video_only`: Only OCR pipeline runs (stages 1–6 + EXPORT)
- `audio_only`: Only audio pipeline runs (AUDIO + EXPORT)
- `video+audio`: Both pipelines run

### OCR 8-Stage Filtering Pipeline (`OCREngine.process_frames`)
1. `_noise_filter` — Remove results with >25% noise characters
2. `_length_filter` — Minimum 2-character texts only
3. `_confidence_filter` — Drop results below 0.50 PaddleOCR confidence
4. `_digit_noise_filter` — Drop if >55% digits (keeps 4-digit years 1900–2100)
5. `_blacklist_filter` — Regex blacklist (URLs, timecodes, watermarks, etc.)
6. `_name_split_pass` — Split concatenated names (e.g., "SEBNEMSONMEZ" → "SEBNEM SONMEZ") using TurkishNameDB DP split
7. `_fuzzy_dedup` — Timecode-aware fuzzy deduplication (RapidFuzz)
8. `_persistence_and_watermark` — Group by appearance; drop if seen ≥15 times (watermark); single-occurrence texts get 30% confidence penalty

### PaddleOCR Init Strategy
`OCREngine._init_paddle()` tries five fallback attempts (3.x GPU → 2.x GPU → 3.x CPU → 2.x CPU → minimal) and automatically prunes unknown/mutually-exclusive arguments. Always check actual device via `paddle.device.get_device()`.

---

## Content Profiles

Defined in `Project/config/content_profiles.json`. Each profile controls:
- `scope`: pipeline scope
- `first_segment_minutes` / `last_segment_minutes`: how much of the video to analyze
- `text_filter_threshold`, `text_filter_mser_min_boxes`, `text_filter_max_per_segment`
- `audio_stages`: list of audio stages to run
- `whisper_model`, `whisper_language`
- `ocr_enabled`, `match_parse_enabled`

Currently active profiles: `FilmDizi`, `Spor`. Placeholders: `StudyoProgram`, `MuzikProgram`, `KisaHaber`.

For `Spor` profile, `match_parse_enabled=true` skips TMDB and runs `match_parser` instead (future implementation).

---

## Dual Virtual Environment Architecture

The project uses **two separate Python environments** due to PyTorch/PaddlePaddle conflicts:

| venv | Purpose | Key packages |
|------|---------|--------------|
| `venv` (main) | Video OCR pipeline + GUI | PaddleOCR, PaddlePaddle-GPU, PySide6, OpenCV, RapidFuzz, Pillow |
| `venv_audio` | Audio transcription | PyTorch (CUDA), WhisperX, pyannote.audio, deepfilternet |

**Installing dependencies:**
```bash
# Main venv (OCR + GUI)
pip install -r Project/requirements.txt
# For GPU: pip install paddlepaddle-gpu  (CUDA-version specific)

# Audio venv
pip install -r Project/requirements_audio.txt
# PyTorch: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note**: NumPy must be `>=1.24,<2.0` in both environments. PaddleOCR 3.x (PP-OCRv5) recommends NumPy 1.26.x.

---

## Configuration & Environment Variables

Copy `.env.template` to `.env` in the `Project/` directory:

```bash
cp Project/.env.template Project/.env
```

| Variable | Purpose | Default |
|----------|---------|---------|
| `FFMPEG_BIN_DIR` | Path to ffmpeg/ffprobe binaries | `~/Source/ffmpeg/bin` |
| `TMDB_API_KEY` | TMDB API key (v3) | from `config/api_keys.json` |
| `TMDB_BEARER_TOKEN` | TMDB Bearer token (v4, optional) | — |
| `TMDB_LANGUAGE` | TMDB response language | `tr-TR` |
| `GOOGLE_KEYS_JSON` | Google service account JSON path | — |
| `GOOGLE_APPLICATION_CREDENTIALS` | Same as above (alt) | — |
| `NAME_DB_DIR` | Turkish name database directory | `~/Source/name_db` |
| `LOGOLAR_DIR` | TV channel logo images directory | `~/Source/Logo` |
| `OLLAMA_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model for cast filtering | `llama3.1:8b` |
| `HF_TOKEN` | HuggingFace token (for pyannote diarization) | — |
| `QWEN_MODEL` | Qwen VL model for OCR verification | `qwen3-vl:8b` |
| `QWEN_THRESHOLD` | Qwen confidence threshold | `0.80` |
| `OCR_LANG` | PaddleOCR language | `tr` |
| `OCR_VERSION` | PaddleOCR model version override | — |
| `NAME_DB_PATH` | Direct path to SQLite names.db | auto-detected |
| `VITOS_DATABASE_ROOT` | Output archive root | `D:\DATABASE` |

All paths are resolved via `config/runtime_paths.py` with `lru_cache` lazy loading.

---

## Testing

### CI Tests (run on every push — no GPU required)
```bash
# 1. Syntax check
python -m compileall Project

# 2. Smoke test (requires test/test.mp4 to be non-empty)
mkdir -p artifacts
python Project/tests/video_smoke.py --video test/test.mp4 --out artifacts/ocr_smoke.json
```

### Unit Tests (pytest)
```bash
cd Project
pytest tests/ -v

# Run specific test file
pytest tests/test_hardening.py -v
pytest tests/test_ocr_fuzzy_dedup.py -v
```

### Full GPU Test (manual, Windows self-hosted runner)
Triggered via `workflow_dispatch` on GitHub Actions. Requires:
- Self-hosted runner with GPU
- `F:\Root\venv\Scripts\python.exe`
- `F:\test\full_test.mp4`
- ffmpeg on PATH

### Local Pipeline Test
```bash
cd Project
# OCR only
python test_runner.py --config test_config_ocr_only.json /path/to/video.mp4

# Full pipeline
python test_runner.py --config test_config.json /path/to/video.mp4

# Watch a folder for new videos
python watch_folder.py /path/to/folder
```

**Test config keys** (`test_config.json`):
```json
{
  "scope": "video+audio",
  "first_min": 2.0,
  "last_min": 6.0,
  "difficulty": "heavy",
  "use_gpu": true,
  "program_type": "film_dizi",
  "output_root": "/path/to/output"
}
```

---

## Output Files

Each run produces a timestamped work directory `arsiv_{video_stem}_{YYYYMMDD_HHMMSS}/`:

| File | Description |
|------|-------------|
| `{stem}_report.json` | Full structured pipeline output |
| `{stem}.txt` | Human-readable credits report |
| `{stem}_transcript.txt` | Audio transcript |

The `DATABASE` archive layer additionally writes:
- `{stem}_{ts}_report.json` — report copy
- `{stem}_{ts}_ocr_scores.json` — per-line dual OCR/pipeline scores
- `{stem}_{ts}_credits_raw.json` — pre-LLM-filter credits snapshot
- `{stem}_{ts}_transcript.json` — structured transcript
- `{stem}_{ts}_debug.log` — full pipeline log messages

---

## Key Conventions

### Paths — Always Cross-Platform
- Use `pathlib.Path` everywhere; never hardcode Windows paths like `F:\` or `D:\`
- Resolve paths through `config/runtime_paths.py`
- Test with `Path(...).resolve()` before use

### Turkish Text Handling
- Turkish has special characters: `ç ğ ı ö ş ü` (and uppercase `Ç Ğ İ Ö Ş Ü`)
- Note: `İ` (dotted I) is the uppercase of `i`; `I` (dotless) is uppercase of `ı`
- Use `utils/turkish.py` for Turkish-specific text operations
- Use `utils/unicode_io.py` (`imread_unicode`) for OpenCV image reads with Unicode paths
- Always open files with `encoding="utf-8"` and handle `errors="replace"` in console output

### PaddleOCR Version Compatibility
- Support both PaddleOCR 2.x and 3.x APIs simultaneously
- `OCREngine._init_paddle()` auto-detects and adapts to the installed version
- `OCREngine._run_paddle()` handles both `o.ocr()` (2.x) and `o.predict()` (3.x)
- `OCREngine._iter_paddle_lines()` normalizes both output formats

### Logging Pattern
All major classes accept a `log_cb` callback. Use the `_log()` pattern:
```python
def _log(self, msg):
    if self._log_cb:
        self._log_cb(msg)
    print(msg)
```
Never use module-level `logging` for pipeline messages — always thread through `log_cb`.

### Configuration Safety
- Always convert config values with type safety:
  ```python
  try:
      val = float(self.config.get("some_key", default))
      val = max(min_val, min(max_val, val))  # clamp
  except (ValueError, TypeError):
      val = default
  ```
- Boolean config values: use `bool(self.config.get("key", True))`
- Never raise on missing optional config keys; always have defaults

### Audio Pipeline Stages
Audio stages are selected based on `program_type`:
- `film_dizi` / `kisa_haber`: `["extract", "transcribe"]` — no denoise/diarize
- Others (mac, muzik_programi, etc.): `["extract", "denoise", "diarize", "transcribe", "post_process"]`

`compute_type` is intentionally left `None` in config so `TranscribeStage` auto-selects (`float16` for CUDA, `int8` for CPU).

### Import Guards for Optional Dependencies
```python
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
```
Use `HAS_PADDLE`, `HAS_FUZZ` etc. guards — never assume all packages are installed.

### No Secrets in Code
- API keys, tokens, paths belong in `.env` or `config/api_keys.json` (gitignored)
- Never commit actual keys; use `.env.template` as reference
- `config/api_keys.json` in the repo is a template (verify it contains no real keys)

### Headless Mode Compatibility
- `main.py` checks `HEADLESS=1` at the very top, before any Qt imports
- Headless mode silences PaddleOCR ppocr logger and reconfigures stdout to UTF-8
- All UI code must be guarded — never import PySide6 at module level in `core/`

---

## External Services Summary

| Service | Used For | Required? |
|---------|---------|-----------|
| **ffmpeg/ffprobe** | Frame extraction, audio extraction | Yes (always) |
| **PaddleOCR (local GPU)** | Credits OCR | Yes (for video scope) |
| **Ollama** (`llama3.1:8b`) | LLM cast noise filtering | Optional (graceful skip) |
| **Ollama** (`qwen3-vl:8b`) | OCR line verification | Optional |
| **TMDB API** | Cast/crew verification | Optional (skipped if no key) |
| **HuggingFace** (pyannote) | Speaker diarization | Optional (audio only) |
| **Google Cloud Vision** | OCR fallback | Optional |
| **Google Video Intelligence** | Video analysis | Optional |

---

## Git Workflow

- Main development branch: `master` / `main`
- Feature branches: descriptive names with issue numbers (e.g., `fix-ocr-dedup-#50`)
- Commit messages: imperative, Turkish or English, reference issue numbers when applicable
- CI runs on all branch pushes; must pass before merge

### Branch for AI Work
When working as an AI assistant on this repo, develop on `claude/claude-md-mm8elf1g0zoprtxc-5A1wG` and push with:
```bash
git push -u origin claude/claude-md-mm8elf1g0zoprtxc-5A1wG
```

---

## Common Pitfalls to Avoid

1. **Hardcoded Windows paths** — always use `pathlib.Path` with env var fallbacks (bug fixed in #43)
2. **Float16 on CPU** — `TranscribeStage` must auto-detect `compute_type`; never force `float16` on non-CUDA (bug fixed in #44, #45)
3. **Double diarization** — do not pass `hf_token` to `transcribe.run()`; it's handled by `diarize` stage (bug fixed in BUG-01)
4. **PaddleOCR generator truthiness** — `_iter_paddle_lines` is a generator; always `list()` it before emptiness check
5. **TVDB** — TVDB was fully removed; do not re-add it; TMDB is the only metadata source
6. **Ollama pre-check** — before LLM calls, verify Ollama is reachable; fail gracefully (bug fixed in #42)
7. **Turkish I/İ confusion** — `"istanbul".upper()` returns `"ISTANBUL"` in English locale but `"İSTANBUL"` in Turkish locale; always use explicit character mapping
8. **OCR variant generator** — `_prepare_variants()` returns a list capped at `max_variants` per difficulty level; don't add variants without updating the cap
