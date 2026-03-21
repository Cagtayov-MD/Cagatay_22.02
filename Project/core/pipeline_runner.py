"""
pipeline_runner.py — DAG execution engine.

v2.0 Değişiklikler:
- TVDB tamamen kaldırıldı (dosya da silinmeli).
- mode (light/medium/heavy) → TextFilter.from_config() ile bağlandı.
- AudioBridge gerçekten entegre edildi — scope="video+audio" artık çalışıyor.
- google_json + logolar_dir parametreleri eklendi (path_resolver sabit yollar).
- BUG-01 (double diarization): transcribe.run() çağrısından hf_token kaldırıldı.
- TMDB: ID girmeden çalışır. film_adı + 2 oyuncu = %100 güven. Dosya adı anchor.
"""

import concurrent.futures
import copy
import json
import re
import shutil
import time
import os
from pathlib import Path
from datetime import datetime

from config.runtime_paths import get_tmdb_api_key, get_gemini_api_key, resolve_name_db_path

from core.frame_extractor import FrameExtractor
from core.text_filter import TextFilter
from core.ocr_engine import OCREngine
from core.qwen_ocr_engine import QwenOCREngine
from core.credits_parser import CreditsParser
from core.export_engine import ExportEngine, _map_crew_to_roles
from core.turkish_name_db import TurkishNameDB
from core.qwen_verifier import QwenVerifier
from core.llm_cast_filter import LLMCastFilter
from core.vlm_reader import VLMReader
from core.name_verify import NameVerifier
from core.person_verify import PersonVerifier
from core.xml_sidecar import resolve_xml_sidecar, XmlSidecarInfo
from utils.stats_logger import StatsLogger

# OneOCR — opsiyonel (sadece Windows 11)
try:
    from core.oneocr_engine import OneOCREngine
    _HAS_ONEOCR = True
except ImportError:
    _HAS_ONEOCR = False

# HybridOCRRouter — oneocr + Qwen VLM hibrit motor
try:
    from core.hybrid_ocr_router import HybridOCRRouter
    _HAS_HYBRID = True
except ImportError:
    _HAS_HYBRID = False


_BLOK2_FUZZY_THRESHOLD = 85  # BLOK2 dedup: iki satır bu eşiğin üzerindeyse aynı kabul edilir


def _safe_path(path: Path) -> Path:
    """Dosya çakışması varsa _2, _3 ... ekleyerek güvenli bir yol döndür."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _merge_tmdb_person_evidence(cdata: dict, person_evidence: list, log_cb) -> int:
    """Strateji C/D kişi kanıtlarını cdata cast/crew'a ekle (kurtarma).

    Yalnızca:
    - Rolü 'cast' veya 'director' olan kişiler eklenir.
    - Mevcut isimlerle çakışan kişiler atlanır (duplicate önleme).
    - Eklenen kayıtlar 'tmdb_person_recovery' provenance etiketiyle işaretlenir.

    Return: eklenen kayıt sayısı.
    """
    def _n(s: str) -> str:
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    # Mevcut cast isimlerini normalize et
    existing_cast = cdata.get("cast") or []
    existing_cast_norm = {
        _n(row.get("actor_name") or row.get("actor") or "")
        for row in existing_cast
        if isinstance(row, dict)
    }

    # Mevcut crew/directors isimlerini normalize et
    existing_crew_norm = {
        _n(row.get("name") or "")
        for row in (cdata.get("crew") or []) + (cdata.get("directors") or [])
        if isinstance(row, dict)
    }

    added = 0

    for pe in person_evidence:
        role = pe.get("role", "cast")
        ocr_name = (pe.get("ocr_name") or "").strip()
        tmdb_name = (pe.get("tmdb_name") or "").strip()
        tmdb_id = pe.get("tmdb_id")
        strategy = pe.get("source_strategy", "?")

        # Kullanılacak isim: TMDB kanonik adı öncelikli
        name_to_use = tmdb_name or ocr_name
        if not name_to_use:
            continue

        name_norm = _n(name_to_use)
        if not name_norm:
            continue

        if role == "cast":
            if name_norm in existing_cast_norm:
                continue
            existing_cast.append({
                "actor_name": name_to_use,
                "actor": name_to_use,
                "confidence": 0.7,
                "raw": "tmdb_person_recovery",
                "source": "tmdb_person_recovery",
                "tmdb_id": tmdb_id,
                "tmdb_strategy": strategy,
            })
            existing_cast_norm.add(name_norm)
            added += 1

        elif role == "director":
            if name_norm in existing_crew_norm:
                continue
            existing_crew = cdata.get("crew") or []
            existing_crew.append({
                "name": name_to_use,
                "job": "Director",
                "role": "Director",
                "role_tr": "Yönetmen",
                "confidence": 0.7,
                "raw": "tmdb_person_recovery",
                "source": "tmdb_person_recovery",
                "tmdb_id": tmdb_id,
                "tmdb_strategy": strategy,
            })
            cdata["crew"] = existing_crew
            existing_crew_norm.add(name_norm)
            added += 1

    if added > 0:
        cdata["cast"] = existing_cast

    return added


class PipelineRunner:
    """Film/Dizi analiz pipeline."""

    def __init__(self, ffmpeg, ffprobe, tesseract="", config=None,
                 output_root="", google_json="", logolar_dir=""):
        self.config       = config or {}
        self.output_root  = output_root
        self.google_json  = google_json
        self.logolar_dir  = logolar_dir
        self.stats        = StatsLogger(
            os.path.join(output_root, "logs") if output_root else "logs")
        self.stage_stats  = {}
        self._log_cb      = None
        self._log_messages: list[str] = []

        self._ffmpeg    = ffmpeg
        self._ffprobe   = ffprobe
        self._tesseract = tesseract
        self._extractor   = None
        self._text_filter = None
        self._ocr_engine  = None

        # TurkishNameDB
        name_db_path = (
            self.config.get("name_db_path") or
            os.environ.get("NAME_DB_PATH") or
            resolve_name_db_path()
        )
        self._name_db = TurkishNameDB(
            sql_path=name_db_path if os.path.isfile(name_db_path) else "",
        )

        # VLM / QwenVerifier — config precedence:
        # 1. vlm_verify / vlm_enabled (config)
        # 2. VLM_ENABLED (env)
        # 3. qwen_verify (config, legacy)
        # 4. default True
        vlm_enabled_cfg = self.config.get("vlm_verify",
                          self.config.get("vlm_enabled", None))
        if vlm_enabled_cfg is None:
            env_enabled = os.environ.get("VLM_ENABLED", "")
            if env_enabled:
                vlm_enabled = env_enabled.lower() not in ("0", "false", "no")
            else:
                vlm_enabled = bool(self.config.get("qwen_verify", True))
        else:
            vlm_enabled = bool(vlm_enabled_cfg)

        # Model precedence: vlm_model (config) → VLM_MODEL (env) → qwen_model (config) → QWEN_MODEL (env) → default
        vlm_model = (
            self.config.get("vlm_model") or
            os.environ.get("VLM_MODEL") or
            self.config.get("qwen_model") or
            os.environ.get("QWEN_MODEL") or
            "glm4.6v-flash:q4_K_M"
        )

        # Threshold precedence: vlm_threshold (config) → VLM_THRESHOLD (env) → qwen_threshold (config) → QWEN_THRESHOLD (env) → default
        _thresh_raw = (
            self.config.get("vlm_threshold") or
            os.environ.get("VLM_THRESHOLD") or
            self.config.get("qwen_threshold") or
            os.environ.get("QWEN_THRESHOLD") or
            0.80
        )
        # Safe float conversion with validation
        try:
            vlm_thresh = float(_thresh_raw)
            # Clamp to valid range [0.0, 1.0]
            vlm_thresh = max(0.0, min(1.0, vlm_thresh))
        except (ValueError, TypeError) as e:
            self._log(f"  [INIT] Geçersiz vlm_threshold değeri, varsayılan 0.80 kullanılıyor: {e}")
            vlm_thresh = 0.80
        self._qwen = QwenVerifier(
            model=vlm_model,
            confidence_threshold=vlm_thresh,
            enabled=vlm_enabled,
            name_checker=self._name_db.is_name,
        )

        # LLMCastFilter config
        self._llm_filter_enabled = bool(self.config.get("llm_cast_filter", True))

        # BLOK2 (VLM derin okuma) — varsayılan KAPALI
        self._blok2_enabled = bool(self.config.get("blok2_enabled", False))

        # VLMReader (VLM-as-OCR) — off by default
        vlm_ocr_enabled = bool(
            self.config.get("vlm_ocr_enabled",
            self.config.get("use_vlm_for_ocr", False))
        )
        self._vlm_reader = VLMReader(
            model=vlm_model,
            enabled=vlm_ocr_enabled,
        )

    def set_log_callback(self, cb):
        self._log_cb = cb
        self._name_db._log = cb

    def _log(self, msg):
        if self._log_cb:
            self._log_cb(msg)
        self._log_messages.append(str(msg))
        print(msg, flush=True)
        # Incremental log — her mesaj anında dosyaya
        if hasattr(self, '_live_log_path') and self._live_log_path:
            try:
                with open(self._live_log_path, "a", encoding="utf-8") as f:
                    f.write(str(msg) + "\n")
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────
    def run(self, video_path, scope=None,
            first_min=6.0, last_min=10.0, content_profile: dict | None = None) -> dict:
        # Scope belirleme: açık parametre > profil > config > varsayılan
        # ÖNEMLİ: Kullanıcı arayüzünden gelen açık scope seçimi EN YÜKSEK ÖNCELİKTEDİR.
        # Profil scope'u yalnızca açık bir seçim yapılmadığında (None) devreye girer.
        profile_name = "FilmDizi-Hybrid"
        if content_profile:
            profile_name = content_profile.get("_name", "FilmDizi-Hybrid")
            if scope is None:
                scope = content_profile.get("scope")
            try:
                first_min = float(content_profile.get("first_segment_minutes", first_min))
            except (ValueError, TypeError):
                print(f"  [Config] Geçersiz first_segment_minutes — varsayılan {first_min} dk kullanılıyor")
            try:
                last_min  = float(content_profile.get("last_segment_minutes", last_min))
            except (ValueError, TypeError):
                print(f"  [Config] Geçersiz last_segment_minutes — varsayılan {last_min} dk kullanılıyor")

            # ── Profil ayarlarını config'e merge et (ASR/OCR parametreleri) ──
            _profile_merge_keys = (
                "audio_stages", "whisper_model", "whisper_language",
                "compute_type", "beam_size", "ocr_engine",
                "tmdb_enabled", "tmdb_person_verify",
                "gemini_enabled", "llm_cast_filter", "blok2_enabled",
                "qwen_fallback_on_handwriting", "imdb_enabled",
            )
            for _pk in _profile_merge_keys:
                if _pk in content_profile:
                    self.config[_pk] = content_profile[_pk]

        # SporMaci profili → ayrı pipeline'a yönlendir
        if profile_name == "SporMaci":
            from core.sport_pipeline import run_sport_pipeline
            cdata: dict = {}
            merged_config = {**self.config}
            if content_profile:
                merged_config.update(content_profile)
            merged_config["ffmpeg"] = self._ffmpeg
            merged_config["ffprobe"] = self._ffprobe
            return run_sport_pipeline(video_path, merged_config, cdata)

        # Scope hâlâ None ise config'den veya varsayılandan al
        if scope is None:
            scope = self.config.get("scope", "video+audio")

        t0 = time.time()
        video_path = str(Path(video_path).resolve())
        vname = Path(video_path).stem
        film_title_from_filename = self._extract_film_title_from_filename(vname)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_root = (
            self.config.get("database_root") or
            os.environ.get("VITOS_DATABASE_ROOT") or
            r"D:\DATABASE\FilmDizi"
        )
        work_dir = os.path.join(db_root, f"arsiv_{vname}_{ts}")
        os.makedirs(work_dir, exist_ok=True)
        self._live_log_path = os.path.join(work_dir, "_live_debug.log")
        # BOM ile başlat — Notepad UTF-8 olarak doğru okusun
        with open(self._live_log_path, "w", encoding="utf-8-sig") as _f:
            pass

        self.stats.start_job(video_path, "WORKSTATION", scope, profile_name)
        self.stage_stats = {}
        self._log_messages = []

        self._log(f"\n{'='*60}")
        self._log(f"  VİTOS — Pipeline v7")
        self._log(f"{'='*60}")
        self._log(f"  Video  : {Path(video_path).name}")
        self._log(f"  Profil : {profile_name}")
        self._log(f"  Scope  : {scope}")
        self._log(f"  Giriş  : {first_min} dk | Çıkış: {last_min} dk")
        self._log(f"  Mod    : {self.config.get('difficulty','medium')}")
        self._log(f"  NameDB : {len(self._name_db)} isim yüklü")

        xml_info: XmlSidecarInfo | None = None
        try:
            # ── TextFilter mode bağlantısı ──────────────────────
            self._text_filter = TextFilter.from_config(self.config)

            self._extractor = FrameExtractor(
                self._ffmpeg, self._ffprobe, log_cb=self._log)

            self._log(f"\n  OCR motoru başlatılıyor...")
            _default_engine = "hybrid" if _HAS_ONEOCR else "qwen"
            _ocr_engine_type = (
                (content_profile or {}).get("ocr_engine") or
                self.config.get("ocr_engine", _default_engine)
            ).lower()

            # audio_only modunda OCR motoru başlatılmıyor — ses analizi için gereksiz
            if scope != "audio_only":
                if _ocr_engine_type == "hybrid":
                    if _HAS_HYBRID and _HAS_ONEOCR:
                        self._log(f"  [OCR] Motor: Hybrid (oneocr + Qwen VLM handwriting fallback)")
                        self._ocr_engine = HybridOCRRouter(
                            cfg=self.config,
                            log_cb=self._log,
                            name_db=self._name_db,
                        )
                    elif _HAS_ONEOCR:
                        self._log(f"  [OCR] Hybrid router yüklenemedi — OneOCR tek başına")
                        self._ocr_engine = OneOCREngine(
                            cfg=self.config,
                            log_cb=self._log,
                            name_db=self._name_db,
                        )
                    else:
                        self._log(f"  [OCR] oneocr kurulu değil — Qwen VLM'e fallback")
                        self._ocr_engine = QwenOCREngine(
                            cfg=self.config,
                            log_cb=self._log,
                            name_db=self._name_db,
                            ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
                        )
                elif _ocr_engine_type == "oneocr":
                    if _HAS_ONEOCR:
                        self._log(f"  [OCR] Motor: OneOCR (Windows Snipping Tool)")
                        self._ocr_engine = OneOCREngine(
                            cfg=self.config,
                            log_cb=self._log,
                            name_db=self._name_db,
                        )
                    else:
                        self._log(f"  [OCR] OneOCR kurulu değil — Qwen VLM'e fallback")
                        self._ocr_engine = QwenOCREngine(
                            cfg=self.config,
                            log_cb=self._log,
                            name_db=self._name_db,
                            ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
                        )
                elif _ocr_engine_type == "qwen":
                    self._log(f"  [OCR] Motor: Qwen2.5-VL (VLM tabanlı)")
                    self._ocr_engine = QwenOCREngine(
                        cfg=self.config,
                        log_cb=self._log,
                        name_db=self._name_db,
                        ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
                    )
                else:
                    self._log(f"  [OCR] Motor: PaddleOCR (GPU)")
                    self._ocr_engine = OCREngine(
                        use_gpu=self.config.get("use_gpu", True),
                        lang=self.config.get("ocr_languages", ["en"])[0],
                        cfg=self.config,
                        log_cb=self._log,
                        name_db=self._name_db,
                    )
            else:
                self._log(f"  [OCR] Scope=audio_only — OCR motoru başlatılmıyor (atlanıyor)")

            # ══ [1/6] INGEST ═══════════════════════════════════
            self._log(f"\n[1/6] INGEST")
            t = time.time()
            self.stats.start_stage("INGEST")
            info = self._extractor.probe_video(video_path)
            self._stage("INGEST", time.time() - t,
                        duration=info["duration_human"],
                        resolution=info["resolution"])
            self._log(f"  OK: {info['duration_human']} | "
                      f"{info['resolution']} | {info['fps']} FPS")
            xml_info = resolve_xml_sidecar(video_path, log_cb=self._log)

            # Audio başlatma: scope'a göre dallan
            audio_result = None
            audio_future = None
            executor = None
            if scope == "audio_only":
                # Sadece ses — seri çalıştır
                audio_result = self._run_audio(video_path, work_dir)
            elif scope == "video+audio":
                # Video + Ses — audio arka planda paralel çalışsın
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                audio_future = executor.submit(self._run_audio, video_path, work_dir)
                self._log("  [AUDIO] Ses analizi arka planda başlatıldı (paralel mod)")
            else:
                # video_only — ses analizi tamamen atlanıyor
                self._log("  [AUDIO] Scope=video_only — ses analizi atlanıyor")
                audio_result = {"status": "skipped", "reason": "scope=video_only"}

            # Varsayılanlar (audio_only için)
            ocr_lines = []
            cdata = self._empty_credits_data(film_title_from_filename)
            cdata_raw = None
            tmdb_result = None
            imdb_result = None

            # Profil bazlı OCR kontrolü
            ocr_enabled = bool(content_profile.get("ocr_enabled", True)) if content_profile else True

            if scope != "audio_only" and ocr_enabled:
                # ══ [2/6] FRAME_EXTRACT ════════════════════════════
                self._log(f"\n[2/6] FRAME_EXTRACT")
                t = time.time()
                self.stats.start_stage("FRAME_EXTRACT")
                fps = self.config.get("ocr_fps", 1.0)
                fd = self._extractor.extract_credits_frames(
                    video_path, work_dir, info, first_min, last_min, fps)
                self._stage("FRAME_EXTRACT", time.time() - t,
                            total=fd["total"],
                            entry=len(fd["entry"]),
                            exit_=len(fd["exit"]))
                self._log(f"  OK: {fd['total']} frame "
                          f"(giriş:{len(fd['entry'])} çıkış:{len(fd['exit'])})")

                # ══ [3/6] TEXT_FILTER ══════════════════════════════
                self._log(f"\n[3/6] TEXT_FILTER "
                          f"[mod={self.config.get('difficulty','medium')}]")
                t = time.time()
                self.stats.start_stage("TEXT_FILTER")
                all_frames = fd["entry"] + fd["exit"]
                candidates = self._text_filter.filter_frames(all_frames)
                if not candidates:
                    self._log(f"  !! Yazı adayı yok — fallback")
                    candidates = self._text_filter.fallback_filter(
                        fd["entry"], fd["exit"], 100)
                rej = 1 - len(candidates) / max(len(all_frames), 1)
                self._stage("TEXT_FILTER", time.time() - t,
                            input=len(all_frames), candidates=len(candidates))
                self._log(f"  OK: {len(all_frames)} -> {len(candidates)} "
                          f"aday (eleme: {rej:.0%})")

                # ══ [4/6] OCR_CREDITS ══════════════════════════════
                _engine_label = type(self._ocr_engine).__name__
                self._log(f"\n[4/6] OCR_CREDITS ({_engine_label})")
                t = time.time()
                self.stats.start_stage("OCR_CREDITS")
                ocr_lines, layout_pairs = self._ocr_engine.process_frames(
                    candidates, log_callback=self._log)
                ocr_time = time.time() - t

                ocr_lines = self._repair_turkish(ocr_lines)
                layout_pairs = self._repair_layout_pairs(layout_pairs)

                # Optional VLM-as-OCR fallback (off by default)
                if self._vlm_reader.enabled and self._vlm_reader.is_available():
                    vlm_r_t = time.time()
                    ocr_lines = self._vlm_reader.augment_ocr_lines(
                        ocr_lines, candidates, log_cb=self._log)
                    self._log(f"  [VLM-OCR] Tamamlandı ({time.time()-vlm_r_t:.1f}s)")

                # ── BLOK1 VLM Skip ──
                # TMDB araması için NameDB onarımı yeterli isim sağlıyor.
                # TMDB miss olursa BLOK2'de Qwen 2.5 (GPU) derin okuma yapacak.
                # VLM'i tekrar aktif etmek için aşağıdaki bloğu uncomment edin.
                self._log("  [VLM] BLOK1'de atlanıyor (TMDB arama için NameDB yeterli)")
                qwen_elapsed = 0.0
                # ── VLM'i geri açmak için bu bloğu uncomment edin: ──
                # qwen_t = time.time()
                # ocr_lines = self._qwen.verify(ocr_lines, log_cb=self._log,
                #                               resolution=info.get("resolution", ""))
                # qwen_elapsed = time.time() - qwen_t
                # if qwen_elapsed > 0.5:
                #     self._log(f"  [VLM] Doğrulama süresi: {qwen_elapsed:.1f}s")

                ocr_total = ocr_time + qwen_elapsed
                self._stage("OCR_CREDITS", ocr_total,
                            processed=len(candidates),
                            lines=len(ocr_lines),
                            layout_pairs=len(layout_pairs))
                self._log(f"  OK: {len(ocr_lines)} benzersiz satır ({ocr_total:.1f}s)")

                # ══ [5/6] CREDITS_PARSE ════════════════════════════
                self._log(f"\n[5/6] CREDITS_PARSE")
                t = time.time()
                self.stats.start_stage("CREDITS_PARSE")
                parser = CreditsParser(turkish_name_db=self._name_db)
                parsed = parser.parse(ocr_lines, layout_pairs=layout_pairs)
                cdata = parser.to_report_dict(parsed)
                self._stage("CREDITS_PARSE", time.time() - t,
                            actors=cdata["total_actors"],
                            crew=cdata["total_crew"],
                            companies=cdata["total_companies"])
                self._log(f"  OK: Oyuncu:{cdata['total_actors']} "
                          f"Ekip:{cdata['total_crew']} "
                          f"Şirket:{cdata['total_companies']}")

                # Anomali uyarıları
                total_actors = cdata.get("total_actors", 0)
                total_crew = cdata.get("total_crew", 0)
                if total_actors > 500:
                    self._log(f"  ⚠️ Anormal cast sayısı: {total_actors} — OCR kalitesi düşük olabilir")
                if total_crew > 2000:
                    self._log(f"  ⚠️ Anormal crew sayısı: {total_crew} — OCR kalitesi düşük olabilir")

                # Snapshot before filters (DATABASE credits_raw için)
                cdata_raw = copy.deepcopy(cdata)

                # ── Dosya adı anchor'ını koru ────────────────────────────────────
                # Gemini cast ayıklaması artık NAME_VERIFY fallback olarak çalışır.
                if film_title_from_filename:
                    cdata["film_title"] = film_title_from_filename

                # ══ [6/6] NAME_VERIFY — İçerik Tipine Göre Doğrulama ══════
                # Yeni strateji: tek tek kişi aramak yerine içerik adı + yönetmen/oyuncu
                # kombinasyonu ile TMDB'de film/dizi eşleşmesi yapılır.
                # Eşleşme bulunamazsa Gemini 2.5 Flash fallback devreye girer.
                self._log(f"\n[6/6] NAME_VERIFY")
                t = time.time()
                self.stats.start_stage("NAME_VERIFY")

                # TMDB client hazırla
                _tmdb_client = None
                _tmdb_enabled = bool((content_profile or {}).get("tmdb_enabled", True))
                if _tmdb_enabled:
                    try:
                        from core.tmdb_verify import TMDBClient
                        _tmdb_api_key = get_tmdb_api_key()
                        if _tmdb_api_key:
                            _tmdb_client = TMDBClient(
                                api_key=_tmdb_api_key,
                                language="tr-TR",
                                log_cb=self._log,
                            )
                    except Exception as e:
                        self._log(f"  [TMDB] Client başlatılamadı: {e}")

                # NameVerifier oluştur
                verifier = NameVerifier(
                    name_db=self._name_db,
                    tmdb_client=_tmdb_client,
                    log_cb=self._log,
                )

                # ── Yönetmen adlarını çıkar ──
                director_names_raw = []
                for d in (cdata.get("directors") or []):
                    if isinstance(d, str):
                        director_names_raw.append(d.strip())
                    elif isinstance(d, dict):
                        director_names_raw.append(str(d.get("name", "")).strip())
                director_names_raw = [n for n in director_names_raw if n]

                # ── Dosya adından dizi/film ayrımını yap ──
                # Format: {prefix}_{YYYY}-{XXXX}-{B}-{SSSS}-{XX}-{X}-{BAŞLIK}
                # B (3. blok) = 0 → dizi, 1 → film
                # _ID_PATTERN benzeri: 4 rakam - 2-4 rakam - 1+ rakam - 4 rakam
                _id_match = re.search(r'\d{4}-\d{2,4}-(\d+)-\d{4}', vname)
                _content_flag = _id_match.group(1) if _id_match else None
                _is_series = (_content_flag == "0")
                self._log(
                    f"  [NAME_VERIFY] İçerik tipi: "
                    f"{'DİZİ' if _is_series else 'FİLM'} (flag={_content_flag})"
                )

                # ── İçerik tipine göre TMDB doğrulama ──
                _nv_title = film_title_from_filename or cdata.get("film_title", "")
                nv_match = None

                if _is_series:
                    # Dizi (flag=0): IMDb DuckDB önce çalışacak — verify_as_series() burada atlanıyor.
                    # IMDb miss sonrası fallback olarak aşağıda (_series_nv_pending) tetiklenecek.
                    cdata["name_verify_matched"] = False
                    cdata["_series_nv_pending"] = True
                else:
                    # Film: başlık + yönetmen + 1-2 oyuncu kombinasyonu
                    _top_actors = []
                    _cast_sorted = sorted(
                        [e for e in (cdata.get("cast") or []) if isinstance(e, dict)],
                        key=lambda x: (float(x["confidence"]) if isinstance(x.get("confidence"), (int, float)) else 0.0),
                        reverse=True,
                    )
                    for _entry in _cast_sorted[:2]:
                        _aname = (_entry.get("actor_name") or "").strip()
                        if _aname:
                            _top_actors.append(_aname)
                    nv_match = verifier.verify_as_film(
                        title=_nv_title,
                        director_names=director_names_raw,
                        top_actors=_top_actors,
                    )

                if nv_match:
                    # ── Eşleşme bulundu: TMDB verisini cdata'ya uygula ──
                    _nv_credits = nv_match.get("credits", {})
                    _nv_tmdb_id = nv_match.get("tmdb_id")
                    _nv_tmdb_title = nv_match.get("tmdb_title", "")
                    _nv_media_type = nv_match.get("media_type", "")
                    _nv_matched_via = nv_match.get("matched_via", "")

                    self._log(
                        f"  [NAME_VERIFY] ✓ Eşleşme: '{_nv_tmdb_title}' "
                        f"({_nv_media_type}, id:{_nv_tmdb_id}, via:{_nv_matched_via})"
                    )

                    _tmdb_cast = [
                        {
                            "actor_name": item.get("name", ""),
                            "is_verified_name": True,
                            "is_tmdb_verified": True,
                            "confidence": 0.9,
                        }
                        for item in (_nv_credits.get("cast") or [])[:50]
                        if item.get("name")
                    ]
                    _tmdb_crew = [
                        {
                            "name": item.get("name", ""),
                            "job": item.get("job", ""),
                            "role": item.get("job", ""),
                            "is_verified_name": True,
                            "is_tmdb_verified": True,
                        }
                        for item in (_nv_credits.get("crew") or [])
                        if item.get("name")
                    ]
                    if _tmdb_cast:
                        cdata["cast"] = _tmdb_cast
                        cdata["total_actors"] = len(_tmdb_cast)
                    if _tmdb_crew:
                        cdata["crew"] = _tmdb_crew
                        cdata["technical_crew"] = _tmdb_crew
                        cdata["total_crew"] = len(_tmdb_crew)

                    cdata["name_verify_matched"] = True
                    cdata["name_verify_method"] = _nv_matched_via
                    cdata["name_verify_tmdb_id"] = _nv_tmdb_id
                    cdata["name_verify_tmdb_title"] = _nv_tmdb_title
                else:
                    # ── Eşleşme bulunamadı: Gemini 2.5 Flash fallback ──
                    # Dizi modunda bu blok IMDb/TMDB sonrasına ertelenmiş olabilir.
                    if cdata.get("_series_nv_pending"):
                        pass  # Dizi: IMDb ve verify_as_series() fallback sonrasına ertelendi
                    else:
                        self._log(
                            "  [NAME_VERIFY] Eşleşme bulunamadı → "
                            "Gemini 2.5 Flash fallback devreye giriyor"
                        )
                        self._run_gemini_cast_extract(ocr_lines, cdata)
                        # Gemini film_title'ı ezmiş olabilir — dosya adı anchor'ını geri yükle
                        if film_title_from_filename and cdata.get("film_title") != film_title_from_filename:
                            gemini_title = cdata.get("film_title", "")
                            cdata["film_title"] = film_title_from_filename
                            cdata["_gemini_suggested_title"] = gemini_title
                            self._log(
                                f"  [Anchor] Gemini başlığı '{gemini_title}' → "
                                f"dosya adı anchor'ı korundu: '{film_title_from_filename}'"
                            )
                        cdata["name_verify_matched"] = False

                    # ── PersonVerifier: film TMDB'de bulunamadı → isim bazlı doğrulama ──
                    # Dizi modunda IMDb/verify_as_series sonrasına ertelendi.
                    # _map_crew_to_roles → NameVerifier.verify_crew → PersonVerifier filtresi
                    _ocr_crew = cdata.get("technical_crew") or cdata.get("crew") or []
                    _ocr_directors = cdata.get("directors") or []
                    if not cdata.get("_series_nv_pending") and (_ocr_crew or _ocr_directors):
                        self._log(
                            "  [PERSON] Film TMDB'de bulunamadı — isim bazlı doğrulama başlıyor"
                        )
                        person_verifier = PersonVerifier(
                            tmdb_client=_tmdb_client,
                            log_cb=self._log,
                        )
                        # Katman 1: rol bazlı filtreleme
                        crew_roles = _map_crew_to_roles(_ocr_crew, _ocr_directors)
                        # Katman 2: NameVerifier blacklist + yapısal kontrol
                        crew_roles = verifier.verify_crew(crew_roles)
                        # Katman 3: PersonVerifier (isim bazlı TMDB doğrulama)
                        verified_crew_roles: dict[str, list[str]] = {}
                        for role_key, names in crew_roles.items():
                            verified_names: list[str] = []
                            for person_name in names:
                                pv_result = person_verifier.verify_name(person_name)
                                if pv_result["found"]:
                                    verified_names.append(person_name)
                                    self._log(
                                        f"    ✅ {person_name} → "
                                        f"{pv_result.get('known_for_department', '?')}"
                                    )
                                elif pv_result.get("source") == "non_person_filter":
                                    self._log(
                                        f"    ❌ {person_name} → REJECT (rol başlığı/mekan)"
                                    )
                                else:
                                    # TMDB'de bulunamadı ama yapısal olarak isim gibi
                                    from core.person_verify import _is_likely_person_name
                                    if _is_likely_person_name(person_name):
                                        verified_names.append(person_name)
                                        self._log(
                                            f"    ⚠️ {person_name} → "
                                            "TMDB'de bulunamadı, düşük güvenle korundu"
                                        )
                                    else:
                                        self._log(
                                            f"    ❌ {person_name} → REJECT "
                                            "(yapısal filtre)"
                                        )
                            verified_crew_roles[role_key] = verified_names
                        cdata["_verified_crew_roles"] = verified_crew_roles
                        self._log(
                            f"  [PERSON] Doğrulama tamamlandı: "
                            + ", ".join(
                                f"{r}={len(v)}"
                                for r, v in verified_crew_roles.items()
                                if v
                            )
                        )

                # ── Verification log kaydet ──
                cdata["_verification_log"] = verifier.get_log()
                cdata["_verification_log_text"] = verifier.get_log_text()

                verify_elapsed = time.time() - t
                self._stage("NAME_VERIFY", verify_elapsed,
                            status="completed")
                self._log(f"  OK: İsim doğrulama tamamlandı ({verify_elapsed:.1f}s)")

                # ── _verified_crew_roles → cdata["crew"] köprüsü ──────────────────────
                # NAME_VERIFY stage'i crew isimlerini _verified_crew_roles dict'ine yazar.
                # Ama cdata["crew"] boşsa (OCR'da crew kredisi bulunamadıysa) TMDB hiçbir
                # isim görmez. Strateji 2 için _verified_crew_roles'u cdata["crew"]'a
                # düz liste olarak aktar.
                _vr = cdata.get("_verified_crew_roles") or {}
                if _vr and len(cdata.get("crew") or []) <= 2:
                    _bridge_crew = []
                    for role_key, persons in _vr.items():
                        if not isinstance(persons, list):
                            persons = [persons]
                        for p in persons:
                            if isinstance(p, str):
                                name = p.strip()
                            elif isinstance(p, dict):
                                name = (p.get("name") or "").strip()
                            else:
                                continue
                            if name and len(name) >= 3:
                                _bridge_crew.append({
                                    "name": name,
                                    "job": role_key,
                                    "role": role_key,
                                    "confidence": float(p.get("confidence", 0.85)) if isinstance(p, dict) else 0.85,
                                    "raw": "name_verify_bridge",
                                })
                    if _bridge_crew:
                        cdata["crew"] = _bridge_crew
                        cdata["total_crew"] = len(_bridge_crew)
                        self._log(
                            f"  [TMDB Bridge] _verified_crew_roles → cdata['crew'] aktarıldı: "
                            f"{len(_bridge_crew)} kişi"
                        )

                # ══ IMDb DuckDB DOĞRULAMA (TMDB'den önce) ══
                imdb_matched = False
                _imdb_enabled = bool((content_profile or {}).get("imdb_enabled", True))
                if _imdb_enabled:
                    try:
                        from core.imdb_lookup import IMDBLookup
                        _imdb_lookup = IMDBLookup(log_cb=self._log)
                        if _imdb_lookup.enabled():
                            self._log("  [IMDb] DuckDB doğrulama başlatılıyor...")
                            imdb_result = _imdb_lookup.lookup(cdata)
                            if imdb_result and imdb_result.matched:
                                self._log(
                                    f"  [IMDb] ✓ Eşleşme: '{imdb_result.title}' "
                                    f"({imdb_result.title_type}, tconst:{imdb_result.tconst}, "
                                    f"via:{imdb_result.matched_via})"
                                )
                                imdb_matched = True
                                cdata = self._apply_imdb_credits(cdata, imdb_result)
                                self._log(
                                    f"  [IMDb] LOCK aktif — "
                                    f"cast:{cdata['total_actors']} crew:{cdata['total_crew']}"
                                )
                                # ══ IMDb eşleşti → TMDB aggregate_credits ile zenginleştir ══
                                if _tmdb_enabled:
                                    self._enrich_cdata_with_tmdb(cdata, work_dir, _is_series)
                                # ══ Kredi alanı yanlış sınıflandırma kontrolü ══
                                self._check_credit_field_misclassification(cdata)
                                # ══ Kaynak güven annotasyonu (IMDb path) ══
                                self._annotate_crew_confidence(
                                    cdata, imdb_matched=True, tmdb_matched=False
                                )
                            else:
                                reason = imdb_result.reason if imdb_result else "lookup failed"
                                self._log(f"  [IMDb] Eşleşme bulunamadı ({reason}) → TMDB'ye geçiliyor")
                        else:
                            self._log("  [IMDb] DuckDB mevcut değil — atlanıyor")
                    except Exception as e:
                        self._log(f"  [IMDb] Hata: {e} — TMDB'ye geçiliyor")

                # ══ DİZİ: IMDb miss → verify_as_series() fallback ══
                if _is_series and not imdb_matched and cdata.get("_series_nv_pending"):
                    self._log("  [NAME_VERIFY/Dizi] IMDb miss — TMDB verify_as_series fallback başlatılıyor")
                    _nv_title = film_title_from_filename or cdata.get("film_title", "")
                    nv_match = verifier.verify_as_series(
                        title=_nv_title,
                        director_names=director_names_raw,
                    )
                    if nv_match:
                        _nv_credits  = nv_match.get("credits", {})
                        _nv_tmdb_id  = nv_match.get("tmdb_id")
                        _nv_tmdb_title = nv_match.get("tmdb_title", "")
                        _nv_media_type = nv_match.get("media_type", "")
                        _nv_matched_via = nv_match.get("matched_via", "")
                        self._log(
                            f"  [NAME_VERIFY/Dizi] ✓ Eşleşme: '{_nv_tmdb_title}' "
                            f"({_nv_media_type}, id:{_nv_tmdb_id}, via:{_nv_matched_via})"
                        )
                        _tmdb_cast = [
                            {
                                "actor_name": item.get("name", ""),
                                "is_verified_name": True,
                                "is_tmdb_verified": True,
                                "confidence": 0.9,
                            }
                            for item in (_nv_credits.get("cast") or [])[:50]
                            if item.get("name")
                        ]
                        # aggregate_credits crew: jobs[] array; regular credits: tek job string
                        _tmdb_crew = []
                        for item in (_nv_credits.get("crew") or []):
                            name = item.get("name", "")
                            if not name:
                                continue
                            department = item.get("department", "")
                            jobs = item.get("jobs")  # aggregate_credits formatı
                            if jobs:
                                for job_entry in jobs:
                                    _tmdb_crew.append({
                                        "name": name,
                                        "job": job_entry.get("job", department),
                                        "role": job_entry.get("job", department),
                                        "department": department,
                                        "episode_count": job_entry.get("episode_count", 0),
                                        "is_verified_name": True,
                                        "is_tmdb_verified": True,
                                    })
                            else:
                                job = item.get("job", department)
                                _tmdb_crew.append({
                                    "name": name,
                                    "job": job,
                                    "role": job,
                                    "department": department,
                                    "is_verified_name": True,
                                    "is_tmdb_verified": True,
                                })
                        if _tmdb_cast:
                            cdata["cast"] = _tmdb_cast
                            cdata["total_actors"] = len(_tmdb_cast)
                        if _tmdb_crew:
                            cdata["crew"] = _tmdb_crew
                            cdata["technical_crew"] = _tmdb_crew
                            cdata["total_crew"] = len(_tmdb_crew)
                        cdata["name_verify_matched"] = True
                        cdata["name_verify_method"] = _nv_matched_via
                        cdata["name_verify_tmdb_id"] = _nv_tmdb_id
                        cdata["name_verify_tmdb_title"] = _nv_tmdb_title
                    else:
                        # IMDb da TMDB de miss → Gemini fallback
                        self._log(
                            "  [NAME_VERIFY/Dizi] TMDB da eşleşmedi → Gemini fallback"
                        )
                        self._run_gemini_cast_extract(ocr_lines, cdata)
                        if film_title_from_filename and cdata.get("film_title") != film_title_from_filename:
                            gemini_title = cdata.get("film_title", "")
                            cdata["film_title"] = film_title_from_filename
                            cdata["_gemini_suggested_title"] = gemini_title
                        cdata["name_verify_matched"] = False
                    cdata.pop("_series_nv_pending", None)

                # ══ ESKI TMDB FILM ARAMASINI KORU (opsiyonel, eski profiller için) ══
                tmdb_matched = False
                tmdb_result = None
                if _tmdb_enabled and not imdb_matched and not (content_profile or {}).get("match_parse_enabled"):
                    if film_title_from_filename:
                        cdata["film_title"] = film_title_from_filename
                    # XML sidecar'dan gelen orijinal başlığı cdata'ya ekle (TMDB Strategy 1b için)
                    if xml_info and xml_info.original_title:
                        cdata["original_title"] = xml_info.original_title
                        self._log(f"  [XML→TMDB] Orijinal başlık TMDB'ye iletiliyor: '{xml_info.original_title}'")
                    try:
                        tmdb_result = self._run_tmdb(cdata, work_dir, is_series=_is_series)
                        if tmdb_result and (tmdb_result.updated or tmdb_result.matched_id):
                            self._log(f"  [TMDB Film] '{tmdb_result.matched_title}' — "
                                      f"hits:{tmdb_result.hits} misses:{tmdb_result.misses}")
                            tmdb_matched = True

                            # ── Profil bazlı dallanma ──
                            _matched_via = getattr(tmdb_result, 'matched_via', '')
                            _reverse_score = getattr(tmdb_result, 'reverse_score', 0.0)
                            _reverse_threshold = 3.4  # varsayılan minimum eşik
                            if tmdb_result.reverse_breakdown:
                                _reverse_threshold = tmdb_result.reverse_breakdown.get('threshold', 3.4)
                            _lock_eligible = (
                                _matched_via == "title"
                                or (_reverse_score >= _reverse_threshold and tmdb_result.matched_id > 0)
                            )
                            if (profile_name == "FilmDizi-Hybrid" and _lock_eligible):
                                if _matched_via == "cast_only":
                                    self._log(f"  [TMDB] cast_only eşleşme ama reverse_score={_reverse_score:.1f} ≥ {_reverse_threshold:.1f} — LOCK açılıyor")
                                # TMDB LOCK: OCR verisi silinir, TMDB kanonik veri yazılır
                                cdata = self._apply_tmdb_credits(cdata, tmdb_result)
                                self._log(f"  [TMDB] LOCK aktif — cast:{cdata['total_actors']} crew:{cdata['total_crew']}")
                                # ══ Kredi alanı yanlış sınıflandırma kontrolü ══
                                self._check_credit_field_misclassification(cdata)
                                # ══ Kaynak güven annotasyonu (TMDB path) ══
                                self._annotate_crew_confidence(
                                    cdata, imdb_matched=False, tmdb_matched=True
                                )
                                # QA: OCR'da olup TMDB'de olmayan oyuncular
                                if ocr_lines:
                                    try:
                                        from core.credits_qa import check_missing_actors
                                        qa = check_missing_actors(
                                            ocr_results=ocr_lines,
                                            tmdb_cast=cdata.get("cast", []),
                                        )
                                        if qa.missing_actors:
                                            cdata["credits_qa"] = qa.to_dict()
                                            self._log(f"  {qa.summary}")
                                    except Exception as e:
                                        self._log(f"  [QA] {e}")
                                if ocr_lines:
                                    try:
                                        from core.credits_qa import check_missing_crew
                                        crew_qa = check_missing_crew(
                                            ocr_results=ocr_lines,
                                            tmdb_crew=cdata.get("crew", []),
                                        )
                                        if crew_qa.missing_crew:
                                            cdata["crew_qa"] = crew_qa.to_dict()
                                            self._log(f"  {crew_qa.summary}")
                                    except Exception as e:
                                        self._log(f"  [Crew QA] {e}")
                            else:
                                # REFERANS MODU: LOCK koşulu sağlanamadı
                                # (_lock_eligible=False: başlık eşleşmedi ve reverse_score < eşik veya matched_id=0)
                                if _matched_via == "cast_only":
                                    self._log(f"  [TMDB] cast_only eşleşme — reverse_score={_reverse_score:.1f} < {_reverse_threshold:.1f} — LOCK açılmıyor, referans modu")
                                cdata["_tmdb_film_match"] = {
                                    "title": tmdb_result.matched_title,
                                    "id": tmdb_result.matched_id,
                                    "hits": tmdb_result.hits,
                                    "misses": tmdb_result.misses,
                                }
                                cdata["_tmdb_cast_ref"] = tmdb_result.cast or []
                                cdata["_tmdb_crew_ref"] = tmdb_result.crew or []
                                self._log(f"  [TMDB] Referans modu — OCR verisi korunuyor")
                                # TMDB crew verisini raw:"tmdb" etiketiyle cdata["crew"]'a ekle
                                # export_engine bu etiketi kullanarak TMDB'yi önceliklendirir
                                _tmdb_backup_crew = []
                                for item in (tmdb_result.crew or []):
                                    _name = (item.get("name") or "").strip()
                                    if not _name:
                                        continue
                                    _job = (item.get("job") or "").strip()
                                    _tmdb_backup_crew.append({
                                        "name": _name,
                                        "job": _job,
                                        "role": _job,
                                        "role_tr": self._JOB_TR.get(_job, _job),
                                        "raw": "tmdb",
                                        "confidence": 0.8,
                                    })
                                # TMDB directors'ı da ekle
                                for d in (tmdb_result.directors or []):
                                    _dname = (d.get("name") or "").strip() if isinstance(d, dict) else str(d).strip()
                                    if not _dname:
                                        continue
                                    _tmdb_backup_crew.append({
                                        "name": _dname,
                                        "job": "Director",
                                        "role": "Director",
                                        "role_tr": "Yönetmen",
                                        "raw": "tmdb",
                                        "confidence": 0.8,
                                    })
                                if _tmdb_backup_crew:
                                    existing_crew = cdata.get("crew") or []
                                    cdata["crew"] = existing_crew + _tmdb_backup_crew
                                    self._log(
                                        f"  [TMDB Backup] reverse_validation_rejected ama "
                                        f"{len(_tmdb_backup_crew)} crew kaydı yedeklendi"
                                    )
                        else:
                            if tmdb_result:
                                self._log(f"  [TMDB Film] {tmdb_result.reason}")
                                # ── Strateji C/D kişi kanıtı kurtarma ──
                                # Film reddedildi/bulunamadı ama kişi kanıtı varsa cast'a ekle
                                _person_evidence = (tmdb_result.person_evidence or [])
                                if (_person_evidence and tmdb_result.reason in (
                                    "reverse_validation_rejected", "tmdb match not found"
                                )):
                                    self._log(
                                        f"  [TMDB PersonRecovery] "
                                        f"{len(_person_evidence)} kişi kanıtı korunuyor "
                                        f"(film reddedildi/bulunamadı)"
                                    )
                                    _merged = _merge_tmdb_person_evidence(
                                        cdata, _person_evidence, self._log
                                    )
                                    if _merged > 0:
                                        self._log(
                                            f"  [TMDB PersonRecovery] "
                                            f"{_merged} kişi cast çıktısına eklendi"
                                        )
                    except Exception as e:
                        self._log(f"  [TMDB Film] Hata: {e}")

                # ══ BLOK2: TMDB/IMDb eşleşmedi → VLM ile derin okuma ══
                if not tmdb_matched and not imdb_matched:
                    if self._blok2_enabled and self._vlm_reader.enabled and self._vlm_reader.is_available():
                        self._log("\n[BLOK2] TMDB/IMDb eşleşmedi — VLM ile derin okuma başlatılıyor...")
                        vlm_t = time.time()
                        vlm_ocr_lines = []
                        for frame_info in candidates:
                            result_vlm = self._vlm_reader.read_text_from_frame(
                                frame_info["path"], lang="tr")
                            if result_vlm and result_vlm.get("text"):
                                for line_text in result_vlm["text"].splitlines():
                                    line_text = line_text.strip()
                                    if line_text:
                                        vlm_ocr_lines.append({
                                            "text": line_text,
                                            "avg_confidence": 0.85,
                                            "bbox": [],
                                            "frame_path": frame_info["path"],
                                            "source": "vlm_blok2",
                                        })
                        if vlm_ocr_lines:
                            ocr_lines = self._merge_blok2_results(ocr_lines, vlm_ocr_lines)
                            parser = CreditsParser(turkish_name_db=self._name_db)
                            parsed = parser.parse(ocr_lines, layout_pairs=layout_pairs)
                            cdata = parser.to_report_dict(parsed)
                            cdata_raw = copy.deepcopy(cdata)
                            self._log(
                                f"  [BLOK2] VLM okuma: {len(vlm_ocr_lines)} satır, "
                                f"toplam: {len(ocr_lines)} satır ({time.time()-vlm_t:.1f}s)"
                            )
                    else:
                        if not tmdb_matched and not imdb_matched:
                            self._log("  [BLOK2] Pasif — config'de blok2_enabled=false")
                            if not cdata.get("gemini_extracted"):
                                self._run_gemini_cast_extract(ocr_lines, cdata)

                # ══ [LLM] LLM_CAST_FILTER (opsiyonel) ════════════════
                # Sadece TMDB/IMDb eşleşmezse LLM devreye girsin
                if not tmdb_matched and not imdb_matched:
                    if self._llm_filter_enabled:
                        self._log(f"\n[LLM] Cast Filtreleme (TMDB/IMDb eşleşmedi)")
                        t = time.time()
                        import core.llm_provider as _llm_prov
                        _active_provider = _llm_prov.get_provider()
                        if _active_provider == "gemini":
                            _llm_filter_model = self.config.get("llm_filter_model") or None
                        else:
                            _llm_filter_model = self.config.get("llm_filter_model") or self.config.get("ollama_model", "qwen2.5vl:7b")
                        llm_filter = LLMCastFilter(
                            ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
                            model=_llm_filter_model,
                            enabled=self._llm_filter_enabled,
                            log_cb=self._log,
                            name_checker=self._name_db.is_name,
                        )
                        cdata["cast"] = llm_filter.filter_cast(cdata.get("cast", []), log_cb=self._log)
                        cdata["total_actors"] = len(cdata["cast"])
                        llm_elapsed = time.time() - t
                        if llm_elapsed > 0.1:
                            self._log(f"  [LLM] Cast filtreleme: {llm_elapsed:.1f}s")
                else:
                    self._log(f"\n[LLM] IMDb/TMDB eşleşti — LLM filtresi atlanıyor")

                # ══ [GEMINI CREW] TMDB eşleşmedi → Gemini ile crew doğrulama ══
                # Sadece TMDB/IMDb miss durumunda ve TMDB backup crew yoksa çalışır
                if not imdb_matched and not tmdb_matched and not any(
                    c.get("raw") == "tmdb" for c in (cdata.get("crew") or [])
                ):
                    try:
                        from core.gemini_crew_validator import validate_crew_with_gemini
                        _gcv_ocr_scores = self._build_ocr_scores(ocr_lines, cdata).get("scores", [])
                        _gcv_result = validate_crew_with_gemini(
                            film_title=cdata.get("film_title", ""),
                            ocr_crew=cdata.get("technical_crew") or cdata.get("crew") or [],
                            ocr_lines=ocr_lines,
                            ocr_scores=_gcv_ocr_scores,
                        )
                        if _gcv_result and _gcv_result.get("verified_roles"):
                            cdata["_gemini_crew_roles"] = _gcv_result["verified_roles"]
                            self._log(
                                f"  [Gemini Crew] Doğrulama başarılı: "
                                f"{sum(len(v) for v in _gcv_result['verified_roles'].values())} kişi"
                            )
                    except Exception as _gcv_err:
                        self._log(f"  [Gemini Crew] Hata: {_gcv_err}")

                # ══ Kaynak güven annotasyonu (neither-match path) ══
                self._annotate_crew_confidence(
                    cdata, imdb_matched=False, tmdb_matched=False
                )

                # ══ [GOOGLE_VI] Akıllı tetik ════════════════════════════
                # TMDB/IMDb miss + düşük çözünürlük veya non-standard font ise Google VI çağır
                vi_decision = self._decide_google_vi(
                    tmdb_matched=(tmdb_matched or imdb_matched),
                    resolution=info.get("resolution", ""),
                    ocr_lines=ocr_lines,
                    segment_duration_min=(first_min + last_min),
                    cast_count=cdata.get("total_actors", 0),
                )
                if vi_decision.should_run:
                    self._log(f"\n[GOOGLE_VI] {vi_decision.reason}")
                    for trigger in vi_decision.triggers:
                        self._log(f"  → {trigger}")
                    vi_lines = self._run_google_vi(video_path, info, first_min, last_min)
                    if vi_lines:
                        try:
                            from core.google_video_intelligence import GoogleVITextEngine
                            vi_engine = GoogleVITextEngine(self.config, log_cb=self._log)
                            ocr_lines = vi_engine.merge_with_paddle(ocr_lines, vi_lines)
                        except Exception as _e:
                            self._log(f"  [GOOGLE_VI] Merge hatası: {_e}")
                        # VI sonrası tekrar parse et
                        parser = CreditsParser(turkish_name_db=self._name_db)
                        parsed = parser.parse(ocr_lines, layout_pairs=layout_pairs)
                        cdata = parser.to_report_dict(parsed)
                        self._log(
                            f"  [GOOGLE_VI] Tekrar parse: "
                            f"Oyuncu:{cdata['total_actors']} "
                            f"Ekip:{cdata['total_crew']}"
                        )
                else:
                    self._log(f"\n[GOOGLE_VI] {vi_decision.reason}")

            else:
                if not ocr_enabled:
                    self._log(f"  [OCR] Profil '{profile_name}' — OCR devre dışı")
                for stage_name in ("FRAME_EXTRACT", "TEXT_FILTER", "OCR_CREDITS",
                                   "CREDITS_PARSE", "TMDB_VERIFY"):
                    reason = (f"ocr_enabled=false in {profile_name}"
                              if not ocr_enabled else "scope=audio_only")
                    self._stage(stage_name, 0.0, status="skipped", reason=reason)

            # ── Audio sonucunu topla (paralel mod tamamlanmış olmalı) ──
            if audio_future is not None:
                # Bridge timeout (default 3600s) + 120s marj
                _audio_timeout = self.config.get("audio_timeout", 3600) + 120
                try:
                    self._log("  [AUDIO] Ses sonucu bekleniyor...")
                    audio_result = audio_future.result(timeout=_audio_timeout)
                    executor.shutdown(wait=False)
                    if audio_result and audio_result.get("status") != "error":
                        self._log("  [AUDIO] Ses analizi tamamlandı (OK)")
                    else:
                        self._log(f"  [AUDIO] Ses analizi sonuç: HATA")
                except TimeoutError:
                    self._log(f"  [AUDIO] TIMEOUT — {_audio_timeout}s aşıldı, audio atlanıyor")
                    audio_result = {"status": "error", "error": f"future timeout ({_audio_timeout}s)"}
                    executor.shutdown(wait=False)
                except Exception as ae:
                    self._log(f"  [AUDIO] HATA: {ae}")
                    audio_result = {"status": "error", "error": str(ae)}

            # ══ TRANSCRIPT ÖZETİ (Gemini) ═════════════════════════
            if audio_result and isinstance(audio_result, dict) and audio_result.get("status") != "skipped":
                transcript_lang = audio_result.get("detected_language") or "unknown"
                audio_result.setdefault("transcript_language", transcript_lang)
                audio_result.setdefault("report_language", "tr")
                raw_transcript = audio_result.get("transcript", "")
                # transcript may be a list of segment dicts or a plain string
                if isinstance(raw_transcript, list):
                    transcript_text = " ".join(
                        seg.get("text", "").strip()
                        for seg in raw_transcript
                        if isinstance(seg, dict) and seg.get("text", "").strip()
                    )
                else:
                    transcript_text = str(raw_transcript) if raw_transcript else ""
                if transcript_text.strip():
                    gemini_api_key = get_gemini_api_key()
                    if gemini_api_key:
                        try:
                            from core.gemini_summarizer import summarize_transcript
                            _detected_lang = audio_result.get("detected_language", "tr")
                            # TMDB cast'i al (varsa) — yabancı dil çevirisinde isim doğrulama için
                            _tmdb_cast_for_summary = cdata.get("cast") or cdata.get("_tmdb_cast_ref") or []
                            summary = summarize_transcript(
                                transcript_text,
                                api_key=gemini_api_key,
                                log_cb=self._log,
                                variant="en",
                                detected_language=_detected_lang,
                                tmdb_cast=_tmdb_cast_for_summary,
                            )
                            if summary:
                                audio_result["summary"] = summary
                                audio_result["summary_model"] = summary.get("model_used", "")
                                audio_result["summary_language"] = "tr"
                        except Exception as _sum_e:
                            self._log(f"  [Summarizer] Özet hatası (pipeline etkilenmedi): {_sum_e}")
                    else:
                        self._log("  [Summarizer] Gemini API key yok — özet atlanıyor")

            # ══ EXPORT ════════════════════════════════════════════
            self._log(f"\n[EXPORT]")
            t = time.time()
            self.stats.start_stage("EXPORT")
            export_ts = datetime.now().strftime("%d%m%y-%H%M")
            exp = ExportEngine(work_dir, name_db=self._name_db)
            jp, tp, tr_p, user_tp = exp.generate(
                info, cdata, ocr_lines, self.stage_stats,
                "WORKSTATION", scope, first_min, last_min,
                keywords=cdata.get("_tmdb_keywords") or None,
                content_profile_name=profile_name,
                audio_result=audio_result,
                ts=export_ts)
            self._stage("EXPORT", time.time() - t)

            # Kullanıcı çıktı klasörüne (output_root) yalnızca kullanıcı TXT'yi kopyala
            if self.output_root:
                out_root = Path(self.output_root)
                out_root.mkdir(parents=True, exist_ok=True)
                for _src in (user_tp,):
                    _src_p = Path(_src)
                    if _src_p.is_file():
                        _dst = _safe_path(out_root / _src_p.name)
                        shutil.copy2(_src_p, _dst)
                        self._log(f"  [EXPORT] Kullanıcı raporu kopyalandı: {_dst.name}")

            # E-posta bildirimi (opsiyonel — config'de email_enabled: true ise çalışır)
            if self.config.get("email_enabled", False) and tp and Path(tp).is_file():
                try:
                    from core.email_notifier import send_result_email
                    send_result_email(tp, log_cb=self._log)
                except Exception as _email_exc:
                    self._log(f"  [Email] Gönderilemedi: {_email_exc}")

            # ══ DATABASE ══════════════════════════════════════════
            try:
                self._write_database(
                    video_info=info,
                    credits_data=cdata,
                    credits_raw=cdata_raw,
                    ocr_lines=ocr_lines,
                    stage_stats=self.stage_stats,
                    audio_result=audio_result,
                    work_dir=work_dir,
                    content_profile_name=profile_name,
                    ts=export_ts,
                    xml_path=xml_info.xml_path if xml_info else "",
                )
            except Exception as e:
                self._log(f"  [DATABASE] Yazma hatası (pipeline etkilenmedi): {e}")

            total = time.time() - t0
            self.stats.finish_job(total)

            self._log(f"\n{'='*60}")
            self._log(f"  TAMAMLANDI — {total:.1f}s "
                      f"({info['duration_seconds']/max(total,0.1):.1f}x)")
            self._log(f"  JSON      : {jp}")
            self._log(f"  Rapor     : {tp}")
            self._log(f"  Transcript: {tr_p}")
            self._log(f"  Kullanıcı : {user_tp}")
            self._log(f"{'='*60}")

            xml_sidecar_dict = {
                "path": xml_info.xml_path,
                "original_title": xml_info.original_title,
                "turkish_title": xml_info.turkish_title,
            } if xml_info else None

            return {
                "report_json":       jp,
                "report_txt":        tp,
                "transcript_txt":    tr_p,
                "user_report_txt":   user_tp,
                "work_dir":          work_dir,
                "video_info":        info,
                "credits":           cdata,
                "ocr_lines":         len(ocr_lines),
                "tmdb_result":       tmdb_result,
                "imdb_result":       imdb_result,
                "audio_result":      audio_result,
                "xml_sidecar":       xml_sidecar_dict,
            }

        except Exception as e:
            self.stats.log_error(str(e))
            self.stats.finish_job(time.time() - t0)
            self._log(f"\n!! HATA: {e}")
            import traceback
            traceback.print_exc()
            raise

    # ──────────────────────────────────────────────────────────────
    # SES PİPELİNE — BUG-02 DÜZELTİLDİ
    # ──────────────────────────────────────────────────────────────
    def _run_audio(self, video_path: str, work_dir: str) -> dict:
        """AudioBridge üzerinden ses pipeline'ını çalıştır."""
        try:
            from core.audio_bridge import AudioBridge
        except ImportError:
            self._log("  [Audio] audio_bridge modülü yok — ses atlanıyor")
            return {"status": "skipped", "reason": "audio_bridge not found"}

        self._log(f"\n[AUDIO] Ses analizi başlatılıyor...")

        # Kullanıcı config'de açıkça belirtmişse onu kullan; yoksa profil bazlı seç
        stages = self.config.get("audio_stages", None)
        if stages is None:
            program_type = self.config.get("program_type", "film_dizi")
            if program_type in ("film_dizi", "kisa_haber"):
                # Sadece extract + transcribe (düz deşifre) — denoise/diarize gerekmez
                stages = ["detect_language", "extract", "transcribe"]
            else:
                # mac, muzik_programi vb. tam pipeline gerektirir (ileride genişletilecek)
                stages = ["detect_language", "extract", "denoise", "diarize", "transcribe", "post_process"]

        self._log(f"  [Audio] Stages: {stages}")

        audio_cfg = {
            "program_type": self.config.get("program_type", "film_dizi"),
            "hf_token":     self.config.get("hf_token", ""),
            "ffmpeg":       self._ffmpeg,
            "ollama_url":   self.config.get("ollama_url", "http://localhost:11434"),
            "tmdb_cast":    self.config.get("tmdb_cast", []),
            "stages": stages,
            "options": {
                "denoise_enabled":  "denoise" in stages,
                "whisper_model":    self.config.get("whisper_model", "large-v3"),
                "whisper_language": self.config.get("whisper_language", "tr"),
                # float16 varsayılan; CPU'da TranscribeStage otomatik int8'e döner.
                "compute_type":     self.config.get("compute_type", "float16"),
                "max_speakers":     self.config.get("max_speakers", 10),
                "ollama_model":     self.config.get("ollama_model", "llama3.1:8b"),
                "beam_size":        self.config.get("beam_size", 1),
            },
        }

        bridge = AudioBridge(log_cb=self._log, config=self.config)
        result = bridge.run(video_path, work_dir, audio_cfg)

        if result.get("status") == "ok":
            segs = result.get("transcript", [])
            spks = result.get("speakers", {})
            self._log(f"  [Audio] OK — {len(segs)} segment, "
                      f"{len(spks)} konuşmacı")
        else:
            err = result.get("error", "bilinmiyor")
            self._log(f"  [Audio] HATA: {err}")

        return result

    _JOB_TR = {
        "Director": "Yönetmen",
        "Screenplay": "Senaryo",
        "Screenwriter": "Senarist",
        "Writer": "Yazar",
        "Story": "Hikaye",
        "Producer": "Yapımcı",
        "Executive Producer": "Baş Yapımcı",
        "Cinematography": "Görüntü Yönetmeni",
        "Director of Photography": "Görüntü Yönetmeni",
        "Editor": "Kurgu",
        "Original Music Composer": "Müzik",
        "Music": "Müzik",
        "Costume Design": "Kostüm",
        "Production Design": "Yapım Tasarımı",
        "Sound": "Ses",
        "Makeup": "Makyaj",
        "Makeup Artist": "Makyaj",
        "Art Direction": "Sanat Yönetmeni",
        "Set Decoration": "Set Dekorasyonu",
        "Casting": "Oyuncu Seçimi",
        "Visual Effects": "Görsel Efekt",
        "Stunt Coordinator": "Dublör Koordinatörü",
    }

    def _apply_tmdb_credits(self, cdata: dict, tmdb_result):
        """TMDB doğrulama başarılıysa rapor içeriğini TMDB kanonik verisiyle sınırla."""
        import re as _re
        import unicodedata as _unicodedata

        def _norm_name(s: str) -> str:
            nfkd = _unicodedata.normalize("NFKD", s)
            ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
            return _re.sub(r"[^a-z0-9]", "", ascii_.lower())

        cast = []
        for item in (tmdb_result.cast or []):
            name = (item.get("name") or "").strip()
            if not name:
                continue
            cast.append({
                "actor_name": name,
                "character_name": (item.get("character") or "").strip(),
                "role": "Cast",
                "role_category": "cast",
                "raw": "tmdb",
                "confidence": 1.0,
                "frame": "tmdb",
                "is_verified_name": True,
                "is_tmdb_verified": True,
                "tmdb_order": item.get("order", 999),
            })

        # Uncredited oyuncuları filtrele
        cast = [c for c in cast if "uncredited" not in (c.get("character_name") or "").lower()]
        # tmdb_order'a göre sırala
        cast.sort(key=lambda x: x.get("tmdb_order", 999))
        # Max 15 ile sınırla
        cast = cast[:15]

        _STUNT_JOBS = {
            "stunts", "stunt double", "stunt coordinator", "stunt",
            "stunt driver", "stunt performer", "stunt actor", "utility stunts",
            "dublör", "dublör koordinatörü",
            "stuntkoordinator", "stunt-koordinator",
            "cascadeur", "coordinateur des cascades",
        }

        crew = []
        for item in (tmdb_result.crew or []):
            name = (item.get("name") or "").strip()
            if not name:
                continue
            job = (item.get("job") or "").strip()
            job_lower = job.lower().strip()
            # Stunt/dublör rollerini tamamen atla
            if job_lower in _STUNT_JOBS or "stunt" in job_lower:
                continue
            crew.append({
                "name": name,
                "job": job,
                "role": job or "Crew",
                "role_tr": self._JOB_TR.get(job, job),
                "department": (item.get("department") or "").strip(),
                "role_category": "crew",
                "raw": "tmdb",
                "confidence": 1.0,
                "frame": "tmdb",
            })

        directors = [
            {"name": c.get("name", "")}
            for c in crew
            if (c.get("job") or "").strip().lower() in ("director", "yonetmen", "yönetmen")
        ]

        cdata["cast"] = cast
        cdata["crew"] = crew
        cdata["technical_crew"] = crew
        if directors:
            cdata["directors"] = directors
        cdata["total_actors"] = len(cast)
        cdata["total_crew"] = len(crew)
        cdata["verification_status"] = "tmdb_verified"
        cdata["keywords_source"] = "tmdb_only"

        # ── OCR'da var, TMDB'de yok → crew'a merge et ────────────────────────
        # _verified_crew_roles: {job_key: ["Name", ...]} veya {job_key: [{"name": ..., "confidence": ...}, ...]}
        _DEFAULT_OCR_CONFIDENCE = 0.85
        ocr_crew_roles = cdata.get("_verified_crew_roles") or {}
        if ocr_crew_roles:
            tmdb_crew_names = {_norm_name(c["name"]) for c in crew if c.get("name")}
            for role_key, persons in ocr_crew_roles.items():
                if not isinstance(persons, list):
                    persons = [persons]
                for person in persons:
                    if isinstance(person, str):
                        name = person.strip()
                        confidence = _DEFAULT_OCR_CONFIDENCE
                    elif isinstance(person, dict):
                        name = (person.get("name") or "").strip()
                        confidence = float(person.get("confidence", _DEFAULT_OCR_CONFIDENCE))
                    else:
                        continue
                    if not name:
                        continue
                    if _norm_name(name) in tmdb_crew_names:
                        continue  # Zaten TMDB'de var
                    crew.append({
                        "name": name,
                        "job": role_key,
                        "role": role_key,
                        "role_tr": self._JOB_TR.get(role_key, role_key),
                        "role_category": "crew",
                        "raw": "ocr_verified",
                        "confidence": confidence,
                        "frame": "ocr",
                    })
                    tmdb_crew_names.add(_norm_name(name))  # tekrar eklemeyi önle
            # merge sonrası toplamları güncelle
            cdata["crew"] = crew
            cdata["technical_crew"] = crew
            cdata["total_crew"] = len(crew)

        # TMDB lock aktif — NAME_VERIFY'dan kalan _verified_crew_roles geçersiz
        cdata.pop("_verified_crew_roles", None)
        cdata.pop("_verification_log_text", None)
        cdata.pop("_verification_log", None)

        # OCR başlığını koru — TMDB başlığıyla ezilmeden önce sakla
        if tmdb_result.matched_title:
            _current_film_title = (cdata.get("film_title") or "").strip()
            if _current_film_title and _current_film_title != tmdb_result.matched_title:
                if not cdata.get("ocr_title"):
                    cdata["ocr_title"] = _current_film_title
            cdata["film_title"] = tmdb_result.matched_title

        # Bug 2: year'ı TMDB'den güncelle
        if tmdb_result.year:
            cdata["year"] = tmdb_result.year

        # Bug 5: TMDB keywords ve genres'ı topla
        tmdb_keywords = []
        if tmdb_result.matched_title:
            tmdb_keywords.insert(0, tmdb_result.matched_title)
        if tmdb_result.genres:
            tmdb_keywords.extend(tmdb_result.genres)
        if tmdb_result.keywords:
            tmdb_keywords.extend(tmdb_result.keywords)
        tmdb_keywords.extend([
            c.get("name", "") for c in (tmdb_result.crew or [])
            if (c.get("job") or "").lower() in ("director", "yönetmen", "yonetmen")
        ])
        cdata["_tmdb_keywords"] = [k for k in tmdb_keywords if k]

        return cdata

    def _apply_imdb_credits(self, cdata: dict, imdb_result) -> dict:
        """IMDb doğrulama başarılıysa rapor içeriğini IMDb kanonik verisiyle doldur."""
        import re as _re
        import unicodedata as _unicodedata

        def _norm_name(s: str) -> str:
            nfkd = _unicodedata.normalize("NFKD", s)
            ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
            return _re.sub(r"[^a-z0-9]", "", ascii_.lower())

        # cast listesi: imdb_result.cast → cdata["cast"]
        cast = []
        for item in (imdb_result.cast or []):
            name = (item.get("actor_name") or item.get("name") or "").strip()
            if not name:
                continue
            cast.append({
                "actor_name": name,
                "character_name": (item.get("character_name") or "").strip(),
                "role": "Cast",
                "role_category": "cast",
                "raw": "imdb",
                "confidence": float(item.get("confidence", 1.0)),
                "frame": "imdb",
                "is_verified_name": True,
                "is_imdb_verified": True,
            })

        # crew listesi: imdb_result.crew → cdata["crew"], cdata["technical_crew"]
        crew = []
        for item in (imdb_result.crew or []):
            name = (item.get("name") or "").strip()
            if not name:
                continue
            job = (item.get("job") or item.get("role") or "").strip()
            crew.append({
                "name": name,
                "job": job,
                "role": job or "Crew",
                "role_tr": self._JOB_TR.get(job, job),
                "role_category": "crew",
                "raw": "imdb",
                "confidence": float(item.get("confidence", 1.0)),
                "frame": "imdb",
                "is_imdb_verified": True,
            })

        # directors: imdb_result.directors → cdata["directors"]
        directors = list(imdb_result.directors or [])

        cdata["cast"] = cast
        cdata["crew"] = crew
        cdata["technical_crew"] = crew
        if directors:
            cdata["directors"] = directors
        cdata["total_actors"] = len(cast)
        cdata["total_crew"] = len(crew)
        cdata["verification_status"] = "imdb_verified"
        cdata["keywords_source"] = "imdb_only"

        # ── OCR'da var, IMDb'de yok → crew'a merge et ────────────────────────
        _DEFAULT_OCR_CONFIDENCE = 0.85
        ocr_crew_roles = cdata.get("_verified_crew_roles") or {}
        if ocr_crew_roles:
            imdb_crew_names = {_norm_name(c["name"]) for c in crew if c.get("name")}
            for role_key, persons in ocr_crew_roles.items():
                if not isinstance(persons, list):
                    persons = [persons]
                for person in persons:
                    if isinstance(person, str):
                        name = person.strip()
                        confidence = _DEFAULT_OCR_CONFIDENCE
                    elif isinstance(person, dict):
                        name = (person.get("name") or "").strip()
                        confidence = float(person.get("confidence", _DEFAULT_OCR_CONFIDENCE))
                    else:
                        continue
                    if not name:
                        continue
                    if _norm_name(name) in imdb_crew_names:
                        continue  # Zaten IMDb'de var
                    crew.append({
                        "name": name,
                        "job": role_key,
                        "role": role_key,
                        "role_tr": self._JOB_TR.get(role_key, role_key),
                        "role_category": "crew",
                        "raw": "ocr_verified",
                        "confidence": confidence,
                        "frame": "ocr",
                    })
                    imdb_crew_names.add(_norm_name(name))
            # merge sonrası toplamları güncelle
            cdata["crew"] = crew
            cdata["technical_crew"] = crew
            cdata["total_crew"] = len(crew)

        # IMDb lock aktif — NAME_VERIFY'dan kalan _verified_crew_roles geçersiz
        cdata.pop("_verified_crew_roles", None)
        cdata.pop("_verification_log_text", None)
        cdata.pop("_verification_log", None)

        # film_title: imdb_result.title → cdata["film_title"] (ocr_title'ı sakla)
        if imdb_result.title:
            _current_film_title = (cdata.get("film_title") or "").strip()
            if _current_film_title and _current_film_title != imdb_result.title:
                if not cdata.get("ocr_title"):
                    cdata["ocr_title"] = _current_film_title
            cdata["film_title"] = imdb_result.title

        # year: imdb_result.year → cdata["year"]
        if imdb_result.year:
            cdata["year"] = imdb_result.year

        # _tmdb_keywords: [imdb_result.title] + director isimleri
        imdb_keywords = []
        if imdb_result.title:
            imdb_keywords.append(imdb_result.title)
        for d in (imdb_result.directors or []):
            name = (d.get("name") or "").strip() if isinstance(d, dict) else str(d).strip()
            if name:
                imdb_keywords.append(name)
        cdata["_tmdb_keywords"] = [k for k in imdb_keywords if k]

        return cdata

    # ──────────────────────────────────────────────────────────────
    # GEMINI CAST EXTRACT
    # ──────────────────────────────────────────────────────────────
    def _run_gemini_cast_extract(self, ocr_lines: list, cdata: dict) -> bool:
        """Gemini ile OCR skor verisinden cast/crew ayıkla ve cdata'yı güncelle.

        Crew ayıklama için ham OCR satırları yerine OCR skor yapısı kullanılır
        (text, ocr_confidence, seen_count, verdict, name_db_match, llm_verified).
        Bu sayede REJECTED satırlar varsayılan olarak dışlanır ve Gemini
        sadece Yönetmen/Yapımcı/Yazar rollerine odaklanır.

        Returns:
            True — Gemini sonuç döndürdü ve cdata güncellendi.
            False — API key yok veya Gemini sonuç döndürmedi.
        """
        api_key = get_gemini_api_key()
        if not api_key:
            self._log("  [Gemini] API key bulunamadı — atlanıyor")
            return False

        from core.gemini_cast_extractor import GeminiCastExtractor
        extractor = GeminiCastExtractor(api_key=api_key, log_cb=self._log)
        film_title = cdata.get("film_title", "")

        # ── Cast ayıklama (ham metin — oyuncu tespiti için geniş kapsam gerekir) ──
        cast_result = extractor.extract(
            ocr_lines=[
                line.get("text", "") if isinstance(line, dict) else line.text
                for line in ocr_lines
            ],
            film_title=film_title,
        )
        if extractor.timed_out:
            cdata["gemini_timeout"] = True

        # ── Crew ayıklama (OCR skor verisiyle — sadece Yönetmen/Yapımcı/Yazar) ──
        ocr_scores = self._build_ocr_scores(ocr_lines, cdata).get("scores", [])
        crew_score_result = extractor.extract_crew_from_scores(
            ocr_scores=ocr_scores,
            film_title=film_title,
        )
        if extractor.timed_out:
            cdata["gemini_timeout"] = True

        has_cast = cast_result and cast_result.get("cast")
        has_crew = bool(
            crew_score_result.get("directors")
            or crew_score_result.get("producers")
            or crew_score_result.get("writers")
        )

        if has_cast or has_crew:
            if has_cast:
                self._log(
                    f"  [Gemini] Cast: {len(cast_result.get('cast', []))} oyuncu"
                )
                cdata["cast"] = cast_result.get("cast", [])
                cdata["total_actors"] = len(cdata["cast"])

            if has_crew:
                # Crew skor sonucunu _verified_crew_roles'a yaz
                # (_OUTPUT_ROLES key'leriyle eşleşecek şekilde map et)
                crew_roles = cdata.get("_verified_crew_roles") or {}
                directors = crew_score_result.get("directors") or []
                producers = crew_score_result.get("producers") or []
                writers = crew_score_result.get("writers") or []
                if directors:
                    crew_roles["YÖNETMEN"] = directors
                if producers:
                    crew_roles["YAPIMCI"] = producers
                if writers:
                    crew_roles["SENARYO"] = writers
                cdata["_verified_crew_roles"] = crew_roles
                self._log(
                    f"  [Gemini] Crew (skor): "
                    f"Yönetmen={len(directors)}, "
                    f"Yapımcı={len(producers)}, "
                    f"Yazar={len(writers)}"
                )

            cdata["gemini_extracted"] = True
            return True
        return False

    # ──────────────────────────────────────────────────────────────
    # TMDB
    # ──────────────────────────────────────────────────────────────
    def _run_tmdb(self, cdata: dict, work_dir: str, is_series: bool = False):
        from core.tmdb_verify import TMDBVerify, TMDBVerifyResult

        api_key = (
            self.config.get("tmdb_api_key") or
            get_tmdb_api_key()
        ).strip()
        token = (
            self.config.get("tmdb_bearer_token") or
            os.environ.get("TMDB_BEARER_TOKEN") or ""
        ).strip()
        lang = (
            self.config.get("tmdb_language") or
            os.environ.get("TMDB_LANGUAGE") or
            "tr-TR"
        ).strip()

        if not (api_key or token):
            self._log(f"  [TMDB] API key yok — .env dosyasına TMDB_API_KEY ekle")
            return TMDBVerifyResult(False, "no tmdb api key")

        verifier = TMDBVerify(
            work_dir=work_dir,
            api_key=api_key,
            bearer_token=token,
            language=lang,
            log_cb=self._log,
        )
        return verifier.verify_credits(cdata, is_series=is_series)

    def _enrich_cdata_with_tmdb(self, cdata: dict, work_dir: str, is_series: bool = False):
        """IMDb eşleşmesi sonrası TMDB aggregate_credits ile crew'u zenginleştir.

        İki koşul birden sağlanmalı:
          1) Dizi/film adı TMDB sonucuyla fuzzy eşleşiyor (≥80%)
          2) IMDb'deki yönetmen TMDB credits'te bulunuyor
        Her ikisi sağlanırsa → aggregate_credits'ten eksik crew eklenir.
        Sağlanmazsa → IMDb verisi korunur, hiçbir şey değişmez.
        """
        from core.tmdb_verify import TMDBClient

        api_key = (
            self.config.get("tmdb_api_key") or get_tmdb_api_key()
        ).strip()
        token = (
            self.config.get("tmdb_bearer_token") or
            os.environ.get("TMDB_BEARER_TOKEN") or ""
        ).strip()

        if not (api_key or token):
            return

        # TMDB araması için Türkçe başlığı tercih et (ocr_title = dosya adından gelen başlık,
        # film_title bu noktada IMDb primaryTitle ile ezilmiş olabilir)
        film_title = (cdata.get("ocr_title") or cdata.get("film_title") or "").strip()
        if not film_title:
            return

        try:
            # Fuzzy karşılaştırma yardımcıları
            try:
                from rapidfuzz import fuzz as _fuzz
                def _title_score(a, b):
                    return _fuzz.token_sort_ratio(a.lower(), b.lower())
                def _name_score(a, b):
                    return _fuzz.ratio(a.lower(), b.lower())
            except ImportError:
                def _title_score(a, b):
                    a, b = a.lower().strip(), b.lower().strip()
                    return 100 if a == b else (80 if (a in b or b in a) else 0)
                def _name_score(a, b):
                    return 100 if a.lower().strip() == b.lower().strip() else 0

            client = TMDBClient(
                api_key=api_key or "",
                bearer_token=token or "",
                language="tr-TR",
                log_cb=self._log,
            )

            # TMDB'de ara
            results = client.search_multi(film_title)
            tmdb_id = None
            kind = None
            tmdb_title = ""
            for r in (results or []):
                k = r.get("media_type", "")
                if is_series and k == "tv":
                    tmdb_id = r["id"]
                    kind = "tv"
                    tmdb_title = r.get("name") or r.get("title") or ""
                    break
                elif not is_series and k == "movie":
                    tmdb_id = r["id"]
                    kind = "movie"
                    tmdb_title = r.get("name") or r.get("title") or ""
                    break

            if not tmdb_id:
                self._log(f"  [TMDB Enrich] '{film_title}' için TMDB ID bulunamadı — atlanıyor")
                return

            # ── Koşul 1: Başlık fuzzy kontrolü (≥90) ───────────────────────
            title_score = _title_score(film_title, tmdb_title)
            self._log(
                f"  [TMDB Enrich] Başlık: '{film_title}' ↔ '{tmdb_title}' ({title_score:.0f}%)"
            )
            if title_score < 90:
                self._log(
                    f"  [TMDB Enrich] Başlık eşleşmedi ({title_score:.0f}% < 90) "
                    f"— IMDb verisi korunuyor"
                )
                return

            # ── Koşul 2: IMDb cast'ten en az 2 oyuncu TMDB cast'te bulunmalı ──
            try:
                if kind == "tv":
                    reg_credits = client.get_tv_credits(tmdb_id)
                else:
                    reg_credits = client.get_movie_credits(tmdb_id)
                tmdb_cast_names = [
                    (c.get("name") or "").strip()
                    for c in (reg_credits.get("cast") or [] if reg_credits else [])
                    if (c.get("name") or "").strip()
                ]
            except Exception:
                tmdb_cast_names = []

            imdb_cast = cdata.get("cast") or []
            matched_actors = []
            for entry in imdb_cast:
                actor = (entry.get("actor_name") or "").strip()
                if not actor:
                    continue
                for tmdb_name in tmdb_cast_names:
                    if _name_score(actor, tmdb_name) >= 80:
                        matched_actors.append(f"'{actor}' ↔ '{tmdb_name}'")
                        break

            self._log(
                f"  [TMDB Enrich] Oyuncu eşleşmesi: {len(matched_actors)}/{len(imdb_cast)} "
                f"({', '.join(matched_actors[:3])}{'...' if len(matched_actors) > 3 else ''})"
            )
            if len(matched_actors) < 2:
                self._log(
                    f"  [TMDB Enrich] Yeterli oyuncu eşleşmedi "
                    f"({len(matched_actors)} < 2) — IMDb verisi korunuyor"
                )
                return

            # ── Her iki koşul sağlandı → aggregate_credits ekle ─────────────
            if kind == "tv":
                raw_credits = client.get_tv_aggregate_credits(tmdb_id)
            else:
                raw_credits = client.get_movie_credits(tmdb_id)

            if not raw_credits:
                return

            self._log(
                f"  [TMDB Enrich] id:{tmdb_id} — "
                f"cast:{len(raw_credits.get('cast') or [])} "
                f"crew:{len(raw_credits.get('crew') or [])}"
            )

            # Mevcut crew isimlerini normalize et (dedup için)
            existing_crew_keys = {
                (c.get("name") or "").lower().strip()
                for c in (cdata.get("crew") or [])
            }

            # TMDB crew ekle (IMDb'de olmayanları)
            added = 0
            for item in (raw_credits.get("crew") or []):
                name = (item.get("name") or "").strip()
                if not name:
                    continue
                dept = item.get("department", "")
                key = name.lower()
                jobs_list = item.get("jobs")  # aggregate_credits formatı
                if jobs_list:
                    for job_entry in jobs_list:
                        job = job_entry.get("job", dept)
                        if key not in existing_crew_keys:
                            cdata["crew"].append({
                                "name": name,
                                "job": job,
                                "role": job,
                                "department": dept,
                                "episode_count": job_entry.get("episode_count", 0),
                                "raw": "tmdb",
                                "is_tmdb_verified": True,
                            })
                            existing_crew_keys.add(key)
                            added += 1
                        elif dept:
                            # Zaten crew'da var — dept hint'i sakla (yanlış sınıflandırma tespiti için)
                            _hints = cdata.setdefault("_tmdb_dept_hints", {})
                            if key not in _hints:
                                _hints[key] = {"department": dept.lower(), "job": job}
                else:
                    job = item.get("job", dept)
                    if key not in existing_crew_keys:
                        cdata["crew"].append({
                            "name": name,
                            "job": job,
                            "role": job,
                            "department": dept,
                            "raw": "tmdb",
                            "is_tmdb_verified": True,
                        })
                        existing_crew_keys.add(key)
                        added += 1
                    elif dept:
                        # Zaten crew'da var — dept hint'i sakla (yanlış sınıflandırma tespiti için)
                        _hints = cdata.setdefault("_tmdb_dept_hints", {})
                        if key not in _hints:
                            _hints[key] = {"department": dept.lower(), "job": job}

            if added:
                cdata["total_crew"] = len(cdata.get("crew") or [])
                cdata["technical_crew"] = list(cdata["crew"])
            self._log(
                f"  [TMDB Enrich] {added} yeni crew eklendi "
                f"(toplam: {cdata.get('total_crew', 0)})"
            )
            cdata["_tmdb_enrich_id"] = tmdb_id

        except Exception as e:
            self._log(f"  [TMDB Enrich] Hata: {e}")

    # ──────────────────────────────────────────────────────────────
    # KAYNAK GÜVEN ANNOTASYONU
    # ──────────────────────────────────────────────────────────────

    def _annotate_crew_confidence(
        self,
        cdata: dict,
        imdb_matched: bool,
        tmdb_matched: bool,
    ) -> None:
        """Cast ve crew kişilerine kaynak güven metadata'sı ekle.

        Her kişiye eklenen alanlar:
            ocr_field          : OCR'ın bu kişi için atadığı alan
            matched_source     : imdb / tmdb / gemini / ocr_only
            matched_department : TMDB department (varsa)
            matched_job        : TMDB job (varsa)
            final_field        : final canonical alan adı
            source_confidence  : "high" / "medium" / "low"
            flags              : list[str]

        Hiçbir veri silinmez. Tüm değişiklikler in-place ekleme olarak yapılır.
        """
        try:
            from core.gemini_crew_validator import (
                field_from_tmdb, verify_crew_role,
                _ANA_FIELDS, _DIGER_FIELDS, _crew_norm,
            )
        except ImportError:
            self._log("  [ConfAnnotation] gemini_crew_validator import edilemedi — atlanıyor")
            return

        film_title = cdata.get("film_title", "")

        # _tmdb_dept_hints: IMDb crew için TMDB cross-check sinyali
        # (_enrich_cdata_with_tmdb tarafından doldurulur; _check_credit_field_misclassification
        # da kullanır ama artık pop etmiyor — biz burada temizleyeceğiz)
        dept_hints: dict[str, dict] = dict(cdata.get("_tmdb_dept_hints") or {})
        cdata.pop("_tmdb_dept_hints", None)

        # _gemini_crew_roles → {field: [name, ...]} — neither-match path için
        gemini_by_name: dict[str, str] = {}
        for field, names in (cdata.get("_gemini_crew_roles") or {}).items():
            for n in (names or []):
                if isinstance(n, str) and n.strip():
                    gemini_by_name[_crew_norm(n.strip())] = field

        # ── CAST ──────────────────────────────────────────────────────────
        for person in (cdata.get("cast") or []):
            raw = person.get("raw", "")
            person.setdefault("flags", [])
            person["final_field"] = "OYUNCU"
            if raw == "imdb":
                # IMDb LOCK — TMDB film-level batch match ile dolaylı doğrulama
                person["matched_source"] = "imdb"
                person["source_confidence"] = "medium"
            elif raw == "tmdb":
                # TMDB LOCK — batch cast match ile doğrulandı
                person["matched_source"] = "tmdb"
                person["source_confidence"] = "medium"
            else:
                person["matched_source"] = "ocr_only"
                person["source_confidence"] = "low"
                if "no_external_match" not in person["flags"]:
                    person["flags"].append("no_external_match")

        # ── CREW ──────────────────────────────────────────────────────────
        for person in (cdata.get("crew") or []):
            name = (person.get("name") or "").strip()
            raw  = person.get("raw", "")
            job  = (person.get("job") or "").strip()
            dept = (person.get("department") or "").strip()
            person.setdefault("flags", [])

            name_norm = _crew_norm(name)

            # TMDB department+job → canonical field (crew dict'ten)
            mapped_field = field_from_tmdb(dept, job) if dept else ""

            # _tmdb_dept_hints: IMDb kişi için TMDB cross-check bilgisi
            hint      = dept_hints.get(name_norm) or {}
            hint_dept = (hint.get("department") or "").strip()   # lowercase
            hint_job  = (hint.get("job") or "").strip()
            hint_field = field_from_tmdb(hint_dept, hint_job) if hint_dept else ""

            # ocr_field: _check_credit_field_misclassification tarafından _ocr_job
            # olarak saklanmış olabilir; yoksa job'ı dene
            ocr_field = (person.get("_ocr_job") or "").strip().upper()
            if not ocr_field:
                job_u = job.upper()
                if job_u in (_ANA_FIELDS | _DIGER_FIELDS):
                    ocr_field = job_u

            person["ocr_field"] = ocr_field

            # Canonical field: FIELD_MAP > hint_field > ocr_field > job.upper()
            canonical = mapped_field or hint_field or ocr_field or job.upper()
            person["final_field"] = canonical

            # matched_source + TMDB dept/job
            if raw == "imdb":
                person["matched_source"] = "imdb"
                person["matched_department"] = hint_dept.title() if hint_dept else ""
                person["matched_job"] = hint_job
            elif raw == "tmdb":
                person["matched_source"] = "tmdb"
                person["matched_department"] = dept
                person["matched_job"] = job
            else:
                person["matched_source"] = (
                    "gemini" if name_norm in gemini_by_name else "ocr_only"
                )
                person["matched_department"] = ""
                person["matched_job"] = ""

            # ── Ana Alanlar (YÖNETMEN, YAPIMCI) ──────────────────────────
            if canonical in _ANA_FIELDS:
                if raw == "imdb":
                    if hint_field == canonical:
                        # IMDb + TMDB ikisi de aynı alanı onaylıyor
                        person["source_confidence"] = "high"
                    else:
                        # Sadece IMDb — ya da IMDb+TMDB çelişiyor
                        person["source_confidence"] = "medium"
                        if hint_field and hint_field != canonical:
                            person["flags"].append("source_conflict")
                elif raw == "tmdb":
                    if imdb_matched:
                        # IMDb lock sonrası TMDB enrich — IMDb zaten otorize etti
                        person["source_confidence"] = "medium"
                    else:
                        # Sadece TMDB → Gemini doğrulasın (Ana Alan için makul sayı)
                        gans = verify_crew_role(film_title, name, canonical)
                        if gans == "YES":
                            person["source_confidence"] = "medium"
                            person["flags"].append("tmdb_gemini_confirmed")
                        else:
                            # Veri silinmez — işaretlenir
                            person["source_confidence"] = "low"
                            person["flags"].append("tmdb_only_unconfirmed")
                elif person["matched_source"] == "gemini":
                    person["source_confidence"] = "medium"
                    if "gemini_verified" not in person["flags"]:
                        person["flags"].append("gemini_verified")
                else:
                    person["source_confidence"] = "low"
                    if "no_external_match" not in person["flags"]:
                        person["flags"].append("no_external_match")

            # ── Diğer Ekip ────────────────────────────────────────────────
            elif canonical in _DIGER_FIELDS:
                if raw in ("imdb", "tmdb"):
                    if mapped_field and ocr_field and mapped_field == ocr_field:
                        # TMDB dept+job ile OCR alanı uyumlu
                        person["source_confidence"] = "medium"
                        if "tmdb_dept_match" not in person["flags"]:
                            person["flags"].append("tmdb_dept_match")
                    elif raw == "tmdb" and not imdb_matched:
                        # TMDB var ama OCR uyumsuz/yok → Gemini hakem
                        check_field = ocr_field or canonical
                        gans = verify_crew_role(film_title, name, check_field)
                        if gans == "YES":
                            person["source_confidence"] = "medium"
                            person["flags"].append("gemini_override")
                        else:
                            # Veri silinmez — işaretlenir
                            person["source_confidence"] = "low"
                            person["flags"].append("unverified_mismatch")
                            if mapped_field and mapped_field != ocr_field:
                                person["tmdb_suggests"] = mapped_field
                    else:
                        # IMDb + Diğer Ekip → IMDb'ye güven
                        person["source_confidence"] = "medium"
                elif person["matched_source"] == "gemini":
                    person["source_confidence"] = "medium"
                    if "gemini_verified" not in person["flags"]:
                        person["flags"].append("gemini_verified")
                else:
                    person["source_confidence"] = "low"
                    if "no_external_match" not in person["flags"]:
                        person["flags"].append("no_external_match")

            # ── Diğer (bilinmeyen alan) ───────────────────────────────────
            else:
                if raw in ("imdb", "tmdb") or person["matched_source"] == "gemini":
                    person["source_confidence"] = "medium"
                else:
                    person["source_confidence"] = "low"
                    if "no_external_match" not in person["flags"]:
                        person["flags"].append("no_external_match")

        ann_crew  = sum(1 for p in (cdata.get("crew")  or []) if "source_confidence" in p)
        ann_cast  = sum(1 for p in (cdata.get("cast")  or []) if "source_confidence" in p)
        self._log(f"  [ConfAnnotation] {ann_crew} crew + {ann_cast} cast annotate edildi")

    # ──────────────────────────────────────────────────────────────
    # KREDİ ALANI YANLIŞ SINIFLANDIRMA KONTROLÜ
    # ──────────────────────────────────────────────────────────────

    def _check_credit_field_misclassification(self, cdata: dict) -> None:
        """OCR-atanan alan ile TMDB departman bilgisini karşılaştır.

        OCR bir kişiyi YÖNETMEN olarak işaretleyebilir; TMDB ise aynı kişiyi
        'writing' (SENARYO) departmanında listeler. Bu yöntem bu çakışmaları
        tespit eder, TMDB'yi otoriter kaynak olarak kabul edip job'ı düzeltir
        ve her uyuşmazlığı _field_mismatch=True ile işaretler.

        Hiçbir veri silinmez. Düzeltmeler in-place yapılır.

        Kategori 1 (YÖNETMEN/YAPIMCI/OYUNCU): IMDb + TMDB çift kaynak.
          IMDb zaten crew'a raw='imdb' olarak yazıldığından, yalnızca
          IMDb'de bulunmayan ama OCR'ın yanlış atadığı kişiler kontrol edilir.
        Kategori 2 (SENARYO/GÖRÜNTÜ YÖNETMENİ/KAMERA/KURGU/YÖNETMEN YARDIMCISI):
          TMDB otoriter kaynak; uyuşmazlıkta TMDB kazanır.
        """
        import re as _re
        import unicodedata as _unicodedata

        def _norm(s: str) -> str:
            nfkd = _unicodedata.normalize("NFKD", s)
            ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
            return _re.sub(r"[^a-z0-9]", "", ascii_.lower())

        # OCR field → beklenen TMDB department
        _OCR_TO_DEPT = {
            "YONETMEN": "directing",      "YÖNETMEN": "directing",
            "YAPIMCI": "production",
            "OYUNCU": "acting",
            "SENARYO": "writing",
            "GÖRÜNTÜ YÖNETMENİ": "camera", "GORUNTU YONETMENI": "camera",
            "KAMERA": "camera",
            "KURGU": "editing",
            "YONETMEN YARDIMCISI": "directing", "YÖNETMEN YARDIMCISI": "directing",
        }

        # Kategori 1: hem IMDb hem TMDB kaynaklı alanlar
        _CAT1 = {"YONETMEN", "YÖNETMEN", "YAPIMCI", "OYUNCU"}

        crew = cdata.get("crew") or []

        # TMDB dept lookup: normalized_name → {department, job}
        # Kaynak 1: raw="tmdb" crew girişleri (department alanı dolu olanlar)
        dept_lookup: dict = {}
        for entry in crew:
            if entry.get("raw") == "tmdb" and entry.get("name") and entry.get("department"):
                key = _norm(entry["name"])
                if key not in dept_lookup:
                    dept_lookup[key] = {
                        "department": (entry.get("department") or "").lower(),
                        "job": entry.get("job") or "",
                    }

        # Kaynak 2: _tmdb_dept_hints (_enrich_cdata_with_tmdb tarafından doldurulur)
        for name_key, hint in (cdata.get("_tmdb_dept_hints") or {}).items():
            key = _norm(name_key)
            if key not in dept_lookup:
                dept_lookup[key] = hint

        if not dept_lookup:
            return  # TMDB verisi yoksa kontrol yapılamaz

        mismatches = []
        for entry in crew:
            if entry.get("raw") != "ocr_verified":
                continue
            ocr_job = entry.get("job") or ""
            expected_dept = _OCR_TO_DEPT.get(ocr_job)
            if not expected_dept:
                continue

            person_key = _norm(entry.get("name") or "")
            if not person_key:
                continue

            hint = dept_lookup.get(person_key)
            if not hint:
                continue  # TMDB'de yok → karşılaştırılamaz

            tmdb_dept = (hint.get("department") or "").lower()
            tmdb_job = hint.get("job") or ""

            if tmdb_dept == expected_dept:
                continue  # Eşleşiyor → sorun yok

            # ── Uyuşmazlık tespit edildi ──────────────────────────────────
            cat = "1" if ocr_job in _CAT1 else "2"
            entry["_field_mismatch"] = True
            entry["_ocr_job"] = ocr_job
            entry["_tmdb_department"] = tmdb_dept
            entry["_tmdb_job"] = tmdb_job

            # TMDB kazanır: job'ı düzelt (veri silinmez, sadece güncellenir)
            if tmdb_job:
                entry["job"] = tmdb_job
                entry["role"] = tmdb_job
                entry["role_tr"] = self._JOB_TR.get(tmdb_job, tmdb_job)

            mismatches.append({
                "name": entry.get("name"),
                "ocr_job": ocr_job,
                "tmdb_dept": tmdb_dept,
                "tmdb_job": tmdb_job,
                "category": cat,
            })
            self._log(
                f"  [KrediAlan] Uyuşmazlık (Cat-{cat}): "
                f"'{entry.get('name')}' OCR={ocr_job} TMDB={tmdb_dept}/{tmdb_job}"
                + (f" → job={tmdb_job} olarak düzeltildi" if tmdb_job else " → job korundu")
            )

        if mismatches:
            cdata["_credit_field_mismatches"] = mismatches
            self._log(
                f"  [KrediAlan] {len(mismatches)} yanlış sınıflandırma düzeltildi"
            )

    # ──────────────────────────────────────────────────────────────
    # GOOGLE VI
    # ──────────────────────────────────────────────────────────────
    def _decide_google_vi(
        self,
        tmdb_matched: bool,
        resolution: str,
        ocr_lines: list,
        segment_duration_min: float,
        cast_count: int = 0,
    ):
        """GoogleVITextEngine.decide_after_tmdb() çağrısını lazy import ile sarmala."""
        try:
            from core.google_video_intelligence import GoogleVITextEngine, GVIDecision
        except ImportError:
            from dataclasses import dataclass, field

            @dataclass
            class GVIDecision:
                should_run: bool
                reason: str
                triggers: list = field(default_factory=list)

            return GVIDecision(False, "google_vi modülü yok — VI atlanıyor")

        try:
            vi_engine = GoogleVITextEngine(self.config, log_cb=self._log)

            # Font tipini bbox geometrisinden tahmin et
            font_type = "unknown"
            try:
                from core.ocr_engine import OCREngine
                frame_bboxes = []
                for line in ocr_lines:
                    fp = (line.get("frame_path", "") if isinstance(line, dict)
                          else getattr(line, "frame_path", ""))
                    bbox = (line.get("bbox", []) if isinstance(line, dict)
                            else getattr(line, "bbox", []))
                    if fp and bbox:
                        frame_bboxes.append((fp, [bbox]))
                if frame_bboxes:
                    font_type = OCREngine.estimate_font_type(frame_bboxes)
            except Exception as fe:
                self._log(f"  [GOOGLE_VI] Font tahmin hatası: {fe}")

            # Conf ortalaması
            ocr_avg_conf = 0.0
            if ocr_lines:
                confs = []
                for line in ocr_lines:
                    c = (line.get("avg_confidence", line.get("confidence", 0.0))
                         if isinstance(line, dict)
                         else getattr(line, "avg_confidence",
                                      getattr(line, "confidence", 0.0)))
                    confs.append(float(c))
                if confs:
                    ocr_avg_conf = sum(confs) / len(confs)

            return vi_engine.decide_after_tmdb(
                tmdb_matched=tmdb_matched,
                resolution=resolution,
                font_type=font_type,
                segment_duration_min=segment_duration_min,
                ocr_avg_conf=ocr_avg_conf,
                total_ocr_lines=len(ocr_lines),
                cast_count=cast_count,
            )
        except Exception as e:
            self._log(f"  [GOOGLE_VI] Karar hatası: {e}")
            try:
                from core.google_video_intelligence import GVIDecision
                return GVIDecision(False, f"VI karar hatası: {e}")
            except ImportError:
                pass
            from dataclasses import dataclass, field

            @dataclass
            class _GVIDecision:
                should_run: bool
                reason: str
                triggers: list = field(default_factory=list)

            return _GVIDecision(False, f"VI karar hatası: {e}")

    def _run_google_vi(
        self,
        video_path: str,
        info: dict,
        first_min: float,
        last_min: float,
    ) -> list:
        """Google VI'yı entry + exit segmentleriyle çalıştır."""
        try:
            from core.google_video_intelligence import GoogleVITextEngine
        except ImportError:
            self._log("  [GOOGLE_VI] google_video_intelligence modülü yok — atlanıyor")
            return []

        try:
            vi_engine = GoogleVITextEngine(self.config, log_cb=self._log)
            duration = float(info.get("duration_seconds", 0))
            entry_start = 0.0
            entry_end = min(first_min * 60.0, duration)
            exit_end = duration
            exit_start = max(0.0, duration - last_min * 60.0)
            return vi_engine.process_segments(
                video_path, entry_start, entry_end, exit_start, exit_end
            )
        except Exception as e:
            self._log(f"  [GOOGLE_VI] İşlem hatası: {e}")
            return []

    # ──────────────────────────────────────────────────────────────
    # YARDIMCI
    # ──────────────────────────────────────────────────────────────
    def _stage(self, name, elapsed, status="ok", **details):
        self.stage_stats[name] = {
            "duration_sec": round(elapsed, 2),
            "status": status,
            **details,
        }
        self.stats.end_stage(name, status)

    @staticmethod
    def _extract_film_title_from_filename(stem: str) -> str:
        """evoArcadmin format dosya adından film başlığını ayıkla.

        Desteklenen formatlar:
          1. {prefix}_{FILM_ADI}_{YIL}-{rest}  (yıl bazlı segment, çok kelimeli başlık)
             evoArcadmin_DÖNÜŞÜ OLMAYAN NEHİR_1989-0624-... → "Dönüşü Olmayan Nehir"
             evoArcadmin_KÜL_KEDİSİ_1955-0019-... → "Kül Kedisi"
          2. {prefix}_{code}_{YIL}-NNNN-N-NNNN-NN-N-BAŞLIK  (sayısal blok sonunda başlık)
             evoArcadmin_TEST1_1955-0019-1-0000-00-1-KÜL_KEDİSİ → "Kül Kedisi"

        Eğer format tanınmazsa orijinal stem döndürülür.
        """
        def _tr_capitalize(w: str) -> str:
            if not w:
                return w
            # Keep first character as-is (already correct uppercase form)
            # For the rest: replace İ→i and I→ı, then lowercase
            rest = w[1:].replace('\u0130', 'i').replace('I', 'ı').lower()
            return w[0] + rest

        def _to_title(raw: str) -> str:
            return ' '.join(_tr_capitalize(w) for w in raw.replace('_', ' ').split())

        # Strateji 1 (en güvenilir): Sayısal blok sonundaki başlık (4-4-1-3/4-2-1-BAŞLIK)
        m = re.search(r'\d{4}-\d{3,4}-\d-\d{3,4}-\d{2}-\d-(.+)$', stem)
        if m:
            title = _to_title(m.group(1))
            if title:
                return title

        # Strateji 2 (fallback): {prefix}_{FILM_ADI}_{YIL}-{rest} formatı
        # Alt çizgiyle ayrılmış segmentlerde yıl (4 rakam) ile başlayan bloğu bul
        # Sadece çok kelimeli orta segment gerçek başlık olarak değerlendirilir
        parts = stem.split('_')
        for i, part in enumerate(parts):
            if i > 0 and re.match(r'^\d{4}-', part):
                if i > 1:
                    mid_raw = '_'.join(parts[1:i])
                    # Birden fazla kelime varsa (boşluk veya alt çizgiyle) gerçek başlık
                    mid_words = mid_raw.replace('_', ' ').split()
                    if len(mid_words) >= 2:
                        title = _to_title(mid_raw)
                        if title:
                            return title
                break

        return stem

    @staticmethod
    def _empty_credits_data(film_title: str = "") -> dict:
        """audio_only kapsamı için minimal credits şablonu."""
        return {
            "film_title": film_title,
            "year": None,
            "cast": [],
            "crew": [],
            "technical_crew": [],
            "directors": [],
            "production_companies": [],
            "production_info": [],
            "total_actors": 0,
            "total_crew": 0,
            "total_companies": 0,
            "verification_status": "unverified",
        }

    def _write_database(self, video_info: dict, credits_data: dict,
                         credits_raw, ocr_lines: list, stage_stats: dict,
                         audio_result, work_dir: str,
                         content_profile_name: str, ts: str,
                         xml_path: str = "") -> None:
        """DATABASE dizinine pipeline çıktılarının bir kopyasını yaz."""
        if not self.config.get("database_enabled", True):
            return

        db_root = (
            self.config.get("database_root") or
            os.environ.get("VITOS_DATABASE_ROOT") or
            r"D:\DATABASE\FilmDizi"
        )

        film_title = (credits_data.get("film_title") or "").strip()
        if film_title:
            stem = film_title.replace(" ", "_")
        else:
            stem = Path(video_info.get("filename", "out")).stem

        db_dir = Path(db_root) / stem
        db_dir.mkdir(parents=True, exist_ok=True)

        # 1. work_dir içindeki tüm dosyaları DB klasörüne kopyala
        # work_dir zaten db_dir altındaysa (veya aynıysa) gereksiz kopyalamayı atla
        work_path = Path(work_dir).resolve()
        db_dir_resolved = db_dir.resolve()
        if not (work_path == db_dir_resolved or db_dir_resolved in work_path.parents):
            for src in Path(work_dir).iterdir():
                if src.is_file():
                    dst = _safe_path(db_dir / src.name)
                    shutil.copy2(src, dst)

        # 1b. XML sidecar dosyasını DB klasörüne kopyala
        if xml_path and Path(xml_path).is_file():
            xml_dst = _safe_path(db_dir / Path(xml_path).name)
            shutil.copy2(xml_path, xml_dst)
            self._log(f"  [DATABASE] XML sidecar kopyalandı: {Path(xml_path).name}")

        # 2. OCR dual-score JSON yaz
        ocr_scores = self._build_ocr_scores(ocr_lines, credits_data)
        with open(_safe_path(db_dir / f"{stem}_{ts}_ocr_scores.json"), "w", encoding="utf-8") as f:
            json.dump(ocr_scores, f, ensure_ascii=False, indent=2)

        # 3. Ham credits JSON yaz (LLM filtre öncesi)
        with open(_safe_path(db_dir / f"{stem}_{ts}_credits_raw.json"), "w", encoding="utf-8") as f:
            json.dump(credits_raw if credits_raw is not None else credits_data,
                      f, ensure_ascii=False, indent=2)

        # 4. Transcript JSON yaz
        transcript = []
        if audio_result and isinstance(audio_result, dict):
            for seg in audio_result.get("transcript", []):
                transcript.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", ""),
                })
        with open(_safe_path(db_dir / f"{stem}_{ts}_transcript.json"), "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        # 5. Debug log yaz
        with open(_safe_path(db_dir / f"{stem}_{ts}_debug.log"), "w", encoding="utf-8-sig") as f:
            f.write("\n".join(self._log_messages))

        # 6. ADA1 — OCR Ham Çıktı Listesi
        vname = Path(video_info.get("filename", "out")).stem
        ada1_path = _safe_path(db_dir / f"{vname}_ada1.json")
        self._write_ada1(ocr_lines, vname, credits_data, ada1_path)

        self._log(f"  [DATABASE] Yazıldı: {db_dir}")

    def _write_ada1(self, ocr_lines: list, vname: str,
                    cdata: dict, out_path) -> None:
        """ADA1 — OCR Ham Çıktı Listesi.

        Her OCR satırı için:
          ocr   = motorden gelen ham metin (_repair_turkish öncesi)
          norm1 = _repair_turkish sonrası (NameDB + Türkçe karakter)
          norm2 = pipeline sonundaki nihai isim (Gemini/IMDb/TMDB düzeltmesi)
          confidence = OCR güven skoru
          kategori   = oyuncu / ekip / sirket / bilinmiyor

        Bağımsız dosyadır — pipeline kararlarına hiç dokunmaz.
        """
        import re as _re
        from datetime import datetime as _dt
        from pathlib import Path as _Path

        # ── bolum_no: dosya adından çıkar ────────────────────────────
        _bm = _re.search(r'\d{4}-\d{2,4}-\d+-(\d{4})-', vname)
        bolum_no = _bm.group(1) if _bm else ""

        # ── ASCII normalize yardımcısı (Türkçe karakter farkını dengeler) ──
        _TR_MAP = str.maketrans("ğışöüçĞİŞÖÜÇı", "gisoucGISOUCi")
        def _ascii_key(s: str) -> str:
            return s.lower().translate(_TR_MAP)

        # ── cdata'dan isim → {kategori, norm2} haritası ──────────────
        # İki anahtar: orijinal lower + ASCII normalize (Türkçe fark toleransı)
        _lookup: dict[str, dict] = {}
        for entry in (cdata.get("cast") or []):
            name = (entry.get("actor_name") or "").strip()
            if name:
                val = {"kategori": "oyuncu", "norm2": name}
                _lookup[name.lower()] = val
                _lookup[_ascii_key(name)] = val
        for entry in (cdata.get("crew") or []):
            name = (entry.get("name") or "").strip()
            if name:
                val = {"kategori": "ekip", "norm2": name}
                _lookup[name.lower()] = val
                _lookup[_ascii_key(name)] = val
        for entry in (cdata.get("companies") or []):
            name = (entry.get("name") or "").strip()
            if name:
                val = {"kategori": "sirket", "norm2": name}
                _lookup[name.lower()] = val
                _lookup[_ascii_key(name)] = val

        # ── OCR satırlarını dönüştür ──────────────────────────────────
        satirlar = []
        for line in (ocr_lines or []):
            if isinstance(line, dict):
                norm1      = (line.get("text") or "").strip()
                ocr_raw    = (line.get("text_original") or norm1).strip()
                confidence = float(line.get("avg_confidence") or 0.0)
            else:
                norm1      = (getattr(line, "text", "") or "").strip()
                ocr_raw    = (getattr(line, "text_original", "") or norm1).strip()
                confidence = float(getattr(line, "avg_confidence", 0.0) or 0.0)

            if not norm1:
                continue

            # norm2 + kategori: tam → ASCII normalize → substring sıralamasıyla ara
            hit = (
                _lookup.get(norm1.lower()) or
                _lookup.get(_ascii_key(norm1)) or
                next(
                    (val for key, val in _lookup.items()
                     if _ascii_key(norm1) in key or key in _ascii_key(norm1)),
                    None,
                )
            )

            norm2    = hit["norm2"]    if hit else norm1
            kategori = hit["kategori"] if hit else "bilinmiyor"

            satirlar.append({
                "ocr"       : ocr_raw,
                "norm1"     : norm1,
                "norm2"     : norm2,
                "confidence": round(confidence, 4),
                "kategori"  : kategori,
            })

        ada1 = {
            "video_id"    : vname,
            "video_title" : (cdata.get("film_title") or "").strip(),
            "bolum_no"    : bolum_no,
            "olusturulma" : _dt.now().isoformat(timespec="seconds"),
            "satirlar"    : satirlar,
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ada1, f, ensure_ascii=False, indent=2)
            self._log(f"  [ADA1] {len(satirlar)} satır → {_Path(out_path).name}")
        except Exception as e:
            self._log(f"  [ADA1] Yazma hatası: {e}")

    @staticmethod
    def _build_ocr_scores(ocr_lines: list, credits_data: dict) -> dict:
        """OCR satırları ile pipeline sonuçlarını eşleştirerek ikili skor JSON üret."""
        import re

        def _norm(s: str) -> str:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9]", "", s)
            return s

        cast = credits_data.get("cast", [])
        cast_lookup: dict[str, dict] = {}
        for entry in cast:
            for field in ("actor_name", "character_name"):
                name = (entry.get(field) or "").strip()
                if name:
                    cast_lookup[_norm(name)] = entry

        scores = []
        for line in ocr_lines:
            text = (line.text if hasattr(line, "text") else
                    line.get("text", "") if isinstance(line, dict) else "")
            ocr_conf = (getattr(line, "avg_confidence", None)
                        if hasattr(line, "avg_confidence") else
                        line.get("avg_confidence", line.get("confidence", 0))
                        if isinstance(line, dict) else 0)
            seen_count = (getattr(line, "seen_count", 1)
                          if hasattr(line, "seen_count") else
                          line.get("seen_count", 1)
                          if isinstance(line, dict) else 1)

            cast_entry = cast_lookup.get(_norm(text))
            if cast_entry:
                pipeline_conf = cast_entry.get("confidence")
                name_db_match = bool(cast_entry.get("is_verified_name", False))
                llm_verified = bool(cast_entry.get("is_llm_verified", False))
                verdict = "KEEP"
            else:
                pipeline_conf = None
                name_db_match = False
                llm_verified = False
                verdict = "REJECTED"

            scores.append({
                "text": text,
                "ocr_confidence": ocr_conf,
                "pipeline_confidence": pipeline_conf,
                "seen_count": seen_count,
                "name_db_match": name_db_match,
                "llm_verified": llm_verified,
                "verdict": verdict,
            })

        return {"scores": scores}

    def _repair_turkish(self, ocr_lines: list) -> list:
        if not ocr_lines:
            return ocr_lines
        repaired = 0
        for line in ocr_lines:
            original = (line.get("text", "") if isinstance(line, dict)
                        else getattr(line, "text", ""))
            if not original:
                continue
            fixed = self._name_db.correct_line(original)
            if fixed != original:
                if isinstance(line, dict):
                    line["text"] = fixed
                    line["text_original"] = original
                else:
                    line.text = fixed
                    line.text_original = original
                repaired += 1
                self._log(f"    [NameDB] '{original}' → '{fixed}'")
        if repaired:
            self._log(f"  [NameDB] {repaired} satır Türkçe onarımı yapıldı")
        return ocr_lines

    def _repair_layout_pairs(self, layout_pairs: list) -> list:
        """
        Layout pair'lerden gelen bozuk actor isimlerini NameDB ile onar.
        TurkishNameDB.repair_layout_pairs() kullanır (threshold 0.85).
        """
        if not layout_pairs:
            return layout_pairs

        if hasattr(self._name_db, 'repair_layout_pairs'):
            repaired_pairs, diff = self._name_db.repair_layout_pairs(layout_pairs)
            if diff:
                self._log(f"  [NameDB] {diff} layout pair ismi onarıldı")
            return repaired_pairs

        repaired = 0
        for pair in layout_pairs:
            original = getattr(pair, 'actor_name', '')
            if not original:
                continue
            words = original.strip().split()
            fixed_words = []
            changed = False
            for w in words:
                result, score = self._name_db.find(w)
                if result and score >= 0.85 and result != w:
                    fixed_words.append(result)
                    changed = True
                else:
                    fixed_words.append(w)
            if changed:
                pair.actor_name = ' '.join(fixed_words)
                repaired += 1
        if repaired:
            self._log(f"  [NameDB] {repaired} layout pair ismi onarıldı")
        return layout_pairs

    def _merge_blok2_results(self, paddle_lines: list, vlm_lines: list) -> list:
        """PaddleOCR ve VLM (BLOK2) sonuçlarını fuzzy dedup ile birleştir.

        VLM satırları için _repair_turkish() çağrılmaz (VLM zaten doğru okur).
        PaddleOCR'da zaten bulunan satırlarla fuzzy benzerlik kontrolü yapılır;
        yeterince benzer olanlar eklenmez (tekrar önleme).
        """
        if not vlm_lines:
            return paddle_lines

        try:
            from rapidfuzz.fuzz import ratio as _fuzz_ratio
            _has_rapidfuzz = True
        except ImportError:
            _has_rapidfuzz = False

        paddle_texts = [
            (line.get("text", "") if isinstance(line, dict)
             else getattr(line, "text", "")).strip().lower()
            for line in paddle_lines
        ]

        merged = list(paddle_lines)
        added = 0
        for vlm_line in vlm_lines:
            vtext = (vlm_line.get("text", "")).strip()
            if not vtext:
                continue
            vlow = vtext.lower()
            # Exact match check
            if vlow in paddle_texts:
                continue
            # Fuzzy dedup (rapidfuzz mevcutsa)
            if _has_rapidfuzz:
                is_dup = any(
                    _fuzz_ratio(vlow, pt) >= _BLOK2_FUZZY_THRESHOLD
                    for pt in paddle_texts
                    if pt
                )
                if is_dup:
                    continue
            merged.append(vlm_line)
            paddle_texts.append(vlow)
            added += 1

        if added:
            self._log(f"  [BLOK2] {added} benzersiz VLM satırı eklendi")
        return merged
