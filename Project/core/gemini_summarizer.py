"""
gemini_summarizer.py — Transcript'ten son kullanıcı için Türkçe özet çıkarma.

Model Stratejisi: Gemini 2.5 Pro → OpenAI GPT-5.4 mini → Gemini Flash

Dil Stratejisi:
  • Türkçe transcript → model ekibiyle doğrudan Türkçe özet (tek adım)
  • `tr` dışındaki tüm transcript dilleri → orijinal dilde ara özet → Türkçeye çeviri

Final çıktı sözleşmesi:
  • Son kullanıcı özeti her zaman Türkçe olmalı
  • Tek paragraf, kısa ve spoiler içeren anlatım olmalı
  • Yabancı özel isimler Latin alfabede kalmalı
  • Latin dışı script (Arapça, Yunanca, Kiril, Devanagari vb.) son kullanıcı özetine sızmamalı

summarize_transcript(...) -> dict | None
  → {
        "text": "<final Türkçe özet>",
        "language": "tr",
        "model_used": "gemini-2.5-pro"|"gpt-5.4-mini"|"gemini-2.5-flash",
        "translation_model_used": "gemini-2.5-pro"|"gpt-5.4-mini"|"gemini-2.5-flash",
        "flow": "single_step"|"two_step",
     }
"""

import re
import unicodedata

import core.llm_provider as _llm
from config.runtime_paths import get_openai_api_key

_TIMEOUT_SEC = 90
_MAX_CHARS = 120000
_TRANSLATE_TIMEOUT_SEC = 60
_DEFAULT_MODEL_TEAM = (
    {"provider": "gemini", "model": "gemini-2.5-pro"},
    {"provider": "openai", "model": "gpt-5.4-mini"},
    {"provider": "gemini", "model": "gemini-2.5-flash"},
)

# ─────────────────────────────────────────────────────────────────────────────
# DİL ETİKETLERİ
# ─────────────────────────────────────────────────────────────────────────────
LANG_LABELS = {
    "tr": "TR", "en": "İNG", "de": "ALM", "fr": "FRA",
    "es": "İSP", "it": "İTA", "ru": "RUS", "ar": "ARA",
    "ja": "JAP", "ko": "KOR", "zh": "ÇİN", "pt": "POR",
    "nl": "HOL", "pl": "POL", "sv": "İSV", "da": "DAN",
    "fi": "FİN", "no": "NOR", "el": "YUN", "hu": "MAC",
    "cs": "ÇEK", "ro": "ROM", "hi": "HİN",
    "ku": "KÜR", "fa": "FAR",
}


def get_language_label(lang_code: str) -> str:
    """ISO dil kodundan kısa Türkçe etiket döndür."""
    if not lang_code:
        return "VERİ YOK"
    return LANG_LABELS.get(lang_code.lower(), lang_code.upper())


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Türkçe özet (Türkçe transcript için)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TR = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n"
    "The transcript is in Turkish. Analyze the content and write the summary in Turkish.\n\n"
    "Your task: analyze the story in the transcript and produce a concise, "
    "direct summary in narrative prose format.\n\n"
    "STRICT RULES — THESE MUST BE FOLLOWED WITHOUT EXCEPTION:\n\n"
    "LENGTH: Target 50 words, absolute maximum 55 words. "
    "Target 3–4 sentences (4 is ideal, 3 is acceptable). "
    "If the draft exceeds 55 words, rewrite it shorter before returning.\n\n"
    "STRUCTURE (MANDATORY): The summary must follow a 4-beat arc:\n"
    "  Sentence 1 — Setup (who / where / initial situation).\n"
    "  Sentence 2 — Development (the central conflict or turning point).\n"
    "  Sentence 3 — Escalation (the decisive action or crisis).\n"
    "  Sentence 4 — Ending (explicit final outcome — see ENDING rule below).\n"
    "4 sentences is ideal. 3 sentences is acceptable ONLY IF the ending sentence "
    "still contains the explicit concrete outcome (sentences 2 and 3 may be merged, "
    "but the ENDING rule is never negotiable).\n\n"
    "ZERO SUSPENSE: Attempting to make the reader curious about the film is STRICTLY FORBIDDEN. "
    "Do NOT use phrases like 'faces life's challenges', 'a big decision awaits', "
    "'what will happen next?' or any marketing/teaser language.\n\n"
    "ENDING (MANDATORY — NON-NEGOTIABLE): The LAST sentence MUST state the film's "
    "concrete final outcome explicitly. Vague closings like 'hayatı değişir', "
    "'bir karar verir', 'yolculuğu başlar' are STRICTLY FORBIDDEN.\n"
    "The final sentence MUST use one of these explicit Turkish ending patterns "
    "(choose whichever fits the story):\n"
    "  • 'Film, ... ile biter.'\n"
    "  • 'Sonunda ...'\n"
    "  • '... ölür.'\n"
    "  • '... yakalanır.'\n"
    "  • '... evlenir.'\n"
    "  • '... ayrılır.'\n"
    "  • '... geri döner.'\n"
    "  • '... patron olur.'\n"
    "  • '... barışır.'\n"
    "  • '... bir araya gelir.'\n"
    "  • '... kaçar.' / '... kurtulur.' / '... kaybeder.' / '... kazanır.'\n"
    "If none of these patterns literally fits, still write a concrete result sentence "
    "naming WHO ends up in WHAT state. Never end on suspense, implication, or a "
    "rhetorical question.\n\n"
    "FORMAT: No bullet points. Write ONE short paragraph of 3–4 sentences. "
    "No line breaks inside the paragraph.\n\n"
    "SENTENCE STYLE: Every sentence must be short and direct. No comma-chains. "
    "One idea per sentence. If a sentence feels overloaded, split it.\n\n"
    "FOCUS: Focus only on the main character's key turning point and their final decision. "
    "Completely ignore subplots and secondary characters.\n\n"
    "NO CHARACTER INTRODUCTIONS: Do NOT introduce or describe characters by name/age/job "
    "at the start. Just tell the story — who they are will emerge from the events.\n\n"
    "FINAL OUTPUT CONTRACT: Output language must be Turkish. No title. Start directly "
    "with the story. Every sentence must be clear, meaningful, and natural.\n\n"
    "TURKISH ORTHOGRAPHY: Except for foreign proper names, everything in the summary "
    "must use correct Turkish spelling and Turkish characters.\n\n"
    "FOREIGN NAMES: Write all foreign proper names (character names, place names, "
    "person names) using their original Latin spelling in plain ASCII form only. "
    "NEVER use Turkish-specific characters (ç, ğ, ı, ö, ş, ü, İ) in foreign names. "
    "NEVER write foreign names in Cyrillic, Greek, Arabic, or any other non-Latin script. "
    "Example: Write 'Vichita' not 'Viçita', 'Tom' not 'Töm'.\n\n"
    "FOREIGN NAME TAGGING: Wrap EVERY foreign proper name with double square brackets. "
    "Turkish words and common Turkish names (Mehmet, Ayşe, Fatma, Ali, Hasan, etc.) "
    "must NOT be wrapped.\n"
    "  Multi-word names: wrap the full name together: [[Jack Sparrow]], [[New York]].\n"
    "  Single-word names: [[Amy]], [[Charles]], [[Oliver]].\n"
    "  CORRECT: [[Amy]] şehre geldi. Kadın [[Charles]]'a aşık olur.\n"
    "  WRONG: Amy şehre geldi.  (brackets missing)\n"
    "  WRONG: [[Mehmet]] eve döndü.  (Turkish name, no brackets)\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Orijinal dilde özet (yabancı dil transcript için, adım 1)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_FOREIGN = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n\n"
    "IMPORTANT: Write the summary in the SAME LANGUAGE as the transcript. "
    "Do NOT translate to any other language.\n\n"
    "Your task: analyze the story in the transcript and produce a concise, "
    "direct summary in narrative prose format.\n\n"
    "STRICT RULES — THESE MUST BE FOLLOWED WITHOUT EXCEPTION:\n\n"
    "LENGTH: Target 80 words, maximum 100 words.\n\n"
    "ZERO SUSPENSE: Do NOT use teaser/marketing language.\n\n"
    "SPOILER (ENDING) IS MANDATORY: Include the final outcome.\n\n"
    "FORMAT: No bullet points. Write one or two short paragraphs.\n\n"
    "SENTENCE STYLE: Avoid long, comma-heavy sentences. Prefer shorter, meaningful sentences. "
    "Each sentence should carry one clear idea. If a sentence feels overloaded, split it.\n\n"
    "FOCUS: Main character's key turning point and final decision only.\n\n"
    "NO CHARACTER INTRODUCTIONS: Just tell the story.\n\n"
    "No title. Start directly with the story."
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Çeviri + isim doğrulama (yabancı dil, adım 2)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TRANSLATE = (
    "You are a professional translator and name verification specialist "
    "for a Turkish video archive database.\n\n"
    "Your task: Translate the given summary to Turkish.\n\n"
    "RULES:\n"
    "1. Translate accurately and naturally to Turkish.\n"
    "2. Except for foreign proper names, everything must use correct Turkish spelling "
    "   and Turkish characters.\n"
    "3. Prefer shorter, meaningful sentences. Do NOT keep chaining clauses with commas "
    "   just to preserve a flowing rhythm. If needed, split long sentences into two or more "
    "   shorter Turkish sentences while preserving meaning.\n"
    "4. The final output may be one paragraph or two short paragraphs, but every sentence "
    "   must be clear, meaningful, and natural.\n"
    "5. FOREIGN NAMES: Keep all foreign proper names (character names, person names, "
    "   place names) in their ORIGINAL spelling, but in plain ASCII form only. Do NOT turkify them.\n"
    "   Example: 'Jack Sparrow' stays 'Jack Sparrow', NOT 'Cek Sparov'.\n"
    "6. TURKISH NAME RULES: In Turkish text, 'i' uppercases to 'İ' and 'I' lowercases "
    "   to 'ı'. But this rule applies ONLY to Turkish words, NOT to foreign names.\n"
    "7. FOREIGN NAME TAGGING: Wrap EVERY foreign proper name with double square brackets. "
    "   Turkish words and Turkish names must NOT be wrapped.\n"
    "   Example: [[Jack Sparrow]] gemiden kaçtı. [[Elizabeth]] onu takip etti.\n"
    "8. FINAL OUTPUT MUST BE TURKISH.\n"
    "9. FOREIGN PROPER NAMES MUST REMAIN IN THE LATIN ALPHABET AND IN ASCII ONLY. "
    "   Never output Turkish-specific characters, Arabic, Greek, Cyrillic, Devanagari, "
    "   or any other non-Latin script for foreign names.\n"
    "10. If a foreign name appears in a non-Latin script, transliterate it to standard ASCII Latin "
    "    spelling, or use the exact ASCII-compatible spelling from the cast reference list if provided.\n"
    "11. Output ONLY the Turkish translation. No explanations, no notes.\n"
    "12. No bullet points.\n"
)

_SYSTEM_PROMPT_TRANSLATE_RETRY = (
    _SYSTEM_PROMPT_TRANSLATE
    + "\nRETRY OVERRIDE:\n"
    + "Your previous answer was rejected. The new answer must be valid Turkish and must not "
      "contain any non-Latin foreign names. If needed, rewrite the whole summary so that the "
      "final output satisfies the rules exactly.\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot örnek (Türkçe özet için)
# ─────────────────────────────────────────────────────────────────────────────
_FEW_SHOT = (
    "Aşağıdaki örnek DOĞRU formattır — tam olarak böyle yaz:\n\n"
    "ÖRNEK:\n"
    "Başarılı bir reklamcı olan [[David]], önemli bir müşteriyle duygusal bir ilişki "
    "içindeyken, babasının şeker hastalığı yüzünden bacağının kesilmesi gerektiğini "
    "öğrenir. Kariyeri için hayati önem taşıyan bir sunum ile babasının ameliyatı "
    "aynı güne denk gelir. İşi ve sevgilisi yerine babasını tercih eden [[David]], "
    "her şeyi elinin tersiyle iterek hastanede onun yanında kalır. Annesi de "
    "hastabakıcılık yapmayı reddedince, ameliyattan sakat çıkan babasının tüm "
    "sorumluluğu [[David]]'e kalır. Film, yıllarca birbirinden uzak kalan baba-oğulun "
    "geçmişi geride bırakıp birlikte yeni bir hayata başlamasıyla biter.\n\n"
    "Şimdi aşağıdaki transcript için aynı formatta özet yaz:\n"
)

# Türkçe'ye özgü karakterler (kontrol için)
_TR_CHARS = set("çÇğĞıİöÖşŞüÜ")
_TURKISH_MARKER_WORDS = frozenset({
    "ve", "bir", "bu", "ile", "icin", "için", "gibi", "daha", "sonra", "ancak",
    "kadar", "degil", "değil", "film", "hikaye", "karar", "olur", "eder", "olan",
    "olarak", "kendi", "ona", "onu", "onun", "bunu", "böyle", "yine", "artik", "artık",
})
_ENGLISH_MARKER_WORDS = frozenset({
    "the", "and", "with", "from", "that", "this", "into", "after", "before",
    "while", "when", "their", "they", "his", "her", "ultimately", "chooses",
    "teacher", "soldier", "exchange", "relationship", "secret",
})
_TURKISH_SUFFIX_HINTS = ("iyor", "iyorlar", "erek", "arak", "madan", "meden", "ince", "unca", "ip")
_TR_APOSTROPHE_SUFFIX_RE = re.compile(
    r"['’](?:i|ı|u|ü|in|ın|un|ün|e|a|de|da|den|dan|ye|ya|nin|nın|nun|nün)\b",
    re.IGNORECASE,
)


def _normalize_summary_text(text: str) -> str:
    """LLM çıktısındaki boşlukları düzenle, paragraf kırılımlarını koru."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""

    paragraphs = []
    for chunk in re.split(r"\n\s*\n+", raw):
        paragraph = re.sub(r"[ \t\f\v]+", " ", chunk)
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def normalize_final_summary_text(text: str) -> str:
    """Final özet metnini kullanıcıya uygun biçimde normalize et."""
    return _normalize_summary_text(text)


def _contains_non_latin_script(text: str) -> bool:
    """Latin dışı alfabetik karakter var mı? Türkçe harfler Latin kabul edilir."""
    for ch in text or "":
        if not ch.isalpha():
            continue
        if "LATIN" not in unicodedata.name(ch, ""):
            return True
    return False


def _is_turkish_text(text: str) -> bool:
    """Metnin Türkçe olup olmadığını kontrol et.

    Arapça/Kiril gibi Latin-dışı metinler (ör. yanlış dil tespiti sonucu gelen özetler)
    Latin karakter oranı kontrolüyle reddedilir.
    """
    text = _normalize_summary_text(text)
    if not text or len(text) < 20:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    latin_count = sum(
        1 for c in alpha_chars
        if "LATIN" in unicodedata.name(c, "")
    )
    if latin_count / len(alpha_chars) < 0.7:
        return False  # Latin oranı %70 altında → Türkçe değil (Arapça, Kiril vb.)
    if any(c in _TR_CHARS for c in text):
        return True
    if _TR_APOSTROPHE_SUFFIX_RE.search(text):
        return True

    words = [w.casefold() for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿÇĞİÖŞÜçğıöşü']+", text)]
    tr_hits = sum(1 for w in words if w in _TURKISH_MARKER_WORDS)
    en_hits = sum(1 for w in words if w in _ENGLISH_MARKER_WORDS)
    morph_hits = sum(1 for w in words if any(w.endswith(s) for s in _TURKISH_SUFFIX_HINTS))
    return (tr_hits >= 2 and tr_hits > en_hits) or (morph_hits >= 1 and en_hits == 0)


def _validate_final_summary(text: str) -> tuple[bool, str]:
    """Final kullanıcı özeti sözleşmesini doğrula."""
    normalized = _normalize_summary_text(text)
    if not normalized:
        return False, "empty"
    if not _is_turkish_text(normalized):
        return False, "not_turkish"
    if _contains_non_latin_script(normalized):
        return False, "non_latin_script"
    return True, ""


def is_valid_final_summary(text: str) -> bool:
    """Son kullanıcıya gösterilecek özet geçerli mi?"""
    return _validate_final_summary(text)[0]


def _infer_provider_for_model(model: str) -> str:
    """Model adına göre provider tahmini yap."""
    model_l = str(model or "").strip().lower()
    if model_l.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return "gemini"


def _build_model_team(
    preferred_model: str = "",
    *,
    openai_api_key: str = "",
) -> list[dict[str, str]]:
    """Özet/çeviri için kullanılacak model sırasını üret."""
    if preferred_model:
        return [{
            "provider": _infer_provider_for_model(preferred_model),
            "model": preferred_model,
        }]

    team = []
    for item in _DEFAULT_MODEL_TEAM:
        if item["provider"] == "openai" and not openai_api_key:
            continue
        team.append(dict(item))
    return team


def _try_summarize(
    prompt: str,
    system: str,
    *,
    provider: str,
    model: str,
    gemini_api_key: str,
    openai_api_key: str,
    log_cb,
    timeout: int = _TIMEOUT_SEC,
) -> str | None:
    """Tek provider/model ile özet ya da çeviri denemesi yap."""
    try:
        if provider == "openai":
            result = _llm._openai_generate(
                prompt,
                system=system,
                api_key=openai_api_key or None,
                model=model,
                timeout=timeout,
                log_cb=log_cb,
            )
        else:
            result = _llm._gemini_generate(
                prompt,
                system=system,
                api_key=gemini_api_key or None,
                model=model,
                timeout=timeout,
                log_cb=log_cb,
            )
        return result if result and result.strip() else None
    except Exception as e:
        if log_cb:
            log_cb(f"  [Summarizer] {model} hatası: {e}")
        return None


def _build_cast_reference(tmdb_cast: list) -> str:
    """TMDB cast listesinden çeviri prompt'u için referans metni oluştur."""
    if not tmdb_cast:
        return ""

    lines = []
    for entry in tmdb_cast[:20]:  # max 20 kişi
        actor = ""
        char = ""
        if isinstance(entry, dict):
            actor = (entry.get("actor_name") or entry.get("name") or "").strip()
            char = (entry.get("character_name") or entry.get("character") or "").strip()
        if actor and char:
            lines.append(f"  {actor} → {char}")
        elif actor:
            lines.append(f"  {actor}")

    if not lines:
        return ""

    return (
        "\n\nIMPORTANT — CAST REFERENCE LIST (use these exact spellings):\n"
        + "\n".join(lines)
        + "\n\nIf the summary mentions any character that matches this list, "
        "use the EXACT spelling from this list. "
        "For example, if transcript says 'Cek Sparov' but the list says "
        "'Jack Sparrow', use 'Jack Sparrow'.\n"
    )


def _build_translate_prompt(
    foreign_summary: str,
    detected_language: str,
    cast_ref: str,
    retry_reason: str = "",
) -> str:
    """Çeviri çağrısı için kullanıcı prompt'unu oluştur."""
    lang_label = get_language_label(detected_language)
    prompt = (
        f"Source language: {detected_language.upper()} ({lang_label})\n\n"
        f"Summary to translate:\n{foreign_summary}"
        f"{cast_ref}"
    )
    if retry_reason:
        prompt += (
            "\n\nSTRICT RETRY REQUIREMENT:\n"
            f"The previous output was rejected because: {retry_reason}.\n"
            "Rewrite the summary so the final output is Turkish and all foreign proper names "
            "stay in Latin alphabet only.\n"
        )
    return prompt


def _translate_and_verify(
    foreign_summary: str,
    gemini_api_key: str,
    openai_api_key: str,
    tmdb_cast: list | None,
    detected_language: str,
    log_cb,
) -> tuple[str | None, str]:
    """
    Yabancı dildeki özeti TR'ye çevir ve final sözleşmesini doğrula.

    Sıra: Gemini Pro -> OpenAI gpt-5.4-mini -> Gemini Flash
    """
    cast_ref = _build_cast_reference(tmdb_cast or [])
    attempts = [
        (_SYSTEM_PROMPT_TRANSLATE, ""),
        (
            _SYSTEM_PROMPT_TRANSLATE_RETRY,
            "final summary was not valid Turkish or contained non-Latin script leakage",
        ),
    ]
    model_team = _build_model_team(openai_api_key=openai_api_key)

    for idx, (system_prompt, retry_reason) in enumerate(attempts, start=1):
        prompt = _build_translate_prompt(
            foreign_summary,
            detected_language,
            cast_ref,
            retry_reason=retry_reason,
        )
        for spec in model_team:
            provider = spec["provider"]
            model = spec["model"]
            result = _try_summarize(
                prompt,
                system_prompt,
                provider=provider,
                model=model,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                log_cb=log_cb,
                timeout=_TRANSLATE_TIMEOUT_SEC,
            )

            normalized = _normalize_summary_text(result)
            if not normalized:
                if log_cb:
                    log_cb(
                        f"  [Summarizer] {model} çeviri boş döndü "
                        f"(deneme {idx}/2)"
                    )
                continue

            is_valid, reason = _validate_final_summary(normalized)
            if is_valid:
                if log_cb:
                    has_cast = "evet" if cast_ref else "hayır"
                    log_cb(
                        f"  [Summarizer] {model} çeviri kabul edildi "
                        f"({len(normalized)} kar, cast_ref={has_cast}, deneme={idx})"
                    )
                return normalized, model

            if log_cb:
                log_cb(
                    f"  [Summarizer] {model} çeviri reddedildi "
                    f"(neden={reason}, deneme={idx}/2)"
                )

    return None, ""


def summarize_transcript(
    transcript_text: str,
    api_key: str = "",
    model: str = "",
    log_cb=None,
    variant: str = "en",
    detected_language: str = "tr",
    tmdb_cast: list | None = None,
) -> dict | None:
    """Transcript metninden Gemini ile Türkçe özet üret.

    Dil Stratejisi (v3):
      • Türkçe → model ekibi ile doğrudan Türkçe özet (tek adım)
      • Yabancı → model ekibi ile orijinal dilde özet → model ekibi ile Türkçeye çevir

    Args:
        transcript_text:  ASR transcript metni
        api_key:          Gemini API key
        model:            Belirli model (boşsa Pro→gpt-5.4-mini→Flash fallback)
        log_cb:           Log callback
        variant:          "en" (varsayılan)
        detected_language: Tespit edilen dil kodu ("tr", "en", "ar", ...)
        tmdb_cast:        TMDB cast listesi (opsiyonel, yabancı dil çevirisinde kullanılır)

    Returns:
        {
            "text": "<Türkçe özet>",
            "language": "tr",
            "model_used": "gemini-2.5-pro"|"gpt-5.4-mini"|"gemini-2.5-flash",
            "flow": "single_step"|"two_step",
        }
        Hata durumunda None.
    """
    if not transcript_text or not transcript_text.strip():
        return None

    snippet = transcript_text.strip()[:_MAX_CHARS]
    is_turkish = (detected_language == "tr" or not detected_language)

    if variant in ("en", "both"):
        final_summary_tr = None
        model_used = ""
        translation_model_used = ""
        flow = "single_step" if is_turkish else "two_step"
        openai_api_key = get_openai_api_key()

        if is_turkish:
            # ══════════════════════════════════════════════════════
            # TÜRKÇE AKIŞ — tek adım, mevcut davranış
            # ══════════════════════════════════════════════════════
            if log_cb:
                log_cb("  [Summarizer] Türkçe transcript → tek adım özet")

            system = _SYSTEM_PROMPT_TR
            prompt = _FEW_SHOT + f"Transcript:\n{snippet}"

            model_team = _build_model_team(model, openai_api_key=openai_api_key)
            for spec in model_team:
                m = spec["model"]
                if log_cb:
                    log_cb(f"  [Summarizer] Türkçe özet oluşturuluyor ({m})...")
                final_summary_tr = _try_summarize(
                    prompt,
                    system,
                    provider=spec["provider"],
                    model=m,
                    gemini_api_key=api_key,
                    openai_api_key=openai_api_key,
                    log_cb=log_cb,
                )
                if final_summary_tr:
                    model_used = m
                    if log_cb:
                        log_cb(
                            f"  [Summarizer] Türkçe özet alındı "
                            f"({len(final_summary_tr)} kar, {m})"
                        )
                    break
                else:
                    if log_cb:
                        log_cb(f"  [Summarizer] {m} başarısız — sonraki modele geçiliyor")

            # Güvenlik: Türkçe demiştik ama özet Türkçe gelmezse Flash ile çevir
            if final_summary_tr and not is_valid_final_summary(final_summary_tr):
                if log_cb:
                    log_cb(
                        "  [Summarizer] Tek adım özeti final sözleşmesini geçmedi "
                        "— çeviri ekibiyle yeniden kuruluyor"
                    )
                final_summary_tr, translation_model_used = _translate_and_verify(
                    final_summary_tr,
                    api_key,
                    openai_api_key,
                    tmdb_cast,
                    detected_language,
                    log_cb,
                )

        else:
            # ══════════════════════════════════════════════════════
            # YABANCI DİL AKIŞI — iki adım
            # Adım 1: Orijinal dilde özet (Pro -> gpt-5.4-mini -> Flash)
            # Adım 2: Türkçeye çevir + isim doğrula (Pro -> gpt-5.4-mini -> Flash)
            # ══════════════════════════════════════════════════════
            lang_label = get_language_label(detected_language)
            if log_cb:
                log_cb(
                    f"  [Summarizer] Yabancı dil ({detected_language.upper()}/{lang_label}) "
                    f"→ iki adımlı akış"
                )

            # ── Adım 1: Orijinal dilde özet ──
            system = _SYSTEM_PROMPT_FOREIGN
            lang_info = f"\nTranscript language: {detected_language.upper()} ({lang_label})\n"
            prompt = lang_info + f"Transcript:\n{snippet}"

            model_team = _build_model_team(model, openai_api_key=openai_api_key)
            foreign_summary = None

            for spec in model_team:
                m = spec["model"]
                if log_cb:
                    log_cb(
                        f"  [Summarizer] Adım 1: {detected_language.upper()} özet "
                        f"oluşturuluyor ({m})..."
                    )
                foreign_summary = _try_summarize(
                    prompt,
                    system,
                    provider=spec["provider"],
                    model=m,
                    gemini_api_key=api_key,
                    openai_api_key=openai_api_key,
                    log_cb=log_cb,
                )
                if foreign_summary:
                    model_used = m
                    if log_cb:
                        log_cb(
                            f"  [Summarizer] Adım 1 tamamlandı: {detected_language.upper()} "
                            f"özet ({len(foreign_summary)} kar, {m})"
                        )
                    break
                else:
                    if log_cb:
                        log_cb(f"  [Summarizer] {m} başarısız — sonraki modele geçiliyor")

            if not foreign_summary:
                if log_cb:
                    log_cb("  [Summarizer] Adım 1 başarısız — hiçbir modelden özet alınamadı")
                return None

            # ── Adım 2: Türkçeye çevir + isim doğrula ──
            if log_cb:
                cast_count = len(tmdb_cast) if tmdb_cast else 0
                log_cb(
                    f"  [Summarizer] Adım 2: Türkçeye çeviri ekibi başlatılıyor "
                    f"(cast_ref={cast_count} kişi)..."
                )

            final_summary_tr, translation_model_used = _translate_and_verify(
                foreign_summary,
                api_key,
                openai_api_key,
                tmdb_cast,
                detected_language,
                log_cb,
            )

            if final_summary_tr:
                if log_cb:
                    log_cb(
                        f"  [Summarizer] Adım 2 tamamlandı: Türkçe özet "
                        f"({len(final_summary_tr)} kar, {translation_model_used})"
                    )
            elif log_cb:
                log_cb(
                    "  [Summarizer] Adım 2 başarısız: final özet Türkçe/Latin sözleşmesini geçmedi"
                )

        # Sonuç
        if not final_summary_tr:
            if log_cb:
                log_cb("  [Summarizer] Hiçbir modelden özet alınamadı")
            return None

        return {
            "text": _normalize_summary_text(final_summary_tr),
            "language": "tr",
            "model_used": model_used,
            "translation_model_used": translation_model_used,
            "flow": flow,
        }

    return None
