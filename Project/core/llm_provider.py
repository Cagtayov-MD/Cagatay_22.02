"""llm_provider.py — Provider-agnostic LLM client (Gemini or Ollama).

Reads configuration from environment variables:
    LLM_PROVIDER   : "gemini" (default) or "ollama"
    GEMINI_API_KEY : Google AI Studio API key (required for Gemini)
    GEMINI_MODEL   : model name (default "gemini-2.5-flash")
    GEMINI_BASE_URL: base URL override (default official endpoint)

When LLM_PROVIDER=ollama the caller must pass ollama_url / model kwargs.

Public API:
    generate(prompt, *, system=None, provider=None, **kwargs) -> str | None
        Send a text-only prompt and return the response string, or None on error.
"""

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from core._ollama_url import normalize_ollama_url

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_PROVIDER = "gemini"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
_TIMEOUT_SEC = 60
_GEMINI_RETRY_CODES = {429, 503}
_GEMINI_MAX_RETRIES = 3
_GEMINI_RETRY_DELAY_SEC = 12
_GEMINI_MODEL_COOLDOWNS: dict[str, float] = {}


def _clear_expired_gemini_cooldown(model: str, now_ts: float | None = None) -> None:
    """Süresi dolan cooldown kaydını temizle."""
    now_ts = now_ts if now_ts is not None else time.time()
    until = _GEMINI_MODEL_COOLDOWNS.get(model)
    if until is not None and until <= now_ts:
        _GEMINI_MODEL_COOLDOWNS.pop(model, None)


def _get_gemini_cooldown_remaining(model: str, now_ts: float | None = None) -> int:
    """Model cooldown'daysa kalan süreyi saniye olarak döndür."""
    now_ts = now_ts if now_ts is not None else time.time()
    _clear_expired_gemini_cooldown(model, now_ts=now_ts)
    until = _GEMINI_MODEL_COOLDOWNS.get(model)
    if until is None or until <= now_ts:
        return 0
    return max(0, int(math.ceil(until - now_ts)))


def _set_gemini_model_cooldown(model: str, seconds: int, *, now_ts: float | None = None) -> None:
    """Modeli geçici olarak cooldown'a al."""
    if not model or seconds <= 0:
        return
    now_ts = now_ts if now_ts is not None else time.time()
    _GEMINI_MODEL_COOLDOWNS[model] = now_ts + seconds


def _get_gemini_error_cooldown_sec(code: int, body: str) -> int:
    """HTTP hata kodu/gövdesine göre uygulanacak cooldown süresi."""
    body_l = (body or "").lower()
    if code == 503:
        if "high demand" in body_l or '"status": "unavailable"' in body_l or '"status":"unavailable"' in body_l:
            return 600
        return 180
    if code == 429:
        return 180
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: read env (supports python-dotenv if present, falls back to os.environ)
# ─────────────────────────────────────────────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini backend
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_generate(
    prompt: str,
    *,
    system: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int = _TIMEOUT_SEC,
    log_cb=None,
    timeout_flag: list | None = None,
) -> str | None:
    """Call Gemini generateContent endpoint and return text response.

    Args:
        timeout_flag: Optional mutable list; ``True`` is appended if a timeout
                      occurs so callers can detect the condition without
                      parsing log messages.
    """
    api_key = api_key or _env("GEMINI_API_KEY")
    if not api_key:
        if log_cb:
            log_cb("  [LLM/Gemini] GEMINI_API_KEY tanımlı değil — atlanıyor")
        return None

    model = model or _env("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
    base_url = (base_url or _env("GEMINI_BASE_URL", _DEFAULT_GEMINI_BASE_URL)).rstrip("/")

    cooldown_remaining = _get_gemini_cooldown_remaining(model)
    if cooldown_remaining:
        if log_cb:
            log_cb(
                f"  [LLM/Gemini] model={model} geçici cooldown'da "
                f"({cooldown_remaining}s kaldı) — istek atlandı"
            )
        return None

    # Build contents list
    contents = []
    if system:
        # Gemini uses "system_instruction" for system prompts; include as first user turn
        # when using the basic generateContent API without SDK
        system_part = {"role": "user", "parts": [{"text": system}]}
        model_ack = {"role": "model", "parts": [{"text": "Understood."}]}
        contents.extend([system_part, model_ack])
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = json.dumps({"contents": contents}).encode("utf-8")
    url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"

    if log_cb:
        log_cb(f"  [LLM/Gemini] İstek gönderiliyor: model={model}")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for _attempt in range(_GEMINI_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if not candidates:
                    if log_cb:
                        log_cb("  [LLM/Gemini] Yanıt boş (no candidates)")
                    return None
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text or None
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            if e.code in _GEMINI_RETRY_CODES and _attempt < _GEMINI_MAX_RETRIES - 1:
                if log_cb:
                    log_cb(
                        f"  [LLM/Gemini] HTTP {e.code} — "
                        f"{_attempt + 1}/{_GEMINI_MAX_RETRIES} deneme, {_GEMINI_RETRY_DELAY_SEC}s bekleniyor..."
                    )
                time.sleep(_GEMINI_RETRY_DELAY_SEC)
                continue
            cooldown_sec = _get_gemini_error_cooldown_sec(e.code, body)
            if cooldown_sec:
                _set_gemini_model_cooldown(model, cooldown_sec)
                if log_cb:
                    log_cb(
                        f"  [LLM/Gemini] model={model} "
                        f"{cooldown_sec}s cooldown'a alındı"
                    )
            if log_cb:
                log_cb(f"  [LLM/Gemini] HTTP {e.code}: {e.reason} — {body}")
            return None
        except urllib.error.URLError as e:
            if log_cb:
                log_cb(f"  [LLM/Gemini] Bağlantı hatası: {e.reason}")
            return None
        except TimeoutError:
            if timeout_flag is not None:
                timeout_flag.append(True)
            if log_cb:
                log_cb(f"  [LLM/Gemini] Zaman aşımı ({timeout}s)")
            return None
        except Exception as e:
            if log_cb:
                log_cb(f"  [LLM/Gemini] Beklenmedik hata: {e}")
            return None
    return None  # tüm denemeler tükendi


# ─────────────────────────────────────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_generate(
    prompt: str,
    *,
    system: str | None = None,
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: int = _TIMEOUT_SEC,
    log_cb=None,
) -> str | None:
    """Call Ollama /api/chat and return response text."""
    base_url = normalize_ollama_url(
        ollama_url or _env("OLLAMA_URL", _DEFAULT_OLLAMA_URL)
    )
    model = model or _env("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
    endpoint = f"{base_url}/api/chat"

    if log_cb:
        log_cb(f"  [LLM/Ollama] İstek gönderiliyor: {endpoint}, model={model}")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content or None
    except urllib.error.HTTPError as e:
        if log_cb:
            log_cb(f"  [LLM/Ollama] HTTP {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        if log_cb:
            log_cb(f"  [LLM/Ollama] Bağlantı hatası: {e.reason}")
        return None
    except TimeoutError:
        if log_cb:
            log_cb(f"  [LLM/Ollama] Zaman aşımı ({timeout}s)")
        return None
    except Exception as e:
        if log_cb:
            log_cb(f"  [LLM/Ollama] Beklenmedik hata: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI backend
# ─────────────────────────────────────────────────────────────────────────────

def _openai_generate(
    prompt: str,
    *,
    system: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int = _TIMEOUT_SEC,
    log_cb=None,
) -> str | None:
    """Call OpenAI chat/completions endpoint and return text response."""
    api_key = api_key or _env("OPENAI_API_KEY")
    if not api_key:
        if log_cb:
            log_cb("  [LLM/OpenAI] OPENAI_API_KEY tanımlı değil — atlanıyor")
        return None

    model = model or _env("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
    base_url = (base_url or _env("OPENAI_BASE_URL", _DEFAULT_OPENAI_BASE_URL)).rstrip("/")
    endpoint = f"{base_url}/v1/chat/completions"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }).encode("utf-8")

    if log_cb:
        log_cb(f"  [LLM/OpenAI] İstek gönderiliyor: model={model}")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                or ""
            ).strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text or None
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        if log_cb:
            log_cb(f"  [LLM/OpenAI] HTTP {e.code}: {e.reason} — {body}")
        return None
    except urllib.error.URLError as e:
        if log_cb:
            log_cb(f"  [LLM/OpenAI] Bağlantı hatası: {e.reason}")
        return None
    except TimeoutError:
        if log_cb:
            log_cb(f"  [LLM/OpenAI] Zaman aşımı ({timeout}s)")
        return None
    except Exception as e:
        if log_cb:
            log_cb(f"  [LLM/OpenAI] Beklenmedik hata: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ollama availability check (used by Ollama-path callers)
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_availability(
    ollama_url: str | None = None,
    model: str | None = None,
    log_cb=None,
) -> bool:
    """Return True if Ollama is reachable and the requested model is available."""
    base_url = normalize_ollama_url(
        ollama_url or _env("OLLAMA_URL", _DEFAULT_OLLAMA_URL)
    )
    model = model or _env("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
    try:
        req = urllib.request.Request(
            f"{base_url}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            model_base = model.split(":")[0]
            for m in models:
                if m.startswith(model_base):
                    return True
            if models and log_cb:
                log_cb(
                    f"  [LLM/Ollama] Uyarı: '{model}' bulunamadı. "
                    f"Mevcut: {', '.join(models[:3])}"
                )
            return False
    except Exception as e:
        if log_cb:
            log_cb(f"  [LLM/Ollama] Erişilemiyor: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_provider() -> str:
    """Return the active LLM provider name ('gemini' or 'ollama')."""
    return _env("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()


def generate(
    prompt: str,
    *,
    system: str | None = None,
    provider: str | None = None,
    log_cb=None,
    **kwargs,
) -> str | None:
    """Send a text-only prompt to the configured LLM provider.

    Args:
        prompt:   User prompt text.
        system:   Optional system message.
        provider: Override active provider ("gemini" or "ollama").
                  Defaults to LLM_PROVIDER env var (default "gemini").
        log_cb:   Optional logging callback(str).
        **kwargs: Provider-specific keyword args:
                  Gemini:  api_key, model, base_url, timeout
                  Ollama:  ollama_url, model, timeout

    Returns:
        Response text string, or None on error.
    """
    active = (provider or get_provider()).lower()

    if active == "gemini":
        return _gemini_generate(
            prompt,
            system=system,
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),
            base_url=kwargs.get("base_url"),
            timeout=kwargs.get("timeout", _TIMEOUT_SEC),
            log_cb=log_cb,
        )
    elif active == "openai":
        return _openai_generate(
            prompt,
            system=system,
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),
            base_url=kwargs.get("base_url"),
            timeout=kwargs.get("timeout", _TIMEOUT_SEC),
            log_cb=log_cb,
        )
    else:
        return _ollama_generate(
            prompt,
            system=system,
            ollama_url=kwargs.get("ollama_url"),
            model=kwargs.get("model"),
            timeout=kwargs.get("timeout", _TIMEOUT_SEC),
            log_cb=log_cb,
        )
