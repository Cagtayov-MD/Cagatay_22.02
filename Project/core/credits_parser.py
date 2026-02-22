"""
credits_parser.py — OCR ciktisından yapisal credits verisi.

v2.2:
- Normalize rol eslesmesi + sirket fix + fragment reddi
- TurkishNameDB entegrasyonu:
    * split_concatenated → DP-tabanlı 356k kayıt destekli bölme (lazy fallback)
    * check_swap_risk    → layout pair oyuncu↔karakter swap tespiti
"""
import re
from dataclasses import dataclass, field
from utils.turkish import (
    normalize_tr,
    ascii_key,
    split_concatenated_name,
)

# ----------------------------
# Dataclass'lar
# ----------------------------

@dataclass
class CastItem:
    actor: str = ""
    character: str = ""
    role: str = "cast"
    role_category: str = "cast"
    raw: str = ""
    confidence: float = 0.0
    frame: str = ""
    y: float | None = None
    h: float | None = None


@dataclass
class CrewItem:
    name: str = ""
    job: str = ""
    role: str = "crew"
    role_category: str = "crew"
    raw: str = ""
    confidence: float = 0.0
    frame: str = ""
    y: float | None = None
    h: float | None = None


@dataclass
class ParsedCredits:
    title: str = ""
    year: str = ""
    companies: list[str] = field(default_factory=list)
    cast: list[CastItem] = field(default_factory=list)
    crew: list[CrewItem] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ----------------------------
# Parser
# ----------------------------

class CreditsParser:
    def __init__(self, turkish_name_db=None):
        self._name_db = turkish_name_db

        # Bazı kısa/yanlış OCR fragmentlerini reddetmek için
        self._reject_fragments = {
            "l", "i", "ı", "|", "•", "-", "—", "_", ":", ";", ".", ",", "…"
        }

        # Role normalization (örnek)
        self._role_map = {
            "directed by": ("Director", "crew"),
            "director": ("Director", "crew"),
            "cast": ("Cast", "cast"),
            "starring": ("Cast", "cast"),
            "produced by": ("Producer", "crew"),
            "producer": ("Producer", "crew"),
        }
        # prefix yakalamada daha doğru sonuç için uzun anahtarları öne al
        self._role_keys_desc = sorted(self._role_map.keys(), key=len, reverse=True)

    # ----------------------------
    # yardımcılar
    # ----------------------------

    def _norm_role(self, text: str):
        """
        Role satırını normalize eder.
        Hem exact match, hem de 'directed by ...' gibi prefix match destekler.
        """
        t = normalize_tr(text).strip().lower()
        t = re.sub(r"\s+", " ", t)

        for k in self._role_keys_desc:
            if t == k or t.startswith(k + " "):
                role, cat = self._role_map[k]
                return role, cat

        return text.strip(), "crew"

    def _is_fragment(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        if t.lower() in self._reject_fragments:
            return True
        if len(t) < 2:
            return True
        # sadece noktalama/işaret
        if re.fullmatch(r"[\W_]+", t):
            return True
        return False

    def _maybe_split_name(self, text: str) -> str:
        """
        Türkçe isim DB varsa önce DP split dene; olmazsa basit fallback.
        """
        text = (text or "").strip()
        if not text:
            return ""

        if self._name_db:
            parts = self._name_db.split_concatenated(text)
            if isinstance(parts, (list, tuple)) and len(parts) > 1:
                return " ".join(p for p in parts if p)
        return split_concatenated_name(text)

    # ----------------------------
    # parse
    # ----------------------------

    def parse(self, ocr_lines: list, layout_pairs=None) -> ParsedCredits:
        cr = ParsedCredits()
        current_role = ""
        current_role_cat = "crew"
        seen_names: set[str] = set()

        # ── Layout çiftlerini önce ekle (karakter↔oyuncu) ──
        # seen_name_index: ascii_key → cast index
        seen_name_index: dict[str, int] = {}

        if layout_pairs:
            # layout_pairs bazen list/tuple/set, bazen dict (id->pair) gelebiliyor.
            # dict iterasyonu key döndürür (çoğu zaman int). values() ile normalize et.
            pairs_iter = layout_pairs.values() if isinstance(layout_pairs, dict) else layout_pairs

            for pair in pairs_iter:
                if pair is None:
                    continue
                if isinstance(pair, (int, float)):
                    # key/score/index yanlışlıkla gelmiş olabilir
                    continue

                actor = ""
                char_name = ""

                if isinstance(pair, dict):
                    actor = str(pair.get("actor_name") or pair.get("actor") or "").strip()
                    char_name = str(pair.get("character_name") or pair.get("character") or "").strip()

                elif isinstance(pair, (list, tuple)):
                    # [character_name, actor_name] veya (character_name, actor_name)
                    char_name = str(pair[0]).strip() if len(pair) > 0 else ""
                    actor = str(pair[1]).strip() if len(pair) > 1 else ""

                else:
                    # attribute'lu objeler
                    actor = str(getattr(pair, "actor_name", "") or "").strip()
                    char_name = str(getattr(pair, "character_name", "") or "").strip()

                if not actor or len(actor) < 3:
                    continue

                # ── Swap tespiti: oyuncu↔karakter sütunları yer değiştirmiş mi? ──
                if self._name_db and char_name:
                    a0 = actor.split()[0] if actor.split() else ""
                    c0 = char_name.split()[0] if char_name.split() else ""
                    if a0 and c0 and self._name_db.check_swap_risk(a0, c0):
                        actor, char_name = char_name, actor

                actor = self._maybe_split_name(actor)
                actor_key = ascii_key(actor)

                if actor_key in seen_name_index:
                    idx = seen_name_index[actor_key]
                    if char_name and len(char_name) > len(cr.cast[idx].character):
                        cr.cast[idx].character = char_name
                    continue

                cr.cast.append(CastItem(actor=actor, character=char_name, raw=f"{actor} — {char_name}"))
                seen_name_index[actor_key] = len(cr.cast) - 1

        # ── OCR satırlarını işle ──
        for line in ocr_lines:
            text = ""
            conf = 0.0
            frame = ""
            y = None
            h = None

            if isinstance(line, str):
                text = line
            elif isinstance(line, dict):
                text = line.get("text", "") or ""
                conf = float(line.get("conf", 0.0) or 0.0)
                frame = line.get("frame", "") or ""
                y = line.get("y", None)
                h = line.get("h", None)
            else:
                text = getattr(line, "text", "") or ""
                conf = float(getattr(line, "conf", 0.0) or 0.0)
                frame = getattr(line, "frame", "") or ""
                y = getattr(line, "y", None)
                h = getattr(line, "h", None)

            text = (text or "").strip()
            if self._is_fragment(text):
                continue

            norm_l = re.sub(r"\s+", " ", normalize_tr(text)).strip().lower()

            # Role line mı? (exact veya prefix)
            is_role = any(norm_l == k or norm_l.startswith(k + " ") for k in self._role_keys_desc)
            if is_role:
                current_role, current_role_cat = self._norm_role(text)
                continue

            # Cast / Crew ayrımı
            if current_role_cat == "cast":
                # "Actor  Character" gibi satırları yakalamaya çalış
                parts = re.split(r"\s{2,}|\s-\s|—|–", text)
                parts = [p.strip() for p in parts if p and p.strip()]

                actor = parts[0] if parts else ""
                char_name = parts[1] if len(parts) >= 2 else ""

                actor = self._maybe_split_name(actor)
                actor_key = ascii_key(actor)
                if not actor_key or actor_key in seen_names:
                    continue
                seen_names.add(actor_key)

                cr.cast.append(
                    CastItem(
                        actor=actor,
                        character=char_name,
                        role=current_role or "cast",
                        role_category="cast",
                        raw=text,
                        confidence=conf,
                        frame=frame,
                        y=y,
                        h=h,
                    )
                )

            else:
                # crew satırları: "Name  Job"
                parts = re.split(r"\s{2,}|—|–", text)
                parts = [p.strip() for p in parts if p and p.strip()]

                name = parts[0] if parts else ""
                job = parts[1] if len(parts) >= 2 else (current_role or "")

                name = self._maybe_split_name(name)
                name_key = ascii_key(name)
                if not name_key or name_key in seen_names:
                    continue
                seen_names.add(name_key)

                cr.crew.append(
                    CrewItem(
                        name=name,
                        job=job,
                        role=current_role or "crew",
                        role_category="crew",
                        raw=text,
                        confidence=conf,
                        frame=frame,
                        y=y,
                        h=h,
                    )
                )

        return cr