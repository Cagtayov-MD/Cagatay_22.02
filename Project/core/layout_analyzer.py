"""
layout_analyzer.py — OCR bbox'lardan 2D layout analizi.

Jeneriklerde yaygın formatlar:
1. Tek sütun: rol satırı + isim satırı (dikey akış)
2. İki sütun: karakter adı (sol) ←→ oyuncu adı (sağ)
3. Rol başlığı + altında isim listesi

Bu modül bbox koordinatlarını kullanarak:
- Aynı frame'deki metinleri satırlara gruplar (y-overlap)
- Her satırda sol/sağ sütun belirler (x-merkez)
- Karakter↔Oyuncu eşleştirmesi yapar
"""
from dataclasses import dataclass, field


@dataclass
class LayoutRow:
    """Aynı satırdaki metin blokları."""
    y_center: float
    items: list = field(default_factory=list)  # [(text, bbox, confidence)]


@dataclass
class CastPair:
    """Karakter adı ↔ Oyuncu adı eşleştirmesi."""
    character_name: str = ""
    actor_name: str = ""
    confidence: float = 0.0
    timecode_sec: float = 0.0
    method: str = "layout"  # "layout" | "sequential" | "api"
    char_bbox: list = field(default_factory=list)   # BUG-K4 FIX
    actor_bbox: list = field(default_factory=list)   # BUG-K4 FIX


class LayoutAnalyzer:
    """
    Frame-bazlı 2D layout analizi.

    ISSUE-03 FIX: Cross-frame tutarlılık eklendi.
    Önceden her frame bağımsız analiz ediliyordu. Şimdi:
    - Çoklu frame'den gelen layout kararları oylanır (majority vote)
    - Sütun mid_x değeri birden fazla frame'den ortalamayla güçlendirilir
    - Aynı sütun yapısı N frame boyunca görülüyorsa bu bilgi korunur
    """

    def __init__(self, y_threshold_ratio=0.03, min_x_gap_ratio=0.15):
        """
        Args:
            y_threshold_ratio: İki bbox'ın aynı satırda sayılması için
                               max y-merkez farkı (frame yüksekliğine oran)
            min_x_gap_ratio: Sol ve sağ sütun ayrımı için minimum boşluk
                             (frame genişliğine oran)
        """
        self.y_threshold_ratio = y_threshold_ratio
        self.min_x_gap_ratio = min_x_gap_ratio

    def analyze_frame_results(self, ocr_results: list, frame_width: int = 0,
                              frame_height: int = 0) -> dict:
        """
        Tek frame'in OCR sonuçlarını analiz et.

        Args:
            ocr_results: [{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
            frame_width: Frame genişliği (0 ise bbox'lardan tahmin)
            frame_height: Frame yüksekliği

        Returns:
            {
                "layout_type": "two_column" | "single_column" | "mixed",
                "rows": [LayoutRow, ...],
                "pairs": [CastPair, ...],   # İki sütun tespit edilirse
                "header": str,              # "oynayanlar" gibi başlık
            }
        """
        if not ocr_results:
            return {"layout_type": "empty", "rows": [], "pairs": [], "header": ""}

        # Frame boyutlarını tahmin et
        if not frame_width or not frame_height:
            all_x = [r["bbox"][2] for r in ocr_results if r.get("bbox")]
            all_y = [r["bbox"][3] for r in ocr_results if r.get("bbox")]
            frame_width = max(all_x) if all_x else 800
            frame_height = max(all_y) if all_y else 600

        y_threshold = frame_height * self.y_threshold_ratio
        min_x_gap = frame_width * self.min_x_gap_ratio

        # ── ADIM 1: Y-merkezine göre satırlara grupla ──
        rows = self._group_into_rows(ocr_results, y_threshold)

        # ── ADIM 2: Header tespiti ──
        header = self._detect_header(rows)

        # ── ADIM 3: Layout tipi tespit ──
        two_col_rows = 0
        for row in rows:
            if len(row.items) >= 2:
                # İki öğe arasında yeterli x-boşluk var mı?
                sorted_items = sorted(row.items, key=lambda x: x[1][0])  # x1'e göre
                for i in range(len(sorted_items) - 1):
                    gap = sorted_items[i + 1][1][0] - sorted_items[i][1][2]
                    if gap > min_x_gap:
                        two_col_rows += 1
                        break

        content_rows = [r for r in rows if len(r.items) > 0 and
                        not self._is_header_text(r.items[0][0])]

        if content_rows and two_col_rows >= len(content_rows) * 0.4:
            layout_type = "two_column"
        else:
            layout_type = "single_column"

        # ── ADIM 4: İki sütunsa eşleştirme yap ──
        pairs = []
        if layout_type == "two_column":
            mid_x = self._estimate_mid_x(rows, frame_width)
            pairs = self._extract_pairs(rows, frame_width, mid_x, min_x_gap)

        return {
            "layout_type": layout_type,
            "rows": rows,
            "pairs": pairs,
            "header": header,
        }

    def _group_into_rows(self, results: list, y_threshold: float) -> list[LayoutRow]:
        """OCR sonuçlarını y-merkezine göre satırlara grupla."""
        items_with_y = []
        for r in results:
            bbox = r.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            y_center = (bbox[1] + bbox[3]) / 2.0
            items_with_y.append((r["text"], bbox, r.get("confidence", 0), y_center))

        # Y-merkezine göre sırala
        items_with_y.sort(key=lambda x: x[3])

        # Gruplama
        rows: list[LayoutRow] = []
        current_row = None

        for text, bbox, conf, y_center in items_with_y:
            if current_row is None or abs(y_center - current_row.y_center) > y_threshold:
                current_row = LayoutRow(y_center=y_center)
                rows.append(current_row)
            # Satırın y-merkezini güncelle (ağırlıklı ortalama)
            n = len(current_row.items)
            current_row.y_center = (current_row.y_center * n + y_center) / (n + 1)
            current_row.items.append((text, bbox, conf))

        # Her satır içinde x'e göre sırala (soldan sağa)
        for row in rows:
            row.items.sort(key=lambda x: x[1][0])

        return rows

    def _detect_header(self, rows: list[LayoutRow]) -> str:
        """İlk satırda "oynayanlar/oyuncular/cast" gibi başlık var mı?"""
        if not rows:
            return ""

        header_keywords = {
            "oynayanlar", "oyuncular", "cast", "starring", "basroller",
            "oyuncular:", "basroller:", "oynayshlar",  # OCR hatalı versiyonlar
        }

        first_row = rows[0]
        if len(first_row.items) == 1:
            text = first_row.items[0][0].lower().strip()
            for kw in header_keywords:
                if kw in text:
                    return first_row.items[0][0]  # Orijinal metni döndür
        return ""

    def _is_header_text(self, text: str) -> bool:
        low = text.lower().strip()
        headers = {"oynayanlar", "oyuncular", "cast", "starring",
                   "basroller", "oynayshlar", "yazanlar", "yonetmen"}
        return low in headers or any(h in low for h in headers)

    def _estimate_mid_x(self, rows: list[LayoutRow], frame_width: int) -> float:
        """İki sütun için dinamik orta eksen tahmini.

        Sabit mid_x = w/2 bazı jeneriklerde kayar. Burada 1D k-means (k=2)
        ile bbox x-merkezlerini iki kümeye ayırıp kümeler arası orta noktayı
        mid_x olarak seçeriz.

        Fallback: frame_width/2.
        """
        x_centers = []
        for row in rows:
            if not row.items:
                continue
            if len(row.items) == 1 and self._is_header_text(row.items[0][0]):
                continue
            for _, bbox, _ in row.items:
                if not bbox or len(bbox) < 4:
                    continue
                xc = (bbox[0] + bbox[2]) / 2.0
                if 0 <= xc <= frame_width:
                    x_centers.append(xc)

        if len(x_centers) < 6:
            return frame_width / 2.0

        x_centers.sort()
        c1 = x_centers[len(x_centers) // 4]
        c2 = x_centers[(len(x_centers) * 3) // 4]
        if abs(c2 - c1) < frame_width * 0.10:
            return frame_width / 2.0

        for _ in range(10):
            g1, g2 = [], []
            for x in x_centers:
                if abs(x - c1) <= abs(x - c2):
                    g1.append(x)
                else:
                    g2.append(x)
            if not g1 or not g2:
                break
            nc1 = sum(g1) / len(g1)
            nc2 = sum(g2) / len(g2)
            if abs(nc1 - c1) < 0.5 and abs(nc2 - c2) < 0.5:
                c1, c2 = nc1, nc2
                break
            c1, c2 = nc1, nc2

        left_c, right_c = (c1, c2) if c1 < c2 else (c2, c1)
        mid = (left_c + right_c) / 2.0
        mid = max(frame_width * 0.25, min(frame_width * 0.75, mid))
        return mid


    def _extract_pairs(self, rows: list[LayoutRow],
                       frame_width: int, mid_x: float, min_x_gap: float) -> list[CastPair]:
        """İki sütunlu satırlardan karakter↔oyuncu eşleştirmesi çıkar."""
        pairs = []
        for row in rows:
            if len(row.items) < 2:
                continue

            # Başlık satırını atla
            if self._is_header_text(row.items[0][0]):
                continue

            # Sol ve sağ sütuna ayır
            left_items = []
            right_items = []

            for text, bbox, conf in row.items:
                x_center = (bbox[0] + bbox[2]) / 2.0
                if x_center < mid_x:
                    left_items.append((text, bbox, conf))           # BUG-K4: bbox eklendi
                else:
                    right_items.append((text, bbox, conf))          # BUG-K4: bbox eklendi

            # Eşleştir
            if left_items and right_items:
                # Sol = karakter adı (genelde tek), sağ = oyuncu adı
                char_name = " ".join(t for t, _, _ in left_items)
                actor_name = " ".join(t for t, _, _ in right_items)
                avg_conf = sum(c for _, _, c in left_items + right_items) / (
                    len(left_items) + len(right_items))

                # Sütun ayrımına güven: gap ne kadar büyükse, çift o kadar güvenilir.
                try:
                    left_max_x2 = max(b[2] for _, b, _ in row.items if ((b[0] + b[2]) / 2.0) < mid_x)
                    right_min_x1 = min(b[0] for _, b, _ in row.items if ((b[0] + b[2]) / 2.0) >= mid_x)
                    gap = max(0.0, right_min_x1 - left_max_x2)
                except Exception:
                    gap = 0.0
                gap_factor = min(1.0, gap / max(min_x_gap, 1.0))
                conf = avg_conf * (0.60 + 0.40 * gap_factor)

                pairs.append(CastPair(
                    character_name=char_name.strip(),
                    actor_name=actor_name.strip(),
                    confidence=round(conf, 3),
                    method="layout",
                    char_bbox=list(left_items[0][1]) if len(left_items) == 1 else [],
                    actor_bbox=list(right_items[0][1]) if len(right_items) == 1 else [],
                ))

        return pairs

    def merge_sequential_with_layout(
            self, sequential_lines: list, layout_pairs: list[CastPair]
    ) -> list[CastPair]:
        """
        Layout eşleştirmesi ile sıralı OCR sonuçlarını birleştir.
        Layout'tan gelen çiftler öncelikli.
        Sequential'dan gelenler 'doğrulama' amaçlı kullanılır.
        """
        # Layout sonuçları zaten varsa onları kullan
        if layout_pairs:
            return layout_pairs

        # Layout yoksa (tek sütun), sıralı mantıkla devam et
        # Bu durumda pairs boş döner, parser eski mantığını kullanır
        return []

    # ─────────────────────────────────────────────────────────────
    # CROSS-FRAME ANALİZ — ISSUE-03 FIX
    # ─────────────────────────────────────────────────────────────

    def analyze_multi_frame(
        self,
        frames_ocr: list[tuple[list, int, int]],
        min_frame_agreement: float = 0.4
    ) -> dict:
        """
        Birden fazla frame'den gelen OCR sonuçlarını analiz ederek
        tutarlı layout kararı üret.

        Eleştiri: "Frame'ler arası tutarlılık kontrolü yok — aynı sütun
        yapısı 50 frame boyunca devam ediyorsa bu bilgi kullanılmıyor.
        Her frame bağımsız analiz ediliyor."

        Args:
            frames_ocr: [(ocr_results, frame_width, frame_height), ...]
                         Her eleman bir frame'in OCR çıktısı.
            min_frame_agreement: Kararı kabul etmek için minimum frame oranı.

        Returns:
            {
                'layout_type': 'two_column' | 'single_column',
                'consensus_mid_x': float,    # Ortalama sütun ayırıcı
                'all_pairs': [CastPair],      # Tüm frame'lerden toplanan çiftler
                'frame_votes': dict,          # {layout_type: count}
                'agreement_ratio': float,     # En çok oylanan tip oranı
            }
        """
        if not frames_ocr:
            return {
                'layout_type': 'single_column',
                'consensus_mid_x': 0.0,
                'all_pairs': [],
                'frame_votes': {},
                'agreement_ratio': 0.0,
            }

        votes: dict[str, int] = {}
        mid_x_values: list[float] = []
        all_pairs: list[CastPair] = []

        for ocr_results, fw, fh in frames_ocr:
            frame_result = self.analyze_frame_results(ocr_results, fw, fh)
            lt = frame_result['layout_type']
            votes[lt] = votes.get(lt, 0) + 1

            # mid_x değerlerini topla (iki sütunluysa)
            if lt == 'two_column' and frame_result['pairs']:
                # Çiftlerdeki ortalama x ayırıcı
                mid_xs = []
                for pair in frame_result['pairs']:
                    char_x2 = pair.char_bbox[2] if pair.char_bbox else 0
                    act_x1  = pair.actor_bbox[0] if pair.actor_bbox else 0
                    if char_x2 and act_x1:
                        mid_xs.append((char_x2 + act_x1) / 2.0)
                if mid_xs:
                    mid_x_values.append(float(sum(mid_xs) / len(mid_xs)))

            all_pairs.extend(frame_result['pairs'])

        total_frames = len(frames_ocr)
        # Majority vote
        winner = max(votes, key=lambda k: votes[k]) if votes else 'single_column'
        agreement_ratio = votes.get(winner, 0) / max(total_frames, 1)

        # Eğer anlaşma oranı yetersizse → single_column (güvenli taraf)
        if agreement_ratio < min_frame_agreement:
            winner = 'single_column'

        consensus_mid_x = (
            float(sum(mid_x_values) / len(mid_x_values))
            if mid_x_values else 0.0
        )

        # Tekrar eden çiftleri birleştir (aynı karakter+oyuncu birden çok frame'de)
        merged_pairs = self._merge_duplicate_pairs(all_pairs)

        return {
            'layout_type': winner,
            'consensus_mid_x': consensus_mid_x,
            'all_pairs': merged_pairs,
            'frame_votes': votes,
            'agreement_ratio': round(agreement_ratio, 3),
        }

    def _merge_duplicate_pairs(self, pairs: list) -> list:
        """Aynı karakter+oyuncu eşini birleştir, en yüksek confidence'lı tut."""
        seen: dict[tuple, CastPair] = {}
        for pair in pairs:
            key_char  = (pair.character_name or '').strip().lower()
            key_actor = (pair.actor_name or '').strip().lower()
            if not key_char and not key_actor:
                continue
            key = (key_char, key_actor)
            if key not in seen:
                seen[key] = pair
            else:
                # Daha yüksek confidence'lıyı tut
                if getattr(pair, 'confidence', 0) > getattr(seen[key], 'confidence', 0):
                    seen[key] = pair
        return list(seen.values())

