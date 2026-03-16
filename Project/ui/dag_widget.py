"""dag_widget.py — Pipeline DAG diyagramı (QPainter native dark theme)."""

from PySide6.QtCore import Qt, QRect, QPoint, QTimer
from PySide6.QtGui import (
    QPainter, QColor, QPen, QFont, QBrush,
)
from PySide6.QtWidgets import QWidget

from ui.dag_definitions import get_dag

# ── Renk paleti ────────────────────────────────────────────────────
BG_COLOR       = QColor("#0a0c10")
HEADER_BG      = QColor("#0d1117")
LOG_BG         = QColor("#0d1117")
BORDER_COLOR   = QColor("#1e2a3a")
ACCENT_COLOR   = QColor("#4a7aff")
STAGE_COLOR    = QColor("#c084fc")
TEXT_COLOR     = QColor("#dde0e8")
MUTED_COLOR    = QColor("#6b7280")
CONN_COLOR     = QColor("#1e3a5f")

NODE_COLORS = {
    "pending": QColor("#374151"),
    "running": QColor("#c084fc"),
    "success": QColor("#22c55e"),
    "error":   QColor("#ef4444"),
    "skipped": QColor("#ca8a04"),
}
NODE_TEXT_COLORS = {
    "pending": QColor("#dde0e8"),
    "running": QColor("#ffffff"),
    "success": QColor("#ffffff"),
    "error":   QColor("#ffffff"),
    "skipped": QColor("#ffffff"),
}

# ── Boyutlar ───────────────────────────────────────────────────────
HEADER_H   = 28
LOGSTRIP_H = 26
LEGEND_H   = 22
FOOTER_H   = 32
NODE_W     = 128
NODE_H     = 30
STEP_H     = 22
V_GAP      = 6
H_GAP      = 70


class DAGWidget(QWidget):
    """Pipeline DAG diyagramı — QPainter ile native dark theme."""

    def __init__(self, profile_name="FilmDizi-Paddle", parent=None):
        super().__init__(parent)
        self.profile_name = profile_name
        self._dag = get_dag(profile_name)
        self._node_states: dict[str, str] = {}
        self._step_states: dict[tuple[str, str], tuple[str, float | None]] = {}
        self._pipeline_status = "Bekleniyor"
        self._last_log = ""
        self._stats: dict[str, str] = {}
        self._blink = False
        self._running = False
        self.setMinimumHeight(260)

        # Blink timer (running animasyonu için)
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._on_blink)
        self._blink_timer.start(600)

    # ── Public API ─────────────────────────────────────────────────

    def set_profile(self, profile_name: str):
        self.profile_name = profile_name
        self._dag = get_dag(profile_name)
        self._node_states.clear()
        self._step_states.clear()
        self.update()

    def update_node_state(self, node_name: str, state: str):
        self._node_states[node_name] = state
        self._running = any(s == "running" for s in self._node_states.values())
        self.update()

    def update_step_state(self, branch_name: str, step_name: str,
                          state: str, elapsed_sec: float | None = None):
        self._step_states[(branch_name, step_name)] = (state, elapsed_sec)
        self.update()

    def set_pipeline_status(self, status: str):
        """Üst header şeridini güncelle ('Bekleniyor'/'İşleniyor'/'Tamamlandı'/'Hata')."""
        self._pipeline_status = status
        self._running = (status == "İşleniyor")
        self.update()

    def set_last_log(self, message: str):
        """Alt log şeridini güncelle."""
        self._last_log = message
        self.update()

    def set_stats(self, stats: dict):
        """Footer istatistiklerini güncelle. Örn: {'Süre': '12.3s', 'OCR': '47'}"""
        self._stats = stats
        self.update()

    # ── Private ────────────────────────────────────────────────────

    def _on_blink(self):
        self._blink = not self._blink
        if self._running:
            self.update()

    def _status_color(self) -> QColor:
        mapping = {
            "Bekleniyor": QColor("#6b7280"),
            "İşleniyor":  STAGE_COLOR,
            "Tamamlandı": QColor("#22c55e"),
            "Hata":       QColor("#ef4444"),
        }
        return mapping.get(self._pipeline_status, QColor("#6b7280"))

    # ── paintEvent ─────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # ── Arka plan ──────────────────────────────────────────────
        p.fillRect(0, 0, w, h, BG_COLOR)

        y = 0

        # ── 1. Header şeridi ───────────────────────────────────────
        y = self._draw_header(p, w, y)

        # ── 2. Log şeridi ──────────────────────────────────────────
        y = self._draw_log_strip(p, w, y)

        # ── 3. Legend ──────────────────────────────────────────────
        y = self._draw_legend(p, w, y)

        # ── 4. DAG diyagramı ───────────────────────────────────────
        dag_h = h - y - FOOTER_H
        if dag_h > 40:
            self._draw_dag(p, w, y, dag_h)

        # ── 5. Stats footer ────────────────────────────────────────
        self._draw_footer(p, w, h - FOOTER_H)

    def _draw_header(self, p: QPainter, w: int, y: int) -> int:
        rect = QRect(0, y, w, HEADER_H)
        p.fillRect(rect, HEADER_BG)
        p.setPen(QPen(BORDER_COLOR, 1))
        p.drawLine(0, y + HEADER_H - 1, w, y + HEADER_H - 1)

        # Durum noktası
        sc = self._status_color()
        dot_x, dot_y = 10, y + HEADER_H // 2
        if self._pipeline_status == "İşleniyor" and self._blink:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(sc.lighter(130)))
            p.drawEllipse(QPoint(dot_x, dot_y), 7, 7)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(sc))
        p.drawEllipse(QPoint(dot_x, dot_y), 5, 5)

        # Durum metni
        font = QFont("Consolas", 9, QFont.Bold)
        p.setFont(font)
        p.setPen(QPen(sc))
        p.drawText(QRect(22, y, 120, HEADER_H), Qt.AlignVCenter | Qt.AlignLeft,
                   self._pipeline_status)

        # Profil adı (sağa hizalı)
        p.setPen(QPen(MUTED_COLOR))
        font2 = QFont("Consolas", 8)
        p.setFont(font2)
        p.drawText(QRect(0, y, w - 8, HEADER_H), Qt.AlignVCenter | Qt.AlignRight,
                   f"profil: {self.profile_name}")

        return y + HEADER_H

    def _draw_log_strip(self, p: QPainter, w: int, y: int) -> int:
        rect = QRect(0, y, w, LOGSTRIP_H)
        p.fillRect(rect, LOG_BG)
        p.setPen(QPen(BORDER_COLOR, 1))
        p.drawLine(0, y + LOGSTRIP_H - 1, w, y + LOGSTRIP_H - 1)

        if self._last_log:
            font = QFont("Consolas", 8)
            p.setFont(font)
            p.setPen(QPen(MUTED_COLOR))
            # Metni sağdan kırp
            fm = p.fontMetrics()
            txt = fm.elidedText(self._last_log.strip(), Qt.ElideRight, w - 16)
            p.drawText(QRect(8, y, w - 16, LOGSTRIP_H), Qt.AlignVCenter | Qt.AlignLeft, txt)

        return y + LOGSTRIP_H

    def _draw_legend(self, p: QPainter, w: int, y: int) -> int:
        rect = QRect(0, y, w, LEGEND_H)
        p.fillRect(rect, BG_COLOR)

        items = [
            ("Bekliyor", "#374151"),
            ("İşleniyor", "#c084fc"),
            ("Tamamlandı", "#22c55e"),
            ("Hata", "#ef4444"),
            ("Atlandı", "#ca8a04"),
        ]
        font = QFont("Consolas", 7)
        p.setFont(font)
        x = 8
        dot_r = 4
        cy = y + LEGEND_H // 2
        for label, color in items:
            c = QColor(color)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(c))
            p.drawEllipse(QPoint(x + dot_r, cy), dot_r, dot_r)
            x += dot_r * 2 + 3
            p.setPen(QPen(MUTED_COLOR))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label)
            p.drawText(x, y, tw + 2, LEGEND_H, Qt.AlignVCenter | Qt.AlignLeft, label)
            x += tw + 12

        p.setPen(QPen(BORDER_COLOR, 1))
        p.drawLine(0, y + LEGEND_H - 1, w, y + LEGEND_H - 1)
        return y + LEGEND_H

    def _draw_dag(self, p: QPainter, w: int, y: int, dag_h: int):
        dag = self._dag
        if not dag:
            return

        nodes = list(dag.keys())
        root = nodes[0] if nodes else None
        if not root:
            return

        children = dag[root].get("children", [])

        node_font = QFont("Consolas", 8, QFont.Bold)
        step_font = QFont("Consolas", 7)

        def branch_height(nk: str) -> int:
            steps = dag.get(nk, {}).get("steps", [])
            return NODE_H + (len(steps) * (STEP_H + V_GAP) if steps else 0)

        # Kök node ortalanmış
        total_bh = (sum(branch_height(c) for c in children)
                    + max(0, len(children) - 1) * 16)
        root_x = 10
        root_y = y + dag_h // 2 - NODE_H // 2
        branch_x = root_x + NODE_W + H_GAP
        cy = y + dag_h // 2 - total_bh // 2

        # ── Kök node ──────────────────────────────────────────────
        r_state = self._node_states.get(root, "pending")
        self._draw_node(p, root_x, root_y, NODE_W, NODE_H,
                        dag[root].get("label", root), r_state, node_font,
                        is_root=True)

        for child in children:
            info = dag.get(child, {})
            steps = info.get("steps", [])
            bh = branch_height(child)
            node_y = cy

            # ── Bağlantı çizgisi (kök → dal) ──────────────────────
            mid_root_y = root_y + NODE_H // 2
            mid_node_y = node_y + NODE_H // 2
            p.setPen(QPen(CONN_COLOR, 1, Qt.DotLine))
            p.drawLine(root_x + NODE_W, mid_root_y, branch_x, mid_node_y)
            p.setPen(Qt.NoPen)

            # ── Dal node ───────────────────────────────────────────
            c_state = self._node_states.get(child, "pending")
            self._draw_node(p, branch_x, node_y, NODE_W, NODE_H,
                            info.get("label", child), c_state, node_font)

            # ── Step kutucukları ───────────────────────────────────
            sy = node_y + NODE_H + V_GAP
            for step in steps:
                sk = (child, step)
                s_state, s_elapsed = self._step_states.get(sk, ("pending", None))

                # Bağlantı çizgisi (dal → step)
                p.setPen(QPen(CONN_COLOR, 1))
                p.drawLine(branch_x + NODE_W // 2, sy - V_GAP,
                           branch_x + NODE_W // 2, sy)
                p.setPen(Qt.NoPen)

                self._draw_step(p, branch_x + 4, sy, NODE_W - 8, STEP_H,
                                step, s_state, s_elapsed, step_font)
                sy += STEP_H + V_GAP

            cy += bh + 16

    def _draw_node(self, p: QPainter, x: int, y: int, nw: int, nh: int,
                   label: str, state: str, font: QFont, is_root: bool = False):
        nc = NODE_COLORS.get(state, NODE_COLORS["pending"])
        tc = NODE_TEXT_COLORS.get(state, TEXT_COLOR)

        # Glow efekti (running)
        if state == "running" and self._blink:
            glow_c = QColor(nc)
            glow_c.setAlpha(60)
            p.setPen(QPen(glow_c, 6))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(x - 3, y - 3, nw + 6, nh + 6, 8, 8)

        # Node arkaplanı
        bg = QColor(nc)
        bg.setAlpha(40 if state == "pending" else 70)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(nc, 1 if state == "pending" else 2))
        p.drawRoundedRect(x, y, nw, nh, 6, 6)

        # Label
        p.setPen(QPen(tc))
        p.setFont(font)
        p.drawText(QRect(x, y, nw, nh), Qt.AlignCenter, label)

    def _draw_step(self, p: QPainter, x: int, y: int, sw: int, sh: int,
                   label: str, state: str, elapsed: float | None, font: QFont):
        sc = NODE_COLORS.get(state, NODE_COLORS["pending"])
        tc = NODE_TEXT_COLORS.get(state, TEXT_COLOR)

        bg = QColor(sc)
        bg.setAlpha(30 if state == "pending" else 55)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(sc, 1))
        p.drawRoundedRect(x, y, sw, sh, 3, 3)

        icon = {"pending": "·", "running": "▶", "success": "✓",
                "error": "✗", "skipped": "–"}.get(state, "·")
        txt = f"{icon} {label}"
        if elapsed is not None:
            txt += f" {elapsed:.1f}s"

        p.setPen(QPen(tc))
        p.setFont(font)
        fm = p.fontMetrics()
        clipped = fm.elidedText(txt, Qt.ElideRight, sw - 8)
        p.drawText(QRect(x + 4, y, sw - 8, sh), Qt.AlignVCenter | Qt.AlignLeft, clipped)

    def _draw_footer(self, p: QPainter, w: int, y: int):
        rect = QRect(0, y, w, FOOTER_H)
        p.fillRect(rect, HEADER_BG)
        p.setPen(QPen(BORDER_COLOR, 1))
        p.drawLine(0, y, w, y)

        if not self._stats:
            return

        items = list(self._stats.items())
        if not items:
            return

        font = QFont("Consolas", 8)
        p.setFont(font)
        cell_w = w // max(len(items), 1)
        for i, (key, val) in enumerate(items):
            cx = i * cell_w
            # Key
            p.setPen(QPen(MUTED_COLOR))
            p.drawText(QRect(cx + 4, y, cell_w - 8, FOOTER_H // 2),
                       Qt.AlignBottom | Qt.AlignLeft, key)
            # Value
            p.setPen(QPen(ACCENT_COLOR))
            font_b = QFont("Consolas", 8, QFont.Bold)
            p.setFont(font_b)
            p.drawText(QRect(cx + 4, y + FOOTER_H // 2, cell_w - 8, FOOTER_H // 2),
                       Qt.AlignTop | Qt.AlignLeft, val)
            p.setFont(font)
            # Separator
            if i > 0:
                p.setPen(QPen(BORDER_COLOR, 1))
                p.drawLine(cx, y + 4, cx, y + FOOTER_H - 4)

