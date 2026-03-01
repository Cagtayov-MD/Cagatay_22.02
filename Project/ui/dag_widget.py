"""dag_widget.py — Profil bazlı DAG diagram widget'ı (QPainter ile çizim)."""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QBrush

from ui.dag_definitions import get_dag

# Node durum renkleri
NODE_COLORS = {
    "pending": QColor("#333333"),
    "running": QColor("#FFD700"),
    "success": QColor("#22b33a"),
    "error":   QColor("#e94560"),
}

NODE_W  = 120
NODE_H  = 32
STEP_H  = 22
V_GAP   = 10   # step satırları arası boşluk
H_GAP   = 60   # sütunlar arası boşluk
ROOT_W  = 90   # kök node genişliği


class DAGWidget(QWidget):
    """Pipeline DAG diyagramı — solda kök, sağda dal node'ları."""

    def __init__(self, profile_name="FilmDizi", parent=None):
        super().__init__(parent)
        self.profile_name = profile_name
        self._dag = get_dag(profile_name)
        self._node_states: dict[str, str] = {}
        self._step_states: dict[tuple[str, str], tuple[str, float | None]] = {}
        self.setMinimumHeight(200)

    def set_profile(self, profile_name: str):
        """Profil değiştiğinde DAG tanımını güncelle."""
        self.profile_name = profile_name
        self._dag = get_dag(profile_name)
        self._node_states.clear()
        self._step_states.clear()
        self.update()

    def update_node_state(self, node_name: str, state: str):
        """Node durumunu güncelle (pending/running/success/error)."""
        self._node_states[node_name] = state
        self.update()

    def update_step_state(self, branch_name: str, step_name: str,
                          state: str, elapsed_sec: float | None = None):
        """Bir dal'ın alt adımının durumunu güncelle."""
        self._step_states[(branch_name, step_name)] = (state, elapsed_sec)
        self.update()

    # ─────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#16213e"))

        dag = self._dag
        if not dag:
            return

        nodes = list(dag.keys())
        root = nodes[0] if nodes else None
        if not root:
            return

        children = dag[root].get("children", [])

        node_font  = QFont("Segoe UI", 9, QFont.Bold)
        step_font  = QFont("Segoe UI", 8)
        small_font = QFont("Segoe UI", 7)

        # ── Kök node ──────────────────────────────────────────────
        root_x = 12
        root_y = self.height() // 2 - NODE_H // 2
        root_state = self._node_states.get(root, "pending")
        root_color = NODE_COLORS.get(root_state, NODE_COLORS["pending"])
        root_rect  = QRect(root_x, root_y, ROOT_W, NODE_H)

        painter.setBrush(QBrush(root_color.darker(200)))
        painter.setPen(QPen(root_color, 2))
        painter.drawRoundedRect(root_rect, 6, 6)
        painter.setPen(QPen(QColor("#e0e0e0"), 1))
        painter.setFont(node_font)
        painter.drawText(root_rect, Qt.AlignCenter,
                         dag[root].get("label", root))

        if not children:
            return

        # ── Dal node'ları ──────────────────────────────────────────
        # Her dal için toplam yüksekliği hesapla
        def branch_height(node_key: str) -> int:
            steps = dag.get(node_key, {}).get("steps", [])
            return NODE_H + (len(steps) * (STEP_H + V_GAP) if steps else 0)

        total_h = sum(branch_height(c) for c in children) + (len(children) - 1) * 20
        branch_x = root_x + ROOT_W + H_GAP

        cy = self.height() // 2 - total_h // 2

        for child in children:
            info   = dag.get(child, {})
            steps  = info.get("steps", [])
            bh     = branch_height(child)
            node_y = cy  # dal header'ı üstten başlar

            # Kök → dal okunu çiz
            arrow_pen = QPen(QColor("#4a6fa5"), 2)
            painter.setPen(arrow_pen)
            mid_root_y = root_y + NODE_H // 2
            mid_node_y = node_y + NODE_H // 2
            painter.drawLine(root_x + ROOT_W, mid_root_y,
                             branch_x, mid_node_y)

            # Dal node kutusu
            c_state = self._node_states.get(child, "pending")
            c_color = NODE_COLORS.get(c_state, NODE_COLORS["pending"])
            node_rect = QRect(branch_x, node_y, NODE_W, NODE_H)

            painter.setBrush(QBrush(c_color.darker(200)))
            painter.setPen(QPen(c_color, 2))
            painter.drawRoundedRect(node_rect, 6, 6)
            painter.setPen(QPen(QColor("#e0e0e0"), 1))
            painter.setFont(node_font)
            painter.drawText(node_rect, Qt.AlignCenter,
                             info.get("label", child))

            # Step satırları
            sy = node_y + NODE_H + V_GAP
            for step in steps:
                step_key = (child, step)
                s_state, s_elapsed = self._step_states.get(step_key, ("pending", None))
                s_color = NODE_COLORS.get(s_state, NODE_COLORS["pending"])

                step_rect = QRect(branch_x + 6, sy, NODE_W - 12, STEP_H)
                painter.setBrush(QBrush(s_color.darker(250)))
                painter.setPen(QPen(s_color, 1))
                painter.drawRoundedRect(step_rect, 3, 3)

                # İkon
                icon = {"pending": "○", "running": "⏳", "success": "✅", "error": "❌"}.get(s_state, "○")
                label = f"{icon} {step}"
                if s_elapsed is not None:
                    label += f" ({s_elapsed:.1f}s)"

                painter.setPen(QPen(QColor("#cccccc"), 1))
                painter.setFont(step_font)
                painter.drawText(step_rect, Qt.AlignVCenter | Qt.AlignLeft,
                                 "  " + label)

                sy += STEP_H + V_GAP

            cy += bh + 20

