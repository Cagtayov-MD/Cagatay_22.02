"""dag_widget.py — Profil bazlı DAG diagram widget'ı (QPainter ile çizim)."""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QBrush

from ui.dag_definitions import get_dag

# Node durum renkleri
NODE_COLORS = {
    "pending": QColor("#888888"),
    "running": QColor("#FFD700"),
    "success": QColor("#22b33a"),
    "error":   QColor("#e94560"),
}

NODE_W = 140
NODE_H = 32
STEP_H = 24
H_GAP  = 30
V_GAP  = 18


class DAGWidget(QWidget):
    """Pipeline DAG diyagramı çizen widget."""

    def __init__(self, profile_name="FilmDizi", parent=None):
        super().__init__(parent)
        self.profile_name = profile_name
        self._dag = get_dag(profile_name)
        self._node_states: dict[str, str] = {}
        self.setMinimumHeight(200)

    def set_profile(self, profile_name: str):
        """Profil değiştiğinde DAG tanımını güncelle."""
        self.profile_name = profile_name
        self._dag = get_dag(profile_name)
        self._node_states.clear()
        self.update()

    def update_node_state(self, node_name: str, state: str):
        """Node durumunu güncelle ve yeniden çiz. state: pending/running/success/error"""
        self._node_states[node_name] = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#16213e"))

        dag = self._dag
        if not dag:
            return

        # Basit soldan-sağa layout: her node bir sütun
        nodes = list(dag.keys())
        col_w = NODE_W + H_GAP
        x_start = 20
        y_center = self.height() // 2

        positions = {}
        for i, node in enumerate(nodes):
            x = x_start + i * col_w
            positions[node] = (x, y_center)

        # Okları çiz
        pen = QPen(QColor("#0f3460"), 2)
        painter.setPen(pen)
        root = nodes[0] if nodes else None
        if root and "children" in dag[root]:
            rx, ry = positions[root]
            for child in dag[root]["children"]:
                if child in positions:
                    cx, cy = positions[child]
                    painter.drawLine(
                        rx + NODE_W, ry,
                        cx, cy,
                    )

        # Node'ları çiz
        font = QFont("Segoe UI", 9, QFont.Bold)
        step_font = QFont("Segoe UI", 8)
        painter.setFont(font)

        for node, (x, y) in positions.items():
            state = self._node_states.get(node, "pending")
            color = NODE_COLORS.get(state, NODE_COLORS["pending"])

            # Node arka planı
            painter.setBrush(QBrush(color.darker(180)))
            painter.setPen(QPen(color, 2))
            node_rect = QRect(x, y - NODE_H // 2, NODE_W, NODE_H)
            painter.drawRoundedRect(node_rect, 6, 6)

            # Node etiketi
            painter.setPen(QPen(QColor("#e0e0e0"), 1))
            painter.setFont(font)
            label = dag[node].get("label", node)
            painter.drawText(node_rect, Qt.AlignCenter, label)

            # Alt adımlar (steps)
            steps = dag[node].get("steps", [])
            painter.setFont(step_font)
            for j, step in enumerate(steps):
                sy = y + NODE_H // 2 + V_GAP + j * (STEP_H + 4)
                step_rect = QRect(x + 8, sy, NODE_W - 16, STEP_H)
                painter.setPen(QPen(QColor("#888888"), 1))
                painter.setBrush(QBrush(QColor("#1a1a2e")))
                painter.drawRoundedRect(step_rect, 4, 4)
                painter.setPen(QPen(QColor("#aaaaaa"), 1))
                painter.drawText(step_rect, Qt.AlignCenter, step)
