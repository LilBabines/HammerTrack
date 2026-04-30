"""Training page with progress bar, epoch metrics table, and console log."""

import os
from typing import Optional, TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from ..windows import LauncherWindow


class TrainPage(QtWidgets.QWidget):
    """Training UI: progress + metrics table + console log.

    The actual training is launched from :class:`AnnotatePage` (which owns
    the dataset state). This page just exposes a "Launch" button and slots
    that the training worker can stream progress / metrics / logs into.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._launcher: Optional["LauncherWindow"] = None
        self._build_ui()

    # ==================== UI construction ====================

    def _build_ui(self):
        self.train_btn = QtWidgets.QPushButton("Launch Training (Finetune)")
        self.train_btn.setFixedHeight(40)
        self.train_btn.clicked.connect(self._on_train)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle")

        self.metrics_table = QtWidgets.QTableWidget()
        self.metrics_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setMaximumHeight(600)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily(
            "Consolas" if os.name == "nt" else "monospace"
        )

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)
        v.addWidget(self.train_btn)
        v.addWidget(self.progress_bar)
        v.addWidget(QtWidgets.QLabel("Epoch metrics:"))
        v.addWidget(self.metrics_table)
        v.addWidget(QtWidgets.QLabel("Console output:"))
        v.addWidget(self.log_text, stretch=1)

    def set_launcher(self, launcher: "LauncherWindow"):
        self._launcher = launcher

    # ==================== Slots ====================

    def log(self, msg: str):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_progress(self, msg: str, frac: float):
        pct = int(frac * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{msg}  —  {pct}%")

    def update_metrics(self, epoch: int, total_epochs: int, metrics: dict):
        if not metrics:
            return

        # Build the union of column names
        all_keys = ["Epoch"]
        for col in range(self.metrics_table.columnCount()):
            header = self.metrics_table.horizontalHeaderItem(col)
            if header and header.text() != "Epoch":
                all_keys.append(header.text())
        for k in sorted(metrics.keys()):
            if k not in all_keys:
                all_keys.append(k)

        self.metrics_table.setColumnCount(len(all_keys))
        self.metrics_table.setHorizontalHeaderLabels(all_keys)

        # Find or create the row for this epoch
        row = -1
        for r in range(self.metrics_table.rowCount()):
            item = self.metrics_table.item(r, 0)
            if item and item.text() == str(epoch):
                row = r
                break
        if row < 0:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)

        self.metrics_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(epoch)))
        for k, val in metrics.items():
            if k in all_keys:
                col = all_keys.index(k)
                self.metrics_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem(f"{val:.4f}")
                )

        self.metrics_table.resizeColumnsToContents()
        self.metrics_table.scrollToBottom()

    def reset_for_new_run(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.metrics_table.setRowCount(0)
        self.metrics_table.setColumnCount(0)
        self.log_text.clear()

    # ==================== Actions ====================

    def _on_train(self):
        if self._launcher:
            self._launcher.annotate_page.finetune_model()
