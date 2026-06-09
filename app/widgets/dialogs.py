from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from app.notes import load_notes, save_notes


_px_func: Callable[..., int] | None = None


def configure_dialog_widgets(*, px) -> None:  # noqa: ANN001
    global _px_func
    _px_func = px


def _px(value: int | float, scale: float | None = None) -> int:
    if _px_func is None:
        return max(1, round(float(value)))
    return _px_func(value, scale)


class ClickableImageLabel(QLabel):
    """A QLabel that calls on_click() when the user left-clicks on it."""

    def __init__(self, on_click=None) -> None:  # noqa: ANN001
        super().__init__()
        self._on_click = on_click

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton and self._on_click is not None:
            self._on_click()
        super().mousePressEvent(event)


class FullImageDialog(QDialog):
    """Floating overlay that shows a full-size image with Save / Close controls."""

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self.setWindowTitle("Image")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        screen = QApplication.screenAt(self.mapToGlobal(QPoint(0, 0))) or QApplication.primaryScreen()
        if screen is not None:
            sg = screen.availableGeometry()
            max_w, max_h = int(sg.width() * 0.88), int(sg.height() * 0.88)
        else:
            max_w, max_h = 1600, 900

        scaled = pixmap
        if pixmap.width() > max_w or pixmap.height() > max_h:
            scaled = pixmap.scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        frame = QFrame()
        frame.setObjectName("imageViewerFrame")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(_px(10), _px(10), _px(10), _px(10))
        frame_layout.setSpacing(_px(8))

        img_label = QLabel()
        img_label.setPixmap(scaled)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setCursor(Qt.CursorShape.PointingHandCursor)
        img_label.mousePressEvent = lambda _e: self.reject()  # type: ignore[method-assign]

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(_px(8))
        btn_row.addStretch(1)

        save_btn = QPushButton("Сохранить")
        save_btn.setObjectName("ghostButton")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._save_image)

        close_btn = QPushButton("Закрыть")
        close_btn.setObjectName("ghostButton")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.reject)

        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)

        frame_layout.addWidget(img_label)
        frame_layout.addLayout(btn_row)
        outer.addWidget(frame)

        frame.setStyleSheet(
            "QFrame#imageViewerFrame {"
            "  background: #141c22;"
            "  border-radius: 16px;"
            "  border: 1px solid rgba(154,214,189,0.22);"
            "}"
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.reject()
            return
        super().keyPressEvent(event)

    def _save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "image.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if path:
            self._pixmap.save(path)


class NotesDialog(QDialog):
    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = path
        self._dirty = False
        self.setObjectName("notesDialog")
        self.setWindowTitle("Notes")
        self.setMinimumSize(_px(620), _px(460))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_px(18), _px(16), _px(18), _px(16))
        layout.setSpacing(_px(12))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(_px(10))

        title = QLabel("Notes")
        title.setObjectName("notesTitle")
        self.status = QLabel("")
        self.status.setObjectName("notesStatus")

        save_button = QPushButton("Save")
        save_button.setObjectName("dialogPrimaryButton")
        save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        save_button.clicked.connect(self.save_now)

        close_button = QPushButton("Close")
        close_button.setObjectName("ghostButton")
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.clicked.connect(self.reject)

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status)
        header.addWidget(save_button)
        header.addWidget(close_button)

        self.editor = QTextEdit()
        self.editor.setObjectName("notesEditor")
        self.editor.setAcceptRichText(False)
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.editor.setPlainText(self._load_text())
        self.editor.textChanged.connect(self._schedule_save)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self.save_now)

        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self.save_now)

        layout.addLayout(header)
        layout.addWidget(self.editor, 1)
        self.status.setText("Saved")

    def _load_text(self) -> str:
        try:
            return load_notes(self._path)
        except Exception:
            self.status.setText("Load failed")
            return ""

    def _schedule_save(self) -> None:
        self._dirty = True
        self.status.setText("Unsaved")
        self._save_timer.start()

    def save_now(self) -> bool:
        if not self._dirty:
            self.status.setText("Saved")
            return True
        try:
            save_notes(self.editor.toPlainText(), self._path)
        except Exception:
            self.status.setText("Save failed")
            return False
        self._dirty = False
        self.status.setText("Saved")
        return True

    def accept(self) -> None:
        if self.save_now():
            super().accept()

    def reject(self) -> None:
        if self.save_now():
            super().reject()



class ActionPopup(QFrame):
    def __init__(self, parent: QWidget, items: tuple[tuple[str, str], ...], callback) -> None:  # noqa: ANN001
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("actionPopup")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        for value, label in items:
            button = QPushButton(label)
            button.setObjectName("popupButton")
            button.clicked.connect(lambda _checked=False, selected=value: self._select(callback, selected))
            layout.addWidget(button)

    def _select(self, callback, value: str) -> None:  # noqa: ANN001
        self.hide()
        callback(value)

    def show_below(self, anchor: QWidget) -> None:
        self.adjustSize()
        point = anchor.mapToGlobal(anchor.rect().bottomLeft())
        self.move(point.x(), point.y() + _px(4))
        self.show()
        self.raise_()

