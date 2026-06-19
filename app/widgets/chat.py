from __future__ import annotations

import math
import os
import random
from collections.abc import Callable

import shiboken6
from PySide6.QtCore import QEasingCurve, QLineF, QPointF, QPropertyAnimation, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QTextBrowser, QVBoxLayout, QWidget


_px_func: Callable[..., int] | None = None
_icon_pixmap_func: Callable[..., object] | None = None
_flat_icon_button_func: Callable[..., QPushButton] | None = None
ACCENT = "#9ad6bd"


def configure_chat_widgets(*, px, icon_pixmap, flat_icon_button, accent: str) -> None:  # noqa: ANN001
    global _px_func, _icon_pixmap_func, _flat_icon_button_func, ACCENT
    _px_func = px
    _icon_pixmap_func = icon_pixmap
    _flat_icon_button_func = flat_icon_button
    ACCENT = accent


def _px(value: int | float, scale: float | None = None) -> int:
    if _px_func is None:
        return max(1, round(float(value)))
    return _px_func(value, scale)


def icon_pixmap(kind: str, size: int, color: str):  # noqa: ANN001
    if _icon_pixmap_func is None:
        raise RuntimeError("chat widgets are not configured")
    return _icon_pixmap_func(kind, size, color)


def _flat_icon_button(kind: str, tooltip: str, on_click) -> QPushButton:  # noqa: ANN001
    if _flat_icon_button_func is None:
        raise RuntimeError("chat widgets are not configured")
    return _flat_icon_button_func(kind, tooltip, on_click)


class NeuralBackground(QWidget):
    """Animated particle / neural-network backdrop shown on the welcome screen.

    A drifting set of nodes connected by fading lines, plus a sparse dust layer.
    Painted natively with QPainter, transparent to mouse events, and only ticks
    while visible so it costs nothing during a conversation.
    """

    NODE_COUNT = 40
    DUST_COUNT = 48
    LINK_DISTANCE = 150.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._nodes: list[list[float]] = []  # [x, y, vx, vy]
        self._dust: list[list[float]] = []  # [x, y, vx, vy, radius]
        self._w = 0.0
        self._h = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps for smooth motion
        self._timer.timeout.connect(self._tick)

    def _seed(self) -> None:
        w = max(1, self.width())
        h = max(1, self.height())
        self._w, self._h = float(w), float(h)
        self._nodes = [
            [random.uniform(0, w), random.uniform(0, h), random.uniform(-0.14, 0.14), random.uniform(-0.14, 0.14)]
            for _ in range(self.NODE_COUNT)
        ]
        self._dust = [
            [random.uniform(0, w), random.uniform(0, h), random.uniform(-0.06, 0.06), random.uniform(-0.10, 0.02), random.uniform(0.6, 1.9)]
            for _ in range(self.DUST_COUNT)
        ]

    def start(self) -> None:
        if not self._nodes:
            self._seed()
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        w = max(1.0, self._w)
        h = max(1.0, self._h)
        for node in self._nodes:
            node[0] += node[2]
            node[1] += node[3]
            if node[0] <= 0 or node[0] >= w:
                node[2] *= -1
                node[0] = min(max(node[0], 0.0), w)
            if node[1] <= 0 or node[1] >= h:
                node[3] *= -1
                node[1] = min(max(node[1], 0.0), h)
        for dust in self._dust:
            dust[0] = (dust[0] + dust[2]) % (w + 6)
            dust[1] = (dust[1] + dust[3]) % (h + 6)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        new_w = float(max(1, self.width()))
        new_h = float(max(1, self.height()))
        # Rescale existing points instead of reseeding so the field does not
        # shimmer/jump while the window is being dragged.
        if self._nodes and self._w > 0 and self._h > 0:
            sx = new_w / self._w
            sy = new_h / self._h
            for node in self._nodes:
                node[0] *= sx
                node[1] *= sy
            for dust in self._dust:
                dust[0] *= sx
                dust[1] *= sy
        self._w, self._h = new_w, new_h
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        if not self._nodes:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(154, 214, 189, 22))
        for dust in self._dust:
            painter.drawEllipse(QPointF(dust[0], dust[1]), dust[4], dust[4])

        link = self.LINK_DISTANCE
        for i, a in enumerate(self._nodes):
            ax, ay = a[0], a[1]
            for b in self._nodes[i + 1 :]:
                dx = ax - b[0]
                dy = ay - b[1]
                dist2 = dx * dx + dy * dy
                if dist2 >= link * link:
                    continue
                alpha = int(44 * (1.0 - (dist2 ** 0.5) / link))
                if alpha <= 0:
                    continue
                painter.setPen(QPen(QColor(154, 214, 189, alpha), 1.0))
                painter.drawLine(QLineF(ax, ay, b[0], b[1]))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(154, 214, 189, 130))
        for node in self._nodes:
            painter.drawEllipse(QPointF(node[0], node[1]), 2.2, 2.2)
        painter.end()


class ThinkingDots(QWidget):
    """Three softly pulsing mint dots shown while the assistant is thinking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = 0.0
        self.setFixedHeight(_px(20))
        self.setMinimumWidth(_px(60))
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._phase += 0.16
        self.update()

    def stop(self) -> None:
        self._timer.stop()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = _px(4)
        gap = _px(15)
        cx = radius + _px(2)
        cy = self.height() / 2
        for index in range(3):
            wave = math.sin(self._phase - index * 0.7) * 0.5 + 0.5
            alpha = int(90 + 150 * wave)
            scale = 0.65 + 0.5 * wave
            painter.setBrush(QColor(154, 214, 189, alpha))
            r = radius * scale
            painter.drawEllipse(QPointF(cx + index * gap, cy), r, r)
        painter.end()


class VoiceWave(QWidget):
    """Animated audio-level equalizer shown while speech is being recognized.

    Feed it with push_level(0..1) from the STT worker; bars travel and decay
    smoothly (voice-assistant style). Ticks only while running, so it costs
    nothing when idle/hidden.
    """

    BARS = 26

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFixedHeight(_px(26))
        self._energy = 0.0
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)  # ~25fps
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self) -> None:
        self.show()
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._energy = 0.0
        self.hide()

    def push_level(self, level: float) -> None:
        # Rise fast to the loudest recent block; _tick decays it between updates.
        self._energy = max(self._energy, max(0.0, min(1.0, float(level))))

    def _tick(self) -> None:
        self._energy *= 0.88
        self._phase += 0.38
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        width, height = self.width(), self.height()
        n = self.BARS
        gap = _px(3)
        bar_w = max(2, (width - gap * (n - 1)) // n)
        center_y = height / 2
        base = max(0.05, self._energy)
        alpha = int(110 + 130 * min(1.0, self._energy * 1.3))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(154, 214, 189, alpha))
        for i in range(n):
            envelope = math.sin(math.pi * (i + 0.5) / n)             # taller in the middle
            osc = 0.5 + 0.5 * math.sin(self._phase + i * 0.55)        # travelling wave
            amp = base * envelope * osc
            bar_h = max(_px(2), int(amp * (height - _px(4))))
            x = int(i * (bar_w + gap))
            painter.drawRoundedRect(x, int(center_y - bar_h / 2), int(bar_w), bar_h, _px(2), _px(2))
        painter.end()


class ChatMessageBrowser(QTextBrowser):
    """A per-message rich-text view that auto-sizes to its content height so the
    outer scroll area (not the browser) does the scrolling."""

    def __init__(self, on_anchor) -> None:  # noqa: ANN001
        super().__init__()
        self.setObjectName("msgBrowser")
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.anchorClicked.connect(on_anchor)

    def set_html(self, markup: str) -> None:
        self.setHtml(markup)
        self._fit()

    def _fit(self) -> None:
        doc = self.document()
        doc.setTextWidth(max(1, self.viewport().width()))
        self.setFixedHeight(int(doc.size().height()) + _px(4))

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._fit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()  # let the outer scroll area handle scrolling


class AssistantRow(QWidget):
    """Assistant message: 'РђСЃСЃРёСЃС‚РµРЅС‚' label + content (thinking dots or rich text) + copy."""

    def __init__(self, index: int, on_anchor, on_copy, model_name: str = "", on_regenerate=None, on_reasoning=None, has_reasoning: bool = False, reasoning_shown: bool = False) -> None:  # noqa: ANN001
        super().__init__()
        self.index = index
        self._on_anchor = on_anchor
        self.browser: ChatMessageBrowser | None = None
        self.thinking: ThinkingDots | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_px(2), _px(2), _px(2), _px(2))
        layout.setSpacing(_px(4))
        # Role line: small brand avatar + "Assistant".
        role_row = QHBoxLayout()
        role_row.setContentsMargins(0, 0, 0, 0)
        role_row.setSpacing(_px(7))
        avatar = QLabel()
        avatar.setPixmap(icon_pixmap("mark", _px(16), ACCENT))
        role = QLabel("ОТВЕТ")
        role.setObjectName("roleLabel")
        _role_font = role.font()
        _role_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, _px(1.4))
        role.setFont(_role_font)
        model_badge = QLabel(model_name)
        model_badge.setObjectName("assistantModelBadge")
        model_badge.setVisible(bool(model_name.strip()))
        role_row.addWidget(avatar)
        role_row.addWidget(role)
        role_row.addWidget(model_badge)
        role_row.addStretch(1)
        layout.addLayout(role_row)
        self._holder = QWidget()
        self._hl = QVBoxLayout(self._holder)
        self._hl.setContentsMargins(0, 0, 0, 0)
        self._hl.setSpacing(0)
        layout.addWidget(self._holder)
        self._copy = _flat_icon_button("copy", "Copy", lambda: on_copy(self.index))
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self._copy)
        if on_regenerate is not None:
            self._regen = _flat_icon_button("regen", "Переписать / другой вариант", lambda: on_regenerate(self.index))
            actions.addWidget(self._regen)
        if has_reasoning and on_reasoning is not None:
            _tip = "Скрыть размышления" if reasoning_shown else "Показать размышления"
            self._reasoning = _flat_icon_button("deepthink", _tip, lambda: on_reasoning(self.index))
            actions.addWidget(self._reasoning)
        actions.addStretch(1)
        self._actions_holder = QWidget()
        self._actions_holder.setLayout(actions)
        layout.addWidget(self._actions_holder)
        self._actions_holder.setVisible(False)

    def _clear_holder(self) -> None:
        while self._hl.count():
            item = self._hl.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # remove from view immediately (deleteLater is async)
                widget.deleteLater()
        self.thinking = None
        self.browser = None

    def show_thinking(self) -> None:
        self._clear_holder()
        self.thinking = ThinkingDots()
        self._hl.addWidget(self.thinking)
        self._actions_holder.setVisible(False)

    def show_html(self, markup: str, *, final: bool = False) -> None:
        if self.browser is None:
            self._clear_holder()
            self.browser = ChatMessageBrowser(self._on_anchor)
            self._hl.addWidget(self.browser)
        self.browser.set_html(markup)
        self._actions_holder.setVisible(final)


class ChatArea(QWidget):
    """Modern chat surface: an animated welcome backdrop when empty, and a scrollable
    column of message-bubble widgets when there is a conversation."""

    def __init__(self, background: "NeuralBackground") -> None:
        super().__init__()
        self.setObjectName("chatArea")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._background = background
        background.setParent(self)

        self.welcome = QWidget(self)
        self.welcome.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        wl = QVBoxLayout(self.welcome)
        wl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl.setSpacing(_px(8))
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setPixmap(icon_pixmap("mark", _px(56), ACCENT))
        title = QLabel("StackWire")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from app.i18n import t as _t
        sub = QLabel(_t("welcome_sub"))
        sub.setObjectName("welcomeSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl.addWidget(logo)
        wl.addWidget(title)
        wl.addWidget(sub)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("chatScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.container.setObjectName("chatContainer")
        # Full-width column: assistant rows hug the left, user bubbles hug the right.
        self.col = QVBoxLayout(self.container)
        self.col.setContentsMargins(_px(16), _px(12), _px(16), _px(12))
        self.col.setSpacing(_px(8))
        self.top_spacer = QWidget()
        self.top_spacer.setObjectName("chatTopSpacer")
        self.col.addWidget(self.top_spacer, 1)
        self.end_spacer = QWidget()
        self.end_spacer.setObjectName("chatEndSpacer")
        self.end_spacer.setFixedHeight(_px(24))
        self.col.addWidget(self.end_spacer)
        self.scroll_area.setWidget(self.container)
        self._scroll_animation: QPropertyAnimation | None = None

        background.lower()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        rect = self.rect()
        self._background.setGeometry(rect)
        self.welcome.setGeometry(rect)
        self.scroll_area.setGeometry(rect)
        self.end_spacer.setFixedHeight(_px(24))
        super().resizeEvent(event)

    def add_row(self, widget: QWidget) -> None:
        # Insert between the top filler and the small bottom gap.
        self.col.insertWidget(max(1, self.col.count() - 1), widget)

    def clear_rows(self) -> None:
        while self.col.count() > 2:  # keep top filler + bottom gap
            item = self.col.takeAt(1)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # remove from view immediately (deleteLater is async)
                widget.deleteLater()

    def show_welcome(self) -> None:
        self._background.show()
        self._background.start()
        self.welcome.show()
        self.welcome.raise_()
        self.scroll_area.hide()

    def show_list(self) -> None:
        self._background.stop()
        self._background.hide()
        self.welcome.hide()
        self.scroll_area.show()
        self.scroll_area.raise_()

    def _scroll_to_target(self, target: int, *, animated: bool = True, start_value: int | None = None) -> None:
        bar = self.scroll_area.verticalScrollBar()
        scroll_max = bar.maximum()
        target = max(0, min(target, scroll_max))
        if self._scroll_animation is not None:
            self._scroll_animation.stop()
            self._scroll_animation.deleteLater()
            self._scroll_animation = None
        if start_value is not None:
            bar.setValue(max(0, min(start_value, scroll_max)))
        distance = abs(bar.value() - target)
        if not animated or distance <= 2:
            bar.setValue(target)
            return
        animation = QPropertyAnimation(bar, b"value", self)
        animation.setDuration(max(140, min(260, int(distance * 0.28))))
        animation.setStartValue(bar.value())
        animation.setEndValue(target)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        def cleanup() -> None:
            if self._scroll_animation is animation:
                self._scroll_animation = None
            animation.deleteLater()

        animation.finished.connect(cleanup)
        self._scroll_animation = animation
        animation.start()

    def scroll_to_bottom(self, *, animated: bool = True, start_value: int | None = None) -> None:
        self._scroll_to_target(self.scroll_area.verticalScrollBar().maximum(), animated=animated, start_value=start_value)

    def scroll_to_message_start(self, widget: QWidget, *, animated: bool = True, start_value: int | None = None, _attempt: int = 0) -> None:
        # The widget may have been deleted (e.g. chat cleared / new session) between the
        # QTimer.singleShot scheduling and this firing — bail out if its C++ side is gone.
        if not shiboken6.isValid(widget) or not shiboken6.isValid(self):
            return
        layout = self.container.layout()
        if layout is not None:
            layout.activate()
        self.container.updateGeometry()
        viewport_height = max(1, self.scroll_area.viewport().height())
        target = widget.y() - int(viewport_height * 0.18)
        if self.scroll_area.verticalScrollBar().maximum() < target and _attempt < 6:
            QTimer.singleShot(
                16,
                lambda row=widget, start=start_value, attempt=_attempt + 1: self.scroll_to_message_start(
                    row,
                    animated=animated,
                    start_value=start,
                    _attempt=attempt,
                ),
            )
            return
        self._scroll_to_target(target, animated=animated, start_value=start_value)

    def restore_scroll_position(self, value: int) -> None:
        bar = self.scroll_area.verticalScrollBar()
        if self._scroll_animation is not None:
            return
        bar.setValue(max(0, min(value, bar.maximum())))

