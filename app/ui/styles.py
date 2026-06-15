from __future__ import annotations


def build_window_styles(
    *,
    scale: float | None = None,
    px,
    accent: str,
    coral: str,
    elevated: str,
    font_display: str,
    font_stack: str,
    muted: str,
    rail: str,
    surface: str,
    text: str,
    ui_zoom: float,
) -> str:
    _px = px
    ACCENT = accent
    CORAL = coral
    ELEVATED = elevated
    FONT_DISPLAY = font_display
    FONT_STACK = font_stack
    MUTED = muted
    RAIL = rail
    SURFACE = surface
    TEXT = text
    UI_ZOOM = ui_zoom
    return f"""
QWidget#root {{
    background: transparent;
}}

QFrame#shell {{
    background: transparent;
    border: none;
    border-radius: {_px(22, scale)}px;
}}

QFrame#rail {{
    background: {RAIL};
    border: 1px solid rgba(154, 214, 189, 0.09);
    border-right: none;
    border-top-left-radius: {_px(22, scale)}px;
    border-bottom-left-radius: {_px(22, scale)}px;
}}

QPushButton#railBrandButton {{
    color: #eef8f4;
    background: transparent;
    border: 1px solid transparent;
    border-radius: {_px(12, scale)}px;
    padding: 0 {_px(8, scale)}px;
    font-family: {FONT_DISPLAY};
    font-size: {_px(15, scale)}px;
    font-weight: 780;
    text-align: left;
}}

QPushButton#railBrandButton:hover {{
    background: rgba(154, 214, 189, 0.07);
    border: 1px solid rgba(154, 214, 189, 0.12);
}}

QPushButton#railBrandButton[brandInteractive="false"]:hover {{
    background: transparent;
    border: 1px solid transparent;
}}

QPushButton#railBrandButton[railCollapsed="true"] {{
    padding: 0;
    border-radius: {_px(18, scale)}px;
    text-align: center;
}}

QWidget#railHeader {{
    background: transparent;
}}

QPushButton#railButton {{
    color: #9fb7bd;
    background: transparent;
    border: 1px solid transparent;
    border-radius: {_px(13, scale)}px;
    padding: 0 {_px(14, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(14, scale)}px;
    font-weight: 720;
    text-align: left;
}}

QPushButton#railButton:hover {{
    color: #d7f1e7;
    background: rgba(42, 58, 62, 0.62);
    border: 1px solid rgba(154, 214, 189, 0.18);
}}

QPushButton#railButton:checked {{
    color: #e8fff5;
    background: rgba(154, 214, 189, 0.13);
    border: 1px solid rgba(154, 214, 189, 0.25);
}}

QPushButton#railButton[railCollapsed="true"] {{
    padding: 0;
    color: #93bdc9;
    background: transparent;
    border: 1px solid transparent;
    border-radius: {_px(12, scale)}px;
    text-align: center;
}}

QPushButton#railButton[railCollapsed="true"]:hover {{
    color: #d7f1e7;
    background: rgba(42, 58, 62, 0.58);
    border: 1px solid rgba(154, 214, 189, 0.18);
}}

QPushButton#railButton[railCollapsed="true"]:checked {{
    color: #e8fff5;
    background: rgba(154, 214, 189, 0.16);
    border: 1px solid rgba(154, 214, 189, 0.24);
    border-left: {_px(3, scale)}px solid {ACCENT};
}}

QWidget#railNav {{
    background: transparent;
}}

QFrame#chatPanel {{
    background: {RAIL};
    border-left: none;
    border-right: 1px solid rgba(154, 214, 189, 0.07);
}}

QLabel#chatPanelTitle {{
    color: #dcebe6;
    font-family: {FONT_DISPLAY};
    font-size: {_px(13, scale)}px;
    font-weight: 700;
}}

QPushButton#newChatButton {{
    color: #9ad6bd;
    background: rgba(154, 214, 189, 0.06);
    border: 1px solid rgba(154, 214, 189, 0.28);
    border-radius: {_px(13, scale)}px;
    padding: 0 {_px(14, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(14, scale)}px;
    font-weight: 780;
    text-align: left;
}}

QPushButton#newChatButton:hover {{
    background: rgba(154, 214, 189, 0.13);
    border: 1px solid rgba(154, 214, 189, 0.42);
}}

QLabel#railSectionLabel {{
    color: #5f7079;
    font-family: {FONT_STACK};
    font-size: {_px(10, scale)}px;
    font-weight: 700;
    padding: {_px(4, scale)}px {_px(4, scale)}px {_px(1, scale)}px;
}}

QFrame#railDivider {{
    background: rgba(154, 214, 189, 0.16);
    border: none;
    margin: {_px(5, scale)}px {_px(2, scale)};
}}

QFrame#railGroupDivider {{
    background: rgba(154, 214, 189, 0.22);
    border: none;
}}

QScrollArea#chatSessionsScroll,
QWidget#chatSessionsContainer {{
    background: transparent;
    border: none;
}}

QFrame#chatSessionItem {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {_px(13, scale)}px;
}}

QFrame#chatSessionItem:hover {{
    background: rgba(37, 49, 58, 0.56);
    border: 1px solid rgba(154, 214, 189, 0.13);
}}

QFrame#chatSessionItem[active="true"] {{
    background: rgba(154, 214, 189, 0.13);
    border: 1px solid rgba(154, 214, 189, 0.10);
}}

QPushButton#chatSessionButton {{
    color: #d6e5e2;
    background: transparent;
    border: none;
    padding: 0 {_px(4, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(14, scale)}px;
    font-weight: 720;
    text-align: left;
}}

QPushButton#chatSessionButton:hover {{
    color: #e7f5f0;
    background: transparent;
    border: none;
}}

QPushButton#chatRenameButton,
QPushButton#chatDeleteButton {{
    background: transparent;
    border: none;
    border-radius: {_px(8, scale)}px;
    padding: 0;
}}

QPushButton#chatRenameButton:hover {{
    background: rgba(154, 214, 189, 0.12);
}}

QPushButton#chatDeleteButton:hover {{
    background: rgba(246, 102, 102, 0.12);
}}

QPushButton#railUserButton {{
    color: #d8e7e3;
    background: transparent;
    border: 1px solid transparent;
    border-radius: {_px(14, scale)}px;
    padding: 0 {_px(10, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(14, scale)}px;
    font-weight: 760;
    text-align: left;
}}

QPushButton#railUserButton:hover {{
    background: rgba(42, 58, 62, 0.56);
    border: 1px solid rgba(154, 214, 189, 0.14);
}}

QPushButton#railUserButton[railCollapsed="true"] {{
    padding: 0;
    border-radius: {_px(20, scale)}px;
    text-align: center;
}}

QFrame#content {{
    background: rgba(22, 27, 34, 0.97);
    border-top-right-radius: {_px(22, scale)}px;
    border-bottom-right-radius: {_px(22, scale)}px;
}}

/* Mini mode: content is the only visible panel, so round all four corners (more pronounced). */
QFrame#content[mini="true"] {{
    border-radius: {_px(28, scale)}px;
    border: 1px solid rgba(154, 214, 189, 0.14);
}}

QLabel#title {{
    color: #d8d5db;
    font-size: {_px(14, scale)}px;
    font-weight: 850;
}}

QLabel#titleMark {{
    min-width: {_px(20, scale)}px;
    min-height: {_px(20, scale)}px;
}}

QLabel#subtitle,
QLabel#status {{
    color: {MUTED};
    font-size: {_px(11, scale)}px;
}}

QLabel#modelChip,
QLabel#apiChip,
QLabel#accountChip,
QLabel#status {{
    min-height: {_px(28, scale)}px;
    padding: 0 {_px(10, scale)}px;
    border-radius: {_px(9, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(11, scale)}px;
}}

QLabel#accountChip {{
    color: #c7d1db;
    background: rgba(4, 8, 11, 0.52);
    border: 1px solid rgba(154, 214, 189, 0.13);
}}

QLabel#modelChip {{
    color: #b5d7c8;
    background: rgba(4, 8, 11, 0.52);
    border: 1px solid rgba(154, 214, 189, 0.13);
}}

QLabel#apiChip {{
    color: #9ad6bd;
    background: rgba(154, 214, 189, 0.08);
    border: 1px solid rgba(154, 214, 189, 0.16);
}}

QLabel#status {{
    color: #6f8793;
    background: rgba(4, 8, 11, 0.35);
    border: 1px solid rgba(154, 214, 189, 0.08);
}}

QLabel#status[listening="true"] {{
    color: #e8896b;
    background: rgba(232, 137, 107, 0.10);
    border: 1px solid rgba(232, 137, 107, 0.24);
}}

QLabel#brandDot {{
    background: transparent;
}}

QLabel#headerWordmark {{
    color: #eafaf3;
    font-family: {FONT_DISPLAY};
    font-size: {_px(15, scale)}px;
    font-weight: 700;
}}

QLabel#ghostPill {{
    color: #8ab4f0;
    background: rgba(138, 180, 240, 0.10);
    border: 1px solid rgba(138, 180, 240, 0.22);
    border-radius: {_px(9, scale)}px;
    min-height: {_px(28, scale)}px;
    padding: 0 {_px(10, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(11, scale)}px;
    font-weight: 600;
}}

QLabel#listenPill {{
    color: #e8896b;
    background: rgba(232, 137, 107, 0.10);
    border: 1px solid rgba(232, 137, 107, 0.24);
    border-radius: {_px(9, scale)}px;
    min-height: {_px(28, scale)}px;
    padding: 0 {_px(10, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(11, scale)}px;
    font-weight: 600;
}}

QLabel#debugPanel {{
    color: #a9cdbd;
    background: rgba(5, 7, 10, 170);
    border: 1px solid rgba(154, 214, 189, 0.11);
    border-radius: {_px(14, scale)}px;
    padding: {_px(6, scale)}px {_px(8, scale)}px;
    font-family: Consolas, Courier New, monospace;
    font-size: {_px(10, scale)}px;
}}

QWidget#chatArea {{
    background: transparent;
    border: none;
}}

QScrollArea#chatScroll {{
    background: transparent;
    border: none;
}}

QWidget#chatContainer {{
    background: transparent;
}}

QScrollBar:vertical {{
    background: transparent;
    width: {_px(10, scale)}px;
    margin: {_px(4, scale)}px {_px(2, scale)}px;
}}

QScrollBar::handle:vertical {{
    background: rgba(154, 214, 189, 0.26);
    border-radius: {_px(5, scale)}px;
    min-height: {_px(32, scale)}px;
}}

QScrollBar::handle:vertical:hover {{
    background: rgba(154, 214, 189, 0.46);
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QLabel#userBubble {{
    background: rgba(154, 214, 189, 0.13);
    border: 1px solid rgba(154, 214, 189, 0.18);
    border-radius: {_px(15, scale)}px;
    border-bottom-right-radius: {_px(5, scale)}px;
    padding: {_px(10, scale)}px {_px(14, scale)}px;
    color: #e7f6ef;
    font-family: {FONT_STACK};
    font-size: {_px(15, scale)}px;
}}

QLabel#shotLabel {{
    background: transparent;
    border-radius: {_px(12, scale)}px;
}}

QLabel#roleLabel {{
    color: #9ad6bd;
    font-family: {FONT_DISPLAY};
    font-size: {_px(11, scale)}px;
    font-weight: 700;
}}

QLabel#assistantModelBadge {{
    min-height: {_px(19, scale)}px;
    padding: 0 {_px(7, scale)}px;
    color: rgba(130, 149, 160, 0.72);
    background: rgba(5, 7, 10, 0.20);
    border: 1px solid rgba(154, 214, 189, 0.07);
    border-radius: {_px(7, scale)}px;
    font-size: {_px(10, scale)}px;
    font-weight: 650;
}}

QTextBrowser#msgBrowser {{
    background: transparent;
    border: none;
    selection-background-color: #263139;
}}

QPushButton#msgActionButton {{
    background: transparent;
    border: none;
    border-radius: {_px(7, scale)}px;
    padding: 0;
    min-height: 0;
}}

QPushButton#msgActionButton:hover {{
    background: rgba(154, 214, 189, 0.14);
}}

QLabel#welcomeTitle {{
    font-family: {FONT_DISPLAY};
    color: #e6f4ee;
    font-size: {_px(40, scale)}px;
    font-weight: 700;
}}

QLabel#welcomeSub {{
    color: #88a096;
    font-size: {_px(14, scale)}px;
}}

QFrame#composer {{
    background: rgba(16, 21, 26, 0.92);
    border: 1px solid rgba(154, 214, 189, 0.16);
    border-radius: {_px(16, scale)}px;
}}

QFrame#composer[focused="true"] {{
    border: 1px solid rgba(154, 214, 189, 0.38);
}}

QPushButton#composerIcon {{
    min-width: {_px(38, scale)}px;
    max-width: {_px(38, scale)}px;
    min-height: {_px(38, scale)}px;
    max-height: {_px(38, scale)}px;
    padding: 0;
    background: transparent;
    border: none;
    border-radius: {_px(19, scale)}px;
}}

QPushButton#composerIcon:hover {{
    background: rgba(154, 214, 189, 0.14);
}}

QPushButton#composerIcon:pressed {{
    background: rgba(154, 214, 189, 0.22);
}}

QPushButton#composerSend {{
    min-width: {_px(38, scale)}px;
    max-width: {_px(38, scale)}px;
    min-height: {_px(38, scale)}px;
    max-height: {_px(38, scale)}px;
    padding: 0;
    color: #0c1f18;
    background: {ACCENT};
    border: none;
    border-radius: {_px(12, scale)}px;
}}

QPushButton#composerSend:hover {{
    background: #abe2cb;
}}

QPushButton#composerSend:pressed {{
    background: #84c4ad;
}}

QPushButton#composerSend:disabled {{
    color: rgba(13, 20, 17, 0.45);
    background: rgba(154, 214, 189, 0.22);
}}

QFrame#attachBar {{
    background: rgba(154, 214, 189, 0.07);
    border: 1px solid rgba(154, 214, 189, 0.16);
    border-radius: {_px(10, scale)}px;
}}

QLabel#attachChip {{
    color: #b9c6d2;
    font-size: {_px(12, scale)}px;
}}

QPushButton#attachRemove {{
    color: #8fa0ad;
    background: transparent;
    border: none;
    border-radius: {_px(6, scale)}px;
    font-size: {_px(13, scale)}px;
    font-weight: 800;
}}

QPushButton#attachRemove:hover {{
    color: #f06b6b;
    background: rgba(246, 102, 102, 0.12);
}}

QWidget#quickBar {{
    background: transparent;
}}

QPushButton#quickChip {{
    color: #aacdbf;
    background: rgba(20, 28, 34, 0.55);
    border: 1px solid rgba(154, 214, 189, 0.12);
    border-radius: {_px(11, scale)}px;
    padding: {_px(7, scale)}px {_px(12, scale)}px;
    font-size: {_px(12, scale)}px;
    font-family: {FONT_STACK};
    font-weight: 600;
}}

QPushButton#quickChip:hover {{
    background: rgba(154, 214, 189, 0.15);
    color: {TEXT};
}}

QPushButton#quickChip:pressed {{
    background: rgba(154, 214, 189, 0.22);
}}

QLabel#livePanel {{
    color: #9fb3c2;
    background: rgba(154, 214, 189, 0.05);
    border: 1px solid rgba(154, 214, 189, 0.12);
    border-radius: {_px(10, scale)}px;
    padding: {_px(7, scale)}px {_px(11, scale)}px;
    font-size: {_px(12, scale)}px;
    font-family: {FONT_STACK};
}}

QFrame#slashPopup {{
    background: {ELEVATED};
    border: 1px solid rgba(154, 214, 189, 0.20);
    border-radius: {_px(10, scale)}px;
}}

QPushButton#slashRow {{
    text-align: left;
    color: #c7d2dd;
    background: transparent;
    border: none;
    border-radius: {_px(7, scale)}px;
    padding: {_px(7, scale)}px {_px(10, scale)}px;
    font-size: {_px(12, scale)}px;
}}

QPushButton#slashRow:hover {{
    background: rgba(154, 214, 189, 0.12);
}}

QPushButton#slashRow[active="true"] {{
    background: rgba(154, 214, 189, 0.18);
    color: {TEXT};
}}

QTextEdit#prompt {{
    background: transparent;
    border: none;
    color: {TEXT};
    padding: {_px(8, scale)}px {_px(6, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(15, scale)}px;
    selection-background-color: #263139;
}}

QPushButton {{
    min-height: {_px(30, scale)}px;
    border-radius: {_px(9, scale)}px;
    padding: 0 {_px(11, scale)}px;
    font-size: {_px(12, scale)}px;
    font-weight: 760;
}}

QPushButton#askButton {{
    min-width: {_px(44, scale)}px;
    max-width: {_px(44, scale)}px;
    min-height: {_px(44, scale)}px;
    max-height: {_px(44, scale)}px;
    padding: 0;
    color: #10131a;
    background: {ACCENT};
    border: 1px solid rgba(154, 214, 189, 0.30);
    border-radius: {_px(13, scale)}px;
}}

QPushButton#askButton:disabled {{
    color: rgba(201, 237, 244, 0.4);
    background: rgba(154, 214, 189, 0.18);
}}

QPushButton#ghostButton {{
    color: {TEXT};
    background: rgba(31, 38, 54, 126);
    border: 1px solid rgba(154, 214, 189, 0.11);
}}

QPushButton#ghostButton:hover {{
    border: 1px solid rgba(154, 214, 189, 0.22);
}}

QPushButton#ghostButton:disabled {{
    color: rgba(130, 149, 160, 0.45);
    background: rgba(20, 28, 34, 0.26);
    border: 1px solid rgba(154, 214, 189, 0.06);
}}

QWidget#modalOverlay {{
    background: rgba(3, 6, 9, 0.56);
}}

QDialog#settingsDialog {{
    background: #101722;
    color: {TEXT};
    border: 1px solid rgba(154, 214, 189, 0.14);
    border-radius: {_px(18, scale)}px;
}}

QDialog#notesDialog {{
    background: #101722;
    color: {TEXT};
    border: 1px solid rgba(154, 214, 189, 0.14);
    border-radius: {_px(18, scale)}px;
}}

QLabel#notesTitle {{
    color: #eef8f4;
    font-family: {FONT_STACK};
    font-size: {_px(18, scale)}px;
    font-weight: 820;
}}

QLabel#notesStatus {{
    color: {MUTED};
    font-family: {FONT_STACK};
    font-size: {_px(12, scale)}px;
}}

QTextEdit#notesEditor {{
    color: {TEXT};
    background: rgba(8, 12, 17, 0.58);
    border: 1px solid rgba(154, 214, 189, 0.12);
    border-radius: {_px(14, scale)}px;
    padding: {_px(12, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(14, scale)}px;
    selection-background-color: #263139;
}}

QTextEdit#notesEditor:focus {{
    border: 1px solid rgba(154, 214, 189, 0.28);
}}

QDialog#inlineFormDialog {{
    background: #101722;
    color: {TEXT};
    border: 1px solid rgba(154, 214, 189, 0.14);
    border-radius: {_px(16, scale)}px;
}}

QDialog#deleteChatDialog {{
    background: #101722;
    color: {TEXT};
    border: 1px solid rgba(154, 214, 189, 0.16);
    border-radius: {_px(16, scale)}px;
}}

QLabel#deleteChatTitle {{
    color: #eef8f4;
    font-size: {_px(20, scale)}px;
    font-weight: 780;
}}

QLabel#deleteChatText {{
    color: {TEXT};
    font-size: {_px(14, scale)}px;
    font-weight: 520;
}}

QLabel#deleteChatNote {{
    color: {MUTED};
    font-size: {_px(12, scale)}px;
}}

QPushButton#deleteCancelButton {{
    min-width: {_px(86, scale)}px;
    min-height: {_px(36, scale)}px;
    color: {TEXT};
    background: rgba(31, 38, 54, 0.58);
    border: 1px solid rgba(154, 214, 189, 0.14);
    border-radius: {_px(10, scale)}px;
    font-size: {_px(13, scale)}px;
    font-weight: 720;
}}

QPushButton#deleteCancelButton:hover {{
    background: rgba(50, 63, 78, 0.72);
    border: 1px solid rgba(154, 214, 189, 0.26);
}}

QPushButton#deleteDangerButton {{
    min-width: {_px(86, scale)}px;
    min-height: {_px(36, scale)}px;
    color: #170807;
    background: {CORAL};
    border: none;
    border-radius: {_px(10, scale)}px;
    font-size: {_px(13, scale)}px;
    font-weight: 820;
}}

QPushButton#deleteDangerButton:hover {{
    background: #f09a7f;
}}

QLabel#dialogTitle {{
    color: #ffffff;
    font-size: {_px(18, scale)}px;
    font-weight: 800;
}}

QLabel#dialogNote {{
    color: {MUTED};
    font-size: {_px(12, scale)}px;
}}

QPushButton#dialogPrimaryButton {{
    min-width: {_px(92, scale)}px;
    min-height: {_px(32, scale)}px;
    color: #07130d;
    background: {ACCENT};
    border: 0;
    border-radius: {_px(8, scale)}px;
    font-weight: 760;
}}

QPushButton#iconButton {{
    min-width: {_px(34, scale)}px;
    max-width: {_px(34, scale)}px;
    min-height: {_px(34, scale)}px;
    max-height: {_px(34, scale)}px;
    padding: 0;
    color: {TEXT};
    background: rgba(20, 28, 34, 0.55);
    border: 1px solid rgba(154, 214, 189, 0.10);
    border-radius: {_px(11, scale)}px;
}}

QPushButton#iconButton:hover {{
    background: rgba(37, 53, 60, 0.72);
    border: 1px solid rgba(154, 214, 189, 0.28);
}}

QPushButton#iconButton:checked {{
    background: rgba(154, 214, 189, 0.16);
    border: 1px solid rgba(154, 214, 189, 0.34);
}}

QFrame#actionPopup {{
    background: rgba(18, 24, 38, 245);
    border: 1px solid rgba(154, 214, 189, 0.18);
    border-radius: {_px(10, scale)}px;
}}

QPushButton#popupButton {{
    min-width: {_px(190, scale)}px;
    min-height: {_px(30, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: {TEXT};
    background: transparent;
    border: 1px solid transparent;
    border-radius: {_px(7, scale)}px;
    text-align: left;
}}

QPushButton#popupButton:hover {{
    background: rgba(154, 214, 189, 0.14);
    border: 1px solid rgba(154, 214, 189, 0.22);
}}

QPushButton#closeButton {{
    min-width: {_px(34, scale)}px;
    max-width: {_px(34, scale)}px;
    min-height: {_px(34, scale)}px;
    max-height: {_px(34, scale)}px;
    padding: 0;
    color: #6f8793;
    background: rgba(20, 22, 27, 0.68);
    border: 1px solid rgba(154, 214, 189, 0.08);
    border-radius: {_px(11, scale)}px;
}}

QPushButton#closeButton:hover {{
    background: rgba(246, 102, 102, 0.16);
    border: 1px solid rgba(246, 102, 102, 0.30);
}}

QComboBox#deviceCombo {{
    min-width: {_px(220, scale)}px;
    max-width: {_px(280, scale)}px;
    min-height: {_px(28, scale)}px;
    border-radius: {_px(9, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: #a9cdbd;
    background: rgba(5, 7, 10, 0.44);
    border: 1px solid rgba(154, 214, 189, 0.11);
}}

QComboBox#settingsCombo {{
    min-width: {_px(280, scale)}px;
    min-height: {_px(32, scale)}px;
    border-radius: {_px(8, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: {TEXT};
    background: rgba(31, 38, 54, 176);
    border: 1px solid rgba(154, 214, 189, 0.16);
}}

QComboBox QAbstractItemView {{
    color: {TEXT};
    background: #111820;
    selection-background-color: #263139;
}}

QSlider#settingsSlider {{
    min-height: {_px(22, scale)}px;
}}
QSlider#settingsSlider::groove:horizontal {{
    height: {_px(4, scale)}px;
    background: rgba(154, 214, 189, 0.14);
    border-radius: {_px(2, scale)}px;
}}
QSlider#settingsSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: {_px(2, scale)}px;
}}
QSlider#settingsSlider::handle:horizontal {{
    width: {_px(14, scale)}px;
    height: {_px(14, scale)}px;
    margin: {_px(-6, scale)}px 0;
    border-radius: {_px(7, scale)}px;
    background: {ACCENT};
}}

QLineEdit#settingsCombo {{
    min-width: {_px(280, scale)}px;
    min-height: {_px(32, scale)}px;
    border-radius: {_px(8, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: {TEXT};
    background: rgba(31, 38, 54, 176);
    border: 1px solid rgba(154, 214, 189, 0.16);
}}

QLineEdit#settingsCombo:focus {{
    border: 1px solid rgba(154, 214, 189, 0.42);
}}

QFrame#modelHubPanel {{
    background: rgba(5, 7, 10, 0.28);
    border: 1px solid rgba(154, 214, 189, 0.10);
    border-radius: {_px(10, scale)}px;
}}

QScrollArea#modelHubScroll {{
    background: transparent;
    border: none;
}}

QWidget#modelCardsContainer {{
    background: transparent;
}}

QFrame#modelCard {{
    background: rgba(31, 38, 54, 0.42);
    border: 1px solid rgba(154, 214, 189, 0.11);
    border-radius: {_px(10, scale)}px;
}}

QFrame#modelCard:hover {{
    background: rgba(40, 50, 65, 0.52);
    border: 1px solid rgba(154, 214, 189, 0.20);
}}

QLabel#modelCardTitle {{
    color: #e6f4ee;
    font-size: {_px(14, scale)}px;
    font-weight: 800;
}}

QLabel#modelState {{
    min-height: {_px(26, scale)}px;
    padding: 0 {_px(9, scale)}px;
    color: #9ad6bd;
    background: rgba(154, 214, 189, 0.08);
    border: 1px solid rgba(154, 214, 189, 0.18);
    border-radius: {_px(8, scale)}px;
    font-size: {_px(11, scale)}px;
    font-weight: 700;
}}

QProgressBar#modelHubProgress {{
    min-height: {_px(12, scale)}px;
    max-height: {_px(12, scale)}px;
    border: 1px solid rgba(154, 214, 189, 0.16);
    border-radius: {_px(6, scale)}px;
    background: rgba(5, 7, 10, 0.38);
    text-align: center;
    color: transparent;
}}

QProgressBar#modelHubProgress::chunk {{
    border-radius: {_px(5, scale)}px;
    background: {ACCENT};
}}

QLabel#dialogError {{
    color: #f08a8a;
    font-size: {_px(12, scale)}px;
}}

QCheckBox {{
    color: {TEXT};
    font-size: {_px(13, scale)}px;
    spacing: {_px(8, scale)}px;
}}

QCheckBox::indicator {{
    width: {_px(16, scale)}px;
    height: {_px(16, scale)}px;
    border-radius: {_px(4, scale)}px;
    border: 1px solid rgba(154, 214, 189, 0.34);
    background: rgba(5, 7, 10, 0.44);
}}

QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
}}

QTabWidget#settingsTabs::pane {{
    border: 1px solid rgba(154, 214, 189, 0.12);
    border-radius: {_px(10, scale)}px;
    top: -1px;
}}

QTabBar::tab {{
    color: {MUTED};
    background: transparent;
    padding: {_px(7, scale)}px {_px(14, scale)}px;
    margin-right: {_px(2, scale)}px;
    border-top-left-radius: {_px(8, scale)}px;
    border-top-right-radius: {_px(8, scale)}px;
}}

QTabBar::tab:selected {{
    color: #07130d;
    background: {ACCENT};
    font-weight: 700;
}}

QTabBar::tab:hover:!selected {{
    color: {TEXT};
    background: rgba(154, 214, 189, 0.12);
}}

QWidget#suggestionsBar {{
    background: transparent;
}}

QPushButton#suggestionChip {{
    background: rgba(154, 214, 189, 0.08);
    color: {ACCENT};
    border: 1px solid rgba(154, 214, 189, 0.22);
    border-radius: {_px(14, scale)}px;
    padding: {_px(4, scale)}px {_px(10, scale)}px;
    font-size: {round(9.5 * (scale or UI_ZOOM))}pt;
    text-align: left;
}}

QPushButton#suggestionChip:hover {{
    background: rgba(154, 214, 189, 0.17);
    border-color: rgba(154, 214, 189, 0.45);
}}

QPushButton#suggestionChip:pressed {{
    background: rgba(154, 214, 189, 0.28);
}}
"""

