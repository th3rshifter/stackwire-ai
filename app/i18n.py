"""Central UI string catalog for StackWire.

ALL user-facing interface labels live here, so the whole app can be translated by
editing one file. Language comes from STACKWIRE_UI_LANGUAGE ("ru"/"en"), falling
back to STACKWIRE_ANSWER_LANGUAGE, then "ru".

Usage:
    from app.i18n import t, tr
    label.setText(t("new_chat"))            # catalog key (preferred for reused strings)
    label.setText(tr("Готово", "Done"))     # inline pair (one-offs)

The language is read at build time, so switching it in Settings → View takes effect
on the next launch (set_language() also updates it live for newly built widgets).
To add a string: add a key below with "ru"/"en", then use t("key") at the call site.
"""

from __future__ import annotations

import os

_LANG = "ru"


def _detect() -> str:
    value = os.getenv("STACKWIRE_UI_LANGUAGE", os.getenv("STACKWIRE_ANSWER_LANGUAGE", "ru")).strip().lower()
    return value if value in {"ru", "en"} else "ru"


def set_language(lang: str | None = None) -> None:
    """Set the active UI language. Pass None to re-read from the environment."""
    global _LANG
    if lang is None:
        _LANG = _detect()
        return
    lang = (lang or "ru").strip().lower()
    _LANG = lang if lang in {"ru", "en"} else "ru"


def current_language() -> str:
    return _LANG


def tr(ru: str, en: str) -> str:
    """Inline translation pair for one-off strings."""
    return en if _LANG == "en" else ru


def t(key: str) -> str:
    """Catalog lookup by key. Unknown keys return the key itself (visible, not crashing)."""
    entry = CATALOG.get(key)
    if not entry:
        return key
    return entry.get(_LANG) or entry.get("ru") or key


# ── Catalog ───────────────────────────────────────────────────────────────────
# Edit translations here. Keep keys stable (call sites reference them).
CATALOG: dict[str, dict[str, str]] = {
    # Rail / navigation
    "new_chat":   {"ru": "Новый чат", "en": "New chat"},
    "chats":      {"ru": "Чаты",      "en": "Chats"},
    "notes":      {"ru": "Заметки",   "en": "Notes"},
    "capture":    {"ru": "Скриншот",  "en": "Capture"},
    "compare":    {"ru": "Сравнить",  "en": "Compare"},
    "deeper":     {"ru": "Глубже",    "en": "Deeper"},
    "debug":      {"ru": "Debug",     "en": "Debug"},
    "settings":   {"ru": "Настройки", "en": "Settings"},
    # Header / status
    "ready":      {"ru": "Готово",    "en": "Ready"},
    "mini_mode":  {"ru": "Mini mode", "en": "Mini mode"},
    "hidden":     {"ru": "Скрыто",    "en": "Hidden"},
    "generating": {"ru": "Генерирую…", "en": "Generating…"},
    # Composer / welcome
    "composer_placeholder": {"ru": "Задайте вопрос или /команду…", "en": "Ask anything or /command…"},
    "message_placeholder":  {"ru": "Сообщение StackWire…", "en": "Message StackWire…"},
    "welcome_sub": {"ru": "Думай быстрее. Работай локально.", "en": "Think faster. Work locally."},
    # Settings tabs
    "tab_account":     {"ru": "Аккаунт",     "en": "Account"},
    "tab_modelhub":    {"ru": "Модели",      "en": "ModelHub"},
    "tab_speech":      {"ru": "Речь",        "en": "Speech"},
    "tab_view":        {"ru": "Вид",         "en": "View"},
    "tab_knowledge":   {"ru": "Знания",      "en": "Knowledge"},
    "tab_diagnostics": {"ru": "Диагностика", "en": "Diagnostics"},
    "settings_title":  {"ru": "Настройки",   "en": "Settings"},
    "save":            {"ru": "Сохранить",   "en": "Save"},
    "cancel":          {"ru": "Отмена",      "en": "Cancel"},
    # View tab
    "view_opacity":   {"ru": "Прозрачность окна", "en": "Window opacity"},
    "view_language":  {"ru": "Язык",              "en": "Language"},
    "view_hidden":    {"ru": "Скрытый режим",
                       "en": "Hidden mode"},
    "view_hint":      {"ru": "Прозрачность и скрытый режим применяются сразу (в mini окно прозрачнее). "
                             "Язык влияет на ответы модели и интерфейс (интерфейс — после перезапуска). "
                             "Всё сохраняется в stackwire.local.env.",
                       "en": "Opacity and hidden mode apply immediately (mini is more see-through). "
                             "Language affects model answers and the UI (UI on restart). "
                             "Everything is saved to stackwire.local.env."},
}


_RAIL_LABELS_EN = {
    "Чаты": "Chats", "Заметки": "Notes", "Скриншот": "Capture",
    "Сравнить": "Compare", "Глубже": "Deeper", "Debug": "Debug",
    "Настройки": "Settings", "Новый чат": "New chat", "Compact": "Compact",
}


def tr_rail(ru: str) -> str:
    """Translate a hard-coded Russian rail label to the active UI language."""
    return _RAIL_LABELS_EN.get(ru, ru) if _LANG == "en" else ru


set_language(None)
