from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RailActionSpec:
    attr: str
    kind: str
    label: str
    tooltip: str
    handler: str
    checkable: bool = False
    expand_mode: str = ""


MAIN_RAIL_ACTIONS: tuple[RailActionSpec, ...] = (
    RailActionSpec("chats_button", "chats", "Чаты", "Показать чаты", "toggle_chats_panel", checkable=True),
    RailActionSpec("notes_button", "notes", "Заметки", "Заметки", "show_notes_dialog"),
    RailActionSpec("capture_button", "capture", "Скриншот", "Сделать скриншот (F6 из любого приложения)", "start_region_capture"),
    RailActionSpec("diff_button", "diff", "Сравнить", "Сравнить последний ответ", "submit_expand", expand_mode="compare"),
    RailActionSpec("search_button", "search", "Глубже", "Раскрыть последний ответ глубже", "submit_expand", expand_mode="details"),
    RailActionSpec("debug_button", "debug", "Debug", "Разобрать или отладить последний ответ", "submit_expand", expand_mode="troubleshoot"),
)
# Image generation (/image) and clearing the chat (/clear) are now slash commands typed
# into the composer — their rail buttons were removed.
