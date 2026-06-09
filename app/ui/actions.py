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
    RailActionSpec("chats_button", "chats", "Chats", "Show chats", "toggle_chats_panel", checkable=True),
    RailActionSpec("notes_button", "notes", "Notes", "Notes", "show_notes_dialog"),
    RailActionSpec("capture_button", "capture", "Capture", "Capture screen", "start_region_capture"),
    RailActionSpec("diff_button", "diff", "Diff", "Compare the last answer", "submit_expand", expand_mode="compare"),
    RailActionSpec("search_button", "search", "Search", "Search deeper on the last answer", "submit_expand", expand_mode="details"),
    RailActionSpec("debug_button", "debug", "Debug", "Debug or troubleshoot the last answer", "submit_expand", expand_mode="troubleshoot"),
)
# Image generation (/image) and clearing the chat (/clear) are now slash commands typed
# into the composer — their rail buttons were removed.
