"""
ui/add_marker_dialog.py
--------------------------
Small modal for manually adding a kill marker at a user-typed timestamp
(manual correction). Shared by the PREVIEW and ANALYSE pages.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

import ui.dialogs as dialogs
from data.detection import format_seconds, parse_timecode


class AddMarkerDialog(ctk.CTkToplevel):
    def __init__(self, master, colors: dict, on_confirm: Callable[[float], None], max_seconds: Optional[float] = None):
        super().__init__(master)
        self.colors = colors
        self.on_confirm = on_confirm
        self.max_seconds = max_seconds  # the video's length, when known

        self.title("Add Marker")
        self.configure(fg_color=colors["bg"])
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        ctk.CTkLabel(
            self, text="Timestamp (mm:ss or hh:mm:ss)", text_color=colors["text_primary"]
        ).pack(padx=16, pady=(20, 6), anchor="w")

        self.entry_var = tk.StringVar(value="00:00")
        entry = ctk.CTkEntry(self, textvariable=self.entry_var, width=200)
        entry.pack(padx=16)
        entry.focus_set()

        ctk.CTkButton(
            self, text="Add", fg_color=colors["accent"], hover_color=colors["accent_hover"], command=self._confirm
        ).pack(pady=20)

    def _confirm(self) -> None:
        seconds = parse_timecode(self.entry_var.get())
        if seconds is None:
            dialogs.show_error(
                self, self.colors, "Invalid Timestamp", "That timestamp couldn't be read.",
                what_to_do="Use mm:ss or hh:mm:ss, e.g. 1:23 or 01:23:45.",
            )
            return
        if self.max_seconds is not None and seconds > self.max_seconds:
            dialogs.show_error(
                self, self.colors, "Timestamp Beyond The Video",
                f"That is past the end of the video ({format_seconds(self.max_seconds)}).",
                what_to_do="Enter a time inside the video.",
            )
            return

        self.on_confirm(seconds)
        self.destroy()
