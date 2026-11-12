"""
ui/event_timeline.py
-----------------------
Small horizontal timeline showing detected events as clickable markers
along the video's duration (spec item 19). Used live during analysis (new
markers appear as they're found) and in the PREVIEW page for navigation.

Plain tk.Canvas -- CustomTkinter has no canvas widget, and mixing raw tk
widgets into a CTk app for drawing needs is standard practice.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, List, Optional

from data.detection import Event


class EventTimeline(tk.Canvas):
    def __init__(self, master, colors: dict, on_select: Callable[[int], None], height: int = 46, **kwargs):
        super().__init__(master, height=height, bg=colors["panel_alt"], highlightthickness=0, **kwargs)
        self.colors = colors
        self.on_select = on_select
        self.duration_seconds = 0.0
        self.events: List[Event] = []
        self.selected_index: Optional[int] = None
        self._marker_hitboxes: List[tuple] = []

        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<Button-1>", self._on_click)

    def set_duration(self, seconds: float) -> None:
        self.duration_seconds = max(0.001, seconds)
        self._redraw()

    def set_events(self, events: List[Event], selected_index: Optional[int]) -> None:
        self.events = events
        self.selected_index = selected_index
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1:
            return

        pad = 10
        line_y = height // 2
        self.create_line(pad, line_y, width - pad, line_y, fill=self.colors["border"], width=2)

        self._marker_hitboxes = []
        if self.duration_seconds > 0:
            usable_w = width - 2 * pad
            for i, event in enumerate(self.events):
                frac = min(1.0, event.timestamp_seconds / self.duration_seconds)
                x = pad + frac * usable_w
                selected = i == self.selected_index
                radius = 6 if selected else 4
                color = self.colors["accent"] if selected else self.colors["success"]
                self.create_oval(x - radius, line_y - radius, x + radius, line_y + radius, fill=color, outline="")
                self._marker_hitboxes.append((x - 8, x + 8, i))

        self.create_text(
            pad, height - 8, text="00:00", fill=self.colors["text_secondary"], anchor="w", font=("Segoe UI", 9)
        )
        self.create_text(
            width - pad, height - 8, text=self._format_time(self.duration_seconds),
            fill=self.colors["text_secondary"], anchor="e", font=("Segoe UI", 9),
        )

    def _on_click(self, event) -> None:
        for x0, x1, index in self._marker_hitboxes:
            if x0 <= event.x <= x1:
                self.on_select(index)
                return

    @staticmethod
    def _format_time(seconds: float) -> str:
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
