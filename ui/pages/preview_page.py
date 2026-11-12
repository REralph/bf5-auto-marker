"""
ui/pages/preview_page.py
---------------------------
PREVIEW tab: inspect detected events with bounding-box overlays, zoom, and
manual correction, plus a clickable event timeline (spec item 19-20).
"""

from __future__ import annotations

import functools

import customtkinter as ctk

import ui.devtools as devtools
import ui.dialogs as dialogs
from ui.add_marker_dialog import AddMarkerDialog
from ui.event_timeline import EventTimeline
from ui.preview import PreviewPanel


class PreviewPage(ctk.CTkFrame):
    def __init__(self, master, app, colors: dict):
        super().__init__(master, fg_color=colors["bg"])
        self.app = app
        self.colors = colors
        self.rows = []
        self._build()

    def _build(self) -> None:
        colors = self.colors
        ctk.CTkLabel(
            self, text="PREVIEW", font=("Segoe UI", 22, "bold"), text_color=colors["text_primary"]
        ).pack(anchor="w", padx=28, pady=(24, 12))
        devtools.page_label(self, self.app, colors, "PREVIEW", pack_opts={"anchor": "w", "padx": 28, "pady": (0, 8)})

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=28)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.preview_panel = PreviewPanel(body, colors, on_navigate_event=self._navigate)
        self.preview_panel.grid(row=0, column=0, sticky="n", padx=(0, 16))

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")

        table_panel = ctk.CTkFrame(right, fg_color=colors["panel"], corner_radius=10)
        table_panel.pack(fill="both", expand=True)
        devtools.section_label(table_panel, self.app, colors, "EVENTS", "PREVIEW", pack_opts={"anchor": "w", "padx": 16, "pady": (10, 0)})
        ctk.CTkLabel(
            table_panel, text="EVENTS", font=("Segoe UI", 13, "bold"), text_color=colors["text_secondary"]
        ).pack(anchor="w", padx=16, pady=(14, 6))
        self.table_scroll = ctk.CTkScrollableFrame(table_panel, fg_color=colors["panel_alt"], corner_radius=8)
        self.table_scroll.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.empty_label = ctk.CTkLabel(
            self.table_scroll, text="No events yet. Run ANALYSE first.", text_color=colors["text_secondary"]
        )
        self.empty_label.pack(pady=20)

        correction_row = ctk.CTkFrame(table_panel, fg_color="transparent")
        correction_row.pack(fill="x", padx=16, pady=(0, 16))
        devtools.dev_button(
            correction_row, self.app, colors, "ADD_MARKER", "PREVIEW", "EVENTS",
            text="+ Add Marker", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            command=self._on_add_marker, pack_opts={"side": "left"},
        )
        self.delete_btn = devtools.dev_button(
            correction_row, self.app, colors, "DELETE_EVENT", "PREVIEW", "EVENTS",
            text="🗑 Delete", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            state="disabled", command=self._on_delete, pack_opts={"side": "left", "padx": 8},
        )

        ctk.CTkLabel(
            self, text="TIMELINE", font=("Segoe UI", 12, "bold"), text_color=colors["text_secondary"]
        ).pack(anchor="w", padx=28, pady=(16, 4))
        self.timeline = EventTimeline(self, colors, on_select=self.app.select_event)
        self.timeline.pack(fill="x", padx=28, pady=(0, 20))

    def on_show(self) -> None:
        # Auto-select marker 1 on arrival after a fresh video/analysis (never start on "No
        # detection selected" if markers exist) -- but not on every manual re-visit, so a
        # user's in-progress ◀/▶ position within one run survives a tab switch.
        if self.app.preview_needs_reset and self.app.events:
            self.app.preview_needs_reset = False
            self.app.select_event(0)
        else:
            self.refresh()

    def refresh(self) -> None:
        events = self.app.events
        for row in self.rows:
            row.destroy()
        self.rows = []

        if not events:
            self.empty_label.pack(pady=20)
        else:
            self.empty_label.pack_forget()
            for i, event in enumerate(events):
                self.rows.append(self._build_row(i, event))

        self.preview_panel.set_video(self.app.video_path)
        self.preview_panel.show_detection(events, self.app.selected_index)
        self.delete_btn.configure(state="normal" if self.app.selected_index is not None else "disabled")

        if self.app.metadata:
            self.timeline.set_duration(self.app.metadata.duration_seconds)
        self.timeline.set_events(events, self.app.selected_index)

    def _build_row(self, index: int, event) -> ctk.CTkFrame:
        selected = index == self.app.selected_index
        row = ctk.CTkFrame(
            self.table_scroll, fg_color=self.colors["accent_dim"] if selected else "transparent", corner_radius=6
        )
        row.pack(fill="x", pady=1)
        values = [(str(event.event_number), 40), (event.hms_display, 90)]
        labels = []
        for text, width in values:
            label = ctk.CTkLabel(row, text=text, width=width, anchor="w", text_color=self.colors["text_primary"])
            label.pack(side="left", pady=6)
            labels.append(label)

        handler = functools.partial(self.app.select_event, index)
        row.bind("<Button-1>", lambda e: handler())
        for label in labels:
            label.bind("<Button-1>", lambda e: handler())
        return row

    def _navigate(self, direction: int) -> None:
        self.app.navigate_event(direction)

    def _on_add_marker(self) -> None:
        if not self.app.video_path or not self.app.metadata:
            dialogs.show_info(self.app.root, self.colors, "No Video Selected", "Select a video first.")
            return
        AddMarkerDialog(
            self.app.root, self.colors, self.app.add_manual_event, max_seconds=self.app.metadata.duration_seconds,
        )

    def _on_delete(self) -> None:
        self.app.delete_selected_event()
