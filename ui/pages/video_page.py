"""
ui/pages/video_page.py
-------------------------
The 01 VIDEO stage of the WORKFLOW page: one compact drag-and-drop / click-to-browse zone.
Dropping or selecting a valid video loads it in the background (AppSession.load_video() runs on
a worker thread via MainWindow.start_loading_video()); the ANALYSIS stage below updates in
place once it's ready. A video can't be replaced while an analysis is running.
"""

from __future__ import annotations

import os
from tkinter import filedialog

import customtkinter as ctk
from tkinterdnd2 import DND_FILES

import ui.devtools as devtools
import ui.dialogs as dialogs

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")


class VideoPage(ctk.CTkFrame):
    def __init__(self, master, app, colors: dict, standalone: bool = True):
        # Embedded in a WORKFLOW stage card -> transparent, so it blends into the card instead of
        # painting a darker page-coloured rectangle over it.
        super().__init__(master, fg_color=colors["bg"] if standalone else "transparent")
        self.app = app
        self.colors = colors
        self.standalone = standalone
        self._build()

    def _build(self) -> None:
        colors = self.colors
        if self.standalone:
            ctk.CTkLabel(
                self, text="VIDEO", font=("Segoe UI", 22, "bold"), text_color=colors["text_primary"]
            ).pack(anchor="w", padx=28, pady=(24, 4))
            ctk.CTkLabel(
                self, text="Select Battlefield V gameplay video", font=("Segoe UI", 13), text_color=colors["text_secondary"]
            ).pack(anchor="w", padx=28)
            devtools.page_label(self, self.app, colors, "VIDEO")

        panel = ctk.CTkFrame(self, fg_color=colors["panel"] if self.standalone else "transparent", corner_radius=10)
        panel.pack(fill="x", padx=(28 if self.standalone else 0), pady=(20 if self.standalone else 0))
        devtools.section_label(
            panel, self.app, colors, "VIDEO_SOURCE", "VIDEO", pack_opts={"anchor": "w", "padx": 20, "pady": (14, 0)}
        )

        # The drop zone itself is packed once, unconditionally -- only its CONTENTS get
        # destroyed/rebuilt between idle and loading (same swap-a-frame's-children pattern
        # already used for ANALYSE's progress panel), so the zone's own position never drifts.
        self.drop_zone = ctk.CTkFrame(
            panel, fg_color=colors["panel_alt"], corner_radius=10, border_width=2, border_color=colors["border"],
        )
        # Low (compact in height) but spread across the full width of the stage, with its contents centred.
        side_pad = 20 if self.standalone else 0
        self.drop_zone.pack(fill="x", padx=(side_pad, side_pad), pady=(8 if self.standalone else 4, 8))
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)

        self._render_idle()

    # ------------------------------------------------------------------
    def _render_idle(self) -> None:
        for child in self.drop_zone.winfo_children():
            child.destroy()
        colors = self.colors

        row = ctk.CTkFrame(self.drop_zone, fg_color="transparent")
        row.pack(padx=24, pady=(16, 4))
        ctk.CTkLabel(
            row, text="DRAG VIDEO HERE", font=("Segoe UI", 15, "bold"), text_color=colors["text_primary"],
        ).pack(side="left")
        ctk.CTkLabel(row, text="or", text_color=colors["text_secondary"]).pack(side="left", padx=12)
        devtools.dev_button(
            row, self.app, colors, "SELECT_VIDEO", "VIDEO", "VIDEO_SOURCE",
            text="SELECT VIDEO", fg_color=colors["accent"], hover_color=colors["accent_hover"],
            font=("Segoe UI", 13, "bold"), height=34, corner_radius=8, command=self._on_select,
            pack_opts={"side": "left"},
        )
        formats_text = " • ".join(ext.lstrip(".").upper() for ext in SUPPORTED_EXTENSIONS)
        ctk.CTkLabel(
            self.drop_zone, text=formats_text, font=("Segoe UI", 11), text_color=colors["text_secondary"],
        ).pack(pady=(0, 12))
        devtools.field_label(
            self.drop_zone, self.app, colors, "VIDEO_PATH", lambda: self.app.video_path or "(none selected)",
            anchor=self.drop_zone, side="inside_bottom",   # bottom-right: clear of the VIDEO_SOURCE code up top
        )

    def _render_loading(self, filename: str) -> None:
        for child in self.drop_zone.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            self.drop_zone, text=f"Loading {filename}...", font=("Segoe UI", 14, "bold"),
            text_color=self.colors["text_primary"],
        ).pack(padx=24, pady=26)

    # ------------------------------------------------------------------
    def _on_select(self) -> None:
        path = filedialog.askopenfilename(
            title="Select your Battlefield V gameplay video.",
            filetypes=[("Video files", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS))],
        )
        if path:
            self._handle_selected_path(path)

    def _on_drop(self, event) -> None:
        paths = self.tk.splitlist(event.data)
        if paths:
            self._handle_selected_path(paths[0])

    def _handle_selected_path(self, path: str) -> None:
        if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
            dialogs.show_error(
                self.app.root, self.colors, "Unsupported File", "That file can't be used as a source video.",
                why=["The file extension isn't one BF5 Auto Marker supports."],
                what_to_do="Choose a .mp4, .mov, .mkv, or .avi file instead.",
            )
            return
        if self.app.analysis_running:
            dialogs.show_info(
                self.app.root, self.colors, "Analysis In Progress",
                "A video is being analysed right now. Cancel the analysis (or wait for it to finish) "
                "before loading a different video.",
            )
            return
        if self.app.start_loading_video(path):
            self._render_loading(os.path.basename(path))

    # ------------------------------------------------------------------
    def on_video_loaded(self) -> None:
        # MainWindow auto-navigates to CALIBRATE immediately after this -- reset back to idle
        # so returning to VIDEO later (e.g. to load a different clip) starts clean.
        self._render_idle()

    def on_video_load_failed(self) -> None:
        self._render_idle()

    def refresh(self) -> None:
        pass
