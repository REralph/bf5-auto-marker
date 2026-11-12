"""
ui/pages/workflow_page.py
----------------------------
WORKFLOW: the primary nav-bar page, consolidating what used to be three
separate pages (VIDEO, ANALYSE, RESOLVE) into one continuous three-stage
page -- 01 VIDEO -> 02 ANALYSIS -> 03 RESOLVE -- so the normal path through
the app feels like one workflow instead of three unrelated screens
competing for nav-bar space.

This is a UI/layout consolidation, not a rewrite: every stage embeds the
existing, unmodified page class (VideoPage/AnalysePage/ResolvePage,
constructed with standalone=False so they skip their own big page title and
render compactly into this page's own consistently-styled stage cards
instead). Every button, callback, readiness branch, and Developer Mode
field on those three classes is exactly the same code that ran when they
were separate pages -- only where their widgets get packed changed.

CALIBRATE, PREVIEW, and SETTINGS are deliberately NOT embedded here -- they
stay full, separate pages (CALIBRATE reached via the always-visible
"CALIBRATE SETTINGS" link inside the ANALYSIS stage; PREVIEW via the
ANALYSIS stage's REVIEW MARKERS button; SETTINGS via the nav bar), per the
explicit instruction to keep calibration/review/settings as secondary,
on-demand pages rather than main workflow stages.

The stage indicator at the top reflects only real, already-tracked
application state (AppSession.is_loaded, MainWindow.last_analysis_completed)
-- never a fake completion. The page deliberately has no title or tagline of its
own: the stage strip is the first thing on it.
"""

from __future__ import annotations

import customtkinter as ctk

import ui.devtools as devtools
from ui.pages.analyse_page import AnalysePage
from ui.pages.resolve_page import ResolvePage
from ui.pages.video_page import VideoPage

_STAGE_DONE = "done"
_STAGE_ACTIVE = "active"
_STAGE_FUTURE = "future"


class WorkflowPage(ctk.CTkFrame):
    def __init__(self, master, app, colors: dict):
        super().__init__(master, fg_color=colors["bg"])
        self.app = app
        self.colors = colors
        self._stage_dots = {}
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        colors = self.colors
        devtools.page_label(self, self.app, colors, "WORKFLOW")
        self._build_stage_indicator(self, padx=28)

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(6, 10))

        video_body = self._build_stage_card(body, "01", "VIDEO", "Load your Battlefield V gameplay clip.", "STAGE_VIDEO")
        self.video_section = VideoPage(video_body, self.app, colors, standalone=False)
        self.video_section.pack(fill="x")

        analyse_body = self._build_stage_card(
            body, "02", "ANALYSIS", "Scan the loaded clip for confirmed skull events.", "STAGE_ANALYSIS",
        )
        self.analyse_section = AnalysePage(analyse_body, self.app, colors, standalone=False)
        self.analyse_section.pack(fill="x")

        resolve_body = self._build_stage_card(
            body, "03", "RESOLVE", "Send the confirmed markers to DaVinci Resolve.", "STAGE_RESOLVE",
        )
        self.resolve_section = ResolvePage(resolve_body, self.app, colors, standalone=False)
        self.resolve_section.pack(fill="x")

    def _build_stage_card(self, parent, number: str, title: str, subtitle: str, ref_id: str = ""):
        colors = self.colors
        card = ctk.CTkFrame(parent, fg_color=colors["panel"], corner_radius=10)
        card.pack(fill="x", pady=(0, 12))

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(12, 2))
        ctk.CTkLabel(
            header, text=number, font=("Segoe UI", 12, "bold"), text_color=colors["text_secondary"],
            fg_color=colors["panel_alt"], corner_radius=6, width=26, height=20,
        ).pack(side="left")
        title_label = ctk.CTkLabel(
            header, text=title, font=("Segoe UI", 16, "bold"), text_color=colors["text_primary"],
        )
        title_label.pack(side="left", padx=(10, 0))
        if ref_id:
            devtools.section_label(header, self.app, colors, ref_id, "WORKFLOW", anchor=title_label)

        ctk.CTkLabel(
            card, text=subtitle, font=("Segoe UI", 12), text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(0, 4))

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=(2, 14))
        return body

    # ------------------------------------------------------------------
    # Stage indicator -- real state only, no fake completion.
    # ------------------------------------------------------------------
    def _build_stage_indicator(self, parent, padx: int) -> None:
        colors = self.colors
        row = ctk.CTkFrame(parent, fg_color=colors["panel"], corner_radius=8)
        row.pack(fill="x", padx=padx, pady=(20, 6))
        devtools.section_label(row, self.app, colors, "STAGE_INDICATOR", "WORKFLOW")
        inner = ctk.CTkFrame(row, fg_color="transparent")
        inner.pack(padx=14, pady=10)

        for i, (num, name) in enumerate([("01", "VIDEO"), ("02", "ANALYSIS"), ("03", "RESOLVE")]):
            dot = ctk.CTkLabel(inner, text="○", font=("Segoe UI", 13), text_color=colors["text_secondary"], width=16)
            dot.pack(side="left")
            label = ctk.CTkLabel(
                inner, text=f"{num} {name}", font=("Segoe UI", 12, "bold"), text_color=colors["text_secondary"],
            )
            label.pack(side="left", padx=(4, 0))
            self._stage_dots[name] = (dot, label)
            if i < 2:
                ctk.CTkLabel(inner, text="  →  ", text_color=colors["text_secondary"]).pack(side="left")

        self.refresh_stage_indicator()

    def refresh_stage_indicator(self) -> None:
        app = self.app
        video_state = _STAGE_DONE if app.session.is_loaded else _STAGE_ACTIVE
        if not app.session.is_loaded:
            analysis_state = _STAGE_FUTURE
        elif getattr(app, "last_analysis_completed", False):
            analysis_state = _STAGE_DONE
        else:
            analysis_state = _STAGE_ACTIVE
        resolve_state = _STAGE_ACTIVE if getattr(app, "last_analysis_completed", False) else _STAGE_FUTURE

        for name, state in (("VIDEO", video_state), ("ANALYSIS", analysis_state), ("RESOLVE", resolve_state)):
            dot, label = self._stage_dots[name]
            if state == _STAGE_DONE:
                glyph, color = "✓", self.colors["success"]
            elif state == _STAGE_ACTIVE:
                glyph, color = "●", self.colors["accent"]
            else:
                glyph, color = "○", self.colors["text_secondary"]
            dot.configure(text=glyph, text_color=color)
            label.configure(text_color=color)

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.refresh_stage_indicator()
        for section in (self.video_section, self.analyse_section, self.resolve_section):
            if hasattr(section, "refresh"):
                section.refresh()
