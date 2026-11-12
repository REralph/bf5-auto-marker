"""
ui/pages/analyse_page.py
----------------------------
ANALYSE stage: extremely compact. Normally embedded (standalone=False) as the
second stage inside ui/pages/workflow_page.py's consolidated WORKFLOW page,
sitting between the VIDEO and RESOLVE stages -- standalone=True (the default)
still renders it as a full, independent page for any other caller. A small
always-visible CALIBRATE SETTINGS link lets the user open CALIBRATE at any
time, not just as an error recovery. Shows progress, a live read-only preview
of the most recently detected skull, and a large "MARKERS FOUND" counter;
hides shape/colour confidence and other CV internals that used to be
always-visible here. Laid out compactly: the video name and CALIBRATE SETTINGS share
one row, and the progress panel sits beside the live preview instead of above it. On
completion, REVIEW MARKERS opens PREVIEW and SEND TO RESOLVE sends straight away through the
RESOLVE stage's own send_now() (no pop-up), so the SENDING.../CREATED status shows in that card.

Never silently shows "Analysing frame 0 / 0" -- MainWindow.check_analysis_readiness()
is consulted before a scan is allowed to start (and on every plain refresh),
distinguishing VIDEO LOAD ERROR / VIDEO METADATA ERROR / CALIBRATION ERROR / READY
so a real problem is reported explicitly, with a RELOAD VIDEO or GO TO
CALIBRATE recovery action, instead of ever falling through to a frozen,
meaningless progress display.
"""

from __future__ import annotations

import os
import time

import customtkinter as ctk

import ui.devtools as devtools
import ui.dialogs as dialogs
from ui.live_preview import LiveSkullPreview

_ERROR_TITLES = {
    "VIDEO_LOAD_ERROR": "VIDEO LOAD ERROR",
    "VIDEO_METADATA_ERROR": "VIDEO METADATA ERROR",
    "CALIBRATION_ERROR": "CALIBRATION ERROR",
}


class AnalysePage(ctk.CTkFrame):
    def __init__(self, master, app, colors: dict, standalone: bool = True):
        # Embedded in a WORKFLOW stage card -> transparent, so it blends into the card instead of
        # painting a darker page-coloured rectangle over it.
        super().__init__(master, fg_color=colors["bg"] if standalone else "transparent")
        self.app = app
        self.colors = colors
        self.standalone = standalone
        self._current_frame_index = 0
        self._current_timestamp_seconds = 0.0
        self._build()

    def _build(self) -> None:
        colors = self.colors
        pad = 28 if self.standalone else 0
        if self.standalone:
            ctk.CTkLabel(
                self, text="ANALYSE", font=("Segoe UI", 22, "bold"), text_color=colors["text_primary"]
            ).pack(anchor="w", padx=pad, pady=(24, 4))
            devtools.page_label(self, self.app, colors, "ANALYSE")
        devtools.section_label(
            self, self.app, colors, "ANALYSIS_CONTROLS", "ANALYSE", pack_opts={"anchor": "w", "padx": pad, "pady": (8, 0)}
        )

        # Video name and the CALIBRATE SETTINGS link share one row. The link is reachable
        # regardless of readiness state -- not just as an error recovery -- so the user can jump to
        # CALIBRATE to double-check calibration even when it's already fine.
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill="x", padx=pad, pady=(0, 8))
        self.video_name_label = ctk.CTkLabel(
            top_row, text="No video selected.", font=("Segoe UI", 13), text_color=colors["text_secondary"]
        )
        self.video_name_label.pack(side="left")
        devtools.dev_button(
            top_row, self.app, colors, "CALIBRATE_SETTINGS", "ANALYSE", "ANALYSIS_CONTROLS",
            text="CALIBRATE SETTINGS", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            text_color=colors["text_secondary"], height=28, font=("Segoe UI", 11),
            command=lambda: self.app.show_page("CALIBRATE"), pack_opts={"side": "right"},
        )

        # Progress/status on the left, the live "what was just detected" preview beside it on the
        # right -- one compact block instead of two stacked ones.
        body_row = ctk.CTkFrame(self, fg_color="transparent")
        body_row.pack(fill="x", padx=pad, pady=(0, 8))
        body_row.grid_columnconfigure(0, weight=1)
        self.progress_panel = ctk.CTkFrame(
            body_row, fg_color=colors["panel"] if self.standalone else colors["panel_alt"], corner_radius=10,
        )
        self.progress_panel.grid(row=0, column=0, sticky="new", padx=(0, 12))
        self.live_preview = LiveSkullPreview(body_row, colors)
        self.live_preview.grid(row=0, column=1, sticky="n")
        # Codes for the two panels (they live on body_row, not inside the panels, which get rebuilt).
        devtools.section_label(
            body_row, self.app, colors, "ANALYSIS_STATUS", "ANALYSE", anchor=self.progress_panel, side="inside",
        )
        devtools.section_label(
            body_row, self.app, colors, "LIVE_PREVIEW", "ANALYSE", anchor=self.live_preview, side="inside",
        )
        self._render_ready_idle()

        # Developer Mode debug fields (item 16 of the spec this page was hardened against) --
        # always live, not just while a scan is running, so a "why won't analysis start" report
        # can be diagnosed from these values alone.
        # Packed side="left" -- a compact single-row strip of tiny reference chips (hover/click
        # for the real field name+value) instead of one full row per field.
        debug_row = ctk.CTkFrame(self, fg_color="transparent")
        debug_row.pack(fill="x", padx=pad, pady=(0, 8))
        _f = {"side": "left", "padx": (0, 6)}
        devtools.readout_label(debug_row, self.app, colors, "VIDEO_PATH", lambda: self.app.video_path or "(none)", pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "VIDEO_LOADED", lambda: self.app.video_session.is_loaded, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "CAPTURE_OPEN", self._capture_open_display, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "FPS", lambda: self.app.metadata.fps if self.app.metadata else "(unknown)", pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "TOTAL_FRAMES", lambda: self.app.metadata.frame_count if self.app.metadata else "(unknown)", pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "CURRENT_TIMESTAMP", self._current_timestamp_display, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "CALIBRATION_LOADED", lambda: self.app.session.is_calibrated(), pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "DETECTION_REGION", self._detection_region_display, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "ANALYSIS_WORKER", lambda: self.app.analysis_worker_status, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "ANALYSIS_HEARTBEAT", self._heartbeat_display, pack_opts=_f)
        # Live NO_ACTIVE_SKULL/ACTIVE_SKULL event-tracking state (video_scanner.ScanState) --
        # lets you confirm a continuously-visible skull is being tracked as ONE appearance
        # (ACTIVE_SKULL the whole time, one growing best-frame/confidence/count) rather than
        # split into several bookmarks.
        devtools.readout_label(debug_row, self.app, colors, "DETECTION_STATE", lambda: self.app.last_scan_state.detection_state, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "ACTIVE_EVENT_START", self._active_event_start_display, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "ACTIVE_EVENT_BEST_FRAME", lambda: self.app.last_scan_state.active_event_best_frame if self.app.last_scan_state.active_event_best_frame is not None else "(none)", pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "ACTIVE_EVENT_BEST_CONFIDENCE", lambda: f"{self.app.last_scan_state.active_event_best_confidence * 100:.0f}%", pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "ACTIVE_EVENT_SKULL_COUNT", lambda: self.app.last_scan_state.active_event_skull_count, pack_opts=_f)
        # False-positive candidates the neighbor-validation gate rejected before they could ever
        # become a bookmark -- confirms the gate is genuinely filtering something, not just idle.
        devtools.readout_label(debug_row, self.app, colors, "REJECTED_CANDIDATE_COUNT", lambda: self.app.last_scan_state.rejected_candidate_count, pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "LAST_REJECTED_CANDIDATE", self._last_rejected_candidate_display, pack_opts=_f)
        # Per-signal evidence (structure/edge/luminance/colour/size/aspect/position) behind the last
        # candidate that started an event, the last rejection and its reason, and the strongest
        # candidate that fell BELOW the threshold (a skull the detector nearly caught, and why not).
        devtools.readout_label(debug_row, self.app, colors, "LAST_CANDIDATE_EVIDENCE", lambda: self.app.last_scan_state.last_candidate_evidence or "(none)", pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "LAST_REJECTED_REASON", lambda: self.app.last_scan_state.last_rejected_reason or "(none)", pack_opts=_f)
        devtools.readout_label(debug_row, self.app, colors, "BEST_MISSED_CANDIDATE", lambda: self.app.last_scan_state.best_missed_evidence or "(none)", pack_opts=_f)

    # ------------------------------------------------------------------
    # Developer Mode field helpers
    # ------------------------------------------------------------------
    def _capture_open_display(self):
        session = self.app.video_session
        return session.is_loaded and session.ensure_open()

    def _current_timestamp_display(self) -> str:
        total_ms = int(round(max(0.0, self._current_timestamp_seconds) * 1000))
        m, rem = divmod(total_ms, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{m:02d}:{s:02d}.{ms:03d}"

    def _heartbeat_display(self) -> str:
        """Distinguishes 'about to update' from a genuine stall: if the worker hasn't reported
        progress in over 5 real seconds while supposedly running, that's a real stall worth
        flagging explicitly rather than leaving CURRENT_FRAME sitting there with no explanation."""
        if not self.app.analysis_running:
            return "NOT RUNNING"
        last = self.app._last_progress_wall_time
        if last is None:
            return "ACTIVE -- waiting for first frame"
        elapsed = time.time() - last
        frame = self.app._last_progress_frame
        if elapsed > 5.0:
            return f"STALLED -- no update for {elapsed:.1f}s (last frame {frame})"
        return f"ACTIVE -- frame {frame} ({elapsed:.1f}s ago)"

    def _active_event_start_display(self) -> str:
        start_frame = self.app.last_scan_state.active_event_start_frame
        if start_frame is None:
            return "(none)"
        fps = self.app.metadata.fps if self.app.metadata else 0.0
        if not fps:
            return f"frame {start_frame}"
        total_ms = int(round((start_frame / fps) * 1000))
        m, rem = divmod(total_ms, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{m:02d}:{s:02d}.{ms:03d} (frame {start_frame})"

    def _last_rejected_candidate_display(self) -> str:
        state = self.app.last_scan_state
        if state.last_rejected_frame is None:
            return "(none)"
        return f"frame {state.last_rejected_frame} -- {state.last_rejected_symbol} at {state.last_rejected_confidence * 100:.0f}%"

    def _detection_region_display(self) -> str:
        region = self.app.session.detection_region
        if region is None:
            return "(none)"
        x, y, w, h = region
        return f"X={x},Y={y},W={w},H={h}"

    # ------------------------------------------------------------------
    # Progress panel states: ready-idle, cannot-start (with a specific reason), running, complete
    # ------------------------------------------------------------------
    def _render_ready_idle(self) -> None:
        for child in self.progress_panel.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            self.progress_panel, text="✓ ANALYSIS READY", font=("Segoe UI", 13, "bold"), text_color=self.colors["success"]
        ).pack(anchor="w", padx=16, pady=(14, 2))
        ctk.CTkLabel(
            self.progress_panel, text="Video and calibration are both ready.",
            text_color=self.colors["text_secondary"],
        ).pack(anchor="w", padx=16, pady=(0, 10))
        devtools.dev_button(
            self.progress_panel, self.app, self.colors, "ANALYSE_VIDEO", "ANALYSE", "ANALYSIS_CONTROLS",
            text="ANALYSE VIDEO", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=self._start_analysis, pack_opts={"anchor": "w", "padx": 16, "pady": (0, 16)},
        )

    def _render_cannot_start(self, readiness) -> None:
        for child in self.progress_panel.winfo_children():
            child.destroy()
        title = _ERROR_TITLES.get(readiness.state, "ANALYSIS CANNOT START")
        ctk.CTkLabel(
            self.progress_panel, text=title, font=("Segoe UI", 14, "bold"), text_color=self.colors["danger"]
        ).pack(anchor="w", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            self.progress_panel, text=readiness.detail, text_color=self.colors["text_secondary"],
            wraplength=480, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        video_name = os.path.basename(self.app.video_path) if self.app.video_path else "(none selected)"
        frames = self.app.metadata.frame_count if self.app.metadata else 0
        ctk.CTkLabel(
            self.progress_panel, text=f"Video: {video_name}\nFrames detected: {frames}",
            text_color=self.colors["text_primary"], justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 10))

        if readiness.state in ("VIDEO_LOAD_ERROR", "VIDEO_METADATA_ERROR"):
            devtools.dev_button(
                self.progress_panel, self.app, self.colors, "RELOAD_VIDEO_ANALYSE", "ANALYSE", "ANALYSIS_CONTROLS",
                text="RELOAD VIDEO", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
                command=self._on_reload_video, pack_opts={"anchor": "w", "padx": 16, "pady": (0, 16)},
            )
        elif readiness.state == "CALIBRATION_ERROR":
            devtools.dev_button(
                self.progress_panel, self.app, self.colors, "GO_TO_CALIBRATE", "ANALYSE", "ANALYSIS_CONTROLS",
                text="GO TO CALIBRATE", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
                command=lambda: self.app.show_page("CALIBRATE"), pack_opts={"anchor": "w", "padx": 16, "pady": (0, 16)},
            )

    def _on_reload_video(self) -> None:
        if self.app.video_session.reload():
            self._start_analysis()  # retry immediately now that the video is genuinely reloaded
        else:
            dialogs.show_error(
                self.app.root, self.colors, "RELOAD FAILED", "The video could not be reopened.",
                why=["The file may have been moved, renamed, or deleted.", "Another program may be locking it."],
                what_to_do="Select the video again from the VIDEO tab.",
            )
            self.refresh()

    def _render_progress_running(self) -> None:
        parent = self.progress_panel
        for child in parent.winfo_children():
            child.destroy()

        ctk.CTkLabel(
            parent, text="ANALYSING...", font=("Segoe UI", 13, "bold"), text_color=self.colors["accent"]
        ).pack(anchor="w", padx=16, pady=(14, 2))
        ctk.CTkLabel(
            parent, text="✓ Skull calibration loaded", font=("Segoe UI", 11), text_color=self.colors["text_secondary"],
        ).pack(anchor="w", padx=16, pady=(0, 8))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16)
        self.progress_bar = ctk.CTkProgressBar(row, progress_color=self.colors["accent"])
        self.progress_bar.set(0)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_percent_label = ctk.CTkLabel(row, text="--%", width=44, text_color=self.colors["text_secondary"])
        self.progress_percent_label.pack(side="left", padx=(8, 0))

        # Deliberately NOT "frame 0 / 0" -- that literal string is indistinguishable from the
        # real failure state this page exists to catch. A neutral "Starting..." holds the slot
        # until the first genuine progress update (which always carries a real frame count,
        # since check_analysis_readiness() already guaranteed one exists before this can render).
        self.frame_count_label = ctk.CTkLabel(parent, text="Starting analysis...", text_color=self.colors["text_secondary"])
        self.frame_count_label.pack(anchor="w", padx=16, pady=(6, 0))
        devtools.field_label(
            parent, self.app, self.colors, "CURRENT_FRAME", lambda: self._current_frame_index,
            anchor=self.frame_count_label,
        )

        counter_col = ctk.CTkFrame(parent, fg_color="transparent")
        counter_col.pack(pady=(12, 4))
        ctk.CTkLabel(
            counter_col, text="MARKERS FOUND", font=("Segoe UI", 12, "bold"), text_color=self.colors["text_secondary"]
        ).pack()
        self.markers_count_label = ctk.CTkLabel(
            counter_col, text="0", font=("Segoe UI", 34, "bold"), text_color=self.colors["text_primary"]
        )
        self.markers_count_label.pack()
        devtools.field_label(
            counter_col, self.app, self.colors, "DETECTION_COUNT", lambda: self.markers_count_label.cget("text"),
            anchor=self.markers_count_label,
        )

        devtools.dev_button(
            parent, self.app, self.colors, "CANCEL_ANALYSIS", "ANALYSE", "ANALYSIS_CONTROLS",
            text="CANCEL ANALYSIS", fg_color=self.colors["panel_alt"], hover_color=self.colors["danger"],
            text_color=self.colors["text_primary"], command=self.app.cancel_analysis,
            pack_opts={"anchor": "w", "padx": 16, "pady": (12, 16)},
        )

    def _render_progress_complete(self) -> None:
        from data.export import summarize_events

        parent = self.progress_panel
        for child in parent.winfo_children():
            child.destroy()
        summary = summarize_events(self.app.events)
        ctk.CTkLabel(
            parent, text="ANALYSIS COMPLETE", font=("Segoe UI", 13, "bold"), text_color=self.colors["success"]
        ).pack(anchor="w", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            parent, text=f"✓ {summary.total_events} marker(s) found", font=("Segoe UI", 20, "bold"),
            text_color=self.colors["text_primary"],
        ).pack(anchor="w", padx=16, pady=(0, 14))

        button_row = ctk.CTkFrame(parent, fg_color="transparent")
        button_row.pack(fill="x", padx=16, pady=(0, 16))
        devtools.dev_button(
            button_row, self.app, self.colors, "REVIEW_MARKERS", "ANALYSE", "ANALYSIS_CONTROLS",
            text="REVIEW MARKERS", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=lambda: self.app.show_page("PREVIEW"), pack_opts={"side": "left"},
        )
        devtools.dev_button(
            button_row, self.app, self.colors, "SEND_TO_RESOLVE_FROM_ANALYSE", "ANALYSE", "ANALYSIS_CONTROLS",
            text="SEND TO RESOLVE", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"],
            command=self._send_to_resolve,
            pack_opts={"side": "left", "padx": 8},
        )

    def _send_to_resolve(self) -> None:
        self.app.pages["WORKFLOW"].resolve_section.send_now()

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        if self.app.pending_auto_analysis:
            self.app.pending_auto_analysis = False
            self._start_analysis()
        else:
            self.refresh()

    def _start_analysis(self) -> None:
        readiness = self.app.check_analysis_readiness()
        if not readiness.ready:
            self._render_cannot_start(readiness)
            return

        self.video_name_label.configure(text=os.path.basename(self.app.video_path))
        self._current_frame_index = 0
        self._current_timestamp_seconds = 0.0
        started = self.app.start_analysis()
        if started:
            self._render_progress_running()
            self.live_preview.set_video(self.app.video_path)
            self.live_preview.show_latest([])
        else:
            self.refresh()

    def refresh(self) -> None:
        if self.app.video_path:
            self.video_name_label.configure(text=os.path.basename(self.app.video_path))
        self.live_preview.set_video(self.app.video_path)

        if self.app.analysis_running:
            self.live_preview.show_latest(self.app.events)
        elif self.app.events:
            self._render_progress_complete()
            self.live_preview.show_latest(self.app.events)
        else:
            readiness = self.app.check_analysis_readiness()
            if readiness.ready:
                self._render_ready_idle()
            else:
                self._render_cannot_start(readiness)
            self.live_preview.show_latest([])

    def update_progress(self, percent, frame_index: int, events_so_far: int, kills_so_far: int) -> None:
        if not hasattr(self, "progress_bar") or not self.progress_bar.winfo_exists():
            return
        del kills_so_far  # the big counter tracks marker (event) count, not raw skull count
        self._current_frame_index = frame_index
        fps = self.app.metadata.fps if self.app.metadata else 0.0
        self._current_timestamp_seconds = frame_index / fps if fps else 0.0

        total_frames = self.app.metadata.frame_count if self.app.metadata else 0
        if percent is not None and total_frames > 0:
            if self.progress_bar.cget("mode") != "determinate":
                self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(percent / 100.0)
            self.progress_percent_label.configure(text=f"{percent:.0f}%")
            self.frame_count_label.configure(text=f"Analysing frame {frame_index:,} / {total_frames:,}")
        else:
            # Should be rare -- probe_video() now falls back to counting frames manually rather
            # than trusting a lying container/codec -- but never freeze on a stale "0/0"-style
            # display if it ever does happen: show real, live frame progress with an
            # indeterminate bar instead of a fabricated percentage.
            if self.progress_bar.cget("mode") != "indeterminate":
                self.progress_bar.configure(mode="indeterminate")
            self.progress_percent_label.configure(text="--%")
            self.frame_count_label.configure(text=f"Analysing frame {frame_index:,} (frame count unavailable)")

        self.markers_count_label.configure(text=str(events_so_far))
        self.live_preview.show_latest(self.app.events)

    def on_analysis_complete(self) -> None:
        self._render_progress_complete()

    def on_analysis_failed(self) -> None:
        self.refresh()
