"""
ui/main_window.py
-------------------
Application shell: the top navigation bar and the page container it
switches between. Owns the single AppSession (core/session.py) every page
reads and writes through, plus the controller methods pages call into for
cross-cutting concerns -- background analysis threading, Resolve
communication, and keeping every page's view of `events`/`selected_index`
in sync.

Nav is WORKFLOW (the VIDEO -> ANALYSIS -> RESOLVE stages on one page), CALIBRATE and
SETTINGS; PREVIEW is a real page but reached only via the ANALYSIS stage's REVIEW MARKERS
button, not a permanent nav tab. Only the page being shown is mapped (see show_page()).

Threading model: video loading, analysis and Resolve API calls run on worker threads. Workers
never touch widgets directly -- they push messages onto a queue.Queue (or hand a callable to
post_to_main()), and the main thread drains it on a periodic `after()` timer. Each message is
handled in isolation, so one failing handler can't stop the polling.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from typing import List, Optional

import customtkinter as ctk

import core.calibration_store as calibration_store
import ui.devtools as devtools
import ui.dialogs as dialogs
from core.crash_log import log_exception_text
from core.session import AppSession
from data.config import AppConfig, load_config, save_config
from data.detection import Event, SkullSighting
from data.export import AnalysisLogger, get_output_dir, write_csv, write_json, write_txt
from resolve.resolve_controller import (
    ResolveConnectionError,
    add_markers_from_detections,
    get_current_project,
    get_current_timeline,
    get_resolve,
)
from ui.pages.calibrate_page import CalibratePage
from ui.pages.preview_page import PreviewPage
from ui.pages.settings_page import SettingsPage
from ui.pages.workflow_page import WorkflowPage
from ui.welcome_dialog import show_welcome_dialog
from ui.window_perf import DeferredPageScaling, ResizeFreezer
from video.video_probe import VideoOpenError, seconds_to_timecode
from video.video_scanner import ScanState
from video.video_scanner import scan as scan_video

COLORS = {
    "bg": "#121214",
    "panel": "#1B1B1F",
    "panel_alt": "#242429",
    "border": "#2E2E34",
    "text_primary": "#F2F2F0",
    "text_secondary": "#9C9CA3",
    "accent": "#FF5C39",
    "accent_hover": "#FF7A57",
    "accent_dim": "#3A2A25",
    "success": "#3DDC84",
    "danger": "#FF5C5C",
}

NAV_ITEMS = ["WORKFLOW", "CALIBRATE", "SETTINGS"]

_MSG_PROGRESS = "progress"
_MSG_EVENT = "event"
_MSG_ANALYSIS_DONE = "analysis_done"
_MSG_ANALYSIS_ERROR = "analysis_error"
_MSG_SEND_DONE = "send_done"
_MSG_SEND_ERROR = "send_error"
_MSG_VIDEO_LOADED = "video_loaded"
_MSG_VIDEO_LOAD_ERROR = "video_load_error"
_MSG_CALL = "call"  # run a callable on the main thread (see MainWindow.post_to_main)

_REF_CODES_TICK_MS = 15_000  # how often the REF CODES countdown refreshes / checks its deadline


def _format_remaining(seconds: float) -> str:
    """'1h 59m' / '12m' / '<1m' -- the REF CODES time left, rounded UP to the minute so it never reads 0m while still on."""
    minutes = int((max(0.0, seconds) + 59) // 60)
    if seconds < 60:
        return "<1m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


@dataclasses.dataclass
class AnalysisReadiness:
    """
    Distinguishes exactly why analysis can't start instead of collapsing every failure into a
    frozen "Analysing frame 0 / 0". `state` is one of "READY", "VIDEO_LOAD_ERROR" (no video, or
    it failed to open), "VIDEO_METADATA_ERROR" (opened, but frame count/fps couldn't be
    determined), or "CALIBRATION_ERROR" (no detection region or reference image saved yet).
    """
    ready: bool
    state: str
    detail: str = ""


class MainWindow:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("BF5 Auto Marker")
        self.root.geometry("1300x860")
        self.root.configure(fg_color=COLORS["bg"])
        self.root.minsize(1150, 780)

        self.config: AppConfig = load_config()

        # The one shared session every page reads/writes through (video, calibration,
        # detections) -- loaded once here so it's already available even if the user goes
        # straight to ANALYSE without visiting CALIBRATE first.
        self.session = AppSession()
        calibration_store.load_into(self.session)

        self.output_dir: Optional[str] = None
        self.logger: Optional[AnalysisLogger] = None
        self._video_loading = False

        self.selected_index: Optional[int] = None
        self.follow_live = True

        # One-shot flags for the automatic ADD VIDEO -> CALIBRATE -> ANALYSE -> REVIEW chain:
        # set by the step that triggers the automatic transition, consumed (and cleared) by
        # the next page's on_show() so a later manual re-visit to that tab doesn't repeat it.
        self.pending_auto_analysis = False
        self.preview_needs_reset = False

        self.analysis_running = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.cancel_requested = False
        self.msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self._sending_markers = False
        self._reported_handler_errors: set = set()

        # WORKFLOW's stage indicator reads this to decide whether ANALYSIS/RESOLVE show as
        # "done"/"active" -- real state only: True only once a scan has actually finished
        # (regardless of how many markers it found), reset False the moment a new video loads
        # or a new scan starts, never left stale from a previous run.
        self.last_analysis_completed = False

        # Diagnostics for "is the worker actually still alive" (Developer Mode ANALYSIS_HEARTBEAT/
        # ANALYSIS_WORKER fields) -- if a real scan ever genuinely stalls (a hung capture.read()
        # on a specific file, for example), these make that visible immediately instead of an
        # indistinguishable frozen CURRENT_FRAME with no way to tell "stalled" from "about to update".
        self.analysis_worker_status = "NOT_STARTED"  # NOT_STARTED / RUNNING / STOPPED / ERROR
        self._last_progress_wall_time: Optional[float] = None
        self._last_progress_frame = 0
        # Live event-tracking state from the scan loop's NO_ACTIVE_SKULL/ACTIVE_SKULL state
        # machine (video_scanner.ScanState) -- Developer Mode's DETECTION_STATE/ACTIVE_EVENT_*
        # fields read this directly, so a continuously-visible skull can be confirmed as ONE
        # tracked appearance rather than several separate bookmarks.
        self.last_scan_state = ScanState()

        self.pages: dict = {}
        self.nav_buttons: dict = {}
        self.current_page_name = "WORKFLOW"

        # Reference codes (the REF CODES switch, top right): in memory only -- off at every start and
        # off again REFERENCE_CODES_SECONDS after being switched on. `_clock` is a seam for tests.
        self.ref_codes_on = False
        self.ref_codes_var = tk.BooleanVar(master=self.root, value=False)
        self._ref_codes_deadline: Optional[float] = None
        self._ref_codes_job = None
        self._poll_job = None
        self._tab_refs: list = []
        self._clock = time.time

        self._build_shell()
        self._poll_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.config.show_welcome_guide:
            self.root.after(200, self._show_welcome_guide)

    def _on_close(self) -> None:
        """Window closed: write any settings change still waiting for its debounced save, stop
        timers/threads cleanly, then destroy the window."""
        settings = self.pages.get("SETTINGS")
        if settings is not None:
            settings.flush_pending_save()
        self.cancel_requested = True  # a running scan checks this between frames; its thread is a daemon
        self.shutdown()
        self.root.destroy()

    def shutdown(self) -> None:
        """Cancel this window's own repeating timers (queue polling, the REF CODES countdown) so nothing
        is left scheduled against a window that is about to be destroyed. Safe to call twice."""
        for attr in ("_poll_job", "_ref_codes_job"):
            job = getattr(self, attr, None)
            setattr(self, attr, None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass

    def post_to_main(self, callback) -> None:
        """Thread-safe way for a worker thread to run `callback()` on the Tk main thread (Tk widgets
        must only be touched there). The callback runs from the same queue poll as the analysis,
        video-load and Resolve results, so one failing callback can't stop the others."""
        self.msg_queue.put((_MSG_CALL, callback))

    # ------------------------------------------------------------------
    # Convenience properties -- pages read session state through `self.app.X`
    # ------------------------------------------------------------------
    @property
    def video_path(self) -> Optional[str]:
        return self.session.video_path

    @property
    def metadata(self):
        return self.session.metadata

    @property
    def video_session(self) -> AppSession:
        return self.session

    @property
    def events(self) -> List[Event]:
        return self.session.detections

    @events.setter
    def events(self, value: List[Event]) -> None:
        self.session.detections = value

    # ------------------------------------------------------------------
    # Shell / navigation
    # ------------------------------------------------------------------
    def _build_shell(self) -> None:
        self._build_nav_bar()

        self.content_frame = ctk.CTkFrame(self.root, fg_color=COLORS["bg"])
        self.content_frame.pack(fill="both", expand=True)

        self.pages = {
            "WORKFLOW": WorkflowPage(self.content_frame, self, COLORS),
            "CALIBRATE": CalibratePage(self.content_frame, self, COLORS),
            "SETTINGS": SettingsPage(self.content_frame, self, COLORS),
            "PREVIEW": PreviewPage(self.content_frame, self, COLORS),  # reached via REVIEW, not a nav tab
        }
        # Only the page being shown is ever placed (mapped); the others are unmapped, so resizing
        # the window re-lays-out one tab instead of all four -- see show_page(). On a monitor
        # change, hidden tabs also skip the immediate DPI re-scale, and while the window edge is
        # being dragged the visible tab holds its size and re-flows once afterwards
        # (ui/window_perf.py).
        self._visible_page = None
        self._page_scaling = DeferredPageScaling(self.root, self.pages, lambda: self._visible_page)
        self._page_scaling.install()
        self._resize_freezer = ResizeFreezer(self.root, lambda: self._visible_page)

        self.show_page("WORKFLOW")

    def _build_nav_bar(self) -> None:
        header = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=0, height=56)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="BF5 AUTO MARKER", font=("Segoe UI", 15, "bold"), text_color=COLORS["text_primary"]
        ).pack(side="left", padx=(20, 24))

        nav_row = ctk.CTkFrame(header, fg_color="transparent")
        nav_row.pack(side="left", fill="y")
        for name in NAV_ITEMS:
            btn = ctk.CTkButton(
                nav_row, text=name, width=104, height=56, corner_radius=0, fg_color="transparent",
                hover_color=COLORS["accent_dim"], text_color=COLORS["text_secondary"], font=("Segoe UI", 12, "bold"),
                command=lambda n=name: self.show_page(n),
            )
            btn.pack(side="left")
            self.nav_buttons[name] = btn
            # The tab's reference code sits on the button itself (an overlay: no layout space).
            self._tab_refs.append(devtools.tab_label(nav_row, self, COLORS, name, anchor=btn))

        # Top right: the REF CODES switch (+ how long until it turns itself off). Packed on the right
        # of the header, so the title and tabs on the left never move when it changes.
        ref_box = ctk.CTkFrame(header, fg_color="transparent")
        ref_box.pack(side="right", fill="y", padx=(0, 18))
        self.ref_codes_switch = ctk.CTkSwitch(
            ref_box, text="", width=40, variable=self.ref_codes_var, progress_color=COLORS["accent"],
            command=lambda: self.set_reference_codes(self.ref_codes_var.get()),
        )
        self.ref_codes_switch.pack(side="right", pady=14)
        ctk.CTkLabel(
            ref_box, text="REF CODES", font=("Segoe UI", 11, "bold"), text_color=COLORS["text_secondary"],
        ).pack(side="right", padx=(0, 6))
        self.ref_codes_time_label = ctk.CTkLabel(
            ref_box, text="", width=64, anchor="e", font=("Consolas", 11), text_color=devtools.REF_COLOUR,
        )
        self.ref_codes_time_label.pack(side="right", padx=(0, 10))

    # ------------------------------------------------------------------
    # Reference codes (the REF CODES switch) and Developer Mode
    # ------------------------------------------------------------------
    def set_reference_codes(self, on: bool) -> None:
        """Show/hide the faint, click-to-copy reference codes. Switching them on arms a timer that
        switches them off again after devtools.REFERENCE_CODES_SECONDS (they are never saved, so a
        restart starts with them off)."""
        on = bool(on)
        if self._ref_codes_job is not None:
            self.root.after_cancel(self._ref_codes_job)
            self._ref_codes_job = None
        self.ref_codes_on = on
        self._ref_codes_deadline = self._clock() + devtools.REFERENCE_CODES_SECONDS if on else None
        if self.ref_codes_var.get() != on:
            self.ref_codes_var.set(on)
        devtools.refresh_all(self)
        for ref in self._tab_refs:
            ref.retint()
        self._update_ref_codes_countdown()

    def _update_ref_codes_countdown(self) -> None:
        """Once a countdown tick: refresh the 'time left' text, and switch the codes off at the deadline."""
        if self._ref_codes_job is not None:
            self.root.after_cancel(self._ref_codes_job)  # never leave two countdowns running
            self._ref_codes_job = None
        if not self.ref_codes_on or self._ref_codes_deadline is None:
            self.ref_codes_time_label.configure(text="")
            return
        remaining = self._ref_codes_deadline - self._clock()
        if remaining <= 0:
            self.set_reference_codes(False)
            return
        self.ref_codes_time_label.configure(text=_format_remaining(remaining))
        self._ref_codes_job = self.root.after(_REF_CODES_TICK_MS, self._update_ref_codes_countdown)

    def refresh_developer_mode(self) -> None:
        """Call after the DEVELOPER MODE switch changes in Settings."""
        devtools.refresh_all(self)
        # A handful of pages show a bigger Developer-Mode-only block that devtools.refresh_all()
        # doesn't know about since it's not a DevRef -- give those an explicit hook.
        for page in self.pages.values():
            on_changed = getattr(page, "on_developer_mode_changed", None)
            if callable(on_changed):
                on_changed()

    def show_page(self, name: str) -> None:
        if name not in self.pages:
            return
        # PREVIEW isn't in the nav bar, so only recolor real nav buttons.
        if name in self.nav_buttons:
            self.current_page_name = name
        for nav_name, btn in self.nav_buttons.items():
            selected = nav_name == name
            btn.configure(
                fg_color=COLORS["accent"] if selected else "transparent",
                text_color="#FFFFFF" if selected else COLORS["text_secondary"],
            )
        for ref in self._tab_refs:  # a code sits on its tab, so it follows the tab's fill
            ref.retint()

        page = self.pages[name]
        self._resize_freezer.thaw()
        if page is not self._visible_page:
            if self._visible_page is not None:
                self._visible_page.place_forget()
            self._visible_page = page
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._page_scaling.apply_pending(page)
        if hasattr(page, "on_show"):
            page.on_show()
        elif hasattr(page, "refresh"):
            page.refresh()

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    def start_loading_video(self, path: str) -> bool:
        """
        Loads `path` into the shared AppSession on a background thread (AppSession.load_video()
        never touches a widget, so it's safe off the main thread) and resets per-video state
        once it lands. Returns False immediately if a load is already in progress, or while an
        analysis is running (replacing the video then would reset the event list under the scan).
        """
        if self._video_loading or self.analysis_running:
            return False
        self._video_loading = True
        threading.Thread(target=self._video_load_worker, args=(path,), daemon=True).start()
        return True

    def _video_load_worker(self, path: str) -> None:
        try:
            self.session.load_video(path)
            self.msg_queue.put((_MSG_VIDEO_LOADED, path))
        except VideoOpenError as exc:
            self.msg_queue.put((_MSG_VIDEO_LOAD_ERROR, path, str(exc)))
        except Exception as exc:  # noqa: BLE001 - any other failure must still end the "Loading..." state
            log_exception_text(traceback.format_exc())
            self.msg_queue.put((_MSG_VIDEO_LOAD_ERROR, path, f"Unexpected error while opening the video: {exc}"))

    def _on_video_loaded(self, path: str) -> None:
        self._video_loading = False
        self.output_dir = None
        self.events = []
        self.selected_index = None
        self.follow_live = True
        self.preview_needs_reset = True
        self.last_analysis_completed = False
        self._refresh_event_pages()
        video_section = self.pages["WORKFLOW"].video_section
        if hasattr(video_section, "on_video_loaded"):
            video_section.on_video_loaded()
        # No navigation here -- VIDEO is a stage on the same WORKFLOW page, not a separate page
        # to leave; the ANALYSIS stage and stage indicator update in place via
        # _refresh_event_pages() above.

    def _on_video_load_failed(self, path: str, error_message: str) -> None:
        self._video_loading = False
        video_section = self.pages["WORKFLOW"].video_section
        if hasattr(video_section, "on_video_load_failed"):
            video_section.on_video_load_failed()
        dialogs.show_error(
            self.root, COLORS, "VIDEO COULD NOT BE OPENED",
            f"'{os.path.basename(path)}' could not be opened.",
            why=["The file is damaged or incomplete.", "The format/codec isn't supported.",
                 "Another program is locking the file."],
            what_to_do="Try selecting the video again, or pick a different file.",
        )

    # ------------------------------------------------------------------
    # Event selection / manual correction (shared across ANALYSE/PREVIEW/RESOLVE)
    # ------------------------------------------------------------------
    def select_event(self, index: int) -> None:
        if self.analysis_running:
            self.follow_live = False  # manual inspection pauses live-follow
        self.selected_index = index
        self._refresh_event_pages()

    def navigate_event(self, direction: int) -> None:
        if not self.events:
            return
        new_index = 0 if self.selected_index is None else max(0, min(len(self.events) - 1, self.selected_index + direction))
        self.select_event(new_index)

    def add_manual_event(self, timestamp_seconds: float) -> None:
        fps = self.metadata.fps if self.metadata else 30.0
        event = Event(
            event_number=0,
            timestamp_seconds=timestamp_seconds,
            frame_number=int(round(timestamp_seconds * fps)),
            timecode=seconds_to_timecode(timestamp_seconds, fps),
            skulls=[SkullSighting(bbox=(0, 0, 0, 0), shape_confidence=1.0, color_name="Manual", color_confidence=1.0)],
        )
        events = self.events
        events.append(event)
        events.sort(key=lambda e: e.timestamp_seconds)
        for i, e in enumerate(events, start=1):
            e.event_number = i
        self.selected_index = events.index(event)
        self._refresh_event_pages()

    def delete_selected_event(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.events)):
            return
        del self.events[self.selected_index]
        for i, e in enumerate(self.events, start=1):
            e.event_number = i
        self.selected_index = None
        self._refresh_event_pages()

    def _refresh_event_pages(self) -> None:
        """Re-render the event views -- only the page on screen. A page that isn't showing is
        refreshed by its own on_show() when it's next opened; refreshing it now would decode video
        frames and rebuild widgets nobody can see, on every new event during a scan."""
        for name in ("WORKFLOW", "PREVIEW"):
            page = self.pages.get(name)
            if page is not None and page is self._visible_page and hasattr(page, "refresh"):
                page.refresh()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def check_analysis_readiness(self) -> AnalysisReadiness:
        """
        The full validation chain from "is there a video" through "is calibration ready",
        checked explicitly and in order so a failure anywhere reports exactly which stage
        failed -- never silently falls through to a 0/0 progress display. Read-only: never
        raises, never mutates state, safe to call from on_show()/refresh() as often as needed.
        """
        session = self.session
        if not session.video_path:
            return AnalysisReadiness(False, "VIDEO_LOAD_ERROR", "No video has been selected yet.")
        if not session.is_loaded:
            return AnalysisReadiness(
                False, "VIDEO_LOAD_ERROR", session.load_error or "The video could not be opened."
            )
        if session.metadata is None or session.frame_count <= 0:
            return AnalysisReadiness(
                False, "VIDEO_METADATA_ERROR",
                "The video opened, but its total frame count could not be determined.",
            )
        if session.fps <= 0:
            return AnalysisReadiness(
                False, "VIDEO_METADATA_ERROR",
                "The video opened, but its frame rate could not be determined.",
            )
        if not session.is_calibrated():
            return AnalysisReadiness(
                False, "CALIBRATION_ERROR", "Skull calibration has not been completed yet.",
            )
        oversized = session.oversized_reference_labels()
        if oversized:
            return AnalysisReadiness(
                False, "CALIBRATION_ERROR",
                f"The saved reference image(s) {', '.join(oversized)} are larger than the "
                "detection region, so they can never match -- this would silently produce zero "
                "detections for those appearances across the entire video. Go to CALIBRATE and "
                "run AUTO-DETECT FROM TEST FOOTAGE (or FIND SKULL POSITION) again.",
            )
        return AnalysisReadiness(True, "READY")

    def start_analysis(self) -> bool:
        if self.analysis_running:
            return False
        readiness = self.check_analysis_readiness()
        if not readiness.ready:
            return False

        session = self.session
        region, aspect_mismatch = session.scaled_detection_region(session.width, session.height)

        if aspect_mismatch:
            proceed = dialogs.ask_yes_no(
                self.root, COLORS, "Video Resolution Doesn't Match Calibration",
                "This video's aspect ratio is substantially different from the one you "
                "calibrated on, so the saved skull location may not scale to the right part "
                "of the frame. Confirm to analyse anyway, or Cancel and recalibrate on this "
                "video first (CALIBRATE → SELECT SKULL AREA).",
            )
            if not proceed:
                return False

        try:
            self.output_dir = get_output_dir(self.video_path)
            self.logger = AnalysisLogger(self.output_dir)
        except OSError as exc:
            # No result files can be written (typically a full drive, or a read-only folder). The
            # markers themselves don't need them, so offer to carry on without rather than failing.
            self.output_dir = None
            self.logger = None
            proceed = dialogs.ask_yes_no(
                self.root, COLORS, "Result Files Can't Be Saved",
                f"The folder for this video's result files could not be created:\n\n{exc}\n\n"
                "Analyse anyway? The markers will still be found and can be sent to Resolve, but "
                "kill_timestamps.csv / .txt, detections.json and the analysis log won't be written.",
            )
            if not proceed:
                return False
        if self.logger:
            self.logger.log(f"Video: {self.video_path}")
            self.logger.log(
                f"Settings: confidence={self.config.confidence_threshold}, sampling={self.config.frame_sampling}, "
                f"neighbour validation={self.config.enable_neighbor_validation}, "
                f"multiple skulls={self.config.multiple_skull_detection}, region={region}"
            )

        self.events = []
        self.selected_index = None
        self.follow_live = True
        self.cancel_requested = False
        self.analysis_running = True
        self.analysis_worker_status = "STARTING"
        self._last_progress_wall_time = time.time()
        self._last_progress_frame = 0
        self.last_scan_state = ScanState()
        self.last_analysis_completed = False
        self._refresh_event_pages()

        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            args=(
                session.video_path, session.metadata, region, session.scaled_reference_templates(),
                session.colour_reference_set(session.height),
            ),
            daemon=True,
        )
        self.analysis_thread.start()
        self.analysis_worker_status = "RUNNING"
        return True

    def cancel_analysis(self) -> None:
        self.cancel_requested = True

    def _analysis_worker(self, video_path, metadata, region, reference_templates, colour_reference=None) -> None:
        def progress_cb(percent, frame_index, events_so_far, kills_so_far, scan_state) -> None:
            self.msg_queue.put((_MSG_PROGRESS, percent, frame_index, events_so_far, kills_so_far, scan_state))

        def on_event(event: Event) -> None:
            self.msg_queue.put((_MSG_EVENT, event))

        def should_cancel() -> bool:
            return self.cancel_requested

        try:
            scan_video(
                video_path, metadata, region, reference_templates, self.config,
                progress_callback=progress_cb, should_cancel=should_cancel, on_event=on_event,
                colour_reference=colour_reference,
            )
            self.msg_queue.put((_MSG_ANALYSIS_DONE,))
        except VideoOpenError as exc:
            self.msg_queue.put((_MSG_ANALYSIS_ERROR, str(exc), traceback.format_exc()))
        except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the UI, not a crash
            self.msg_queue.put((_MSG_ANALYSIS_ERROR, f"Unexpected error during analysis: {exc}", traceback.format_exc()))

    def _on_new_event(self, event: Event) -> None:
        self.events.append(event)
        if self.follow_live:
            self.selected_index = len(self.events) - 1
        self._refresh_event_pages()

    def _on_analysis_complete(self) -> None:
        self.analysis_running = False
        self.analysis_worker_status = "STOPPED"

        export_error = None
        if self.output_dir:
            try:
                written = [
                    write_csv(self.events, self.output_dir),
                    write_txt(self.events, self.output_dir),
                    write_json(self.events, self.output_dir),
                ]
                if self.logger:
                    self.logger.log(
                        f"Analysis complete. {len(self.events)} events, "
                        f"{sum(e.skull_count for e in self.events)} kills detected."
                    )
                    for path in written:
                        self.logger.log(f"Wrote {path}")
            except OSError as exc:
                export_error = exc  # the results are intact in memory; finish normally and say so below

        # Completion must always be reached: whether or not the result files could be written, the
        # detected markers are valid and can be reviewed and sent to Resolve.
        self.preview_needs_reset = True
        self.last_analysis_completed = True
        analyse_section = self.pages["WORKFLOW"].analyse_section
        analyse_section.on_analysis_complete()
        self._refresh_event_pages()

        if export_error is not None:
            dialogs.show_error(
                self.root, COLORS, "RESULT FILES NOT SAVED",
                "The analysis finished, but its result files could not be written.",
                why=[str(export_error), "The drive may be full or the folder may be read-only."],
                what_to_do="Your markers are still available: review them and send them to Resolve as normal. "
                "Free some disk space and run the analysis again if you also want the files.",
            )

    def _on_analysis_failed(self, error_message: str, traceback_str: str = "") -> None:
        self.analysis_running = False
        self.analysis_worker_status = "ERROR"
        if self.logger:
            self.logger.log(f"ERROR: {error_message}")
            if traceback_str:
                self.logger.log(f"Traceback:\n{traceback_str}")

        analyse_section = self.pages["WORKFLOW"].analyse_section
        if hasattr(analyse_section, "on_analysis_failed"):
            analyse_section.on_analysis_failed()

        dialogs.show_error(
            self.root, COLORS, "ANALYSIS STOPPED", "The analysis stopped before it could finish.",
            why=[error_message], what_to_do="Check the video file and try again.",
        )

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------
    def send_markers(self) -> bool:
        """Sends every marker to Resolve on a worker thread. The events and settings are copied here,
        on the main thread, so the worker never reads a list the UI could be changing (a marker
        added/deleted in PREVIEW, a new analysis). Returns False if nothing was started."""
        if not self.events or self._sending_markers:
            return False
        self._sending_markers = True
        threading.Thread(
            target=self._send_markers_worker,
            args=(
                list(self.events), self.config.marker_name, self.config.marker_color,
                self.session.bookmark_adjustment_seconds,
            ),
            daemon=True,
        ).start()
        return True

    def _send_markers_worker(self, events, marker_name, marker_color, bookmark_adjustment_seconds) -> None:
        try:
            resolve = get_resolve()
            project = get_current_project(resolve)
            timeline = get_current_timeline(project)
            result = add_markers_from_detections(
                timeline, project, events, marker_name=marker_name, marker_color=marker_color,
                bookmark_adjustment_seconds=bookmark_adjustment_seconds,
            )
            self.msg_queue.put((_MSG_SEND_DONE, result))
        except ResolveConnectionError as exc:
            self.msg_queue.put((_MSG_SEND_ERROR, str(exc)))
        except Exception as exc:  # noqa: BLE001
            self.msg_queue.put((_MSG_SEND_ERROR, f"Unexpected error while talking to Resolve: {exc}"))

    # ------------------------------------------------------------------
    def _show_welcome_guide(self) -> None:
        def dont_show_again() -> None:
            self.config.show_welcome_guide = False
            try:
                save_config(self.config)
            except OSError:
                pass  # only a preference; it simply asks again next launch

        show_welcome_dialog(self.root, COLORS, on_dont_show_again=dont_show_again)

    # ------------------------------------------------------------------
    # Main-thread queue polling (the only place worker results touch widgets)
    # ------------------------------------------------------------------
    def _poll_queue(self) -> None:
        """Drains the worker->UI queue every 100 ms. Each message is handled on its own: an error in
        one handler is reported (crash log + one dialog per kind of message) and the loop carries on.
        Rescheduling sits in `finally`, so nothing a handler does can end the polling -- if it ever
        stopped, scan progress, completion and Resolve results would silently never arrive again."""
        self._poll_job = None
        try:
            while True:
                try:
                    message = self.msg_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_message(message)
                except Exception as exc:  # noqa: BLE001 - isolate the failure to this one message
                    self._report_handler_error(message[0], exc)
        finally:
            try:
                self._poll_job = self.root.after(100, self._poll_queue)
            except tk.TclError:
                pass  # the window has been destroyed under us: nothing left to poll for

    def _report_handler_error(self, kind: str, exc: Exception) -> None:
        log_exception_text(f"Error handling '{kind}' message:\n{traceback.format_exc()}")
        if kind in self._reported_handler_errors:
            return  # already told the user about this kind; the log has the rest
        self._reported_handler_errors.add(kind)
        if kind in (_MSG_SEND_DONE, _MSG_SEND_ERROR):
            self._sending_markers = False
        try:
            dialogs.show_error(
                self.root, COLORS, "UNEXPECTED ERROR", "Something went wrong while updating the screen.",
                why=[str(exc)], what_to_do="The details were written to crash_log.txt. The app is still running.",
            )
        except Exception:  # noqa: BLE001 - reporting must never raise
            pass

    def _handle_message(self, message: tuple) -> None:
        kind = message[0]

        if kind == _MSG_PROGRESS:
            _, percent, frame_index, events_so_far, kills_so_far, scan_state = message
            self._last_progress_wall_time = time.time()
            self._last_progress_frame = frame_index
            self.last_scan_state = scan_state
            self.pages["WORKFLOW"].analyse_section.update_progress(percent, frame_index, events_so_far, kills_so_far)

        elif kind == _MSG_EVENT:
            _, event = message
            self._on_new_event(event)

        elif kind == _MSG_ANALYSIS_DONE:
            self._on_analysis_complete()

        elif kind == _MSG_ANALYSIS_ERROR:
            _, error_message, traceback_str = message
            self._on_analysis_failed(error_message, traceback_str)

        elif kind == _MSG_VIDEO_LOADED:
            _, path = message
            self._on_video_loaded(path)

        elif kind == _MSG_VIDEO_LOAD_ERROR:
            _, path, error_message = message
            self._on_video_load_failed(path, error_message)

        elif kind == _MSG_SEND_DONE:
            _, result = message
            self._sending_markers = False
            resolve_section = self.pages["WORKFLOW"].resolve_section
            if hasattr(resolve_section, "on_send_result"):
                resolve_section.on_send_result(result)

        elif kind == _MSG_SEND_ERROR:
            _, error_message = message
            self._sending_markers = False
            resolve_section = self.pages["WORKFLOW"].resolve_section
            if hasattr(resolve_section, "on_send_error"):
                resolve_section.on_send_error(error_message)
            else:
                dialogs.show_error(
                    self.root, COLORS, "Could Not Send Markers", "DaVinci Resolve rejected the request.",
                    why=[error_message], what_to_do="Fix the issue above, then try SEND TO RESOLVE again.",
                )

        elif kind == _MSG_CALL:
            _, callback = message
            callback()
