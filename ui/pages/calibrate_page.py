"""
ui/pages/calibrate_page.py
------------------------------
CALIBRATE tab: a compact, collapsible setup page built around a permanent
TEST FOOTAGE library -- one, two, or three short clips (spec: "1-3 clips
representing different skull appearances"), each showing one controlled
skull appearance with a deliberately-drawn green rectangle around it while
it's visible. The green box is training information only:

    TEST FOOTAGE                        REAL GAMEPLAY VIDEO
    green box -> training signal        no green box -> uses the learned
    (WHERE the skull exists,            region/reference from calibration;
    WHEN the skull exists)              the real Analysis engine never
                                         searches for a green box.

Every clip and reference image added here is copied into
calibration/test_footage/ or calibration/references/ and stays there
permanently (core/calibration_store.py) -- nothing needs to be re-uploaded
on a later launch. SKULL REFERENCE is also a small library (e.g. one White
image + one Gold image): every calibrated reference is combined into ONE
multi-template skull detector, never separate detection systems per
colour. AUTO-DETECT FROM TEST FOOTAGE combines the green-box signal from
every library clip into one shared search region, then genuinely matches
EVERY calibrated reference within that region across every clip to learn
each one's own real on-screen scale -- real, never-invented confidence
scores. FIND SKULL POSITION (single current preview frame) and SELECT
SEARCH AREA (manual rectangle) remain as fallbacks for when the automatic
path doesn't work well or a clip has no green box.

Five sections, all writing into the same AppSession fields and
core.calibration_store files the real ANALYSE stage has always read from,
so that stage, RESOLVE, and the actual detection engine are all untouched
by this page:
    1. SKULL REFERENCE   -- the permanent reference-image library (add/remove/replace).
    2. TEST FOOTAGE       -- the permanent clip library (add/remove/replace).
    3. SEARCH AREA        -- AUTO-DETECT / FIND SKULL POSITION / manual select.
    4. CALIBRATION TEST   -- runs the real multi-reference detector against every library clip.
    5. SAVE CALIBRATION   -- commits the region + each reference's learned scale to calibration/.

Sections 1-4 are collapsible (CollapsibleSection below); only one is open
at a time by default, so the page stays short regardless of how much is
configured.
"""

from __future__ import annotations

import math
import os
import threading
import traceback
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog
from typing import Callable, Dict, List, Optional, Tuple

import cv2
from PIL import Image, ImageTk

import customtkinter as ctk

import core.calibration_store as calibration_store
import ui.devtools as devtools
import ui.dialogs as dialogs
from core.session import AppSession, ReferenceImage, build_colour_reference_set, scaled_region
from data.detection import Event, format_seconds
from detectors.colour_matcher import learn_footage_appearance
from detectors.combined_detector import build_detector
from detectors.green_box import find_green_box_candidates, find_training_spans
from detectors.skull_detector import (
    crop_to_content, expand_to_search_region, locate_reference_in_frame, mode_confidence_breakdown,
)
from resolve.resolve_controller import compute_marker_timestamp
from ui.video_scrubber import VideoScrubber
from video.video_probe import VideoOpenError, probe_video
from video.video_scanner import _apply_region
from video.video_scanner import scan as scan_video

MAX_CANVAS_WIDTH = 640
MAX_CANVAS_HEIGHT = 360
REFERENCE_THUMB_SIZE = 64
VIDEO_FILETYPES = [("Video files", "*.mp4 *.mov *.mkv *.avi")]
REFERENCE_IMAGE_FILETYPES = [("Image files", "*.png *.jpg *.jpeg *.webp")]

# A green box drawn by the user is already a reasonable margin around the skull -- unlike a bare
# icon-only bbox found by FIND SKULL POSITION, which needs a much larger multiplier (see
# detectors/skull_detector.py's own SEARCH_REGION_*_MARGIN, reused unchanged for that path).
TRAINING_REGION_WIDTH_MARGIN = 1.6
TRAINING_REGION_HEIGHT_MARGIN = 1.6
MIN_REFERENCE_MATCH_CONFIDENCE = 0.55  # below this, a detection is reported as failed
MIN_REGION_SIZE = 12  # a manual selection smaller than this in either dimension is rejected

# TEST GOLD SKULL matches library entries by name, not by any dedicated "this one is gold" flag
# -- keeps the reference/test-footage libraries themselves unchanged (still just "a reference
# image" / "a test clip"), while giving this one diagnostic action a concrete file to act on.
GOLD_KEYWORD = "gold"
# Only used to SEARCH for the best candidate to report/display -- the actual PASS/FAIL verdict
# always compares against the real configured confidence_threshold, never this floor (spec:
# "do not lower the threshold blindly" -- this exists purely so a failed test can still report
# a genuine best-found confidence instead of nothing).
GOLD_TEST_SEARCH_FLOOR = 0.15


def _find_by_keyword(filenames: List[str], keyword: str) -> Optional[str]:
    matches = [f for f in filenames if keyword.lower() in f.lower()]
    return matches[0] if matches else None


def _union_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    x2, y2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def _clamp_region(region: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = region
    w, h = min(w, frame_w), min(h, frame_h)
    x, y = max(0, min(x, frame_w - w)), max(0, min(y, frame_h - h))
    return (int(x), int(y), int(w), int(h))


def _open_in_explorer(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    os.startfile(path)


def _reference_colour_text(profile) -> str:
    """User-facing description of what a reference actually contains, derived from its own pixels
    (never its filename)."""
    signature = profile.signature
    if profile.has_source_colour:
        return f"{signature.description}  ·  colour information available"
    return f"{signature.description}  ·  grayscale image file (no hue data in the file itself)"


def _reference_profile_dev_text(profile) -> str:
    sig = profile.signature
    return (
        f"colour: {sig.description} | H {sig.hue:.0f} S {sig.saturation:.0f} V {sig.value:.0f} | "
        f"Lab ({sig.lab_l:.0f},{sig.lab_a:.0f},{sig.lab_b:.0f}) | chromatic {sig.chromatic} | "
        f"source has colour channels {profile.has_source_colour} | background removed {profile.background_removed} | "
        f"silhouette {profile.size[0]}x{profile.size[1]}px | representations: colour, mask, gray, edge, contrast, binary, inverted"
    )


class CalibrationError(Exception):
    """Raised for any expected, explainable calibration failure (no reference uploaded, no test
    footage, no green box found, ...) -- caught and shown as a plain message, never an unhandled
    crash."""


@dataclass
class _ClipAnalysis:
    frames_analysed: int
    frame_w: int
    frame_h: int
    training_events: int
    raw_bbox: Optional[Tuple[int, int, int, int]]  # None if no green box found anywhere in this clip
    frame_cache: Dict[int, "object"] = field(default_factory=dict)  # BGR frames, only within the longest span


@dataclass
class ReferenceMatchResult:
    filename: str
    label: str
    confidence: float
    scale: float
    passed: bool


@dataclass
class AutoDetectResult:
    clips_used: List[str]
    clips_skipped: List[str]
    search_region: Tuple[int, int, int, int]
    calibration_video_width: int
    calibration_video_height: int
    reference_results: List[ReferenceMatchResult]
    skull_detected: bool  # True if at least one reference matched well enough


@dataclass
class ClipTestResult:
    filename: str
    confidence: float
    passed: bool
    detected_as: str = ""
    note: str = ""
    temporal_event_count: Optional[int] = None  # None if the temporal check couldn't run


@dataclass
class _AutoDetectOutcome:
    """What AUTO-DETECT's background step produces; the main thread applies it to the session/UI."""
    result: "AutoDetectResult"
    reference_images: List[ReferenceImage]
    lines: List[str]


@dataclass
class _CalibrationTestOutcome:
    results: List["ClipTestResult"]
    lines: List[str]
    passed: bool


@dataclass
class GoldTestResult:
    clip_filename: str
    reference_filename: str
    frames_analysed: int
    fps: float
    green_box_confirmed: bool  # whether the Gold Test video's own green box was found (ground truth)
    detected: bool             # best_confidence >= the REAL configured threshold
    first_detection_frame: Optional[int]
    best_confidence: float
    detection_frame_count: int
    best_frame_index: Optional[int]
    best_bbox: Optional[Tuple[int, int, int, int]]  # full-frame-relative
    mode_confidences: Dict[str, float]  # empty unless Developer Mode was on when this ran
    best_mode: str
    threshold: float
    temporal_event_count: Optional[int] = None  # how many events the real event-grouping produced

    @property
    def first_detection_timestamp(self) -> Optional[float]:
        return self.first_detection_frame / self.fps if self.first_detection_frame is not None and self.fps else None

    @property
    def best_timestamp(self) -> Optional[float]:
        return self.best_frame_index / self.fps if self.best_frame_index is not None and self.fps else None


# ----------------------------------------------------------------------
class CollapsibleSection(ctk.CTkFrame):
    """One accordion section: a clickable header (arrow + title + a short status summary shown
    only while collapsed) and a body frame shown/hidden as a whole. Toggling is coordinated by
    the page (see CalibratePage._on_section_toggle) so only one section is open by default."""

    def __init__(self, parent, colors: dict, title: str, on_toggle: Callable[["CollapsibleSection"], None]):
        super().__init__(parent, fg_color=colors["panel"], corner_radius=10)
        self.colors = colors
        self.expanded = False
        self._on_toggle = on_toggle

        header = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        header.pack(fill="x")
        self.arrow_label = ctk.CTkLabel(header, text="▶", width=16, text_color=colors["text_secondary"], font=("Segoe UI", 12))
        self.arrow_label.pack(side="left", padx=(16, 4), pady=12)
        ctk.CTkLabel(header, text=title, font=("Segoe UI", 13, "bold"), text_color=colors["text_primary"]).pack(side="left", pady=12)
        self.status_label = ctk.CTkLabel(header, text="", font=("Segoe UI", 11), text_color=colors["text_secondary"])
        self.status_label.pack(side="left", padx=(10, 0), pady=12)

        self.body = ctk.CTkFrame(self, fg_color="transparent")

        for widget in (header, self.arrow_label):
            widget.bind("<Button-1>", self._on_click)

    def _on_click(self, event=None) -> None:
        self._on_toggle(self)

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        if expanded:
            self.body.pack(fill="x", padx=16, pady=(0, 16))
        else:
            self.body.pack_forget()
        self.arrow_label.configure(text="▼" if expanded else "▶")

    def set_status(self, text: str, color: Optional[str] = None) -> None:
        self.status_label.configure(text=text, text_color=color or self.colors["text_secondary"])


# ----------------------------------------------------------------------
class CalibratePage(ctk.CTkFrame):
    def __init__(self, master, app, colors: dict):
        super().__init__(master, fg_color=colors["bg"])
        self.app = app
        self.colors = colors

        # A separate, throwaway AppSession purely for previewing the currently-selected library
        # clip -- deliberately NOT self.app.session (the real gameplay video), so calibration
        # never requires that video loaded at all.
        self.test_session = AppSession()

        self.canvas_scale = 1.0
        self._tk_image = None
        self._reference_ctk_image = None
        self._sections: List[CollapsibleSection] = []
        self._error_trace: Optional[str] = None
        self._last_auto_detect: Optional[AutoDetectResult] = None
        self._last_test_results: Optional[List[ClipTestResult]] = None
        self._last_gold_test: Optional[GoldTestResult] = None
        self._selecting = False
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_rect_id = None
        self._library_changed = False  # references changed since the last AUTO-DETECT
        self._saved_adjustment = app.session.bookmark_adjustment_seconds  # what calibration.json holds
        self._busy = False  # a long calibration action is running on a background thread
        self._heavy_buttons: list = []

        self._build()
        self._refresh_all_statuses()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctk.CTkLabel(
            self, text="CALIBRATE", font=("Segoe UI", 22, "bold"), text_color=self.colors["text_primary"]
        ).pack(anchor="w", padx=28, pady=(24, 4))
        devtools.page_label(self, self.app, self.colors, "CALIBRATE")

        self._build_bookmark_adjustment_bar()

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(10, 10))

        self._build_reference_section(body)
        self._build_footage_section(body)
        self._build_search_area_section(body)
        self._build_calibration_test_section(body)

        footer = ctk.CTkFrame(body, fg_color="transparent")
        footer.pack(fill="x", pady=(4, 16))
        button_row = ctk.CTkFrame(footer, fg_color="transparent")
        button_row.pack(fill="x")
        self.save_calibration_btn = devtools.dev_button(
            button_row, self.app, self.colors, "SAVE_CALIBRATION", "CALIBRATE", "FOOTER",
            text="SAVE CALIBRATION", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            font=("Segoe UI", 14, "bold"), state="disabled", command=self._on_save_calibration,
            pack_opts={"side": "left"},
        )
        devtools.dev_button(
            button_row, self.app, self.colors, "OPEN_CALIBRATION_FOLDER", "CALIBRATE", "FOOTER",
            text="OPEN CALIBRATION FOLDER", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"], command=lambda: _open_in_explorer(calibration_store.CALIBRATION_DIR),
            pack_opts={"side": "left", "padx": (8, 0)},
        )
        self.copy_error_btn = ctk.CTkButton(
            footer, text="COPY ERROR", width=120, fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"], command=self._on_copy_error,
        )
        self.copy_error_btn.pack(anchor="w", pady=(8, 0))
        self.copy_error_btn.pack_forget()

        # Only one section open at a time by default -- start on SKULL REFERENCE.
        self._on_section_toggle(self.reference_section)

    def _register_section(self, section: CollapsibleSection) -> CollapsibleSection:
        section.pack(fill="x", pady=(0, 12))
        self._sections.append(section)
        return section

    def _on_section_toggle(self, section: CollapsibleSection) -> None:
        if section.expanded:
            section.set_expanded(False)
            return
        for other in self._sections:
            if other is not section:
                other.set_expanded(False)
        section.set_expanded(True)

    # ------------------------------------------------------------------
    # BOOKMARK TIME ADJUSTMENT -- always visible at the top of the page (not inside an accordion
    # section), since it's the one setting that affects every future send regardless of which
    # other section is open. Signed seconds: negative moves a marker earlier, positive later, 0 =
    # exact detected timestamp. Typing a value updates app.session.bookmark_adjustment_seconds
    # immediately in-memory (so TEST reflects it live) but -- matching every other calibration
    # field on this page -- only persists to disk once SAVE CALIBRATION is clicked.
    # ------------------------------------------------------------------
    def _build_bookmark_adjustment_bar(self) -> None:
        colors = self.colors
        panel = ctk.CTkFrame(self, fg_color=colors["panel"], corner_radius=10)
        panel.pack(fill="x", padx=28, pady=(0, 12))
        devtools.section_label(
            panel, self.app, colors, "BOOKMARK_TIME_ADJUSTMENT", "CALIBRATE",
            pack_opts={"anchor": "w", "padx": 16, "pady": (12, 0)},
        )
        ctk.CTkLabel(
            panel, text="BOOKMARK TIME ADJUSTMENT", font=("Segoe UI", 13, "bold"), text_color=colors["text_primary"],
        ).pack(anchor="w", padx=16, pady=(12, 6))

        row = ctk.CTkFrame(panel, fg_color="transparent")
        row.pack(fill="x", padx=16)
        self.bookmark_adjustment_var = tk.StringVar(value=f"{self.app.session.bookmark_adjustment_seconds:g}")
        self.bookmark_entry = ctk.CTkEntry(row, textvariable=self.bookmark_adjustment_var, width=90)
        self.bookmark_entry.pack(side="left")
        self._entry_border_default = self.bookmark_entry.cget("border_color")
        ctk.CTkLabel(row, text="seconds", text_color=colors["text_secondary"]).pack(side="left", padx=(6, 0))
        devtools.field_label(
            row, self.app, colors, "BOOKMARK_TIME_ADJUSTMENT", lambda: self.bookmark_adjustment_var.get(),
            anchor=self.bookmark_entry, side="inside",
        )
        self.bookmark_test_btn = devtools.dev_button(
            row, self.app, colors, "TEST_BOOKMARK_ADJUSTMENT", "CALIBRATE", "BOOKMARK_TIME_ADJUSTMENT",
            text="TEST", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            text_color=colors["text_primary"], command=self._on_test_bookmark_adjustment,
            pack_opts={"side": "left", "padx": (16, 0)},
        )
        self.bookmark_unsaved_label = ctk.CTkLabel(
            row, text="", font=("Segoe UI", 11, "bold"), text_color=colors["accent"],
        )
        self.bookmark_unsaved_label.pack(side="left", padx=(12, 0))
        self.bookmark_adjustment_var.trace_add("write", lambda *_: self._on_bookmark_adjustment_changed())

        ctk.CTkLabel(
            panel, text="Negative = earlier   •   Positive = later   •   0 = exact detection",
            text_color=colors["text_secondary"], font=("Segoe UI", 11),
        ).pack(anchor="w", padx=16, pady=(6, 4))

        self.bookmark_test_result_label = ctk.CTkLabel(
            panel, text="", text_color=colors["text_secondary"], font=("Segoe UI", 11), wraplength=560, justify="left",
        )
        self.bookmark_test_result_label.pack(anchor="w", padx=16, pady=(0, 12))

    def _parse_bookmark_adjustment(self) -> Optional[float]:
        """The entry's value in seconds, or None if it isn't a real number within +/- the allowed
        range (text like "nan" or "inf", or something absurd, would otherwise be applied to every
        marker)."""
        limit = calibration_store.MAX_BOOKMARK_ADJUSTMENT_SECONDS
        try:
            value = float(self.bookmark_adjustment_var.get())
        except ValueError:
            return None
        return value if math.isfinite(value) and abs(value) <= limit else None

    def _on_bookmark_adjustment_changed(self) -> None:
        value = self._parse_bookmark_adjustment()
        if value is None:
            # Not a usable number (mid-typing like "-", or invalid): flag it, and leave the last
            # valid adjustment in force rather than applying garbage.
            self.bookmark_entry.configure(border_color=self.colors["danger"])
            return
        self.bookmark_entry.configure(border_color=self._entry_border_default)
        self.app.session.bookmark_adjustment_seconds = value
        self._update_unsaved_marker()

    def _update_unsaved_marker(self) -> None:
        changed = self.app.session.bookmark_adjustment_seconds != self._saved_adjustment
        self.bookmark_unsaved_label.configure(text="● not saved yet -- click SAVE CALIBRATION" if changed else "")

    def _on_test_bookmark_adjustment(self) -> None:
        self._run_guarded(self._test_bookmark_adjustment_impl, self.bookmark_test_result_label)

    def _test_bookmark_adjustment_impl(self) -> None:
        if not self.test_session.is_loaded or self.test_session.current_image is None:
            raise CalibrationError("Load a test clip in SEARCH AREA below and pick a frame first.")
        adjustment = self._parse_bookmark_adjustment()
        if adjustment is None:
            limit = calibration_store.MAX_BOOKMARK_ADJUSTMENT_SECONDS
            raise CalibrationError(f"Enter a number of seconds between -{limit:g} and {limit:g} (e.g. -0.25 or 0.3).")
        self.app.session.bookmark_adjustment_seconds = adjustment

        original_seconds = self.test_session.current_timestamp
        preview_event = Event(
            event_number=0, timestamp_seconds=original_seconds,
            frame_number=self.test_session.current_frame, timecode="",
        )
        adjusted_seconds = compute_marker_timestamp(preview_event, adjustment)
        self.bookmark_test_result_label.configure(
            text=f"Original: {format_seconds(original_seconds)}   ->   Adjustment: {adjustment:+.2f}s"
            f"   ->   New position: {format_seconds(adjusted_seconds)}",
            text_color=self.colors["success"],
        )

    # ------------------------------------------------------------------
    # Shared error handling -- every section's "run something" action funnels through here so a
    # CalibrationError shows a plain message and any genuinely unexpected exception shows
    # CALIBRATION ERROR + a COPY ERROR button, never a crash.
    # ------------------------------------------------------------------
    def _run_guarded(self, action_fn: Callable[[], None], result_label: ctk.CTkLabel) -> None:
        if self._refuse_if_busy():
            return
        self.copy_error_btn.pack_forget()
        self._error_trace = None
        try:
            action_fn()
        except CalibrationError as exc:
            result_label.configure(text=str(exc), text_color=self.colors["danger"])
        except Exception as exc:  # noqa: BLE001 -- never let a calibration action crash the app
            self._show_unexpected_error(result_label, exc, traceback.format_exc())
        self._refresh_all_statuses()

    # ------------------------------------------------------------------
    # Page hooks
    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self._refresh_all_statuses()  # the library/calibration may have changed elsewhere (SETTINGS)

    def release_preview(self) -> None:
        """Close the preview clip. Windows can't delete a file that is still open, so anything that
        removes test footage (RESET CALIBRATION, REMOVE, REPLACE) calls this first."""
        self.test_session.release()
        self.test_session.video_path = None
        self.test_session.is_loaded = False
        self._update_footage_preview_visibility()

    def on_calibration_reset(self) -> None:
        """RESET CALIBRATION (SETTINGS) cleared everything on disk and in the session -- bring this
        page back in line with that."""
        self._last_auto_detect = None
        self._last_test_results = None
        self._last_gold_test = None
        self._library_changed = False
        self._saved_adjustment = 0.0
        self.bookmark_adjustment_var.set("0")
        self._update_unsaved_marker()
        self._refresh_all_statuses()

    def _file_action(self, description: str, action: Callable[[], None]) -> bool:
        """Runs a library/calibration file operation; a failure (full drive, a file another program
        has open) is reported plainly instead of surfacing as an unexpected crash."""
        try:
            action()
        except OSError as exc:
            dialogs.show_error(
                self.app.root, self.colors, "FILE OPERATION FAILED", f"Could not {description}.",
                why=[str(exc), "The drive may be full, or the file may be open in another program."],
                what_to_do="Free some disk space or close the other program, then try again.",
            )
            return False
        return True

    def _drop_reference_from_session(self, filename: str) -> None:
        """A removed/replaced reference file must stop being used right away, not linger in memory
        (and keep detecting) until the app is restarted."""
        label = os.path.splitext(filename)[0]
        self.app.session.reference_images = [r for r in self.app.session.reference_images if r.label != label]

    # ------------------------------------------------------------------
    # Long actions (AUTO-DETECT, RUN CALIBRATION TEST, TEST GOLD SKULL) decode whole clips and take
    # 10-20 s. They run on a worker thread -- `compute` may only touch files, the session (read-only)
    # and the detector, never a widget -- and `apply` then runs back on the main thread. The window
    # stays responsive, the buttons are disabled meanwhile, and anything that would change the
    # library or the calibration underneath the worker is refused until it finishes.
    # ------------------------------------------------------------------
    def _refuse_if_busy(self) -> bool:
        if not self._busy:
            return False
        dialogs.show_info(
            self.app.root, self.colors, "Please Wait",
            "A calibration check is still running. Wait for it to finish, then try again.",
        )
        return True

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for button in self._heavy_buttons:
            button.configure(state="disabled" if busy else "normal")

    def _run_background(self, result_label: ctk.CTkLabel, working_text: str, compute: Callable, apply: Callable) -> None:
        if self._refuse_if_busy():
            return
        self.copy_error_btn.pack_forget()
        self._error_trace = None
        result_label.configure(text=f"{working_text} ... this can take up to a minute.", text_color=self.colors["text_secondary"])
        self._set_busy(True)

        def worker() -> None:
            try:
                outcome = ("ok", compute())
            except CalibrationError as exc:
                outcome = ("calibration_error", str(exc))
            except Exception as exc:  # noqa: BLE001 -- shown as CALIBRATION ERROR, never a crash
                outcome = ("error", exc, traceback.format_exc())
            self.app.post_to_main(lambda: self._finish_background(result_label, apply, outcome))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_background(self, result_label: ctk.CTkLabel, apply: Callable, outcome: tuple) -> None:
        self._set_busy(False)
        kind = outcome[0]
        if kind == "ok":
            try:
                apply(outcome[1])
            except Exception as exc:  # noqa: BLE001
                self._show_unexpected_error(result_label, exc, traceback.format_exc())
        elif kind == "calibration_error":
            result_label.configure(text=outcome[1], text_color=self.colors["danger"])
        else:
            self._show_unexpected_error(result_label, outcome[1], outcome[2])
        self._refresh_all_statuses()

    def _show_unexpected_error(self, result_label: ctk.CTkLabel, exc: Exception, trace_text: str) -> None:
        self._error_trace = trace_text
        result_label.configure(text=f"CALIBRATION ERROR\n{exc}", text_color=self.colors["danger"])
        self.copy_error_btn.pack(anchor="w", pady=(8, 0))

    def _on_copy_error(self) -> None:
        if not self._error_trace:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(self._error_trace)
        except tk.TclError:
            pass
        self.copy_error_btn.configure(text="✓ COPIED")
        self.after(900, lambda: self.copy_error_btn.configure(text="COPY ERROR"))

    def _refresh_all_statuses(self) -> None:
        self._refresh_reference_list()
        self._refresh_footage_list()

        ref_count = len(calibration_store.list_reference_images())
        ref_ok = ref_count > 0
        if ref_ok and self._library_changed:
            self.reference_section.set_status(
                f"⚠ {ref_count} reference{'s' if ref_count != 1 else ''} -- re-run AUTO-DETECT to apply", self.colors["accent"],
            )
        else:
            self.reference_section.set_status(
                f"✓ {ref_count} reference{'s' if ref_count != 1 else ''}" if ref_ok else "Not set",
                self.colors["success"] if ref_ok else self.colors["danger"],
            )
        count = len(calibration_store.list_test_footage())
        self.footage_section.set_status(f"{count} test video{'s' if count != 1 else ''}" if count else "No clips yet")
        region_ok = self.app.session.detection_region is not None
        self.search_area_section.set_status(
            "✓ Ready" if region_ok else "Not set", self.colors["success"] if region_ok else self.colors["danger"]
        )
        if self._last_test_results is not None:
            all_passed = all(r.passed for r in self._last_test_results)
            self.calibration_test_section.set_status(
                "✓ Passed" if all_passed else "✗ Failed", self.colors["success"] if all_passed else self.colors["danger"]
            )
        else:
            self.calibration_test_section.set_status("Not run yet")

        # SAVE CALIBRATION needs a completed, in-session calibration (region + at least one
        # reference actually matched via AUTO-DETECT/FIND SKULL POSITION/SELECT SEARCH AREA) --
        # not merely that reference/test files exist on disk.
        can_save = region_ok and len(self.app.session.reference_images) > 0
        self.save_calibration_btn.configure(state="normal" if can_save else "disabled")

    # ==================================================================
    # 1. SKULL REFERENCE -- a small permanent library (e.g. White + Gold), combined into one
    # multi-template detector rather than separate detection systems.
    # ==================================================================
    def _build_reference_section(self, parent) -> None:
        section = CollapsibleSection(parent, self.colors, "SKULL REFERENCE", self._on_section_toggle)
        self.reference_section = self._register_section(section)
        devtools.section_label(section.body, self.app, self.colors, "SKULL_REFERENCE", "CALIBRATE", pack_opts={"anchor": "w"})
        ctk.CTkLabel(
            section.body,
            text="One or more clean images of the skull/icon -- e.g. one per colour variant "
            "(White, Gold, ...). The background is removed automatically wherever it's used for matching.",
            text_color=self.colors["text_secondary"], font=("Segoe UI", 11), wraplength=520, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self.reference_list_frame = ctk.CTkFrame(section.body, fg_color="transparent")
        self.reference_list_frame.pack(fill="x")
        self._reference_thumb_images: List["object"] = []

        button_row = ctk.CTkFrame(section.body, fg_color="transparent")
        button_row.pack(fill="x", pady=(8, 0))
        devtools.dev_button(
            button_row, self.app, self.colors, "UPLOAD_REFERENCE_IMAGE", "CALIBRATE", "SKULL_REFERENCE",
            text="UPLOAD REFERENCE IMAGE", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=self._on_upload_reference_image, pack_opts={"side": "left"},
        )
        devtools.dev_button(
            button_row, self.app, self.colors, "OPEN_REFERENCE_FOLDER", "CALIBRATE", "SKULL_REFERENCE",
            text="OPEN REFERENCE FOLDER", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"], command=lambda: _open_in_explorer(calibration_store.REFERENCES_DIR),
            pack_opts={"side": "left", "padx": (8, 0)},
        )

    def _on_upload_reference_image(self) -> None:
        if self._refuse_if_busy():
            return
        path = filedialog.askopenfilename(title="Select a skull reference image.", filetypes=REFERENCE_IMAGE_FILETYPES)
        if not path:
            return
        if cv2.imread(path, cv2.IMREAD_GRAYSCALE) is None:
            dialogs.show_error(
                self.app.root, self.colors, "Image Could Not Be Used",
                f"'{os.path.basename(path)}' could not be read as an image.",
                why=["The file may be corrupted or in an unsupported format."],
                what_to_do="Try a different PNG, JPG, JPEG, or WEBP file.",
            )
            return
        # Copied into the permanent library immediately (like TEST FOOTAGE) -- the original
        # upload is never modified; background removal happens only when a reference is
        # actually used for matching (crop_to_content()), never baked into the stored file.
        if not self._file_action("add the reference image", lambda: calibration_store.add_reference_image(path)):
            return
        self._library_changed = True
        self._last_auto_detect = None
        self._last_test_results = None
        self._last_gold_test = None
        self._refresh_all_statuses()

    def _on_remove_reference_image(self, filename: str) -> None:
        if self._refuse_if_busy():
            return
        if not self._file_action("remove the reference image", lambda: calibration_store.remove_reference_image(filename)):
            return
        self._drop_reference_from_session(filename)
        self._library_changed = True
        self._last_auto_detect = None
        self._last_test_results = None
        self._last_gold_test = None
        self._refresh_all_statuses()

    def _on_replace_reference_image(self, filename: str) -> None:
        if self._refuse_if_busy():
            return
        path = filedialog.askopenfilename(title="Select a replacement reference image.", filetypes=REFERENCE_IMAGE_FILETYPES)
        if not path:
            return
        if not self._file_action("replace the reference image", lambda: calibration_store.replace_reference_image(filename, path)):
            return
        self._drop_reference_from_session(filename)
        self._library_changed = True
        self._last_auto_detect = None
        self._last_test_results = None
        self._last_gold_test = None
        self._refresh_all_statuses()

    def _refresh_reference_list(self) -> None:
        for child in self.reference_list_frame.winfo_children():
            child.destroy()
        self._reference_thumb_images = []
        filenames = calibration_store.list_reference_images()
        if not filenames:
            ctk.CTkLabel(
                self.reference_list_frame, text="No reference images yet.", text_color=self.colors["text_secondary"]
            ).pack(anchor="w")
            return
        for filename in filenames:
            row = ctk.CTkFrame(self.reference_list_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            # The thumbnail shows the reference's REAL colour (background removed, skull colour
            # untouched) -- never a grayscale rendering, which is what turned a gold skull white.
            profile = calibration_store.load_reference_profile(filename)
            if profile is not None:
                rgb = cv2.cvtColor(profile.bgr, cv2.COLOR_BGR2RGB)
                thumb_scale = REFERENCE_THUMB_SIZE / max(rgb.shape[0], rgb.shape[1])
                disp_w, disp_h = max(1, round(rgb.shape[1] * thumb_scale)), max(1, round(rgb.shape[0] * thumb_scale))
                resample = Image.NEAREST if thumb_scale > 1.0 else Image.LANCZOS
                pil_image = Image.fromarray(rgb).resize((disp_w, disp_h), resample)
                # ctk.CTkImage, not a raw ImageTk.PhotoImage -- a bare PhotoImage on a CTkLabel
                # risks a real crash ("image pyimageNNN doesn't exist") under reconfiguration.
                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(disp_w, disp_h))
                self._reference_thumb_images.append(ctk_image)
                ctk.CTkLabel(row, text="", image=ctk_image, width=REFERENCE_THUMB_SIZE, height=REFERENCE_THUMB_SIZE).pack(side="left")
            text_frame = ctk.CTkFrame(row, fg_color="transparent")
            text_frame.pack(side="left", padx=(8, 0), fill="x", expand=True)
            ctk.CTkLabel(text_frame, text=f"✓ {filename}", text_color=self.colors["success"], anchor="w").pack(anchor="w")
            if profile is not None:
                ctk.CTkLabel(
                    text_frame, text=_reference_colour_text(profile), text_color=self.colors["text_secondary"],
                    font=("Segoe UI", 11), anchor="w",
                ).pack(anchor="w")
                devtools.field_label(
                    text_frame, self.app, self.colors, "REFERENCE_PROFILE", lambda p=profile: _reference_profile_dev_text(p),
                    pack_opts={"anchor": "w"},
                )
            ctk.CTkButton(
                row, text="REMOVE", width=70, fg_color=self.colors["panel_alt"], hover_color=self.colors["danger"],
                text_color=self.colors["text_primary"], command=lambda f=filename: self._on_remove_reference_image(f),
            ).pack(side="right", padx=(4, 0))
            ctk.CTkButton(
                row, text="REPLACE", width=80, fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
                text_color=self.colors["text_primary"], command=lambda f=filename: self._on_replace_reference_image(f),
            ).pack(side="right", padx=(4, 0))

    # ==================================================================
    # 2. TEST FOOTAGE -- the permanent clip library.
    # ==================================================================
    def _build_footage_section(self, parent) -> None:
        section = CollapsibleSection(parent, self.colors, "TEST FOOTAGE", self._on_section_toggle)
        self.footage_section = self._register_section(section)
        devtools.section_label(section.body, self.app, self.colors, "TEST_FOOTAGE", "CALIBRATE", pack_opts={"anchor": "w"})
        ctk.CTkLabel(
            section.body,
            text="1-3 short (5-10s) clips, each showing one skull appearance with a green box "
            "drawn around it while it's visible. Clips are copied here permanently -- add them once.",
            text_color=self.colors["text_secondary"], font=("Segoe UI", 11), wraplength=520, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self.footage_list_frame = ctk.CTkFrame(section.body, fg_color="transparent")
        self.footage_list_frame.pack(fill="x")

        button_row = ctk.CTkFrame(section.body, fg_color="transparent")
        button_row.pack(fill="x", pady=(8, 0))
        devtools.dev_button(
            button_row, self.app, self.colors, "ADD_TEST_VIDEO", "CALIBRATE", "TEST_FOOTAGE",
            text="ADD TEST VIDEO", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=self._on_add_test_video, pack_opts={"side": "left"},
        )
        devtools.dev_button(
            button_row, self.app, self.colors, "OPEN_TEST_FOOTAGE_FOLDER", "CALIBRATE", "TEST_FOOTAGE",
            text="OPEN TEST FOOTAGE FOLDER", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"], command=lambda: _open_in_explorer(calibration_store.TEST_FOOTAGE_DIR),
            pack_opts={"side": "left", "padx": (8, 0)},
        )

    def _on_add_test_video(self) -> None:
        if self._refuse_if_busy():
            return
        path = filedialog.askopenfilename(title="Select a test footage clip.", filetypes=VIDEO_FILETYPES)
        if not path:
            return
        if not self._file_action("add the test video", lambda: calibration_store.add_test_footage(path)):
            return
        self._refresh_all_statuses()

    def _on_remove_test_video(self, filename: str) -> None:
        if self._refuse_if_busy():
            return
        # Release the preview capture FIRST if it's currently holding this exact file open --
        # Windows refuses to delete a file that's still open (confirmed directly: removing the
        # currently-previewed clip raised a real PermissionError before this ordering fix).
        if self.test_session.video_path and os.path.basename(self.test_session.video_path) == filename:
            self.release_preview()
        if not self._file_action("remove the test video", lambda: calibration_store.remove_test_footage(filename)):
            self._refresh_all_statuses()
            return
        self._refresh_all_statuses()

    def _on_replace_test_video(self, filename: str) -> None:
        if self._refuse_if_busy():
            return
        path = filedialog.askopenfilename(title="Select a replacement clip.", filetypes=VIDEO_FILETYPES)
        if not path:
            return
        # Same file-locking concern as _on_remove_test_video -- release the preview capture
        # first if it's holding the exact file being replaced.
        if self.test_session.video_path and os.path.basename(self.test_session.video_path) == filename:
            self.release_preview()
        self._file_action("replace the test video", lambda: calibration_store.replace_test_footage(filename, path))
        self._refresh_all_statuses()

    def _refresh_footage_list(self) -> None:
        for child in self.footage_list_frame.winfo_children():
            child.destroy()
        footage = calibration_store.list_test_footage()
        if not footage:
            ctk.CTkLabel(
                self.footage_list_frame, text="No test videos yet.", text_color=self.colors["text_secondary"]
            ).pack(anchor="w")
            return
        for filename in footage:
            row = ctk.CTkFrame(self.footage_list_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"✓ {filename}", text_color=self.colors["success"], anchor="w").pack(
                side="left", fill="x", expand=True
            )
            ctk.CTkButton(
                row, text="REMOVE", width=70, fg_color=self.colors["panel_alt"], hover_color=self.colors["danger"],
                text_color=self.colors["text_primary"], command=lambda f=filename: self._on_remove_test_video(f),
            ).pack(side="right", padx=(4, 0))
            ctk.CTkButton(
                row, text="REPLACE", width=80, fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
                text_color=self.colors["text_primary"], command=lambda f=filename: self._on_replace_test_video(f),
            ).pack(side="right", padx=(4, 0))

        # Keep the SEARCH AREA preview dropdown's options in sync with the library.
        if hasattr(self, "preview_clip_menu"):
            current = self.preview_clip_var.get()
            self.preview_clip_menu.configure(values=footage or [""])
            if not footage:
                self.preview_clip_var.set("")
                if self.test_session.is_loaded:
                    self.test_session.release()
                    self.test_session.video_path = None
                    self.test_session.is_loaded = False
                    self._update_footage_preview_visibility()
            elif current not in footage:
                self.preview_clip_var.set(footage[0])
                self._load_preview_clip(footage[0])
            elif not self.test_session.is_loaded:
                self._load_preview_clip(current)

    # ==================================================================
    # 3. SEARCH AREA
    # ==================================================================
    def _build_search_area_section(self, parent) -> None:
        section = CollapsibleSection(parent, self.colors, "SEARCH AREA", self._on_section_toggle)
        self.search_area_section = self._register_section(section)
        devtools.section_label(section.body, self.app, self.colors, "SEARCH_AREA", "CALIBRATE", pack_opts={"anchor": "w"})

        picker_row = ctk.CTkFrame(section.body, fg_color="transparent")
        picker_row.pack(fill="x", pady=(4, 6))
        ctk.CTkLabel(picker_row, text="Preview clip:", text_color=self.colors["text_secondary"]).pack(side="left")
        self.preview_clip_var = tk.StringVar(value="")
        self.preview_clip_menu = ctk.CTkOptionMenu(
            picker_row, variable=self.preview_clip_var, values=[], command=self._load_preview_clip,
            fg_color=self.colors["panel_alt"], button_color=self.colors["accent"], button_hover_color=self.colors["accent_hover"],
        )
        self.preview_clip_menu.pack(side="left", padx=(8, 0))

        self.no_footage_frame = ctk.CTkFrame(section.body, fg_color=self.colors["panel_alt"], corner_radius=10)
        ctk.CTkLabel(
            self.no_footage_frame, text="Add a test video above to preview it here.", text_color=self.colors["text_secondary"]
        ).pack(pady=20)

        self.canvas = tk.Canvas(
            section.body, width=MAX_CANVAS_WIDTH, height=MAX_CANVAS_HEIGHT, bg=self.colors["panel_alt"], highlightthickness=0,
        )
        self.scrubber = VideoScrubber(section.body, self.colors, on_frame_changed=self._on_frame_changed)

        button_row = ctk.CTkFrame(section.body, fg_color="transparent")
        button_row.pack(fill="x", pady=(8, 4))
        self._heavy_buttons.append(devtools.dev_button(
            button_row, self.app, self.colors, "AUTO_DETECT_SEARCH_AREA", "CALIBRATE", "SEARCH_AREA",
            text="AUTO-DETECT FROM TEST FOOTAGE", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=self._on_auto_detect, pack_opts={"side": "left"},
        ))
        devtools.dev_button(
            button_row, self.app, self.colors, "FIND_SKULL_POSITION", "CALIBRATE", "SEARCH_AREA",
            text="FIND SKULL POSITION", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"], command=self._on_find_skull_position, pack_opts={"side": "left", "padx": (8, 0)},
        )
        devtools.dev_button(
            button_row, self.app, self.colors, "SELECT_SEARCH_AREA", "CALIBRATE", "SEARCH_AREA",
            text="SELECT SEARCH AREA", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"], command=self._on_select_area_clicked, pack_opts={"side": "left", "padx": (8, 0)},
        )

        self.search_area_result_label = ctk.CTkLabel(
            section.body, text="", font=("Segoe UI", 11), text_color=self.colors["text_secondary"], wraplength=520, justify="left",
        )
        self.search_area_result_label.pack(anchor="w", pady=(4, 0))

        self._update_footage_preview_visibility()

    def _load_preview_clip(self, filename: str) -> None:
        if not filename:
            return
        self.preview_clip_var.set(filename)
        path = calibration_store.test_footage_path(filename)
        try:
            self.test_session.load_video(path)
        except VideoOpenError as exc:
            dialogs.show_error(
                self.app.root, self.colors, "TEST VIDEO COULD NOT BE OPENED",
                f"'{filename}' could not be opened.", why=[str(exc)],
                what_to_do="Try REPLACE in TEST FOOTAGE, or remove and re-add this clip.",
            )
            return
        self.scrubber.attach_session(self.test_session)
        self._update_footage_preview_visibility()
        self._render_canvas()

    def _update_footage_preview_visibility(self) -> None:
        if not self.test_session.is_loaded:
            self.no_footage_frame.pack(fill="x", pady=(0, 6))
            self.canvas.pack_forget()
            self.scrubber.pack_forget()
            return
        self.no_footage_frame.pack_forget()
        if self.canvas.winfo_manager() != "pack":
            self.canvas.pack(pady=(0, 6))
        self.scrubber.pack_forget()
        self.scrubber.pack(fill="x")

    def _on_frame_changed(self, frame_number: int, frame_bgr) -> None:
        self._render_canvas()

    def _render_canvas(self) -> None:
        self.canvas.delete("all")
        frame = self.test_session.current_image
        if frame is None:
            self.canvas.create_text(
                MAX_CANVAS_WIDTH // 2, MAX_CANVAS_HEIGHT // 2, text="No frame loaded", fill=self.colors["text_secondary"]
            )
            return
        frame_h, frame_w = frame.shape[:2]
        self.canvas_scale = min(MAX_CANVAS_WIDTH / frame_w, MAX_CANVAS_HEIGHT / frame_h, 1.0)
        display_w, display_h = int(frame_w * self.canvas_scale), int(frame_h * self.canvas_scale)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb).resize((display_w, display_h))
        self._tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.configure(width=display_w, height=display_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

        region = self.app.session.detection_region
        if region is not None:
            cal_w = self.app.session.calibration_video_width or frame_w
            cal_h = self.app.session.calibration_video_height or frame_h
            shown_region, _ = scaled_region(region, cal_w, cal_h, frame_w, frame_h)
            x, y, w, h = shown_region
            sx, sy, sw, sh = x * self.canvas_scale, y * self.canvas_scale, w * self.canvas_scale, h * self.canvas_scale
            self.canvas.create_rectangle(sx, sy, sx + sw, sy + sh, outline=self.colors["accent"], width=2)
            self.canvas.create_text(
                sx + 4, max(8, sy - 10), text="SEARCH AREA", anchor="w", fill=self.colors["accent"], font=("Segoe UI", 10, "bold"),
            )

        # TEST GOLD SKULL's candidate overlay -- only while genuinely still looking at the exact
        # clip/frame it was found on, so scrubbing away never leaves a stale, misleading box
        # (the same lesson from an earlier pass's test-overlay bug).
        gold_test = self._last_gold_test
        if (
            gold_test is not None and gold_test.best_bbox is not None
            and self.preview_clip_var.get() == gold_test.clip_filename
            and self.test_session.current_frame == gold_test.best_frame_index
        ):
            gx, gy, gw, gh = gold_test.best_bbox
            gsx, gsy, gsw, gsh = gx * self.canvas_scale, gy * self.canvas_scale, gw * self.canvas_scale, gh * self.canvas_scale
            gold_color = "#E8B923"
            self.canvas.create_rectangle(gsx, gsy, gsx + gsw, gsy + gsh, outline=gold_color, width=2)
            label = "GOLD CANDIDATE" if not gold_test.detected else "GOLD DETECTED"
            self.canvas.create_text(
                gsx + 4, min(display_h - 10, gsy + gsh + 12), text=label, anchor="w", fill=gold_color, font=("Segoe UI", 10, "bold"),
            )

    # -- SELECT SEARCH AREA: a simple fresh-drag rectangle (no resize handles -- the region means
    # "search only here", not an exact position, so pixel-perfect editing isn't needed). --
    def _on_select_area_clicked(self) -> None:
        if self._refuse_if_busy():
            return
        if not self.test_session.is_loaded:
            self.search_area_result_label.configure(
                text="Add and select a test video above first.", text_color=self.colors["danger"]
            )
            return
        self._selecting = True
        self.search_area_result_label.configure(text="Drag a rectangle on the preview above.", text_color=self.colors["text_secondary"])
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

    def _on_canvas_press(self, event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_rect_id = None

    def _on_canvas_drag(self, event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline=self.colors["accent"], width=2)

    def _on_canvas_release(self, event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None
        self._selecting = False
        for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            self.canvas.unbind(seq)

        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        frame_w = int((x_max - x_min) / self.canvas_scale)
        frame_h = int((y_max - y_min) / self.canvas_scale)
        if frame_w < MIN_REGION_SIZE or frame_h < MIN_REGION_SIZE:
            self.search_area_result_label.configure(text="SEARCH AREA TOO SMALL -- try again.", text_color=self.colors["danger"])
            self._render_canvas()
            return

        frame = self.test_session.current_image
        video_h, video_w = frame.shape[:2]
        region = _clamp_region(
            (int(x_min / self.canvas_scale), int(y_min / self.canvas_scale), frame_w, frame_h), video_w, video_h
        )
        self.app.session.set_detection_region(region, video_w, video_h)
        self.search_area_result_label.configure(text=f"✓ Search area set manually: {region[2]}x{region[3]}px", text_color=self.colors["success"])
        self._render_canvas()
        self._refresh_all_statuses()

    # -- FIND SKULL POSITION: single-frame fallback using the reference image directly. --
    def _on_find_skull_position(self) -> None:
        self._run_guarded(self._find_skull_position_impl, self.search_area_result_label)

    def _find_skull_position_impl(self) -> None:
        if not self.test_session.is_loaded:
            raise CalibrationError("Add and select a test video above first.")
        reference_filenames = calibration_store.list_reference_images()
        if not reference_filenames:
            raise CalibrationError("Upload at least one skull reference image first.")
        frame = self.test_session.current_image
        if frame is None:
            raise CalibrationError("No frame is currently loaded to search.")

        # Try every calibrated reference against this one frame -- whichever matches best sets
        # the region; every reference still gets its own scale recorded from this same frame.
        candidates = []  # (label, cropped_image, result_or_None, colour_profile)
        for filename in reference_filenames:
            raw_image = calibration_store.load_reference_gray(filename)
            if raw_image is None:
                continue
            label = os.path.splitext(filename)[0]
            cropped = crop_to_content(raw_image)
            candidates.append((
                label, cropped, locate_reference_in_frame(frame, cropped), calibration_store.load_reference_profile(filename),
            ))

        best_label, best_result = None, None
        for label, _, result, _profile in candidates:
            if result is not None and (best_result is None or result.confidence > best_result.confidence):
                best_label, best_result = label, result

        if best_result is None or best_result.confidence < 0.55:
            best_pct = f"{best_result.confidence * 100:.0f}%" if best_result else "0%"
            self.search_area_result_label.configure(
                text=f"SKULL POSITION NOT FOUND -- best match only {best_pct}. Try a different frame.",
                text_color=self.colors["danger"],
            )
            return

        frame_h, frame_w = frame.shape[:2]
        region = expand_to_search_region(best_result.bbox, frame_w, frame_h)
        self.app.session.set_detection_region(region, frame_w, frame_h)
        self.app.session.reference_images = [
            ReferenceImage(label=label, image=cropped, scale=(result.scale if result is not None else 1.0), profile=profile)
            for label, cropped, result, profile in candidates
        ]
        self.search_area_result_label.configure(
            text=f"✓ SKULL POSITION FOUND ({best_label}) -- confidence {best_result.confidence * 100:.0f}%. Search region: Ready",
            text_color=self.colors["success"],
        )
        self._render_canvas()

    # -- AUTO-DETECT FROM TEST FOOTAGE: the main automatic path, combining every library clip. --
    def _on_auto_detect(self) -> None:
        self._run_background(
            self.search_area_result_label, "Analysing the test footage",
            self._auto_detect_compute, self._auto_detect_apply,
        )

    def _analyse_single_clip(self, path: str) -> _ClipAnalysis:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise CalibrationError(f"'{os.path.basename(path)}' could not be opened for analysis.")
        per_frame_candidates: List[List[Tuple[int, int, int, int]]] = []
        frame_w = frame_h = 0
        all_frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_h, frame_w = frame.shape[:2]
                per_frame_candidates.append(find_green_box_candidates(frame))
                all_frames.append(frame)
        finally:
            capture.release()

        frames_analysed = len(per_frame_candidates)
        spans = find_training_spans(per_frame_candidates, (frame_w, frame_h))
        if not spans:
            return _ClipAnalysis(frames_analysed, frame_w, frame_h, 0, None, {})
        span = spans[0]  # already sorted longest (most genuine) first
        frame_cache = {i: all_frames[i] for i in range(span.start_frame, span.end_frame + 1)}
        return _ClipAnalysis(frames_analysed, frame_w, frame_h, len(spans), span.union_bbox(), frame_cache)

    def _auto_detect_compute(self) -> "_AutoDetectOutcome":
        """Background step: no widget access."""
        footage = calibration_store.list_test_footage()
        if not footage:
            raise CalibrationError("No test footage in the library yet. Add a clip in TEST FOOTAGE first.")
        reference_filenames = calibration_store.list_reference_images()
        if not reference_filenames:
            raise CalibrationError("Upload at least one skull reference image first.")

        clip_analyses = {filename: self._analyse_single_clip(calibration_store.test_footage_path(filename)) for filename in footage}

        canonical_w = canonical_h = 0
        combined_bbox = None
        used, skipped = [], []
        for filename in footage:
            analysis = clip_analyses[filename]
            if analysis.raw_bbox is None:
                skipped.append(filename)
                continue
            if canonical_w == 0:
                canonical_w, canonical_h = analysis.frame_w, analysis.frame_h
            if (analysis.frame_w, analysis.frame_h) != (canonical_w, canonical_h):
                skipped.append(filename)  # a differently-sized clip can't be combined geometrically
                continue
            combined_bbox = analysis.raw_bbox if combined_bbox is None else _union_bbox(combined_bbox, analysis.raw_bbox)
            used.append(filename)

        if combined_bbox is None:
            raise CalibrationError(
                "No green training box was detected in any test footage clip. Make sure each clip "
                "contains a clearly green rectangle drawn around the skull while it's visible."
            )

        search_region = expand_to_search_region(
            combined_bbox, canonical_w, canonical_h,
            width_margin=TRAINING_REGION_WIDTH_MARGIN, height_margin=TRAINING_REGION_HEIGHT_MARGIN,
        )

        # Every calibrated reference is matched against every contributing clip's cached frames --
        # not paired by filename (real filenames won't always correlate), so each reference finds
        # its own best-matching clip/scale on its own merits.
        reference_results: List[ReferenceMatchResult] = []
        new_reference_images: List[ReferenceImage] = []
        for ref_filename in reference_filenames:
            raw_image = calibration_store.load_reference_gray(ref_filename)
            if raw_image is None:
                continue
            label = os.path.splitext(ref_filename)[0]
            cropped_reference = crop_to_content(raw_image)
            best_confidence, best_scale = 0.0, 1.0
            for clip_filename in used:
                for frame in clip_analyses[clip_filename].frame_cache.values():
                    cropped_frame = _apply_region(frame, search_region)
                    match = locate_reference_in_frame(cropped_frame, cropped_reference)
                    if match is not None and match.confidence > best_confidence:
                        best_confidence, best_scale = match.confidence, match.scale
            passed = best_confidence >= MIN_REFERENCE_MATCH_CONFIDENCE
            reference_results.append(ReferenceMatchResult(ref_filename, label, best_confidence, best_scale, passed))
            profile = calibration_store.load_reference_profile(ref_filename)
            footage = None
            if profile is not None and profile.signature.chromatic:
                # Learn HOW this coloured skull really appears in the user's own footage (its real
                # size and colour), not just what the reference image says.
                region_frames = [
                    _apply_region(frame, search_region)
                    for clip_filename in used for frame in clip_analyses[clip_filename].frame_cache.values()
                ]
                footage = learn_footage_appearance(profile, region_frames)
            new_reference_images.append(ReferenceImage(
                label=label, image=cropped_reference, scale=best_scale if passed else 1.0,
                profile=profile, footage=footage,
            ))

        skull_detected = any(r.passed for r in reference_results)
        result = AutoDetectResult(
            clips_used=used, clips_skipped=skipped, search_region=search_region,
            calibration_video_width=canonical_w, calibration_video_height=canonical_h,
            reference_results=reference_results, skull_detected=skull_detected,
        )

        lines = ["✓ SEARCH AREA DETECTED" if skull_detected else "SEARCH AREA FOUND, but no reference matched well enough."]
        if used:
            lines.append(f"Learned from: {', '.join(used)}")
        if skipped:
            lines.append(f"Skipped (no green box or different resolution): {', '.join(skipped)}")
        for r in reference_results:
            lines.append(f"{'✓' if r.passed else '⚠'} {r.label}: confidence {r.confidence * 100:.0f}%")
        for ref in new_reference_images:
            if ref.footage is not None:
                lines.append(
                    f"  {ref.label}: learned from your footage -- size x{ref.footage.scale:.2f} of the reference, "
                    f"colour Lab({ref.footage.lab_l:.0f},{ref.footage.lab_a:.0f},{ref.footage.lab_b:.0f})"
                )
        return _AutoDetectOutcome(result=result, reference_images=new_reference_images, lines=lines)

    def _auto_detect_apply(self, outcome: "_AutoDetectOutcome") -> None:
        """Main-thread step: commit the learned calibration to the session and show the result."""
        result = outcome.result
        self._last_auto_detect = result
        if result.skull_detected:
            self._library_changed = False
            # Every calibrated reference is kept (even a weak one, at scale 1.0) so uploading one
            # doesn't silently vanish from the profile -- the weak result above still flags it.
            self.app.session.reference_images = outcome.reference_images
            self.app.session.set_detection_region(
                result.search_region, result.calibration_video_width, result.calibration_video_height,
            )
        self.search_area_result_label.configure(
            text="\n".join(outcome.lines), text_color=self.colors["success"] if result.skull_detected else self.colors["danger"],
        )
        self._render_canvas()

    # ==================================================================
    # 4. CALIBRATION TEST
    # ==================================================================
    def _build_calibration_test_section(self, parent) -> None:
        section = CollapsibleSection(parent, self.colors, "CALIBRATION TEST", self._on_section_toggle)
        self.calibration_test_section = self._register_section(section)
        devtools.section_label(section.body, self.app, self.colors, "CALIBRATION_TEST", "CALIBRATE", pack_opts={"anchor": "w"})

        self._heavy_buttons.append(devtools.dev_button(
            section.body, self.app, self.colors, "RUN_CALIBRATION_TEST", "CALIBRATE", "CALIBRATION_TEST",
            text="RUN CALIBRATION TEST", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=self._on_run_calibration_test, pack_opts={"anchor": "w", "pady": (4, 8)},
        ))
        self.calibration_test_result_label = ctk.CTkLabel(
            section.body, text="", font=("Segoe UI", 11), text_color=self.colors["text_secondary"], wraplength=520, justify="left",
        )
        self.calibration_test_result_label.pack(anchor="w")

        ctk.CTkFrame(section.body, fg_color=self.colors["panel_alt"], height=1).pack(fill="x", pady=10)
        ctk.CTkLabel(
            section.body,
            text="Targeted diagnostic: tests only the clip/reference whose filenames contain "
            "\"gold\" -- useful while getting one specific colour variant working.",
            text_color=self.colors["text_secondary"], font=("Segoe UI", 11), wraplength=520, justify="left",
        ).pack(anchor="w", pady=(0, 6))
        self._heavy_buttons.append(devtools.dev_button(
            section.body, self.app, self.colors, "TEST_GOLD_SKULL", "CALIBRATE", "CALIBRATION_TEST",
            text="TEST GOLD SKULL", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
            text_color=self.colors["text_primary"], command=self._on_run_gold_skull_test, pack_opts={"anchor": "w", "pady": (0, 8)},
        ))
        self.gold_test_result_label = ctk.CTkLabel(
            section.body, text="", font=("Segoe UI", 11), text_color=self.colors["text_secondary"], wraplength=520, justify="left",
        )
        self.gold_test_result_label.pack(anchor="w")

    def _on_run_calibration_test(self) -> None:
        self._run_background(
            self.calibration_test_result_label, "Testing every clip",
            self._run_calibration_test_compute, self._run_calibration_test_apply,
        )

    def _on_run_gold_skull_test(self) -> None:
        self._run_background(
            self.gold_test_result_label, "Testing the gold skull",
            self._run_gold_skull_test_compute, self._run_gold_skull_test_apply,
        )

    def _run_temporal_confirmation(self, path: str, reference_templates: List[Tuple[str, "object"]], colour_reference=None) -> Optional[int]:
        """Runs the REAL, unchanged event-grouping engine (video/video_scanner.scan(), the exact
        function real ANALYSE uses) against a short calibration clip as a read-only diagnostic --
        confirms a continuous appearance genuinely collapses into ONE event under the user's own
        saved settings, rather than assuming it from a bare per-frame confidence count. Returns
        the resulting event count, or None if the check itself couldn't run (never raises --
        this is a bonus confirmation, not a requirement for the rest of the test to report)."""
        try:
            metadata = probe_video(path)
            region, _ = scaled_region(
                self.app.session.detection_region, self.app.session.calibration_video_width,
                self.app.session.calibration_video_height, metadata.width, metadata.height,
            )
            events = scan_video(path, metadata, region, reference_templates, self.app.config, colour_reference=colour_reference)
            return len(events)
        except Exception:  # noqa: BLE001 -- a diagnostic bonus, never allowed to break the test
            return None

    def _run_calibration_test_compute(self) -> "_CalibrationTestOutcome":
        """Background step: no widget access."""
        session = self.app.session
        if session.detection_region is None or not session.reference_images:
            raise CalibrationError("Run AUTO-DETECT FROM TEST FOOTAGE (or FIND SKULL POSITION / SELECT SEARCH AREA) first.")
        footage = calibration_store.list_test_footage()
        if not footage:
            raise CalibrationError("No test footage in the library yet.")

        # One multi-template detector understanding every calibrated appearance (White + Gold,
        # ...) -- never separate detectors per colour -- but still run independently against
        # each test clip (spec: "do not assume that because white works, gold works").
        templates = session.scaled_reference_templates()

        results: List[ClipTestResult] = []
        for filename in footage:
            clip_path = calibration_store.test_footage_path(filename)
            analysis = self._analyse_single_clip(clip_path)
            if not analysis.frame_cache:
                results.append(ClipTestResult(filename, 0.0, False, note="no skull event found in this clip"))
                continue
            region, _ = scaled_region(
                session.detection_region, session.calibration_video_width,
                session.calibration_video_height, analysis.frame_w, analysis.frame_h,
            )
            # The same combined detector real ANALYSE uses (grayscale templates + the colour-aware
            # matcher), sized for THIS clip's resolution -- so a pass here means a pass there.
            colour_reference = session.colour_reference_set(analysis.frame_h)
            detector = build_detector(templates, colour_reference)
            best_confidence, best_symbol = 0.0, ""
            for frame in analysis.frame_cache.values():
                cropped = _apply_region(frame, region)
                matches = detector.find_matches(cropped, threshold=0.3)
                if matches:
                    best = max(matches, key=lambda m: m.confidence)
                    if best.confidence > best_confidence:
                        best_confidence, best_symbol = best.confidence, best.symbol_name
            temporal_event_count = self._run_temporal_confirmation(clip_path, templates, colour_reference)
            results.append(ClipTestResult(
                filename, best_confidence, best_confidence >= MIN_REFERENCE_MATCH_CONFIDENCE, detected_as=best_symbol,
                temporal_event_count=temporal_event_count,
            ))

        lines = ["CALIBRATION TEST", ""]
        for r in results:
            if r.note:
                lines.append(f"{r.filename}:\n⚠ {r.note}")
            elif r.passed:
                lines.append(f"{r.filename}:\n✓ Detected as {r.detected_as} -- confidence {r.confidence * 100:.0f}%")
            else:
                lines.append(f"{r.filename}:\n⚠ DETECTION FAILED -- best match only {r.confidence * 100:.0f}%")
            if r.temporal_event_count is not None:
                temporal_mark = "✓" if r.temporal_event_count == 1 else "⚠"
                lines.append(f"  Temporal detection: {temporal_mark} {r.temporal_event_count} event(s) under your real analysis settings")
        overall_passed = all(r.passed for r in results)
        lines += ["", f"Overall: {'✓ CALIBRATION PASSED' if overall_passed else '✗ CALIBRATION FAILED'}"]
        return _CalibrationTestOutcome(results=results, lines=lines, passed=overall_passed)

    def _run_calibration_test_apply(self, outcome: "_CalibrationTestOutcome") -> None:
        self._last_test_results = outcome.results
        self.calibration_test_result_label.configure(
            text="\n".join(outcome.lines), text_color=self.colors["success"] if outcome.passed else self.colors["danger"],
        )

    def _run_gold_skull_test_compute(self) -> "GoldTestResult":
        """TEST GOLD SKULL: an isolated, single-colour diagnostic -- runs only against whichever
        library clip/reference have "gold" in their filename, so gold detection can be proven
        (or diagnosed) on its own without touching white detection, the calibration layout, or
        the real Analysis/Resolve systems at all."""
        footage = calibration_store.list_test_footage()
        gold_clip = _find_by_keyword(footage, GOLD_KEYWORD)
        if gold_clip is None:
            raise CalibrationError(
                'No test video with "gold" in its name found in TEST FOOTAGE. Add your Gold Test '
                "video there first."
            )
        reference_filenames = calibration_store.list_reference_images()
        gold_reference_filename = _find_by_keyword(reference_filenames, GOLD_KEYWORD)
        if gold_reference_filename is None:
            raise CalibrationError(
                'No reference image with "gold" in its name found in SKULL REFERENCE. Click '
                "UPLOAD REFERENCE IMAGE and add your gold skull image there first (name it "
                'something containing "gold").'
            )

        raw_reference = calibration_store.load_reference_gray(gold_reference_filename)
        if raw_reference is None:
            raise CalibrationError(f"'{gold_reference_filename}' could not be read as an image.")
        cropped_reference = crop_to_content(raw_reference)

        path = calibration_store.test_footage_path(gold_clip)
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise CalibrationError(f"'{gold_clip}' could not be opened for analysis.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List["object"] = []
        per_frame_candidates: List[List[Tuple[int, int, int, int]]] = []
        frame_w = frame_h = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_h, frame_w = frame.shape[:2]
                frames.append(frame)
                per_frame_candidates.append(find_green_box_candidates(frame))
        finally:
            capture.release()
        frames_analysed = len(frames)
        if frames_analysed == 0:
            raise CalibrationError(f"'{gold_clip}' could not be processed -- no frames could be read from it.")

        # The green box is ground truth for "the gold skull IS visible here" (spec item 9) --
        # confirm it was actually found, and use it to establish a region if none is calibrated
        # yet. When a region IS already saved, that one is authoritative (spec item 13: "the gold
        # detector should ONLY search: Saved Search Region"); the green box then just confirms
        # this clip genuinely has a training appearance to test against.
        spans = find_training_spans(per_frame_candidates, (frame_w, frame_h))
        green_box_confirmed = bool(spans)
        session = self.app.session
        if session.detection_region is not None:
            region, _ = scaled_region(
                session.detection_region, session.calibration_video_width,
                session.calibration_video_height, frame_w, frame_h,
            )
        elif spans:
            region = expand_to_search_region(
                spans[0].union_bbox(), frame_w, frame_h,
                width_margin=TRAINING_REGION_WIDTH_MARGIN, height_margin=TRAINING_REGION_HEIGHT_MARGIN,
            )
        else:
            raise CalibrationError(
                "No calibrated search region is saved yet, and no green training box was found in "
                "the Gold Test video to establish one from -- run AUTO-DETECT FROM TEST FOOTAGE "
                "first, or check that the clip's green box is clearly visible."
            )

        cropped_frames = [_apply_region(frame, region) for frame in frames]

        # Per-representation breakdown (Developer Mode only -- a real diagnostic, not exposed as
        # a user setting, and skipped by default since it's a genuinely more expensive pass).
        mode_confidences: Dict[str, float] = {}
        best_mode = ""
        if devtools.is_enabled(self.app):
            mode_confidences = mode_confidence_breakdown(cropped_reference, cropped_frames)
            if mode_confidences:
                best_mode = max(mode_confidences, key=mode_confidences.get)

        # The real pass/fail signal: the actual production SkullDetector (every preprocessing
        # mode combined, exactly like real ANALYSE), evaluated frame by frame. A low search floor
        # only surfaces a candidate to report/display -- the PASS/FAIL verdict below always
        # compares against the real configured threshold, never the floor (spec item 12).
        # ...built exactly the way real ANALYSE builds it: the grayscale template detector plus the
        # colour-aware matcher, in one combined detector, so this diagnostic can never pass on a
        # path real analysis doesn't use.
        gold_label = os.path.splitext(gold_reference_filename)[0]
        saved_scale = next((r.scale for r in session.reference_images if r.label == gold_label), 1.0)
        colour_reference = build_colour_reference_set(
            [ReferenceImage(
                label=gold_label, image=cropped_reference, scale=saved_scale,
                profile=calibration_store.load_reference_profile(gold_reference_filename),
            )],
            session.calibration_video_height, frame_h,
        )
        detector = build_detector([(gold_label, cropped_reference)], colour_reference)
        threshold = self.app.config.confidence_threshold
        best_confidence, best_frame_index, best_bbox = 0.0, None, None
        first_detection_frame = None
        detection_frame_count = 0
        for idx, cropped in enumerate(cropped_frames):
            matches = detector.find_matches(cropped, threshold=GOLD_TEST_SEARCH_FLOOR)
            if not matches:
                continue
            best_match = max(matches, key=lambda m: m.confidence)
            if best_match.confidence > best_confidence:
                best_confidence = best_match.confidence
                best_frame_index = idx
                best_bbox = (best_match.bbox[0] + region[0], best_match.bbox[1] + region[1], best_match.bbox[2], best_match.bbox[3])
            if best_match.confidence >= threshold:
                detection_frame_count += 1
                if first_detection_frame is None:
                    first_detection_frame = idx

        detected = first_detection_frame is not None
        # Temporal confirmation: proves the real event-grouping engine collapses this ~3s
        # appearance into ONE bookmark under the user's own saved settings (spec: "one continuous
        # appearance = one bookmark"), rather than assuming it from the per-frame count above.
        temporal_event_count = self._run_temporal_confirmation(path, [(gold_label, cropped_reference)], colour_reference) if detected else None
        result = GoldTestResult(
            clip_filename=gold_clip, reference_filename=gold_reference_filename, frames_analysed=frames_analysed,
            fps=fps, green_box_confirmed=green_box_confirmed, detected=detected,
            first_detection_frame=first_detection_frame, best_confidence=best_confidence,
            detection_frame_count=detection_frame_count, best_frame_index=best_frame_index, best_bbox=best_bbox,
            mode_confidences=mode_confidences, best_mode=best_mode, threshold=threshold,
            temporal_event_count=temporal_event_count,
        )
        return result

    def _run_gold_skull_test_apply(self, result: "GoldTestResult") -> None:
        self._last_gold_test = result
        self._show_gold_test_result(result)

        # Jump the SEARCH AREA preview to the gold clip's best candidate frame with its bbox
        # drawn, whether this passed or failed (spec item 4: "automatically show the strongest
        # candidate frame... draw the detector's candidate bounding box").
        if result.best_frame_index is not None:
            if self.preview_clip_var.get() != result.clip_filename:
                self._load_preview_clip(result.clip_filename)
            self.scrubber.set_frame(result.best_frame_index)

    def _show_gold_test_result(self, result: GoldTestResult) -> None:
        def fmt_ts(seconds: Optional[float]) -> str:
            if seconds is None:
                return "--"
            return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"

        lines = ["GOLD SKULL TEST", ""]
        lines.append(f"Frames analysed: {result.frames_analysed}")
        lines.append(f"Gold skull detected: {'YES' if result.detected else 'NO'}")
        if result.detected:
            lines.append(f"First detection: {fmt_ts(result.first_detection_timestamp)}")
        lines.append(f"Best confidence: {result.best_confidence * 100:.0f}%")
        if result.detected:
            lines.append(f"Detection frames: {result.detection_frame_count}")
        if result.temporal_event_count is not None:
            mark = "✓" if result.temporal_event_count == 1 else "⚠"
            lines.append(f"Temporal detection: {mark} {result.temporal_event_count} event(s) under your real analysis settings")
        lines.append(f"Result: {'✓ PASS' if result.detected else '✕ FAILED'}")

        if result.mode_confidences:
            lines += ["", "GOLD DETECTION TEST (per representation)"]
            for mode, confidence in result.mode_confidences.items():
                lines.append(f"{mode}: {confidence * 100:.0f}%")
            lines.append(f"Best method: {result.best_mode} ({result.mode_confidences.get(result.best_mode, 0.0) * 100:.0f}%)")

        if not result.detected:
            lines += ["", "GOLD DETECTION DIAGNOSTIC"]
            lines.append(f"Reference loaded: ✓ ({result.reference_filename})")
            lines.append(f"Gold test loaded: ✓ ({result.clip_filename})")
            lines.append("Search region: ✓")
            lines.append(f"Gold skull visible (green box found): {'✓' if result.green_box_confirmed else '✕ NOT FOUND'}")
            if result.best_frame_index is not None:
                lines.append(f"Best candidate frame: {result.best_frame_index} ({fmt_ts(result.best_timestamp)})")
                lines.append(f"Best candidate confidence: {result.best_confidence * 100:.0f}%")
            if result.best_mode:
                lines.append(f"Best detection method: {result.best_mode}")
            if not result.green_box_confirmed:
                why = "No green training box was found in this clip, so its ground-truth appearance window could not be confirmed."
            elif result.best_frame_index is None:
                why = "No candidate scored above the diagnostic search floor anywhere in the clip -- the reference may not resemble what's on screen at all within the calibrated search region."
            else:
                why = (
                    f"The best match found ({result.best_confidence * 100:.0f}%) is below the "
                    f"configured confidence threshold ({result.threshold * 100:.0f}%)."
                )
            lines.append(f"Why rejected: {why}")

        self.gold_test_result_label.configure(
            text="\n".join(lines), text_color=self.colors["success"] if result.detected else self.colors["danger"],
        )

    # ==================================================================
    # 5. SAVE CALIBRATION
    # ==================================================================
    def _on_save_calibration(self) -> None:
        if self._refuse_if_busy():
            return
        session = self.app.session
        if session.detection_region is None or not session.reference_images:
            return
        # Reference images and test footage are already persisted to calibration/ the moment
        # they're added (library model) -- SAVE CALIBRATION only needs to commit the learned
        # region and each reference's own learned scale.
        if not self._file_action("save the calibration", lambda: calibration_store.save_region(session)):
            return
        self._saved_adjustment = session.bookmark_adjustment_seconds
        self._update_unsaved_marker()
        dialogs.show_info(self.app.root, self.colors, "Calibration Saved", "Calibration saved to the calibration/ folder.")
        self._refresh_all_statuses()
