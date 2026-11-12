"""
resolve/resolve_controller.py
-------------------------------
All communication with DaVinci Resolve goes through this module, using
Blackmagic's official Python scripting API (DaVinciResolveScript). No
keyboard-simulation fallback is used -- if the API isn't reachable, we
raise a clear, specific error instead of faking input.

Resolve's scripting module isn't a pip package -- it ships inside the
Resolve installation itself. See the README's "DaVinci Resolve Scripting
Setup" section for the one-time setup on Windows.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from data.detection import Event

_DEFAULT_RESOLVE_SCRIPT_API = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
_DEFAULT_RESOLVE_SCRIPT_LIB = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"


class ResolveConnectionError(Exception):
    """Raised whenever Resolve, a project, or a timeline can't be reached."""


@dataclass
class ResolveStatus:
    """
    Granular connection state used to drive the ● status dot and to decide
    whether SEND TO RESOLVE is safe to click. `connected` is the single
    boolean the header dot cares about; `detail` is the human-readable
    reason, surfaced verbatim in error dialogs.
    """
    connected: bool
    detail: str
    project_name: Optional[str] = None
    timeline_name: Optional[str] = None


def _ensure_scripting_module_on_path() -> None:
    api_path = os.environ.get("RESOLVE_SCRIPT_API", _DEFAULT_RESOLVE_SCRIPT_API)
    lib_path = os.environ.get("RESOLVE_SCRIPT_LIB", _DEFAULT_RESOLVE_SCRIPT_LIB)
    os.environ.setdefault("RESOLVE_SCRIPT_API", api_path)
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", lib_path)

    modules_path = os.path.join(api_path, "Modules")
    if modules_path not in sys.path and os.path.isdir(modules_path):
        sys.path.append(modules_path)


def get_resolve():
    """Connect to a currently running DaVinci Resolve instance."""
    try:
        import DaVinciResolveScript as dvr_script  # type: ignore
    except ImportError:
        _ensure_scripting_module_on_path()
        try:
            import DaVinciResolveScript as dvr_script  # type: ignore
        except ImportError as exc:
            raise ResolveConnectionError(
                "Could not load DaVinci Resolve's scripting module. Make sure Resolve is "
                "installed and 'External scripting using' is set to Local "
                "(Preferences > System > General). See the README for details."
            ) from exc

    resolve = dvr_script.scriptapp("Resolve")
    if resolve is None:
        raise ResolveConnectionError(
            "DaVinci Resolve does not appear to be running. Please open DaVinci Resolve first."
        )
    return resolve


def get_current_project(resolve):
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject() if project_manager else None
    if project is None:
        raise ResolveConnectionError(
            "No project is currently open in DaVinci Resolve. Please open a project first."
        )
    return project


def get_current_timeline(project):
    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise ResolveConnectionError(
            "No timeline is currently open. Please open (or create) a timeline "
            "containing your gameplay clip first."
        )
    return timeline


def check_connection_status() -> ResolveStatus:
    """
    Full connection check used by the header's ● indicator and by
    SEND TO RESOLVE's pre-flight validation: Resolve running -> project open
    -> timeline open, each step checked explicitly and never raising.
    """
    try:
        resolve = get_resolve()
    except ResolveConnectionError as exc:
        return ResolveStatus(connected=False, detail=str(exc))

    try:
        project = get_current_project(resolve)
    except ResolveConnectionError as exc:
        return ResolveStatus(connected=False, detail=str(exc))

    try:
        timeline = get_current_timeline(project)
    except ResolveConnectionError as exc:
        return ResolveStatus(connected=False, detail=str(exc))

    return ResolveStatus(
        connected=True,
        detail=f"Connected — '{project.GetName()}' / '{timeline.GetName()}'",
        project_name=project.GetName(),
        timeline_name=timeline.GetName(),
    )


def _is_resolve_process_running() -> bool:
    """
    Whether Resolve.exe is running at all, independent of whether its scripting API can be
    reached -- distinguishes "Resolve isn't open" from "Resolve is open but can't be scripted"
    (e.g. the free edition, which has no scripting API regardless of any code here).
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Resolve.exe", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        return "Resolve.exe" in result.stdout
    except Exception:  # noqa: BLE001 - can't verify -> treat as unknown, never crash a diagnostic
        return False


def _load_scripting_module():
    """The DaVinciResolveScript import, isolated from scriptapp() so a diagnostic can tell
    'the module couldn't even be imported' apart from 'it imported but returned nothing'."""
    try:
        import DaVinciResolveScript as dvr_script  # type: ignore
        return dvr_script
    except ImportError:
        _ensure_scripting_module_on_path()
        try:
            import DaVinciResolveScript as dvr_script  # type: ignore
            return dvr_script
        except ImportError as exc:
            raise ResolveConnectionError(
                "Could not load DaVinci Resolve's scripting module (DaVinciResolveScript). "
                "Make sure Resolve is installed and RESOLVE_SCRIPT_API/RESOLVE_SCRIPT_LIB "
                "point to a valid installation."
            ) from exc


@dataclass
class ConnectionStepResult:
    """One row of the TEST RESOLVE CONNECTION checklist -- unlike DiagnosticStep above, this
    breaks 'obtain Resolve' itself into distinct, individually-diagnosable sub-steps."""
    label: str
    passed: bool
    detail: str = ""


@dataclass
class FullDiagnostic:
    steps: List[ConnectionStepResult] = field(default_factory=list)
    connected: bool = False
    project_name: Optional[str] = None
    timeline_name: Optional[str] = None

    @property
    def first_failure(self) -> Optional[ConnectionStepResult]:
        return next((s for s in self.steps if not s.passed), None)


def run_full_diagnostic(on_step: Optional[Callable[[ConnectionStepResult], None]] = None) -> FullDiagnostic:
    """
    RECONNECT TO RESOLVE / TEST RESOLVE CONNECTION: walks the full chain --
        Resolve running -> scripting API loadable -> Resolve instance -> project -> timeline
        -> a harmless read-only bookmark-capability check
    -- stopping at (and clearly reporting) the exact first stage that fails, instead of one
    generic "Resolve connection failed" message. Never raises; every failure mode is reported
    as a normal (non-exceptional) result so the UI can display it directly.

    `on_step`, if given, is called once per stage the moment that stage's result is known
    (not just once at the very end), so a caller running this on a background thread can
    drive a live, progressive "CONNECTING..." display instead of a single dump after the
    whole chain finishes.
    """
    diag = FullDiagnostic()

    def record(step: ConnectionStepResult) -> None:
        diag.steps.append(step)
        if on_step is not None:
            on_step(step)

    process_running = _is_resolve_process_running()
    record(ConnectionStepResult(
        "DaVinci Resolve running", process_running,
        "" if process_running else "No running DaVinci Resolve process was found. Please open DaVinci Resolve.",
    ))
    if not process_running:
        return diag

    try:
        dvr_script = _load_scripting_module()
        record(ConnectionStepResult("Resolve API available", True))
    except ResolveConnectionError as exc:
        record(ConnectionStepResult("Resolve API available", False, str(exc)))
        return diag

    resolve = dvr_script.scriptapp("Resolve")
    if resolve is None:
        record(ConnectionStepResult(
            "Resolve instance", False,
            "DaVinci Resolve is running and its scripting module loaded, but scriptapp(\"Resolve\") "
            "returned nothing. This almost always means one of two things: (1) this is the FREE "
            "edition of DaVinci Resolve, which has no scripting API at all, regardless of any app "
            "configuration -- check Help > About DaVinci Resolve to see whether it says \"DaVinci "
            "Resolve\" or \"DaVinci Resolve Studio\"; or (2) Preferences > System > General > "
            "\"External scripting using\" is set to None instead of Local.",
        ))
        return diag
    record(ConnectionStepResult("Resolve instance", True))

    try:
        project = get_current_project(resolve)
        record(ConnectionStepResult("Project", True, project.GetName()))
        diag.project_name = project.GetName()
    except ResolveConnectionError as exc:
        record(ConnectionStepResult("Project", False, str(exc)))
        return diag

    try:
        timeline = get_current_timeline(project)
        record(ConnectionStepResult("Timeline", True, timeline.GetName()))
        diag.timeline_name = timeline.GetName()
    except ResolveConnectionError as exc:
        record(ConnectionStepResult("Timeline", False, str(exc)))
        return diag

    # Read-only: exercises the same API surface AddMarker() depends on without mutating anything.
    try:
        timeline.GetMarkers()
        record(ConnectionStepResult("Bookmark access", True))
    except Exception as exc:  # noqa: BLE001 - surface whatever Resolve's API actually raised
        record(ConnectionStepResult("Bookmark access", False, f"GetMarkers() failed: {exc}"))
        return diag

    diag.connected = True
    return diag


def create_test_bookmark() -> ConnectionStepResult:
    """
    TEST BOOKMARK: adds one real, clearly-labelled marker to the current timeline to prove
    marker creation genuinely works end-to-end -- the same AddMarker() call
    add_markers_from_detections() depends on -- without touching any of the user's own
    markers. Kept separate from run_full_diagnostic()'s read-only "Bookmark access" check
    (GetMarkers()) since a real write can fail in ways a read never would.
    """
    try:
        resolve = get_resolve()
        project = get_current_project(resolve)
        timeline = get_current_timeline(project)
    except ResolveConnectionError as exc:
        return ConnectionStepResult("Test bookmark", False, str(exc))

    fps = _get_timeline_frame_rate(project, timeline)
    label = "BF5 AUTO MARKER — CONNECTION TEST"
    note = "Created by BF5 Auto Marker's TEST BOOKMARK button to confirm marker creation works. Safe to delete."

    # AddMarker()'s frameId is an OFFSET from the timeline's own start (0 = first frame of the
    # timeline) -- confirmed against Resolve's own shipped scripting README, which describes
    # GetMarkers() results as "timeline offset" positions. GetStartFrame() must NOT be added
    # here: it returns the timeline's start frame in its own timecode numbering (often a large
    # non-zero number, e.g. from a 01:00:00:00 start timecode), and adding it on top placed
    # every marker that many frames past where it was meant to be -- often far outside the
    # timeline's actual edited content entirely, which is why a "successful" test marker was
    # never visible in Resolve.
    #
    # AddMarker() fails outright if a marker already exists at the exact frame -- try a
    # handful of nearby positions so one stray existing marker can't make a working
    # connection look broken.
    for offset_seconds in (1.0, 2.0, 3.0, 4.0, 5.0):
        frame_id = int(round(offset_seconds * fps))
        if timeline.AddMarker(frame_id, "Purple", label, note, 1):
            return ConnectionStepResult(
                "Test bookmark", True,
                f"Marker added at frame {frame_id} on '{project.GetName()}' / '{timeline.GetName()}'. "
                f"Look for a purple marker labelled \"{label}\" -- safe to delete.",
            )

    return ConnectionStepResult(
        "Test bookmark", False,
        "AddMarker() returned False at every position tried -- a marker may already exist there, "
        "or the timeline may be locked.",
    )


def _get_timeline_frame_rate(project, timeline) -> float:
    try:
        return float(project.GetSetting("timelineFrameRate"))
    except (TypeError, ValueError):
        return float(timeline.GetSetting("timelineFrameRate") or 30.0)


@dataclass
class MarkerSendResult:
    """
    Result of add_markers_from_detections() -- `created` reflects a genuine post-write
    re-read of the timeline via GetMarkers(), not just what each AddMarker() call claimed to
    have done. Never report success to the user from anything less than this: a call can
    return True while Resolve silently fails to persist the marker, or vice versa.
    `already_present` counts events whose identical marker was already on the timeline (an earlier
    send of the same events), which are left alone rather than duplicated. `nudged` counts markers
    moved forward by a few frames because their exact frame already held a different marker.
    `verification_error` is a distinct state from a real mismatch -- it means the send itself
    may have worked but the re-read to confirm it failed, which must not be shown as "N
    markers failed" (that would blame markers that were probably fine).
    """
    total: int
    created: int
    failed_events: List[Tuple[Event, int]] = field(default_factory=list)  # (event, frame_id)
    verification_error: Optional[str] = None
    already_present: int = 0
    nudged: int = 0

    @property
    def all_created(self) -> bool:
        return self.verification_error is None and not self.failed_events and self.created + self.already_present == self.total


# Resolve allows one marker per frame. When an event's frame is already taken by a different marker
# (or two events in one send land on the same frame, e.g. a negative bookmark adjustment clamping
# several early events to frame 0) the marker is moved forward to the next free frame, up to this far.
MAX_NUDGE_FRAMES = 30


def _frame_key(key) -> Optional[int]:
    """Resolve's own docs don't promise the key type of GetMarkers() across versions."""
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def _markers_by_frame(markers) -> dict:
    by_frame = {}
    for key, value in (markers or {}).items():
        frame = _frame_key(key)
        if frame is not None:
            by_frame[frame] = value if isinstance(value, dict) else {}
    return by_frame


def compute_marker_timestamp(event: Event, bookmark_adjustment_seconds: float = 0.0) -> float:
    """
    The actual position (in source-video seconds) a marker for this event will be created at --
    shared by the SEND TO RESOLVE preview dialog, CALIBRATE's TEST button, and
    add_markers_from_detections() itself, so no preview can ever drift from what's actually sent.

    `bookmark_adjustment_seconds` is the single, signed, calibrated adjustment (default 0.0 --
    use the detected bookmark exactly): positive moves the marker later, negative moves it
    earlier. Applied against the event's own bookmark time and never allowed to go negative
    (clamped to the source video's own start, per spec: "do not create a negative timestamp").
    """
    return max(0.0, event.bookmark_seconds + bookmark_adjustment_seconds)


def add_markers_from_detections(
    timeline,
    project,
    events: List[Event],
    marker_name: str,
    marker_color: str,
    bookmark_adjustment_seconds: float = 0.0,
) -> MarkerSendResult:
    """
    Add one marker per event onto the given timeline (a multi-kill event
    still gets exactly one marker, per spec item 9 -- the note records how
    many skulls and what colours were involved), then verify every one of
    them actually landed by re-reading the timeline's real marker list.

    Position: AddMarker()'s frameId is an OFFSET from the timeline's own start (0 = first
    frame of the timeline), per Resolve's shipped scripting README -- NOT GetStartFrame()'s
    absolute timecode-based frame number. See compute_marker_timestamp() for exactly how each
    event's source-video timestamp is turned into that position, including the calibrated
    `bookmark_adjustment_seconds` (never permanently altering the event's own stored bookmark --
    this is applied fresh on every send, so sending again after recalibrating it just works).
    GetStartFrame() is deliberately NOT added: doing so previously placed every marker that many
    frames past where it belonged (often far outside the timeline's actual edited content), which
    is why TEST BOOKMARK could report success while the marker was nowhere to be found in Resolve.

    Success is decided by re-reading the timeline, never by AddMarker()'s return value, and never
    by "a marker exists at that frame" alone: a frame that already held a marker before this send
    would otherwise make a failed AddMarker() look like a success. See MarkerSendResult.
    """
    fps = _get_timeline_frame_rate(project, timeline)

    try:
        before = _markers_by_frame(timeline.GetMarkers())
    except Exception:  # noqa: BLE001 - can't see existing markers; proceed and let the re-read decide
        before = {}
    taken = set(before)
    claimed = set()  # existing markers already accounted for by an event of this send

    planned: List[Tuple[Event, int, str, bool]] = []  # (event, frame_id, label, already_there)
    nudged = 0
    for event in events:
        adjusted_seconds = compute_marker_timestamp(event, bookmark_adjustment_seconds)
        wanted_frame = int(round(adjusted_seconds * fps))

        # e.g. "KILL — 3 SKULLS" -- the marker's own label shows the skull count directly on
        # the timeline, not just in its note (spec item 9).
        skull_noun = "SKULL" if event.skull_count == 1 else "SKULLS"
        marker_label = f"{marker_name.upper()} — {event.skull_count} {skull_noun}"

        existing = before.get(wanted_frame)
        if existing is not None and existing.get("name") == marker_label and wanted_frame not in claimed:
            claimed.add(wanted_frame)  # one existing marker can only stand in for one event
            planned.append((event, wanted_frame, marker_label, True))  # an earlier send of this same marker
            continue

        frame_id = wanted_frame
        while frame_id in taken and frame_id - wanted_frame <= MAX_NUDGE_FRAMES:
            frame_id += 1
        if frame_id != wanted_frame:
            nudged += 1
        taken.add(frame_id)
        planned.append((event, frame_id, marker_label, False))

    for event, frame_id, marker_label, already_there in planned:
        if already_there:
            continue
        note = (
            f"Event: {event.event_number}\n"
            f"Skulls: {event.skull_count}\n"
            f"Colours: {', '.join(event.colors)}\n"
            f"Confidence: {event.confidence_percent}%"
        )
        timeline.AddMarker(
            frame_id,       # frameId: position on the timeline, in frames
            marker_color,   # color, e.g. "Red"
            marker_label,   # name, e.g. "KILL — 3 SKULLS"
            note,           # note: event number, skull count, colours, confidence
            1,              # duration: 1 frame
        )

    try:
        after = _markers_by_frame(timeline.GetMarkers())
    except Exception as exc:  # noqa: BLE001 - "sent but unverified" is a real, distinct state
        return MarkerSendResult(total=len(events), created=0, verification_error=str(exc), nudged=nudged)

    created = already_present = 0
    failed_events: List[Tuple[Event, int]] = []
    for event, frame_id, marker_label, already_there in planned:
        if already_there:
            already_present += 1
        elif frame_id in after and frame_id not in before:
            created += 1
        else:
            failed_events.append((event, frame_id))

    return MarkerSendResult(
        total=len(events), created=created, failed_events=failed_events,
        already_present=already_present, nudged=nudged,
    )


def apply_color_grade() -> None:
    """
    PLACEHOLDER -- not yet implemented.

    Intended purpose: automatically apply the user's existing saved
    PowerGrade (or a specific LUT) to the current timeline's gameplay clip
    via the Resolve API, automating step 3 of the manual workflow.

    Realistic implementation paths once the specific PowerGrade name / LUT
    path is known:
      1. Gallery-based PowerGrade:
         gallery = project.GetGallery()
         still = <find the still by name/album via gallery.GetGalleryStillAlbums()>
         gallery.ApplyGradeFromStill(still)   # applies to the currently selected timeline clip(s)
      2. LUT-based:
         timeline_item = timeline.GetItemListInTrack("video", 1)[0]
         timeline_item.SetLUT(nodeIndex, lutPath)

    Left unimplemented deliberately -- it depends on this user's specific
    Resolve Gallery contents / LUT file path, which aren't known yet. Wire
    this into the UI once that's decided; do not fake success in the
    meantime.
    """
    raise NotImplementedError(
        "apply_color_grade() is a placeholder. Implement it with your specific "
        "PowerGrade name or LUT path before calling it."
    )


def apply_auto_highlights(timeline, keep_ranges) -> None:
    """
    PLACEHOLDER -- not yet implemented.

    Intended purpose: given the "keep ranges" computed by
    data/highlights.compute_keep_ranges() (already implemented, see Phase 9),
    actually trim the Resolve timeline down to just those ranges -- i.e.
    perform the "remove boring sections" feature described in the spec.

    This is deliberately left unimplemented in v1: the spec explicitly says
    not to build automatic footage removal reliably until it's been proven
    out, since a bad cut here destroys editorial context (unlike a wrong
    marker, which costs nothing to ignore). A safe future implementation
    would build a *new* timeline from the keep ranges (via
    project.GetMediaPool().AppendToTimeline() with in/out points) rather
    than deleting from the user's existing timeline in place, so the
    original edit is never at risk.
    """
    raise NotImplementedError(
        "apply_auto_highlights() is a placeholder. See compute_keep_ranges() "
        "in data/highlights.py for the (implemented) range calculation this "
        "would consume."
    )
