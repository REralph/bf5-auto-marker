"""
video/video_scanner.py
--------------------------
The frame-by-frame detection scan: crop the calibrated region, match it
against every calibrated skull reference image (one multi-template
detector, not separate detectors per colour/appearance), group consecutive
detections into one event via an explicit NO_ACTIVE_SKULL/ACTIVE_SKULL
state machine, and count simultaneous skulls.

Event grouping is edge-triggered, not timer-based: an event starts the
moment the skull is confirmed present and ends only once it has genuinely
been absent for a short grace period -- not on a fixed cooldown timer.
The previous cooldown-gate design ("re-check every N seconds") fired a
brand-new event every time its timer expired even while the SAME skull
remained continuously visible the whole time -- confirmed directly against
real footage: one continuously-visible kill produced 18+ separate "events"
spaced exactly kill_cooldown_seconds apart, because the gate only measured
time-since-last-event, never whether the icon had actually gone away.

Adapted from the old video/video_analyzer.py's VideoAnalyzer.analyze() for
the parts that didn't change: region cropping, template matching, colour-
as-label classification. What's dropped is the colour-profile-learning/
strict-colour-gate machinery that no longer applies now that colour is
purely a supporting label, never a gate -- shape/template matching across
every calibrated reference image is the real detection signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from data.colour_label import sample_hsv, suggest_name_for_sample
from data.config import AppConfig
from data.detection import Event, SkullSighting
from detectors.colour_matcher import ColourReferenceSet
from detectors.combined_detector import build_detector
from video.video_probe import VideoMetadata, VideoOpenError, seconds_to_timecode

# Consecutive *sampled* checks (not raw video frames) the skull can go undetected before its
# active event is considered genuinely over -- absorbs a frame or two of momentary noise
# (compression, a fast neighbor-frame flicker) without splitting one real appearance into two,
# per the spec's own example: "skull, skull, skull, missed frame, skull, skull" must stay ONE event.
MISSED_SAMPLE_GRACE = 2

NO_ACTIVE_SKULL = "NO_ACTIVE_SKULL"
ACTIVE_SKULL = "ACTIVE_SKULL"


@dataclass
class ScanState:
    """Live event-tracking state, reported alongside progress purely for Developer Mode
    visibility (DETECTION_STATE / ACTIVE_EVENT_* / REJECTED_CANDIDATE_* fields) -- never affects
    scan behavior itself."""
    detection_state: str = NO_ACTIVE_SKULL
    active_event_start_frame: Optional[int] = None
    active_event_best_frame: Optional[int] = None
    active_event_best_confidence: float = 0.0
    active_event_skull_count: int = 0
    # A CANDIDATE frame (matched, but its immediate neighbors didn't hold up) never becomes a
    # bookmark -- these fields expose the most recent one so Developer Mode can confirm the
    # false-positive gate is actually rejecting something, not just silently doing nothing.
    rejected_candidate_count: int = 0
    last_rejected_frame: Optional[int] = None
    last_rejected_confidence: float = 0.0
    last_rejected_symbol: str = ""
    # Developer Mode evidence: the per-signal breakdown of the most recent candidate that started an
    # event, why the most recent candidate was rejected, and the strongest sub-threshold candidate
    # seen so far (a skull the detector nearly caught) -- so "why wasn't this detected?" has an answer.
    last_candidate_evidence: str = ""
    last_rejected_reason: str = ""
    best_missed_confidence: float = 0.0
    best_missed_evidence: str = ""


ProgressCallback = Callable[[Optional[float], int, int, int, ScanState], None]
# percent, frame_index, events_so_far, kills_so_far, live event-tracking state


def _region_bounds(
    frame_shape: Tuple[int, ...], region: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """The (x, y, x2, y2) pixel bounds _apply_region() really crops to -- clamped to the frame --
    or None when it returns the whole frame (no region, or one that doesn't fit this frame)."""
    if region is None:
        return None
    x, y, w, h = region
    frame_h, frame_w = frame_shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(x + w, frame_w), min(y + h, frame_h)
    if x2 <= x or y2 <= y:
        return None  # configured region doesn't fit this frame -- fall back to full frame
    return x, y, x2, y2


def _apply_region(frame: np.ndarray, region: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    bounds = _region_bounds(frame.shape, region)
    if bounds is None:
        return frame
    x, y, x2, y2 = bounds
    return frame[y:y2, x:x2]


def _region_origin(frame_shape: Tuple[int, ...], region: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """Top-left of the crop _apply_region() returns, in full-frame pixels -- (0, 0) when it
    returns the whole frame. Detection runs on the crop, so every match bbox is relative to this."""
    bounds = _region_bounds(frame_shape, region)
    return (bounds[0], bounds[1]) if bounds is not None else (0, 0)


# A false-positive candidate typically scores well only on the one frame it spikes on -- a
# genuine appearance holds up immediately before and after it too. Fixed at a share of the real
# configured threshold (not scaled by the candidate's own confidence, which let a barely-above-
# threshold match get "confirmed" by a neighbor scoring far lower still) -- confirmed directly
# against real footage: both a genuine White and a genuine Gold detection's immediate neighbors
# scored 0.80-0.98, comfortably clearing this floor with real margin to spare.
_NEIGHBOR_THRESHOLD_FRACTION = 0.85


def _validate_with_neighbors(detector, peek_capture, region, frame_index, threshold) -> bool:
    """Requires BOTH immediate neighbors (one frame immediately before, one immediately after --
    not the wider +/-2 window used previously) to also show a reasonably strong match, so a
    single-frame spike that doesn't hold up on both sides is treated as noise, never confirmed as
    a genuine appearance start. A neighbor past either end of the video is simply skipped (not
    treated as a failure) rather than blocking a genuine event that starts at frame 0 or ends on
    the video's last frame; at least one side must still have been checked and passed."""
    neighbor_threshold = threshold * _NEIGHBOR_THRESHOLD_FRACTION
    checked_any = False
    for offset in (-1, 1):
        neighbor_index = frame_index + offset
        if neighbor_index < 0:
            continue
        peek_capture.set(cv2.CAP_PROP_POS_FRAMES, neighbor_index)
        ok, frame = peek_capture.read()
        if not ok:
            continue  # ran past the end of the video -- nothing to check on this side
        checked_any = True
        if not detector.find_matches(_apply_region(frame, region), neighbor_threshold):
            return False  # this side genuinely failed to hold up -- reject the candidate
    return checked_any


def _classify_matches(frame: np.ndarray, matches, origin: Tuple[int, int] = (0, 0)) -> List[SkullSighting]:
    """Colour is a supporting label only -- sampled and named, never used to gate a detection
    that shape matching already confirmed.

    `frame` is the CROPPED detection-region frame the matches were found on, so colour is
    sampled at each match's crop-relative bbox. The bbox each SkullSighting stores is shifted by
    `origin` (the crop's top-left in the source frame) so it is in full-frame source-video
    pixels, as SkullSighting.bbox documents -- that is what PREVIEW draws directly on the frame."""
    ox, oy = origin
    sightings = []
    for match in matches:
        hsv = sample_hsv(frame, match.bbox)
        name = suggest_name_for_sample(hsv) if hsv is not None else "Unknown"
        x, y, w, h = match.bbox
        sightings.append(SkullSighting((x + ox, y + oy, w, h), match.confidence, name, 1.0, match.symbol_name))
    return sightings


def _describe_match(match, frame_index: int, fps: float) -> str:
    """One-line Developer Mode description of a candidate: profile, frame, timestamp, and every
    signal when the colour path found it (the grayscale path has only its single template score)."""
    when = f"frame {frame_index} ({frame_index / fps:.2f}s)" if fps else f"frame {frame_index}"
    if match.evidence is not None:
        return f"{when} | {match.evidence.as_line()}"
    x, y, w, h = match.bbox
    return f"{when} | {match.symbol_name}: conf {match.confidence:.2f} (grayscale template) | bbox {match.bbox}"


def scan(
    video_path: str,
    metadata: VideoMetadata,
    detection_region: Optional[Tuple[int, int, int, int]],
    reference_templates: List[Tuple[str, np.ndarray]],
    config: AppConfig,
    progress_callback: Optional[ProgressCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_event: Optional[Callable[[Event], None]] = None,
    colour_reference: Optional[ColourReferenceSet] = None,
) -> List[Event]:
    """
    Scans the whole video: crop the calibrated region -> match against every calibrated skull
    reference image (e.g. White + Gold, combined into one multi-template detector, never two
    separate detection systems) -> group consecutive detections into one event via the
    NO_ACTIVE_SKULL/ACTIVE_SKULL state machine (module docstring) -> count simultaneous skulls,
    keeping whichever frame during the appearance had the most skulls (ties broken by confidence)
    as the event's final bookmark position and skull list. `on_event` fires the moment each Event
    is confirmed (not just at the end), which is what makes ANALYSE's live preview and MARKERS
    FOUND counter update while the scan is still running.

    `colour_reference` (optional) adds the colour-aware matcher for chromatic references (e.g. gold)
    to the SAME per-frame pass -- see detectors/combined_detector.py. Omitted, the scan behaves
    exactly as it did before colour information was kept.
    """
    detector = build_detector(reference_templates, colour_reference)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise VideoOpenError(f"OpenCV could not open video file: {video_path}")
    # A second handle for neighbor-validation peeks, so random-access reads never disturb the
    # main sequential grab()/retrieve() position.
    peek_capture = cv2.VideoCapture(video_path)
    if not peek_capture.isOpened():
        # Without it every candidate would fail neighbour validation and the scan would quietly
        # find nothing; say so instead.
        capture.release()
        peek_capture.release()
        raise VideoOpenError(f"OpenCV could not open a second reader on the video: {video_path}")

    events: List[Event] = []
    frame_index = 0
    last_progress_report = 0.0
    fps = metadata.fps
    threshold = config.confidence_threshold

    scan_state = ScanState()
    active_missed_checks = 0
    active_best_key: Tuple[int, float] = (0, 0.0)  # (skull_count, confidence) -- lexicographic "best so far"
    active_best_frame_bgr: Optional[np.ndarray] = None
    active_best_origin: Tuple[int, int] = (0, 0)  # top-left of active_best_frame_bgr's crop in the full frame
    active_best_matches = None
    active_last_seen_frame: Optional[int] = None  # last frame the skull was ACTUALLY seen (not "best")

    def start_event(frame_idx: int, matches, cropped: np.ndarray, origin: Tuple[int, int]) -> None:
        nonlocal active_missed_checks, active_best_key, active_best_frame_bgr, active_best_origin, active_best_matches, active_last_seen_frame
        scan_state.detection_state = ACTIVE_SKULL
        scan_state.active_event_start_frame = frame_idx
        active_missed_checks = 0
        active_best_key = (len(matches), max(m.confidence for m in matches))
        active_best_frame_bgr = cropped
        active_best_origin = origin
        active_best_matches = matches
        active_last_seen_frame = frame_idx
        _update_best_fields(frame_idx)

    def update_event_if_better(frame_idx: int, matches, cropped: np.ndarray, origin: Tuple[int, int]) -> None:
        nonlocal active_missed_checks, active_best_key, active_best_frame_bgr, active_best_origin, active_best_matches, active_last_seen_frame
        active_missed_checks = 0
        active_last_seen_frame = frame_idx
        key = (len(matches), max(m.confidence for m in matches))
        if key > active_best_key:
            active_best_key = key
            active_best_frame_bgr = cropped
            active_best_origin = origin
            active_best_matches = matches
            _update_best_fields(frame_idx)

    def _update_best_fields(frame_idx: int) -> None:
        scan_state.active_event_best_frame = frame_idx
        scan_state.active_event_best_confidence = active_best_key[1]
        scan_state.active_event_skull_count = active_best_key[0]

    def finalize_event() -> None:
        nonlocal active_last_seen_frame
        # The BOOKMARK goes at the event's actual START -- confirmation still waited for the
        # skull to disappear (that's what makes the false-positive gate work), but the position
        # on the timeline should point at when the skull genuinely first appeared, not wherever
        # confidence happened to peak or the later moment confirmation itself occurred. The best
        # frame's sightings (skull count, colours, confidence) are still what's reported, since
        # that's the strongest evidence of what was actually there -- only the TIMESTAMP moves.
        sightings = _classify_matches(active_best_frame_bgr, active_best_matches, active_best_origin)
        start_frame_index = scan_state.active_event_start_frame
        bookmark_timestamp = start_frame_index / fps
        end_frame_index = active_last_seen_frame
        new_event = Event(
            event_number=len(events) + 1,
            timestamp_seconds=bookmark_timestamp,
            frame_number=start_frame_index,
            timecode=seconds_to_timecode(bookmark_timestamp, fps),
            skulls=sightings,
            event_end_frame=end_frame_index,
            event_end_seconds=(end_frame_index / fps) if end_frame_index is not None else None,
            confirmation_frame=frame_index,
            confirmation_seconds=frame_index / fps,
            best_frame=scan_state.active_event_best_frame,
        )
        events.append(new_event)
        if on_event is not None:
            on_event(new_event)
        scan_state.detection_state = NO_ACTIVE_SKULL
        scan_state.active_event_start_frame = None
        scan_state.active_event_best_frame = None
        scan_state.active_event_best_confidence = 0.0
        scan_state.active_event_skull_count = 0
        active_last_seen_frame = None

    try:
        while True:
            if should_cancel is not None and should_cancel():
                break

            ok = capture.grab()
            if not ok:
                break  # end of video

            if frame_index % max(1, config.frame_sampling) == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    break

                cropped = _apply_region(frame, detection_region)
                crop_origin = _region_origin(frame.shape, detection_region)
                matches = detector.find_matches(cropped, threshold)
                nearly_caught = detector.last_rejected  # read now: neighbour peeks below overwrite it
                if not matches and nearly_caught is not None and nearly_caught.confidence > scan_state.best_missed_confidence:
                    scan_state.best_missed_confidence = nearly_caught.confidence
                    scan_state.best_missed_evidence = (
                        f"frame {frame_index} ({frame_index / fps:.2f}s) | {nearly_caught.as_line()}"
                    )
                if not config.multiple_skull_detection and matches:
                    matches = [max(matches, key=lambda m: m.confidence)]

                if scan_state.detection_state == NO_ACTIVE_SKULL:
                    # Starting a NEW event is the only place neighbor-validation runs -- it
                    # exists to filter a single-frame false start, which doesn't apply once an
                    # event is already confirmed active (see ACTIVE_SKULL branch below).
                    confirmed = matches and (
                        not config.enable_neighbor_validation
                        or _validate_with_neighbors(detector, peek_capture, detection_region, frame_index, threshold)
                    )
                    if confirmed:
                        start_event(frame_index, matches, cropped, crop_origin)
                        scan_state.last_candidate_evidence = "CONFIRMED | " + _describe_match(
                            max(matches, key=lambda m: m.confidence), frame_index, fps,
                        )
                    elif matches:
                        # A real match that didn't hold up on its immediate neighbors -- a
                        # CANDIDATE the false-positive gate correctly rejected, never a bookmark.
                        scan_state.rejected_candidate_count += 1
                        best_candidate = max(matches, key=lambda m: m.confidence)
                        scan_state.last_rejected_frame = frame_index
                        scan_state.last_rejected_confidence = best_candidate.confidence
                        scan_state.last_rejected_symbol = best_candidate.symbol_name
                        scan_state.last_rejected_reason = (
                            "insufficient temporal confirmation (a neighbouring frame did not hold up) | "
                            + _describe_match(best_candidate, frame_index, fps)
                        )
                else:  # ACTIVE_SKULL -- same appearance continuing, or it just ended
                    if matches:
                        update_event_if_better(frame_index, matches, cropped, crop_origin)
                    else:
                        active_missed_checks += 1
                        if active_missed_checks > MISSED_SAMPLE_GRACE:
                            finalize_event()

            frame_index += 1

            if progress_callback is not None:
                if metadata.frame_count > 0:
                    percent = (frame_index / metadata.frame_count) * 100.0
                    done = frame_index == metadata.frame_count
                else:
                    percent, done = None, False

                # Frame count is genuinely unknown (should be rare after probe_video()'s own
                # fallback counting) -- never silently freeze the UI; report periodically by
                # frame count instead of by percent so CURRENT_FRAME still visibly advances.
                should_report = (
                    (percent is not None and (percent - last_progress_report >= 0.5 or done))
                    or (percent is None and frame_index % 30 == 0)
                )
                if should_report:
                    kills_so_far = sum(e.skull_count for e in events)
                    reported_percent = min(percent, 100.0) if percent is not None else None
                    progress_callback(reported_percent, frame_index, len(events), kills_so_far, scan_state)
                    if percent is not None:
                        last_progress_report = percent

        # The scan ended (cancelled, or ran out of frames) while a skull was still visible --
        # that's a real, genuine appearance and must not be silently dropped.
        if scan_state.detection_state == ACTIVE_SKULL:
            finalize_event()
    finally:
        capture.release()
        peek_capture.release()

    if progress_callback is not None:
        kills_so_far = sum(e.skull_count for e in events)
        final_percent = 100.0 if metadata.frame_count > 0 else None
        progress_callback(final_percent, frame_index, len(events), kills_so_far, scan_state)

    return events
