"""
core/session.py
------------------
AppSession: the single shared object every page reads and writes through
-- one video, one detection region, one or more calibrated skull reference
images, one list of detections. Replaces the old split across video/video_session.py's
VideoSession and data/calibration_profile.py's CalibrationProfile: this
project kept hitting bugs (a marker frame-offset error, an ANALYSE
"frame 0/0" freeze) that traced back to state being scattered across
several objects instead of one place every page could trust completely.
"""

from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from data.detection import Event
from detectors.colour_matcher import ColourReferenceSet, FootageObservation
from detectors.reference_profile import ReferenceProfile
from video.video_probe import VideoMetadata, VideoOpenError, probe_video

_ASPECT_MISMATCH_THRESHOLD = 0.05


@dataclass
class ReferenceImage:
    """One calibrated skull appearance (e.g. "White Skull", "Gold Skull") -- the detector combines
    every ReferenceImage on the session into one skull detector that recognises all of them,
    rather than the app knowing about only a single colour/appearance."""
    label: str          # shown in the UI and used as SkullDetector's symbol_name for this template
    image: np.ndarray   # grayscale, background-isolated (already run through crop_to_content())
    scale: float = 1.0  # this reference's own learned real-world scale
    # The same reference with its COLOUR kept (background removed, skull colour untouched), plus its
    # silhouette mask and colour signature. `image` above is the legacy grayscale template the
    # original detector matches against and is deliberately left exactly as it always was.
    profile: Optional[ReferenceProfile] = None
    # How this reference really appeared in the user's own test footage (real on-screen size and
    # colour, measured at calibration_video_height) -- learned by AUTO-DETECT and saved with the
    # calibration. When present it centres the colour matcher's size band and refines its colour
    # signature; when absent the band falls back on the calibrated `scale` above.
    footage: Optional[FootageObservation] = None


class AppSession:
    def __init__(self):
        # -- Video --
        self.video_path: Optional[str] = None
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.duration: float = 0.0
        self.metadata: Optional[VideoMetadata] = None

        self.current_frame: int = 0
        self.current_timestamp: float = 0.0
        self.current_image: Optional[np.ndarray] = None  # BGR, already decoded
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

        # -- Calibration --
        self.detection_region: Optional[Tuple[int, int, int, int]] = None
        self.calibration_video_width: int = 0   # resolution the region was drawn at
        self.calibration_video_height: int = 0
        # One or more calibrated skull appearances (e.g. White + Gold) -- combined into a single
        # multi-template detector, never treated as separate detection systems. Each image is
        # already background-isolated and carries its own learned scale.
        self.reference_images: List[ReferenceImage] = []
        # Signed seconds applied to every detected bookmark exactly once, at send time --
        # positive moves a marker later, negative earlier, 0.0 uses the exact detected timestamp.
        # Calibrated once (CALIBRATE tab) and persisted with the rest of calibration, not
        # re-entered per send. See resolve.resolve_controller.compute_marker_timestamp().
        self.bookmark_adjustment_seconds: float = 0.0

        # -- Analysis --
        self.detections: List[Event] = []

    # ------------------------------------------------------------------
    @property
    def video_name(self) -> str:
        return os.path.basename(self.video_path) if self.video_path else ""

    def load_video(self, path: str) -> None:
        """Open the video, probe its metadata, and eagerly decode + store frame 0 -- all in one
        call, so by the time this returns successfully every page can rely on `current_image`
        already being populated. Raises VideoOpenError on failure; callers show it."""
        self.release()
        metadata = probe_video(path)  # raises VideoOpenError if the file is missing/invalid

        self.video_path = path
        self.metadata = metadata
        self.width = metadata.width
        self.height = metadata.height
        self.fps = metadata.fps
        self.frame_count = metadata.frame_count
        self.duration = metadata.duration_seconds
        self.current_frame = 0
        self.current_timestamp = 0.0
        self.current_image = None

        self.video_capture = cv2.VideoCapture(path)
        if not self.video_capture.isOpened():
            self.is_loaded = False
            self.load_error = "OpenCV could not open the video for frame decoding."
            raise VideoOpenError(self.load_error)

        self.is_loaded = True
        self.load_error = None
        if not self.read_frame(0):
            # The file opened, but the very first frame wouldn't decode -- surfaced via
            # load_error rather than silently leaving the viewer blank.
            self.load_error = "The video opened, but its first frame could not be decoded."

    def read_frame(self, frame_number: int) -> bool:
        """Central frame-loading function: validate, seek, read, verify, store. Returns True on
        success; on failure, `current_image` is left as whatever it was before (never silently
        blanked), and `load_error` records the reason."""
        if not self.ensure_open():
            return False

        if self.frame_count > 0:
            frame_number = max(0, min(self.frame_count - 1, frame_number))
        else:
            frame_number = max(0, frame_number)

        current_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_number != current_pos:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            self.load_error = f"Frame {frame_number} could not be decoded (OpenCV read() failed)."
            return False

        self.current_image = frame
        self.current_frame = frame_number
        self.current_timestamp = frame_number / self.fps if self.fps else 0.0
        self.load_error = None
        return True

    # ui/video_scrubber.py (reused as-is from before this rewrite) calls session.load_frame()
    # and reads session.total_frames -- aliased here rather than touching that already-correct
    # widget just to rename what it calls.
    load_frame = read_frame

    @property
    def total_frames(self) -> int:
        return self.frame_count

    def ensure_open(self) -> bool:
        """If the capture died mid-session, reopen and reseek automatically."""
        if self.video_capture is not None and self.video_capture.isOpened():
            return True
        return self._reopen()

    def reload(self) -> bool:
        """RELOAD VIDEO: full close/reopen/re-read-frame-0/refresh-metadata cycle."""
        if not self.video_path:
            return False
        path = self.video_path
        try:
            self.load_video(path)
            return self.current_image is not None
        except VideoOpenError:
            return False

    def _reopen(self) -> bool:
        if not self.video_path:
            return False
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = cv2.VideoCapture(self.video_path)
        if self.video_capture.isOpened():
            self.load_error = None
            return True
        self.load_error = "Could not reopen the video file."
        return False

    def release(self) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def set_detection_region(self, region: Tuple[int, int, int, int], video_width: int, video_height: int) -> None:
        self.detection_region = region
        self.calibration_video_width = video_width
        self.calibration_video_height = video_height

    def is_calibrated(self) -> bool:
        return self.detection_region is not None and len(self.reference_images) > 0

    def scaled_reference_templates(self) -> List[Tuple[str, np.ndarray]]:
        """(label, image) pairs actually used for matching -- every calibrated reference, resized
        by its own learned scale. Ready to pass straight into
        SkullDetector(templates=[img for _, img in ...], symbol_names=[label for label, _ in ...]).
        Every detector construction (analysis, calibration test) should use this, never
        reference_images directly, so they always agree on the same real-world sizes."""
        results = []
        for ref in self.reference_images:
            image = ref.image
            if ref.scale != 1.0:
                h, w = image.shape[:2]
                new_w, new_h = max(1, round(w * ref.scale)), max(1, round(h * ref.scale))
                interp = cv2.INTER_AREA if ref.scale < 1.0 else cv2.INTER_LINEAR
                image = cv2.resize(image, (new_w, new_h), interpolation=interp)
            results.append((ref.label, image))
        return results

    def colour_reference_set(self, video_height: int = 0, only_labels: Optional[List[str]] = None) -> Optional[ColourReferenceSet]:
        """The colour information a scan of a `video_height`-tall video needs, or None if no
        reference carries usable colour. See build_colour_reference_set()."""
        refs = self.reference_images
        if only_labels is not None:
            refs = [r for r in refs if r.label in only_labels]
        return build_colour_reference_set(refs, self.calibration_video_height, video_height)

    def oversized_reference_labels(self) -> List[str]:
        """
        Template matching can never succeed if a reference image is larger than the region it's
        searched against (cv2.matchTemplate requires template <= search image in both dimensions)
        -- SkullDetector.find_matches() silently skips an oversized template with no error, which
        otherwise looks exactly like "the detector just never finds anything". Returns the labels
        of any calibrated reference that's currently too large for the saved detection region, so
        callers can name the specific one at fault rather than a generic failure.
        """
        if self.detection_region is None:
            return []
        _, _, region_w, region_h = self.detection_region
        oversized = []
        for label, image in self.scaled_reference_templates():
            ref_h, ref_w = image.shape[:2]
            if ref_h > region_h or ref_w > region_w:
                oversized.append(label)
        return oversized

    def scaled_detection_region(self, video_width: int, video_height: int):
        """(region, aspect_mismatch) -- the saved detection_region scaled to a (possibly
        different) video resolution than it was calibrated at. See scaled_region() below."""
        return scaled_region(
            self.detection_region, self.calibration_video_width, self.calibration_video_height,
            video_width, video_height,
        )


# Fraction of a reference's calibrated scale the colour matcher sweeps (scored down outside the
# full-credit band) -- see build_colour_reference_set().
_COLOUR_SWEEP = (0.75, 1.35)
_COLOUR_FULL_CREDIT = (0.90, 1.10)


def build_colour_reference_set(
    references: List[ReferenceImage], calibration_height: int, video_height: int,
) -> Optional[ColourReferenceSet]:
    """Bundles the colour information one scan needs. Each chromatic reference's size band is its
    calibrated scale, widened upward by the video-vs-calibration resolution ratio: the calibration
    clips are usually a different resolution than real gameplay (measured: 1080p clips vs 1440p
    gameplay), and a learned scale is only exactly right for the footage it was learned on."""
    with_profiles = [r for r in references if r.profile is not None]
    matching = [r for r in with_profiles if r.profile.signature.chromatic]
    if not matching:
        return None
    ratio = 1.0
    if video_height and calibration_height:
        ratio = video_height / float(calibration_height)
    low, high = min(1.0, ratio), max(1.0, ratio)
    bands = {}
    effective = {}
    for ref in matching:
        base = ref.footage.scale if ref.footage is not None else ref.scale
        bands[ref.label] = (
            _COLOUR_SWEEP[0] * base * low, _COLOUR_SWEEP[1] * base * high,
            _COLOUR_FULL_CREDIT[0] * base * low, _COLOUR_FULL_CREDIT[1] * base * high,
        )
        effective[ref.label] = _profile_with_footage_colour(ref)
    return ColourReferenceSet(
        matching=[effective[r.label] for r in matching], scale_bands=bands,
        labelling=[effective.get(r.label, r.profile) for r in with_profiles],
    )


def _profile_with_footage_colour(ref: ReferenceImage) -> ReferenceProfile:
    """The reference's profile with its colour signature averaged with what the user's own footage
    showed (when AUTO-DETECT learned it) -- the reference is one source of truth, the footage the
    other. The reference's own stored profile is never modified."""
    if ref.footage is None:
        return ref.profile
    sig = ref.profile.signature
    blended = dataclasses.replace(
        sig, lab_l=(sig.lab_l + ref.footage.lab_l) / 2.0,
        lab_a=(sig.lab_a + ref.footage.lab_a) / 2.0, lab_b=(sig.lab_b + ref.footage.lab_b) / 2.0,
    )
    return dataclasses.replace(ref.profile, signature=blended)


def scaled_region(
    region: Optional[Tuple[int, int, int, int]],
    cal_w: int, cal_h: int, video_w: int, video_h: int,
):
    """
    Proportionally scales a detection region calibrated at (cal_w, cal_h) to a video whose
    actual resolution is (video_w, video_h) -- a region calibrated at 1920x1080 pointed at a
    2560x1440 clip must scale, not be reused as raw pixel coordinates. Returns
    (scaled_region, aspect_mismatch): aspect_mismatch flags when the two resolutions' aspect
    ratios differ enough that scaling is unlikely to still land on the real detection area --
    the caller should warn and offer recalibration rather than trust the scaled numbers blindly.
    """
    if region is None:
        return None, False
    if not cal_w or not cal_h or (video_w == cal_w and video_h == cal_h):
        return region, False

    cal_aspect = cal_w / cal_h
    video_aspect = video_w / video_h if video_h else cal_aspect
    aspect_mismatch = abs(cal_aspect - video_aspect) / cal_aspect > _ASPECT_MISMATCH_THRESHOLD

    scale_x = video_w / cal_w
    scale_y = video_h / cal_h
    x, y, w, h = region
    scaled = (
        int(round(x * scale_x)), int(round(y * scale_y)),
        int(round(w * scale_x)), int(round(h * scale_y)),
    )
    return scaled, aspect_mismatch
