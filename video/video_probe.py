"""
video/video_probe.py
-----------------------
Video metadata + frame-count validation, used the moment a video is
selected -- before CALIBRATE or ANALYSE ever touch it. Moved verbatim out
of the old video/video_analyzer.py (this logic was already correct and
already tested this session; the rewrite reuses it rather than retyping
it) as part of splitting "read metadata" from "run the detection scan"
into their own modules.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2


class VideoOpenError(Exception):
    """Raised when a video file can't be opened or reports nonsensical metadata."""


@dataclass
class VideoMetadata:
    path: str
    file_size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    @property
    def duration_display(self) -> str:
        total_seconds = int(round(self.duration_seconds))
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @property
    def file_size_display(self) -> str:
        size = float(self.file_size_bytes)
        if size < 1024:
            return f"{int(size)} B"
        for unit in ("KB", "MB", "GB"):
            size /= 1024
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}"
        return f"{size:.1f} GB"

    @property
    def resolution_display(self) -> str:
        return f"{self.width} x {self.height}"

    @property
    def fps_display(self) -> str:
        return f"{self.fps:.0f} FPS" if abs(self.fps - round(self.fps)) < 0.05 else f"{self.fps:.2f} FPS"


def probe_video(path: str) -> VideoMetadata:
    """Read metadata only -- used immediately after video selection, before any analysis."""
    if not os.path.isfile(path):
        raise VideoOpenError(f"File not found: {path}")

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoOpenError(
            f"OpenCV could not open '{os.path.basename(path)}'. The file may be corrupted "
            "or use a codec that isn't supported on this system."
        )

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not (fps and math.isfinite(fps) and fps > 0):
            fps = 30.0  # the container didn't report a usable frame rate
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            # Some containers/codecs (variable frame rate, certain streaming-style muxes)
            # report CAP_PROP_FRAME_COUNT as 0 or negative even though the video plays and
            # decodes fine -- don't trust that number blindly and declare a real, playable
            # video "invalid". Confirm at least one frame actually decodes first (a genuine
            # failure to read anything means the file really is unusable), then fall back to
            # counting frames by reading through the whole file once.
            ok, _ = capture.read()
            if not ok:
                raise VideoOpenError(
                    f"'{os.path.basename(path)}' opened but no frame could be decoded from it. "
                    "The file is likely corrupted or uses an unsupported codec."
                )
            frame_count = _count_frames_by_decoding(path)

        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise VideoOpenError(
            f"'{os.path.basename(path)}' opened but reported invalid metadata "
            "(width/height came back empty). The file is likely corrupted."
        )
    if frame_count <= 0:
        raise VideoOpenError(
            f"'{os.path.basename(path)}' opened and its frames decode, but no frame count could "
            "be determined even after counting manually. Analysis needs a known frame count to "
            "track progress -- try re-exporting the video with standard settings."
        )

    return VideoMetadata(
        path=path,
        file_size_bytes=os.path.getsize(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
    )


def _count_frames_by_decoding(path: str) -> int:
    """Fallback for videos whose CAP_PROP_FRAME_COUNT can't be trusted (0, negative, or wrong
    for certain containers/codecs) -- reads the whole file once just to count real, decodable
    frames. Slower than trusting the container's own metadata, but only runs for the rare video
    that actually needs it, and only once (at probe time), not during analysis itself."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return 0
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def seconds_to_timecode(total_seconds: float, fps: float) -> str:
    """HH:MM:SS:FF timecode on the source video's own frame grid."""
    if fps <= 0:
        fps = 30.0
    total_frames = int(round(total_seconds * fps))
    whole_fps = max(1, int(round(fps)))
    frames = total_frames % whole_fps
    total_whole_seconds = total_frames // whole_fps
    hours, rem = divmod(total_whole_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def extract_frame(video_path: str, frame_number: int):
    """Grab a single BGR frame for preview purposes."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()
