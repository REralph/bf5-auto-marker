"""
detectors/base_detector.py
---------------------------
Common interface every event detector implements. video/video_analyzer.py
runs a list of BaseDetector instances over each sampled frame without
knowing anything about their internals. A detector returns every
shape-matched location in a frame (not just the single best one), since
several skulls can appear together in one event (spec item 9) -- colour
classification, cooldown, and event grouping all live in the analyzer, not
here. That split is what lets new detectors (medal_detector.py,
vehicle_detector.py, objective_detector.py -- described in the project
spec, not yet built) get added later as pure "where does this shape
appear?" plug-ins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class ShapeMatch:
    """One shape-matched skull location in a single frame, before colour/temporal checks."""

    __slots__ = ("bbox", "confidence", "detection_type", "symbol_name", "evidence")

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        detection_type: str,
        symbol_name: str = "Standard Skull",
        evidence=None,
    ):
        self.bbox = bbox  # (x, y, w, h) in the coordinate space of the frame passed in
        self.confidence = confidence
        self.detection_type = detection_type
        self.symbol_name = symbol_name  # which calibrated symbol/shape template produced this match
        self.evidence = evidence  # optional per-signal breakdown (colour_matcher.CandidateEvidence), diagnostics only


class BaseDetector(ABC):
    """Subclass this for each new Battlefield V event type."""

    detection_type: str = "UNKNOWN"

    @abstractmethod
    def find_matches(self, frame_bgr: np.ndarray, threshold: float) -> List[ShapeMatch]:
        """Return every location in this frame matching this detector's shape, above `threshold`."""
        raise NotImplementedError
