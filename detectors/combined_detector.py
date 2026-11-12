"""
detectors/combined_detector.py
--------------------------------
ONE detector for the whole skull-detection pass. Per sampled frame it:

  1. runs the original grayscale multi-template SkullDetector -- completely unchanged, this is the
     path white skulls have always been found by;
  2. runs the colour-aware ColourSilhouetteMatcher for every chromatic reference (e.g. gold) on the
     SAME cropped frame in the SAME call -- never a second pass over the video;
  3. merges the candidates (overlapping hits collapse to the strongest one) and names each
     candidate's skull profile from its own colour evidence.

It exposes the same find_matches(frame, threshold) the scanner and its neighbour-validation already
call, so the event state machine, bookmark timing and Resolve export never know the difference.

Naming by colour matters because the grayscale path is shape-only: measured on real gameplay, the
GOLD template scored real WHITE skulls at ~0.90, so a white kill was reported as a "Gold Skull".
Detection is unaffected (that candidate still passes exactly as before); only its profile name is
corrected from the candidate's actual colour.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from detectors.base_detector import BaseDetector, ShapeMatch
from detectors.colour_matcher import (
    CandidateEvidence, ColourReferenceSet, ColourSilhouetteMatcher, colour_likelihood,
)
from detectors.reference_profile import ReferenceProfile
from detectors.skull_detector import SkullDetector

_OVERLAP_THRESHOLD = 0.3     # same IoU the grayscale detector's own suppression uses
_RELABEL_MIN_COLOUR = 0.40   # a profile must explain at least this much of the candidate's colour
_RELABEL_MARGIN = 0.15       # ...and beat the matched template's own profile by this much to rename


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class CombinedSkullDetector(BaseDetector):
    detection_type = "KILL"

    def __init__(self, base: SkullDetector, colour_reference: Optional[ColourReferenceSet] = None):
        self.base = base
        self.colour_reference = colour_reference
        self.matcher: Optional[ColourSilhouetteMatcher] = None
        if colour_reference is not None and colour_reference.matching:
            self.matcher = ColourSilhouetteMatcher(colour_reference.matching, colour_reference.scale_bands)
        # Developer Mode diagnostics: the strongest candidate from the LAST find_matches() call that
        # did not reach the threshold, and why -- so a missed skull is never a silent nothing.
        self.last_rejected: Optional[CandidateEvidence] = None

    def find_matches(self, frame_bgr: np.ndarray, threshold: float) -> List[ShapeMatch]:
        matches = self.base.find_matches(frame_bgr, threshold)  # original path, unchanged
        self.last_rejected = None

        if self.matcher is not None:
            evidence = self.matcher.find_candidates(frame_bgr)
            accepted = [e for e in evidence if e.confidence >= threshold]
            below = [e for e in evidence if e.confidence < threshold]
            if below:
                weakest_link = max(below, key=lambda e: e.confidence)
                self.last_rejected = weakest_link
                if not weakest_link.reason:
                    weakest_link.reason = "confidence below threshold"
            matches = matches + [
                ShapeMatch(e.bbox, e.confidence, self.detection_type, e.profile, evidence=e) for e in accepted
            ]

        matches = self._merge(matches)
        return self._name_by_colour(frame_bgr, matches)

    @staticmethod
    def _merge(matches: List[ShapeMatch]) -> List[ShapeMatch]:
        kept: List[ShapeMatch] = []
        for match in sorted(matches, key=lambda m: -m.confidence):
            if all(_iou(match.bbox, k.bbox) <= _OVERLAP_THRESHOLD for k in kept):
                kept.append(match)
        return kept

    def _name_by_colour(self, frame_bgr: np.ndarray, matches: List[ShapeMatch]) -> List[ShapeMatch]:
        """Rename grayscale-path candidates by what colour they actually are. Colour-path candidates
        are already named by the profile that found them."""
        if not matches or self.colour_reference is None:
            return matches
        profiles: Sequence[ReferenceProfile] = self.colour_reference.labelling
        if len(profiles) < 2:
            return matches
        by_label = {p.label: p for p in profiles}
        likelihoods = {}
        for match in matches:
            if match.evidence is not None:
                continue
            x, y, w, h = match.bbox
            mx, my = int(w * 0.25), int(h * 0.25)
            score = {}
            for profile in profiles:
                if profile.label not in likelihoods:
                    likelihoods[profile.label] = colour_likelihood(frame_bgr, profile)
                window = likelihoods[profile.label][y + my:y + h - my, x + mx:x + w - mx]
                score[profile.label] = float(window.mean()) if window.size else 0.0
            best_label = max(score, key=score.get)
            own = score.get(match.symbol_name, 0.0) if match.symbol_name in by_label else 0.0
            if best_label != match.symbol_name and score[best_label] >= _RELABEL_MIN_COLOUR and score[best_label] - own >= _RELABEL_MARGIN:
                match.symbol_name = best_label
        return matches


def build_detector(
    reference_templates: List[tuple], colour_reference: Optional[ColourReferenceSet] = None,
) -> CombinedSkullDetector:
    """The one place a scan's detector is assembled: the grayscale multi-template detector built
    exactly as before from `reference_templates` [(label, gray_template)], plus the colour matcher
    when colour information is available."""
    base = SkullDetector(
        [image for _, image in reference_templates], symbol_names=[label for label, _ in reference_templates],
    )
    return CombinedSkullDetector(base, colour_reference)
