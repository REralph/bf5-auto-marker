"""
detectors/skull_detector.py
-----------------------------
Detects Battlefield V kill-confirmed skulls by shape, using OpenCV template
matching against one or more reference templates -- multiple calibrated
examples beat a single screenshot (spec item 5). For each frame, a small
set of lightweight preprocessing variants (grayscale, high-contrast,
threshold, inverted threshold) is tried per spec items 8-10, and the
strongest result wins; this only ever runs on the already-cropped
detection region passed in, never a full 1920x1080 frame (spec item 23).

Colour is deliberately not considered here (colour-aware matching of chromatic references such as
the gold skull lives in detectors/colour_matcher.py, combined with this detector in
detectors/combined_detector.py): shape (this file), position
(the caller's detection-region crop), and size (matching isn't scale-
invariant, so a wrong-sized shape can't score well) are the primary
identification signal (spec item 11). Colour classification happens
afterward, in the analyzer, purely as a verification layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detectors.base_detector import BaseDetector, ShapeMatch

DEFAULT_PREPROCESSING_MODES = ["original", "high_contrast", "threshold", "inverted_threshold"]

# Modest tolerance around the reference image's own size -- calibration_store's reference image
# is a real crop from the user's own footage at roughly the true on-screen size, so this only
# needs to absorb normal variance (a slightly different capture resolution, a HUD scale setting,
# a calibration drawn a little off), not search a huge range. Confirmed empirically: locating a
# real kill-feed skull that was 10x smaller than a mis-sized reference image required sweeping
# scale from 0.07-1.0 -- but once the reference is captured at roughly the right size (the normal
# case), a narrow range around 1.0 is what's actually needed going forward.
DEFAULT_SCALES = [0.85, 0.92, 1.0, 1.08, 1.15]


class TemplateLoadError(Exception):
    """Raised when no usable skull template could be loaded."""


def preprocess_variants(gray: np.ndarray, enabled_modes: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Build the handful of preprocessed versions of a grayscale image the
    spec calls for (item 9). Kept cheap: a couple of OpenCV calls each, run
    on what is -- by the time this is called -- already a small cropped
    region, not a full frame.
    """
    modes = enabled_modes or DEFAULT_PREPROCESSING_MODES
    variants: Dict[str, np.ndarray] = {}

    if "original" in modes:
        variants["original"] = gray

    if "high_contrast" in modes:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        variants["high_contrast"] = clahe.apply(gray)

    thresh = None
    if "threshold" in modes or "inverted_threshold" in modes:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if "threshold" in modes:
        variants["threshold"] = thresh
    if "inverted_threshold" in modes:
        variants["inverted_threshold"] = cv2.bitwise_not(thresh)

    if not variants:
        variants["original"] = gray  # always have at least one usable variant

    return variants


def _non_max_suppress(
    boxes: List[Tuple[int, int, int, int]], scores: List[float], overlap_thresh: float = 0.3
) -> List[int]:
    """Collapse overlapping matches (from different templates/preprocessing modes) around the same skull into one."""
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=np.float64)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    areas = boxes_arr[:, 2] * boxes_arr[:, 3]
    order = np.argsort(scores)[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = np.where(union > 0, intersection / union, 0.0)

        order = order[1:][iou <= overlap_thresh]

    return keep


class SkullDetector(BaseDetector):
    detection_type = "KILL"

    def __init__(
        self,
        templates: List[np.ndarray],
        preprocessing_modes: Optional[List[str]] = None,
        symbol_names: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
    ):
        """
        `templates` are grayscale reference images -- one or many (spec item
        5). `symbol_names` (parallel list, same length) records which
        calibrated symbol/shape each template belongs to, so a match can be
        traced back to "Standard Skull" vs "Special Skull" etc; defaults to
        a single "Standard Skull" symbol for every template when omitted.
        `scales` searches each template resized to several sizes (default
        DEFAULT_SCALES) so the calibrated reference doesn't need to be
        pixel-perfect against the icon's true on-screen size.
        """
        if not templates:
            raise TemplateLoadError("No skull templates were provided.")
        self.templates = templates
        self.symbol_names = symbol_names or ["Standard Skull"] * len(templates)
        if len(self.symbol_names) != len(templates):
            raise TemplateLoadError("symbol_names must be the same length as templates.")
        self.preprocessing_modes = preprocessing_modes or DEFAULT_PREPROCESSING_MODES
        self.scales = scales or DEFAULT_SCALES
        self._template_variants_cache = [
            self._build_scaled_variants(t, self.preprocessing_modes, self.scales) for t in templates
        ]

    @staticmethod
    def _build_scaled_variants(template: np.ndarray, modes: List[str], scales: List[float]) -> Dict[float, Dict[str, np.ndarray]]:
        by_scale: Dict[float, Dict[str, np.ndarray]] = {}
        th, tw = template.shape[:2]
        for s in scales:
            sw, sh = max(1, int(round(tw * s))), max(1, int(round(th * s)))
            resized = template if s == 1.0 else cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            by_scale[s] = preprocess_variants(resized, modes)
        return by_scale

    def find_matches(self, frame_bgr: np.ndarray, threshold: float) -> List[ShapeMatch]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_variants = preprocess_variants(gray, self.preprocessing_modes)

        all_boxes: List[Tuple[int, int, int, int]] = []
        all_scores: List[float] = []
        all_symbol_names: List[str] = []

        for scaled_variants_by_scale, symbol_name in zip(self._template_variants_cache, self.symbol_names):
            for template_variants in scaled_variants_by_scale.values():
                any_variant = next(iter(template_variants.values()))
                th, tw = any_variant.shape[:2]
                if gray.shape[0] < th or gray.shape[1] < tw:
                    continue

                for mode_name, frame_variant in frame_variants.items():
                    template_variant = template_variants.get(mode_name)
                    if template_variant is None:
                        continue
                    result = cv2.matchTemplate(frame_variant, template_variant, cv2.TM_CCOEFF_NORMED)
                    ys, xs = np.where(result >= threshold)
                    for y, x in zip(ys, xs):
                        all_boxes.append((int(x), int(y), tw, th))
                        all_scores.append(float(result[y, x]))
                        all_symbol_names.append(symbol_name)

        if not all_boxes:
            return []

        keep_indices = _non_max_suppress(all_boxes, all_scores, overlap_thresh=0.3)
        return [
            ShapeMatch(
                bbox=all_boxes[i], confidence=all_scores[i], detection_type=self.detection_type,
                symbol_name=all_symbol_names[i],
            )
            for i in keep_indices
        ]


# ----------------------------------------------------------------------
# CALIBRATE's "FIND SKULL POSITION": locate the uploaded reference somewhere in a full frame,
# at whatever scale it actually appears -- replaces manually drawing a box. A real kill-feed
# icon can be a fraction of the uploaded reference's own size (confirmed: a 273x270px reference
# located a real ~24x36px on-screen icon at scale ~0.09), so this searches a wide range, unlike
# SkullDetector's own DEFAULT_SCALES (which assumes the reference is already close to correct).
# ----------------------------------------------------------------------
LOCATE_SCALES = [
    0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15,
    0.18, 0.22, 0.27, 0.33, 0.40, 0.50, 0.65, 0.80, 1.0, 1.25,
]

# Below this, a "found" result is more likely a coincidental background match than the real
# icon -- shown to the user either way (item 4's non-crashing "not found" path), but this is
# the line between "✓ found" and "try another frame".
LOCATE_MIN_CONFIDENCE = 0.55

# The saved search region is generously larger than the detected icon itself (not an exact
# bounding box) -- a kill feed commonly stacks several recent kills as a short vertical list, so
# the true icon position varies by more than its own size between events. Ratios below were
# sized against a real, verified case: a ~24x36px icon needed a ~210x230px search region to
# cover its feed's full stacking range (~8.75x wider, ~6.4x taller) -- these defaults are
# slightly more generous than that measured minimum.
SEARCH_REGION_WIDTH_MARGIN = 9.0
SEARCH_REGION_HEIGHT_MARGIN = 9.0

# A hard cap independent of the margin ratios above -- a real kill-feed icon is a small fraction
# of the frame, so 9x its size is still small in absolute terms (a 24x36px icon -> ~210x230px on
# a 2560x1440 frame, ~8% x ~16% of it). But if the "icon" is large relative to the frame (a
# synthetic test asset, or a reference genuinely close to the frame's own scale), the same 9x
# multiplier could balloon toward the entire frame -- which defeats the whole point of cropping
# (confirmed directly: an uncapped region on a 1280x720 test frame took 40+ seconds to scan vs.
# a small region's near-instant result, because every per-frame preprocessing step then runs on
# almost the full image). Capping to a fraction of the frame keeps the region small regardless.
SEARCH_REGION_MAX_FRAME_FRACTION = 0.35


@dataclass
class LocateResult:
    bbox: Tuple[int, int, int, int]  # full-frame-relative, at the scale it was found
    scale: float
    confidence: float


def locate_reference_in_frame(
    frame_bgr: np.ndarray, reference_gray: np.ndarray,
    scales: Optional[List[float]] = None, min_confidence: float = LOCATE_MIN_CONFIDENCE,
) -> Optional[LocateResult]:
    """
    Searches the ENTIRE frame (not a pre-drawn region) for the reference image at many scales,
    returning the single best match anywhere in the frame. Returns None if nothing scores at
    least `min_confidence` -- callers should still report the best score seen (even below
    threshold) rather than a bare failure, so the user knows whether it was close or nowhere
    near (spec: "SKULL POSITION NOT FOUND ... try another frame").
    """
    scales = scales or LOCATE_SCALES
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = reference_gray.shape[:2]

    best: Optional[LocateResult] = None
    for s in scales:
        tw, th = max(1, round(ref_w * s)), max(1, round(ref_h * s))
        if tw < 6 or th < 6 or tw > gray.shape[1] or th > gray.shape[0]:
            continue
        resized = cv2.resize(reference_gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best.confidence:
            best = LocateResult(bbox=(max_loc[0], max_loc[1], tw, th), scale=s, confidence=float(max_val))

    if best is None or best.confidence < min_confidence:
        return best  # still returned (with its real confidence) so callers can show "best match: N%"
    return best


def expand_to_search_region(
    bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int,
    width_margin: float = SEARCH_REGION_WIDTH_MARGIN, height_margin: float = SEARCH_REGION_HEIGHT_MARGIN,
    max_frame_fraction: float = SEARCH_REGION_MAX_FRAME_FRACTION,
) -> Tuple[int, int, int, int]:
    """Expands a tightly-detected icon bbox into a generous search region centered on the same
    spot, clamped to the frame -- the saved region is deliberately not the exact detected box
    (see module docstring above for why). Also capped to a fraction of the frame regardless of
    the multiplier, so a large detected icon (relative to the frame) can't balloon the region
    toward the whole image -- see SEARCH_REGION_MAX_FRAME_FRACTION for why that matters."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w = min(frame_w, w * width_margin, frame_w * max_frame_fraction)
    new_h = min(frame_h, h * height_margin, frame_h * max_frame_fraction)
    new_x = max(0.0, min(cx - new_w / 2.0, frame_w - new_w))
    new_y = max(0.0, min(cy - new_h / 2.0, frame_h - new_h))
    return (int(round(new_x)), int(round(new_y)), int(round(new_w)), int(round(new_h)))


def crop_to_content(reference_gray: np.ndarray, margin: int = 4) -> np.ndarray:
    """
    Automatic background removal for an uploaded reference image (spec: "the original image
    must remain untouched" -- this returns a new array, never modifies the input). A typical
    uploaded reference is a bright skull icon on a large, mostly-black/uniform background;
    including that background margin in the template hurts matching (the detector would be
    comparing large empty regions, not just the shape that actually matters). Otsu-thresholds
    the image and crops to the bounding box of the largest bright region, with a small margin.
    Falls back to the original image untouched if thresholding finds nothing sensible to crop
    (e.g. an already-tightly-cropped reference, or one with an unusual background) -- this is a
    best-effort convenience, never a hard requirement for matching to work.
    """
    _, binary = cv2.threshold(reference_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The skull is the BRIGHT shape on a dark background in every reference seen so far; if
    # thresholding came out mostly-white instead (a bright background, dark icon), invert so
    # "foreground" consistently means "the icon", not "whichever happened to threshold white".
    if cv2.countNonZero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return reference_gray

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    h_img, w_img = reference_gray.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return reference_gray
    return reference_gray[y0:y1, x0:x1].copy()


def mode_confidence_breakdown(
    reference_gray: np.ndarray,
    cropped_frames_bgr: List[np.ndarray],
    scales: Optional[List[float]] = None,
    modes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Diagnostic only, never used by real detection: for each preprocessing mode (grayscale/
    high-contrast/threshold/inverted-threshold), the single best confidence found across every
    given (already region-cropped) frame. SkullDetector.find_matches() already tries every mode
    together internally and keeps only the single overall-best result -- this exists purely to
    show a calibration diagnostic WHICH representation actually finds a given reference best
    (e.g. "is grayscale carrying gold detection, or is it really the edge/threshold variant"),
    without changing how real detection combines them.
    """
    modes = modes or DEFAULT_PREPROCESSING_MODES
    scales = scales or LOCATE_SCALES
    best_by_mode = {mode: 0.0 for mode in modes}
    ref_h, ref_w = reference_gray.shape[:2]

    for frame_bgr in cropped_frames_bgr:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_variants = preprocess_variants(gray, modes)
        for s in scales:
            tw, th = max(1, round(ref_w * s)), max(1, round(ref_h * s))
            if tw < 6 or th < 6 or tw > gray.shape[1] or th > gray.shape[0]:
                continue
            resized_ref = cv2.resize(reference_gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            ref_variants = preprocess_variants(resized_ref, modes)
            for mode in modes:
                frame_variant, ref_variant = frame_variants.get(mode), ref_variants.get(mode)
                if frame_variant is None or ref_variant is None:
                    continue
                if frame_variant.shape[0] < ref_variant.shape[0] or frame_variant.shape[1] < ref_variant.shape[1]:
                    continue
                match = cv2.matchTemplate(frame_variant, ref_variant, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(match)
                if max_val > best_by_mode[mode]:
                    best_by_mode[mode] = float(max_val)

    return best_by_mode
