"""
detectors/reference_profile.py
--------------------------------
A calibrated skull reference, kept as several coexisting representations of the SAME image rather
than one grayscale template.

The original reference pipeline read every reference with cv2.IMREAD_GRAYSCALE, so a yellow skull
became a white-looking one before anything (the CALIBRATE thumbnail, the detector) ever saw it --
the colour was discarded at load time, permanently. Here the source colour is kept and the
BACKGROUND is what gets removed (by segmentation), not the colour of the skull:

    source image  ->  segment background from skull  ->  silhouette mask (bolt/eyes stay holes)
                  ->  cleaned colour image (background pixels blanked, skull colour untouched)
                  ->  colour signature (HSV / Lab statistics of the skull's own pixels)
    derived on demand from the cleaned image: grayscale, edge, contrast-normalised, binary, inverted

Everything is derived from the reference's own pixels -- nothing is decided from the file name.
The legacy grayscale template the existing detector matches against is deliberately NOT built here
(core/calibration_store.py still builds it exactly as before), so white detection is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# A reference whose foreground has median saturation below this carries no usable hue (white,
# grey, black) -- it is described by brightness instead. Derived from the pixels, not the filename.
CHROMATIC_SATURATION_MIN = 70

_BORDER_RING = 2          # px of outer border sampled to estimate the background colour
_MIN_FOREGROUND_FRACTION = 0.02
_MAX_FOREGROUND_FRACTION = 0.92
_CROP_MARGIN = 1


@dataclass
class ColourSignature:
    """What the skull's own pixels look like, in units that survive video compression better than
    raw RGB: hue/saturation/value (OpenCV units: H 0-179, S/V 0-255) and 8-bit Lab (a/b centred on
    128 = neutral)."""
    hue: float
    saturation: float
    value: float
    lab_l: float
    lab_a: float
    lab_b: float
    hue_spread: float     # half-width (p5..p95) of the reference's own hue -- how much it varies

    @property
    def chromatic(self) -> bool:
        return self.saturation >= CHROMATIC_SATURATION_MIN

    @property
    def description(self) -> str:
        """A human-readable colour name derived purely from the measured HSV values."""
        if not self.chromatic:
            if self.value >= 170:
                return "White"
            if self.value >= 70:
                return "Grey"
            return "Black"
        h = self.hue
        bins = [(10, "Red"), (20, "Orange"), (40, "Gold / Yellow"), (85, "Green"),
                (100, "Cyan"), (130, "Blue"), (160, "Purple"), (180, "Red")]
        for upper, name in bins:
            if h < upper:
                return name
        return "Red"


@dataclass
class ReferenceProfile:
    label: str
    bgr: np.ndarray                 # cleaned colour crop: skull colour untouched, background blanked
    mask: np.ndarray                # uint8 0/255 silhouette, same shape as bgr (holes = bolt/eyes)
    signature: ColourSignature
    has_source_colour: bool         # False if the source file itself was single-channel grayscale
    background_removed: bool        # False if segmentation failed and the whole image was kept
    _derived: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # -- coexisting representations, derived from the cleaned colour image, never one instead of another --
    def gray(self) -> np.ndarray:
        if "gray" not in self._derived:
            self._derived["gray"] = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self._derived["gray"]

    def contrast_normalised(self) -> np.ndarray:
        if "clahe" not in self._derived:
            self._derived["clahe"] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(self.gray())
        return self._derived["clahe"]

    def edges(self) -> np.ndarray:
        if "edges" not in self._derived:
            self._derived["edges"] = gradient_magnitude(self.mask.astype(np.float32) / 255.0)
        return self._derived["edges"]

    def binary(self) -> np.ndarray:
        return self.mask

    def inverted(self) -> np.ndarray:
        if "inverted" not in self._derived:
            self._derived["inverted"] = cv2.bitwise_not(self.mask)
        return self._derived["inverted"]

    def silhouette(self) -> np.ndarray:
        """float32 0..1 silhouette (1 = solid skull, 0 = background / see-through holes)."""
        return self.mask.astype(np.float32) / 255.0

    @property
    def size(self) -> Tuple[int, int]:
        return self.bgr.shape[1], self.bgr.shape[0]


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def _to_bgr_and_alpha(source: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """(bgr uint8, alpha uint8-or-None, source_had_colour) from any cv2.IMREAD_UNCHANGED result."""
    img = source
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / 65535.0 if img.dtype == np.uint16 else 1.0)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, False
    if img.shape[2] == 4:
        return img[:, :, :3].copy(), img[:, :, 3].copy(), True
    return img.copy(), None, True


def _segment_foreground(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """Boolean skull mask. Real transparency wins when present; otherwise the background colour is
    estimated from the image border and everything perceptibly different from it is foreground."""
    if alpha is not None and alpha.min() < 250:
        return alpha > 128

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    ring = np.zeros((h, w), dtype=bool)
    r = min(_BORDER_RING, max(1, min(h, w) // 4))
    ring[:r, :] = ring[-r:, :] = True
    ring[:, :r] = ring[:, -r:] = True
    background = np.median(lab[ring], axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance_u8 = np.clip(distance * (255.0 / max(1.0, float(distance.max()))), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def build_reference_profile(label: str, source: np.ndarray) -> ReferenceProfile:
    """Builds every representation of one reference from its UNCHANGED source pixels (the result
    of cv2.imread(path, cv2.IMREAD_UNCHANGED)). The source file is never modified."""
    bgr, alpha, has_colour = _to_bgr_and_alpha(source)
    foreground = _segment_foreground(bgr, alpha)

    # Keep only the largest connected foreground blob (the skull itself), so a speck of
    # background texture that happened to pass the threshold can't enlarge the crop.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    background_removed = True
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        foreground = labels == biggest
    fraction = float(foreground.mean())
    if count <= 1 or not (_MIN_FOREGROUND_FRACTION <= fraction <= _MAX_FOREGROUND_FRACTION):
        # Segmentation found nothing sensible (e.g. a full-bleed image): keep the whole image
        # rather than guess -- the same "best-effort, never a hard requirement" stance
        # crop_to_content() has always taken.
        foreground = np.ones(bgr.shape[:2], dtype=bool)
        background_removed = False

    ys, xs = np.where(foreground)
    y0, y1 = max(0, ys.min() - _CROP_MARGIN), min(bgr.shape[0], ys.max() + 1 + _CROP_MARGIN)
    x0, x1 = max(0, xs.min() - _CROP_MARGIN), min(bgr.shape[1], xs.max() + 1 + _CROP_MARGIN)
    cropped_bgr = bgr[y0:y1, x0:x1].copy()
    cropped_mask = (foreground[y0:y1, x0:x1].astype(np.uint8)) * 255

    cleaned = cropped_bgr.copy()
    cleaned[cropped_mask == 0] = 0  # blank the background -- the skull's own colour stays untouched

    signature = _measure_signature(cropped_bgr, cropped_mask > 0)
    return ReferenceProfile(
        label=label, bgr=cleaned, mask=cropped_mask, signature=signature,
        has_source_colour=has_colour, background_removed=background_removed,
    )


def _measure_signature(bgr: np.ndarray, mask: np.ndarray) -> ColourSignature:
    pixels = bgr[mask].reshape(-1, 1, 3)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hue = float(np.median(hsv[:, 0]))
    spread = float((np.percentile(hsv[:, 0], 95) - np.percentile(hsv[:, 0], 5)) / 2.0)
    return ColourSignature(
        hue=hue, saturation=float(np.median(hsv[:, 1])), value=float(np.median(hsv[:, 2])),
        lab_l=float(np.median(lab[:, 0])), lab_a=float(np.median(lab[:, 1])), lab_b=float(np.median(lab[:, 2])),
        hue_spread=spread,
    )
