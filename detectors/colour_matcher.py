"""
detectors/colour_matcher.py
-----------------------------
The colour-aware half of skull detection: recognises a CHROMATIC reference (e.g. the gold/yellow
skull) by combining several independent pieces of evidence instead of one grayscale template score.

Why this exists: the original detector (detectors/skull_detector.py) matches a grayscale template,
so a gold skull is only "a bright blob with a skull shape". Its template also carried the reference
screenshot's own dark background, and real gameplay gold skulls are drawn at a different size than
the calibration clips. On real gameplay it peaked at ~0.78 -- below the user's 0.79 threshold on
every frame -- while real white skulls (a clean, high-contrast icon) matched at ~0.90.

Approach -- every candidate is scored on the same frame, in the same pass, from these signals:

    STRUCTURE   the skull silhouette (bolt and eye sockets as holes) correlated against a
                COLOUR-LIKELIHOOD map of the frame -- "how gold is each pixel", derived from the
                reference's own colour signature. Independent of whatever background the game shows
                behind the icon, and of exact RGB values (compared in Lab chroma, not raw RGB).
    LUMINANCE   the same silhouette against the frame's brightness -- colour-independent structure.
    EDGE        silhouette outline against the frame's gradient magnitude.
    COLOUR      how completely the skull's solid area matches the reference colour signature.
    SIZE        matched scale sits inside the size band the calibration implies (soft penalty).
    ASPECT      the matched silhouette's shape proportions vs the reference.
    POSITION    candidate lies fully inside the calibrated search region (guaranteed by the crop,
                reported for diagnostics).

WEIGHTS are proportional to each signal's measured margin -- (mean on real gold skulls minus the 99th
percentile on real skull-free gameplay frames) divided by its spread across skull frames:

    signal      gold mean   skull-free p99   margin (SNR)   weight
    structure     0.890         0.341           27.5         0.47
    colour        0.813         0.246           15.9         0.27
    edge          0.736         0.140            8.1         0.14
    luminance     0.881         0.461            6.4         0.12

(measured on 151 steady-state frames from three independent real gold events and 67 candidates found
in 2,233 real gameplay frames; real white skulls, the hard negative -- same shape, different colour --
score under 0.07 overall). Not equal-weighted: the signals that stay stable and separate best carry the
decision; luminance and edge are supporting evidence that survives a colour shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detectors.reference_profile import ReferenceProfile, gradient_magnitude

# Lab (a,b) distance at which a pixel stops looking like the reference colour. Wide enough to absorb
# compression / HDR-SDR / brightness drift (real gameplay gold sat ~11 units from the reference),
# narrow enough that white (~96 away) and most terrain never resemble it.
CHROMA_SIGMA = 30.0
# Pixels darker than this share of the signature's lightness fade out of the likelihood map.
LIGHTNESS_FADE_LOW, LIGHTNESS_FADE_HIGH = 0.45, 0.70

SCALE_STEP = 1.06
MIN_TEMPLATE_PIXELS = 10
STRUCTURE_FLOOR = 0.30       # candidate-search floor only; the real verdict is the caller's threshold
# A window whose colour-likelihood barely rises above noise has no structure to correlate -- normalised
# correlation of a near-empty map is numerically meaningless (it can read 1.00), so it scores zero.
MIN_WINDOW_LIKELIHOOD_PEAK = 0.35
SIZE_PENALTY_FLOOR = 0.70    # size factor at the very edge of the swept scale range

# Signal weights (sum to 1). See module docstring.
WEIGHTS = {"structure": 0.47, "colour": 0.27, "edge": 0.14, "luminance": 0.12}


@dataclass
class CandidateEvidence:
    """Everything known about ONE candidate, for Developer Mode diagnostics. `signals` are the raw
    per-signal scores; `confidence` is the combined score the detector compares to the threshold."""
    profile: str
    bbox: Tuple[int, int, int, int]
    scale: float
    signals: Dict[str, float]
    confidence: float
    reason: str = ""   # "" = strong enough to be a detection; otherwise the weakest link
    expected_size: Tuple[int, int] = (0, 0)   # (w, h) the calibration implies this skull should be

    def as_line(self) -> str:
        s = self.signals
        return (
            f"{self.profile}: conf {self.confidence:.2f} | structure {s['structure']:.2f} edge {s['edge']:.2f} "
            f"luminance {s['luminance']:.2f} colour {s['colour']:.2f} size {s['size']:.2f} aspect {s['aspect']:.2f} "
            f"position {s['position']:.2f} | bbox {self.bbox} (expected ~{self.expected_size[0]}x{self.expected_size[1]}) "
            f"scale {self.scale:.2f}"
            + (f" | {self.reason}" if self.reason else "")
        )


def colour_likelihood(frame_bgr: np.ndarray, profile: ReferenceProfile) -> np.ndarray:
    """float32 0..1 map: how closely each pixel matches the reference's colour signature. Compared
    in Lab chroma (a, b) -- not raw RGB -- plus a lightness fade, so exact RGB, brightness and
    compression differences don't matter but the actual hue does."""
    sig = profile.signature
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma_distance = np.hypot(lab[:, :, 1] - sig.lab_a, lab[:, :, 2] - sig.lab_b)
    chroma_term = np.exp(-((chroma_distance / CHROMA_SIGMA) ** 2))
    lo, hi = sig.lab_l * LIGHTNESS_FADE_LOW, sig.lab_l * LIGHTNESS_FADE_HIGH
    lightness_term = np.clip((lab[:, :, 0] - lo) / max(1.0, hi - lo), 0.0, 1.0)
    return (chroma_term * lightness_term).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-6 else 0.0


def _non_max_suppress(items: List[Tuple[Tuple[int, int, int, int], float]], overlap: float = 0.3) -> List[int]:
    order = sorted(range(len(items)), key=lambda i: -items[i][1])
    keep: List[int] = []
    for i in order:
        x, y, w, h = items[i][0]
        ok = True
        for j in keep:
            X, Y, W, H = items[j][0]
            iw = max(0, min(x + w, X + W) - max(x, X))
            ih = max(0, min(y + h, Y + H) - max(y, Y))
            inter = iw * ih
            union = w * h + W * H - inter
            if union > 0 and inter / union > overlap:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


@dataclass
class FootageObservation:
    """What one chromatic reference actually looked like in the user's own test footage (the green-box
    training clips): its real on-screen size (as a scale of the reference image) and real colour."""
    scale: float
    lab_l: float
    lab_a: float
    lab_b: float
    confidence: float


def learn_footage_appearance(profile: ReferenceProfile, cropped_frames: List[np.ndarray]) -> Optional[FootageObservation]:
    """Measures how `profile` really appears in `cropped_frames` (region-cropped frames known to
    contain the skull, e.g. the frames inside a test clip's green-box span). The reference says WHAT
    the skull looks like; the footage says HOW it appears in the user's recordings -- size and colour
    -- so calibration is not a single-source-of-truth on the reference image.

    The size sweep is deliberately wide and unpenalised here (this is measuring, not detecting).
    Returns None if no frame yields a convincing match."""
    matcher = ColourSilhouetteMatcher([profile], {profile.label: (0.5, 3.5, 0.5, 3.5)})
    best: Optional[Tuple[CandidateEvidence, np.ndarray]] = None
    for frame in cropped_frames:
        for candidate in matcher.find_candidates(frame):
            if best is None or candidate.confidence > best[0].confidence:
                best = (candidate, frame)
    if best is None or best[0].confidence < 0.55:
        return None
    candidate, frame = best
    x, y, w, h = candidate.bbox
    window = frame[y:y + h, x:x + w]
    solid = colour_likelihood(window, profile) > 0.5
    if int(solid.sum()) < MIN_TEMPLATE_PIXELS:
        return None
    lab = cv2.cvtColor(window, cv2.COLOR_BGR2LAB).reshape(-1, 3)[solid.reshape(-1)]
    return FootageObservation(
        scale=candidate.scale, lab_l=float(np.median(lab[:, 0])), lab_a=float(np.median(lab[:, 1])),
        lab_b=float(np.median(lab[:, 2])), confidence=candidate.confidence,
    )


@dataclass
class ColourReferenceSet:
    """The colour information one scan needs, bundled: `matching` are the chromatic profiles the
    colour matcher searches for (with their size bands for THIS video's resolution); `labelling` is
    every calibrated profile (chromatic or not), used only to name which skull a candidate is."""
    matching: List[ReferenceProfile]
    scale_bands: Dict[str, Tuple[float, float, float, float]]
    labelling: List[ReferenceProfile]


class ColourSilhouetteMatcher:
    """Evaluates every chromatic reference profile against a (region-cropped) frame in one call.

    `scale_bands[label]` = (sweep_lo, sweep_hi, full_lo, full_hi): the range of template scales to
    sweep for that profile (relative to the reference's own pixel size), and the narrower sub-range
    that earns full size credit. Derived by the caller from the calibrated scale and the video
    resolution (see core.session.AppSession.colour_scale_band); scales outside the full-credit
    range are still searched but scored down, so an implausibly-sized match can't win."""

    def __init__(self, profiles: List[ReferenceProfile], scale_bands: Dict[str, Tuple[float, float, float, float]]):
        self.profiles = profiles
        self.scale_bands = scale_bands
        self._silhouettes: Dict[str, List[Tuple[float, np.ndarray, np.ndarray]]] = {}
        for profile in profiles:
            sweep_lo, sweep_hi = scale_bands[profile.label][:2]
            self._silhouettes[profile.label] = self._build_scaled_silhouettes(profile, (sweep_lo, sweep_hi))

    @staticmethod
    def _build_scaled_silhouettes(profile: ReferenceProfile, band: Tuple[float, float]):
        lo, hi = band
        w, h = profile.size
        silhouette = profile.silhouette()
        out = []
        s = lo
        while s <= hi * 1.0001:
            tw, th = int(round(w * s)), int(round(h * s))
            if tw >= MIN_TEMPLATE_PIXELS and th >= MIN_TEMPLATE_PIXELS:
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                template = cv2.resize(silhouette, (tw, th), interpolation=interp)
                out.append((s, template, gradient_magnitude(template)))
            s *= SCALE_STEP
        return out

    def find_candidates(self, frame_bgr: np.ndarray, floor: float = 0.0) -> List[CandidateEvidence]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gradient = gradient_magnitude(gray)
        fh, fw = gray.shape[:2]
        found: List[CandidateEvidence] = []

        for profile in self.profiles:
            likelihood = colour_likelihood(frame_bgr, profile)
            if float(likelihood.max()) < 0.15:
                continue  # nothing anywhere in this crop resembles this colour -- skip the sweep
            raw: List[Tuple[Tuple[int, int, int, int], float, float, np.ndarray, np.ndarray]] = []
            for scale, template, template_edges in self._silhouettes[profile.label]:
                th, tw = template.shape[:2]
                if fh < th or fw < tw:
                    continue
                result = cv2.matchTemplate(likelihood, template, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(result >= STRUCTURE_FLOOR)
                for y, x in zip(ys, xs):
                    raw.append(((int(x), int(y), tw, th), float(result[y, x]), scale, template, template_edges))
            if not raw:
                continue
            for i in _non_max_suppress([(r[0], r[1]) for r in raw]):
                bbox, structure, scale, template, template_edges = raw[i]
                found.append(self._score(profile, bbox, structure, scale, template, template_edges, likelihood, gray, gradient))

        return [c for c in found if c.confidence >= floor]

    def _score(
        self, profile: ReferenceProfile, bbox, structure: float, scale: float, template: np.ndarray,
        template_edges: np.ndarray, likelihood: np.ndarray, gray: np.ndarray, gradient: np.ndarray,
    ) -> CandidateEvidence:
        x, y, w, h = bbox
        window_like = likelihood[y:y + h, x:x + w]
        core = template > 0.5
        # Score colour on the skull's inner area only, so anti-aliased edge pixels (which are a mix
        # of skull and background by construction) don't drag down a genuinely gold skull.
        inner = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        core_pixels = window_like[inner] if inner.sum() >= MIN_TEMPLATE_PIXELS else window_like[core]
        colour = float(core_pixels.mean()) if core_pixels.size else 0.0
        luminance = max(0.0, _ncc(gray[y:y + h, x:x + w], template))
        edge = max(0.0, _ncc(gradient[y:y + h, x:x + w], template_edges))
        structure = max(0.0, structure) if float(window_like.max()) >= MIN_WINDOW_LIKELIHOOD_PEAK else 0.0

        sweep_lo, sweep_hi, full_lo, full_hi = self.scale_bands[profile.label]
        size = self._size_agreement(scale, sweep_lo, sweep_hi, full_lo, full_hi)
        aspect = self._aspect_agreement(profile, window_like)
        position = 1.0  # a matchTemplate hit is by construction fully inside the searched region

        blended = (
            WEIGHTS["structure"] * structure + WEIGHTS["colour"] * colour
            + WEIGHTS["edge"] * edge + WEIGHTS["luminance"] * luminance
        )
        confidence = float(np.clip(blended * (0.5 + 0.5 * size) * (0.5 + 0.5 * aspect) * position, 0.0, 1.0))
        signals = {
            "structure": structure, "colour": colour, "edge": edge, "luminance": luminance,
            "size": size, "aspect": aspect, "position": position,
        }
        centre_scale = float(np.sqrt(full_lo * full_hi))
        expected = (int(round(profile.size[0] * centre_scale)), int(round(profile.size[1] * centre_scale)))
        return CandidateEvidence(
            profile=profile.label, bbox=bbox, scale=scale, signals=signals, confidence=confidence,
            reason=self._weakest_link(signals), expected_size=expected,
        )

    @staticmethod
    def _size_agreement(scale: float, sweep_lo: float, sweep_hi: float, full_lo: float, full_hi: float) -> float:
        """1.0 inside the calibrated size band, falling linearly to SIZE_PENALTY_FLOOR at the edge
        of the swept range."""
        if full_lo <= scale <= full_hi:
            return 1.0
        if scale < full_lo:
            span = max(1e-6, full_lo - sweep_lo)
            t = min(1.0, (full_lo - scale) / span)
        else:
            span = max(1e-6, sweep_hi - full_hi)
            t = min(1.0, (scale - full_hi) / span)
        return 1.0 - (1.0 - SIZE_PENALTY_FLOOR) * t

    @staticmethod
    def _aspect_agreement(profile: ReferenceProfile, window_like: np.ndarray) -> float:
        """1.0 when the colour blob inside the window has the reference's proportions."""
        blob = (window_like > 0.5).astype(np.uint8)
        if blob.sum() < 8:
            return 0.0
        ys, xs = np.where(blob > 0)
        blob_ratio = (xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)
        ref_w, ref_h = profile.size
        ref_ratio = ref_w / float(ref_h)
        return float(max(0.0, 1.0 - abs(blob_ratio - ref_ratio) / ref_ratio))

    @staticmethod
    def _weakest_link(signals: Dict[str, float]) -> str:
        checks = [
            ("colour", 0.55, "colour mismatch"), ("structure", 0.55, "structure mismatch"),
            ("size", 0.9, "size mismatch"), ("aspect", 0.6, "aspect ratio mismatch"),
        ]
        weak = [text for key, floor, text in checks if signals[key] < floor]
        return ", ".join(weak)
