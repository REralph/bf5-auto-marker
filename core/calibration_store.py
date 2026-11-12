"""
core/calibration_store.py
----------------------------
Persistence for the permanent calibration: one detection region, a small
library of calibrated skull reference images (e.g. White + Gold -- multiple
appearances of the same skull, never separate detection systems), and a
small library of test-footage clips. Layout on disk:

    calibration/
        calibration.json        -- detection_region, resolution, per-reference scales
        references/
            <any number of user-provided reference images, copied in by add_reference_image()>
        test_footage/
            <any number of user-provided clips, copied in by add_test_footage()>

Every reference image and test-footage clip is a real copy, never a path
reference, so the app keeps working if the user's original files are later
moved, renamed, or deleted. Both libraries are purely folder-based: whatever
files physically exist in references/ or test_footage/ ARE the library, so
there's no separate list to fall out of sync with what's actually on disk.

Three migrations run automatically (via load_into()), oldest first, so a
calibration completed under an earlier version of this app is never
silently lost just because the storage layout changed again:
1. A much older multi-example-list format (calibration/manifest.json with
   `video_width`/`examples`, images under calibration/skull_examples/).
2. The flat single-file layout used two passes ago (bare
   calibration/skull_reference.png, calibration/manifest.json,
   calibration/test_skull.mp4).
3. The single-reference-image library used one pass ago (already the
   references/ + test_footage/ + calibration.json layout, but with one
   singular `reference_scale` key instead of a `reference_scales` dict --
   handled inline in load_into(), no separate migration function needed
   since the file layout itself didn't change, only the JSON shape).
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from typing import List, Optional

import cv2
import numpy as np

from core.session import ReferenceImage
from detectors.colour_matcher import FootageObservation
from detectors.reference_profile import ReferenceProfile, build_reference_profile
from data.safe_io import as_finite_float, atomic_write_json, read_json_dict
from detectors.skull_detector import crop_to_content

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_DIR = os.path.join(APP_DIR, "calibration")
REFERENCES_DIR = os.path.join(CALIBRATION_DIR, "references")
TEST_FOOTAGE_DIR = os.path.join(CALIBRATION_DIR, "test_footage")
MANIFEST_PATH = os.path.join(CALIBRATION_DIR, "calibration.json")

_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Pass-before-last's flat layout (bare files directly under calibration/) -- migrated into the
# references/ + test_footage/ + calibration.json layout above the first time load_into() runs
# after this update.
_FLAT_REFERENCE_IMAGE_PATH = os.path.join(CALIBRATION_DIR, "skull_reference.png")
_FLAT_MANIFEST_PATH = os.path.join(CALIBRATION_DIR, "manifest.json")
_FLAT_TEST_VIDEO_PATH = os.path.join(CALIBRATION_DIR, "test_skull.mp4")
# The previous pass's single-reference library location -- same references/ folder used today,
# so no file migration is needed for it, only the JSON-shape handling in load_into() below.
_SINGLE_REFERENCE_IMAGE_PATH = os.path.join(REFERENCES_DIR, "skull_reference.png")

# The oldest format of all (predates even the flat layout above).
_OLD_EXAMPLES_DIR = os.path.join(CALIBRATION_DIR, "skull_examples")
_OLD_VIDEO_REFERENCE_FILENAME = "skull_from_video.png"


# Bounds for values read back from calibration.json -- a hand-edited or damaged file must not be able
# to crash the app or silently derail detection/marker placement.
MAX_BOOKMARK_ADJUSTMENT_SECONDS = 600.0
_SCALE_RANGE = (0.01, 20.0)


def _valid_region(value):
    """(x, y, w, h) as ints if `value` is a plausible detection region, else None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v for v in value):
        return None
    x, y, w, h = (int(round(v)) for v in value)
    if x < 0 or y < 0 or w < 1 or h < 1:
        return None
    return (x, y, w, h)


def _nonnegative_int(value) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and value >= 0 else 0


def _footage_observation(value):
    """The learned FootageObservation stored for one reference, or None if it's missing/malformed."""
    if not isinstance(value, dict):
        return None
    try:
        return FootageObservation(**{k: float(value[k]) for k in ("scale", "lab_l", "lab_a", "lab_b", "confidence")})
    except (KeyError, TypeError, ValueError):
        return None


def load_into(session) -> bool:
    """Loads the saved region + every calibrated reference image directly into an AppSession.
    Returns True if a calibration was found and loaded (even partially), False if nothing was
    saved yet. Each reference is background-isolated (crop_to_content()) here at load time, not
    baked permanently into the file on disk -- the stored copy stays exactly as it was added."""
    _migrate_very_old_examples_format_if_needed()
    _migrate_flat_layout_if_needed()

    found = False
    data = _read_manifest()
    region = _valid_region(data.get("detection_region"))
    if region:
        session.detection_region = region
        session.calibration_video_width = _nonnegative_int(data.get("calibration_video_width"))
        session.calibration_video_height = _nonnegative_int(data.get("calibration_video_height"))
        found = True

    session.bookmark_adjustment_seconds = as_finite_float(
        data.get("bookmark_adjustment_seconds"), 0.0, -MAX_BOOKMARK_ADJUSTMENT_SECONDS, MAX_BOOKMARK_ADJUSTMENT_SECONDS
    )

    scales = data.get("reference_scales")
    scales = scales if isinstance(scales, dict) else {}
    footage_learned = data.get("reference_footage")
    footage_learned = footage_learned if isinstance(footage_learned, dict) else {}
    legacy_scale = data.get("reference_scale")  # the previous pass's singular key
    filenames = list_reference_images()
    session.reference_images = []
    for filename in filenames:
        image = load_reference_gray(filename)
        if image is None:
            continue
        label = os.path.splitext(filename)[0]
        if label in scales:
            scale = as_finite_float(scales[label], 1.0, *_SCALE_RANGE)
        elif legacy_scale is not None and len(filenames) == 1:
            scale = as_finite_float(legacy_scale, 1.0, *_SCALE_RANGE)
        else:
            scale = 1.0
        session.reference_images.append(ReferenceImage(
            label=label, image=crop_to_content(image), scale=scale, profile=load_reference_profile(filename),
            footage=_footage_observation(footage_learned.get(label)),
        ))
        found = True

    return found


def _migrate_very_old_examples_format_if_needed() -> None:
    if os.path.isfile(_FLAT_REFERENCE_IMAGE_PATH) or list_reference_images():
        return  # a newer format already has a reference image -- nothing to migrate
    data = _read_json(_FLAT_MANIFEST_PATH)
    if "examples" not in data and "video_width" not in data:
        return  # not the old format (or no calibration at all yet)

    region = data.get("detection_region")
    old_width = data.get("video_width", 0)
    old_height = data.get("video_height", 0)

    # Prefer the video-captured reference (closest to "the real skull as it actually appeared"
    # for this user); fall back to the first uploaded example if that one isn't present.
    examples = data.get("examples", [])
    chosen_filename = None
    if any(e.get("image_filename") == _OLD_VIDEO_REFERENCE_FILENAME for e in examples):
        chosen_filename = _OLD_VIDEO_REFERENCE_FILENAME
    elif examples:
        chosen_filename = examples[0].get("image_filename")

    migrated_image = False
    if chosen_filename:
        old_path = os.path.join(_OLD_EXAMPLES_DIR, chosen_filename)
        if os.path.isfile(old_path):
            image = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                os.makedirs(CALIBRATION_DIR, exist_ok=True)
                cv2.imwrite(_FLAT_REFERENCE_IMAGE_PATH, image)
                migrated_image = True

    if region or migrated_image:
        _write_json(_FLAT_MANIFEST_PATH, {
            "detection_region": region,
            "calibration_video_width": old_width,
            "calibration_video_height": old_height,
        })


def _migrate_flat_layout_if_needed() -> None:
    if os.path.isfile(_FLAT_REFERENCE_IMAGE_PATH) and not list_reference_images():
        os.makedirs(REFERENCES_DIR, exist_ok=True)
        shutil.copyfile(_FLAT_REFERENCE_IMAGE_PATH, _SINGLE_REFERENCE_IMAGE_PATH)

    if os.path.isfile(_FLAT_MANIFEST_PATH) and not os.path.isfile(MANIFEST_PATH):
        _write_manifest(_read_json(_FLAT_MANIFEST_PATH))

    if os.path.isfile(_FLAT_TEST_VIDEO_PATH) and not list_test_footage():
        os.makedirs(TEST_FOOTAGE_DIR, exist_ok=True)
        shutil.copyfile(_FLAT_TEST_VIDEO_PATH, os.path.join(TEST_FOOTAGE_DIR, os.path.basename(_FLAT_TEST_VIDEO_PATH)))


def save_region(session) -> None:
    """Persists detection_region + the resolution it was drawn at (needed later to scale it for a
    differently-sized future video), plus each calibrated reference's own learned scale (keyed by
    label, so a fresh app launch applies the same real-world size each reference was found at)."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    data = _read_manifest()
    data["detection_region"] = list(session.detection_region) if session.detection_region else None
    data["calibration_video_width"] = session.calibration_video_width
    data["calibration_video_height"] = session.calibration_video_height
    data["reference_scales"] = {ref.label: ref.scale for ref in session.reference_images}
    data["reference_footage"] = {
        ref.label: dataclasses.asdict(ref.footage) for ref in session.reference_images if ref.footage is not None
    }
    data.pop("reference_scale", None)  # superseded singular key from the previous pass
    data["bookmark_adjustment_seconds"] = session.bookmark_adjustment_seconds
    _write_manifest(data)


# ----------------------------------------------------------------------
# REFERENCE IMAGE LIBRARY -- one or more calibrated skull appearances (e.g. White + Gold), all
# combined into a single multi-template detector rather than separate detection systems. Purely
# folder-based, mirroring the test-footage library below: whatever image files exist in
# calibration/references/ ARE the library.
# ----------------------------------------------------------------------
def list_reference_images() -> List[str]:
    """Filenames (not full paths) of every calibrated reference image, sorted for a stable
    display order."""
    if not os.path.isdir(REFERENCES_DIR):
        return []
    return sorted(
        name for name in os.listdir(REFERENCES_DIR)
        if name.lower().endswith(_IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(REFERENCES_DIR, name))
    )


def reference_image_path(filename: str) -> str:
    return os.path.join(REFERENCES_DIR, filename)


def load_reference_gray(filename: str) -> Optional[np.ndarray]:
    """The grayscale reference the original detector matches against. Identical to a plain
    cv2.IMREAD_GRAYSCALE read for every opaque image (all current references). The one difference:
    an image with REAL transparency is composited onto black first -- IMREAD_GRAYSCALE ignores alpha,
    so a transparent PNG's hidden background colour (often white) would otherwise become part of the
    template."""
    path = reference_image_path(filename)
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape[2] == 4 and raw[:, :, 3].min() < 250:
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        composited = (raw[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        return cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_reference_profile(filename: str) -> Optional[ReferenceProfile]:
    """The colour-preserving view of a stored reference: read UNCHANGED (colour and any alpha kept,
    unlike the grayscale template load_into() also builds), background segmented out, colour
    signature measured. None if the file can't be read."""
    raw = cv2.imread(reference_image_path(filename), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return build_reference_profile(os.path.splitext(filename)[0], raw)


def add_reference_image(source_path: str) -> str:
    """UPLOAD REFERENCE IMAGE: copies an image into the permanent library, returning its stored
    filename (never the original path -- the user's own original file is never touched or
    deleted). A name collision gets a numeric suffix rather than silently overwriting a
    differently-sourced file that happens to share a filename."""
    os.makedirs(REFERENCES_DIR, exist_ok=True)
    base_name = os.path.basename(source_path)
    stem, ext = os.path.splitext(base_name)
    dest_name = base_name
    counter = 2
    while os.path.isfile(os.path.join(REFERENCES_DIR, dest_name)):
        if os.path.abspath(os.path.join(REFERENCES_DIR, dest_name)) == os.path.abspath(source_path):
            return dest_name  # already the permanent copy -- nothing to do
        dest_name = f"{stem} ({counter}){ext}"
        counter += 1
    shutil.copyfile(source_path, os.path.join(REFERENCES_DIR, dest_name))
    return dest_name


def remove_reference_image(filename: str) -> None:
    path = os.path.join(REFERENCES_DIR, filename)
    if os.path.isfile(path):
        os.remove(path)


def replace_reference_image(old_filename: str, source_path: str) -> str:
    """REPLACE: drops the old library entry and adds the new file fresh. Returns the new entry's
    stored filename."""
    remove_reference_image(old_filename)
    return add_reference_image(source_path)


# ----------------------------------------------------------------------
# TEST FOOTAGE LIBRARY -- a small permanent set of short calibration clips (spec: "1-3 clips
# representing different skull appearances"). Purely folder-based: whatever video files exist in
# calibration/test_footage/ ARE the library, so there's no separate list to keep in sync with the
# files actually on disk.
# ----------------------------------------------------------------------
def list_test_footage() -> List[str]:
    """Filenames (not full paths) of every test-footage clip in the library, sorted for a stable
    display order."""
    if not os.path.isdir(TEST_FOOTAGE_DIR):
        return []
    return sorted(
        name for name in os.listdir(TEST_FOOTAGE_DIR)
        if name.lower().endswith(_VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(TEST_FOOTAGE_DIR, name))
    )


def test_footage_path(filename: str) -> str:
    return os.path.join(TEST_FOOTAGE_DIR, filename)


def add_test_footage(source_path: str) -> str:
    """ADD TEST VIDEO: copies a clip into the permanent library, returning its stored filename
    (never the original path, per "do not permanently depend on the original Downloads/Desktop
    location"). A name collision gets a numeric suffix rather than silently overwriting a
    differently-sourced clip that happens to share a filename."""
    os.makedirs(TEST_FOOTAGE_DIR, exist_ok=True)
    base_name = os.path.basename(source_path)
    stem, ext = os.path.splitext(base_name)
    dest_name = base_name
    counter = 2
    while os.path.isfile(os.path.join(TEST_FOOTAGE_DIR, dest_name)):
        if os.path.abspath(os.path.join(TEST_FOOTAGE_DIR, dest_name)) == os.path.abspath(source_path):
            return dest_name  # already the permanent copy -- nothing to do
        dest_name = f"{stem} ({counter}){ext}"
        counter += 1
    shutil.copyfile(source_path, os.path.join(TEST_FOOTAGE_DIR, dest_name))
    return dest_name


def remove_test_footage(filename: str) -> None:
    path = os.path.join(TEST_FOOTAGE_DIR, filename)
    if os.path.isfile(path):
        os.remove(path)


def replace_test_footage(old_filename: str, source_path: str) -> str:
    """REPLACE: drops the old library entry and adds the new file fresh (rather than forcing the
    new content under the old filename/extension, which could mismatch the new file's actual
    container format). Returns the new entry's stored filename."""
    remove_test_footage(old_filename)
    return add_test_footage(source_path)


def reset() -> None:
    """RESET CALIBRATION: clears the saved region and every reference image and test-footage clip
    from disk -- all together, so a reset calibration can't be mistaken for a completed one next
    launch just because old files were still sitting in calibration/.

    Every deletion is attempted even if one fails (e.g. a clip another program has open); an OSError
    naming whatever could not be removed is raised at the end, rather than stopping half-way."""
    failures = []

    def attempt(description, action):
        try:
            action()
        except OSError as exc:
            failures.append(f"{description}: {exc}")

    if os.path.isfile(MANIFEST_PATH):
        attempt("calibration.json", lambda: os.remove(MANIFEST_PATH))
    for filename in list_reference_images():
        attempt(filename, lambda f=filename: remove_reference_image(f))
    for filename in list_test_footage():
        attempt(filename, lambda f=filename: remove_test_footage(f))
    if failures:
        raise OSError("Could not remove: " + "; ".join(failures))


def _read_manifest() -> dict:
    return _read_json(MANIFEST_PATH)


def _write_manifest(data: dict) -> None:
    _write_json(MANIFEST_PATH, data)


def _read_json(path: str) -> dict:
    return read_json_dict(path)


def _write_json(path: str, data: dict) -> None:
    atomic_write_json(path, data)
