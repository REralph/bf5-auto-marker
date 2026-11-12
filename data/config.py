"""
data/config.py
---------------
All user-adjustable settings in one dataclass, persisted to a local JSON
file (config.json, next to main.py) so choices survive between launches.
No cloud, no telemetry -- just a flat file on disk.

Loading never trusts the file: unknown keys are dropped, wrong types fall back to the default and
numbers are clamped to the range the SETTINGS sliders allow, so a hand-edited or damaged
config.json can't make the app misbehave (or fail to start). Saving is atomic (data/safe_io.py).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

from data.safe_io import as_finite_float, atomic_write_json, read_json_dict

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")

# DaVinci Resolve's marker colours (what AddMarker() accepts).
MARKER_COLORS = [
    "Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple",
    "Fuchsia", "Rose", "Lavender", "Sky", "Mint", "Lemon", "Sand", "Cocoa", "Cream",
]

CONFIDENCE_RANGE = (0.50, 0.99)   # the SETTINGS slider's limits
FRAME_SAMPLING_RANGE = (1, 10)
MARKER_NAME_MAX_LENGTH = 60


@dataclass
class AppConfig:
    """
    Only the settings that still mean something under the multi-reference, single
    detection-region model. Colour tolerance/mode, a separate template path, and a manual
    detection-region override belonged to the old multi-example calibration system; Auto
    Highlights' pre/post-kill fields were placeholders for a feature never built; and the old
    "kill cooldown" slider stopped doing anything when event grouping became edge-triggered
    (video/video_scanner.py). Old config.json files that still carry those keys load fine --
    unknown keys are ignored.
    """
    # --- Detection ---
    confidence_threshold: float = 0.80        # CONFIDENCE_RANGE, enforced by the UI slider
    frame_sampling: int = 3                   # analyze every Nth frame
    enable_neighbor_validation: bool = True   # cross-check the immediate neighbour frames before accepting a match
    multiple_skull_detection: bool = True     # if False, only the single strongest skull per event is kept

    # --- Resolve ---
    marker_name: str = "Kill"
    marker_color: str = "Red"

    # --- App/onboarding ---
    show_welcome_guide: bool = True
    developer_mode: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AppConfig":
        """A fully valid AppConfig from whatever `data` holds (see module docstring)."""
        defaults = AppConfig()
        if not isinstance(data, dict):
            return defaults
        field_names = {f.name for f in dataclasses.fields(AppConfig)}
        merged = {**defaults.to_dict(), **{k: v for k, v in data.items() if k in field_names}}

        low, high = CONFIDENCE_RANGE
        merged["confidence_threshold"] = as_finite_float(
            merged["confidence_threshold"], defaults.confidence_threshold, low, high
        )
        sampling = as_finite_float(
            merged["frame_sampling"], float(defaults.frame_sampling), *map(float, FRAME_SAMPLING_RANGE)
        )
        merged["frame_sampling"] = int(round(sampling))

        for name in ("enable_neighbor_validation", "multiple_skull_detection", "show_welcome_guide", "developer_mode"):
            if not isinstance(merged[name], bool):
                merged[name] = getattr(defaults, name)

        name = merged["marker_name"]
        name = name.strip()[:MARKER_NAME_MAX_LENGTH] if isinstance(name, str) else ""
        merged["marker_name"] = name or defaults.marker_name
        if merged["marker_color"] not in MARKER_COLORS:
            merged["marker_color"] = defaults.marker_color
        return AppConfig(**merged)


def load_config(path: str = None) -> AppConfig:
    """Load config.json, falling back to defaults if it's missing or corrupted."""
    return AppConfig.from_dict(read_json_dict(path or CONFIG_PATH))


def save_config(config: AppConfig, path: str = None) -> None:
    """Atomically writes the settings. Raises OSError if the file can't be written (full disk,
    permissions) -- callers decide how to tell the user; the previous file is left untouched."""
    atomic_write_json(path or CONFIG_PATH, config.to_dict())
