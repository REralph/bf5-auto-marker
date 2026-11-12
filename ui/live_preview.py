"""
ui/live_preview.py
---------------------
LiveSkullPreview: a read-only "what was just detected" view for ANALYSE's live-scan display --
image + box overlay + timestamp + skull count only, no zoom/frame-step/play/event-navigation
controls (those belong to PreviewPanel, used on PREVIEW where manual inspection is the point).
Always shows the single most recent event; no selection state of its own, matching the
simplified workflow's "no manual selection required" rule. Shares its box-drawing logic with
PreviewPanel via ui/preview.py's build_overlay_image() so the two widgets never draw boxes
differently, and hides the raw confidence percentage baked into PreviewPanel's per-skull
labels (the simplified ANALYSE view hides CV internals; PREVIEW is explicitly allowed to keep
them since that's where a user manually inspects a detection).
"""

from __future__ import annotations

from typing import List, Optional

import customtkinter as ctk

from data.detection import Event
from ui.preview import build_overlay_image
from video.video_probe import extract_frame

PREVIEW_WIDTH = 420
PREVIEW_HEIGHT = 260


class LiveSkullPreview(ctk.CTkFrame):
    def __init__(self, master, colors: dict, **kwargs):
        super().__init__(master, fg_color=colors["panel"], corner_radius=10, **kwargs)
        self.colors = colors
        self.video_path: Optional[str] = None
        self._shown_key = None  # what is currently displayed, so an unchanged event isn't re-decoded

        self.image_label = ctk.CTkLabel(
            self, text="Waiting for first detection...", text_color=colors["text_secondary"],
            width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, fg_color=colors["panel_alt"],
        )
        self.image_label.pack(padx=14, pady=(14, 6))
        self.caption_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 13, "bold"), text_color=colors["text_primary"],
        )
        self.caption_label.pack(pady=(0, 14))

    def set_video(self, video_path: Optional[str]) -> None:
        if video_path != self.video_path:
            self._shown_key = None
        self.video_path = video_path

    def show_latest(self, events: List[Event]) -> None:
        if not events or not self.video_path:
            self._shown_key = None
            self.image_label.configure(image=None, text="Waiting for first detection...")
            self.caption_label.configure(text="")
            return

        event = events[-1]
        # Called on every analysis progress tick (hundreds per video) as well as on every new event;
        # decoding a frame out of a large recording is slow, so only do it when the latest event
        # (or the video) actually changed.
        key = (self.video_path, id(event), event.event_number, event.preview_frame)
        if key == self._shown_key:
            return
        # preview_frame (the clearest/best-confidence frame), not frame_number (now the bookmark
        # -- the event's START, which may be a weaker partial glimpse of the skull).
        frame = extract_frame(self.video_path, event.preview_frame)
        if frame is None:
            self._shown_key = None  # try again next time rather than caching a failure
            self.image_label.configure(image=None, text="Frame unavailable")
            return

        pil_image = build_overlay_image(frame, event, box_mode=False, show_confidence=False)
        base_w, base_h = pil_image.size
        fit_scale = min(PREVIEW_WIDTH / base_w, PREVIEW_HEIGHT / base_h, 1.0)
        new_size = (max(1, int(base_w * fit_scale)), max(1, int(base_h * fit_scale)))
        pil_image = pil_image.resize(new_size)

        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image  # keep a reference so Tk doesn't garbage-collect it
        self._shown_key = key

        skull_noun = "skull" if event.skull_count == 1 else "skulls"
        self.caption_label.configure(text=f"{event.timestamp_display}   —   {event.skull_count} {skull_noun}")
