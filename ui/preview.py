"""
ui/preview.py
--------------
Frame-preview panel (spec items 8, 10-12, 16): shows the actual video frame
for the selected event with bounding boxes drawn around every detected
skull (colour-labelled), so a genuine multi-kill can be told apart from a
false positive by eye. Supports:

  - Previous/Next Event -- step between detected events (table selection)
  - Previous/Next Frame -- scrub +/-1 raw frame around the event without
    losing the event selection, to inspect exactly when the icon appeared
  - Play -- flips through the few frames surrounding the event
  - Zoom In/Out, Fit to Screen, 100% -- for the full annotated frame
  - Zoom Detection -- crops in tight on each skull individually and shows
    them enlarged side by side, to make the skull count easy to verify
"""

from __future__ import annotations

from typing import Callable, List, Optional

import customtkinter as ctk
import cv2
from PIL import Image, ImageDraw

from data.detection import Event
from video.video_probe import extract_frame

PREVIEW_WIDTH = 420
PREVIEW_HEIGHT = 260
_BOX_COLOR = (255, 92, 57)  # matches the app's accent, in RGB (PIL, not BGR)
_BOX_BG = (18, 18, 20)


def build_overlay_image(frame_bgr, event: Event, box_mode: bool = False, show_confidence: bool = True) -> Image.Image:
    """
    Draws one box per detected skull (or, if box_mode=True, one box around the whole event)
    onto a full-resolution copy of the frame -- no scaling applied here. Shared between
    PreviewPanel (which applies its own zoom/fit scaling on top) and ANALYSE's lightweight
    live-scan preview (ui/live_preview.py, which just fits it to a small fixed size and, per
    the simplified workflow, hides the raw confidence percentage via show_confidence=False).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    if box_mode and event.skulls:
        # One box around the whole event -- only when explicitly requested, the default is
        # always one box per individual skull, never a single giant box.
        xs = [s.bbox[0] for s in event.skulls]
        ys = [s.bbox[1] for s in event.skulls]
        x2s = [s.bbox[0] + s.bbox[2] for s in event.skulls]
        y2s = [s.bbox[1] + s.bbox[3] for s in event.skulls]
        x, y, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)
        draw.rectangle([x, y, x2, y2], outline=_BOX_COLOR, width=3)
        label = f"Event #{event.event_number} — {event.skull_count} skull(s)"
        text_w = 7 * len(label) + 6
        label_y = max(0, y - 18)
        draw.rectangle([x, label_y, x + text_w, label_y + 16], fill=_BOX_COLOR)
        draw.text((x + 3, label_y + 1), label, fill=_BOX_BG)
    else:
        for skull in event.skulls:
            x, y, w, h = skull.bbox
            draw.rectangle([x, y, x + w, y + h], outline=_BOX_COLOR, width=3)
            label = f"{skull.color_name} {round(skull.shape_confidence * 100)}%" if show_confidence else skull.color_name
            text_w = 7 * len(label) + 6
            label_y = max(0, y - 18)
            draw.rectangle([x, label_y, x + text_w, label_y + 16], fill=_BOX_COLOR)
            draw.text((x + 3, label_y + 1), label, fill=_BOX_BG)

    return pil_image


class PreviewPanel(ctk.CTkFrame):
    def __init__(self, master, colors: dict, on_navigate_event: Callable[[int], None], **kwargs):
        super().__init__(master, fg_color=colors["panel"], corner_radius=10, **kwargs)
        self.colors = colors
        self.on_navigate_event = on_navigate_event

        self.video_path: Optional[str] = None
        self.events: List[Event] = []
        self.current_index: Optional[int] = None
        self.frame_offset = 0
        self.zoom_factor: float = 1.0  # multiplier relative to "fit" baseline
        self.zoom_is_native = False    # True when "100%" was explicitly requested
        self.zoom_detection_mode = False
        self.show_event_region_mode = False  # False = one box per skull (default); True = one box around the whole event
        self._last_base_size = (PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self._play_job = None
        self._play_offset = 0

        ctk.CTkLabel(
            self, text="PREVIEW", font=("Segoe UI", 13, "bold"), text_color=colors["text_secondary"]
        ).pack(anchor="w", padx=14, pady=(12, 4))

        self.image_scroll = ctk.CTkScrollableFrame(
            self, fg_color=colors["panel_alt"], corner_radius=8, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT
        )
        self.image_scroll.pack(padx=14, pady=4)
        self.image_label = ctk.CTkLabel(self.image_scroll, text="No detection selected", text_color=colors["text_secondary"])
        self.image_label.pack()

        self.marker_index_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 12, "bold"), text_color=colors["text_secondary"]
        )
        self.marker_index_label.pack(pady=(8, 0))

        self.timestamp_label = ctk.CTkLabel(
            self, text="--:--.---", font=("Segoe UI", 16, "bold"), text_color=colors["text_primary"]
        )
        self.timestamp_label.pack(pady=(6, 2))
        self.detail_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 12), text_color=colors["text_secondary"], wraplength=380
        )
        self.detail_label.pack(pady=(0, 8))

        zoom_row = ctk.CTkFrame(self, fg_color="transparent")
        zoom_row.pack(pady=(0, 6))
        for label, command in (
            ("Zoom In", self._zoom_in),
            ("Zoom Out", self._zoom_out),
            ("Fit to Screen", self._zoom_fit),
            ("100%", self._zoom_native),
        ):
            ctk.CTkButton(
                zoom_row, text=label, width=88, height=26, font=("Segoe UI", 11),
                fg_color=colors["panel_alt"], hover_color=colors["accent_dim"], command=command,
            ).pack(side="left", padx=2)

        toggle_row = ctk.CTkFrame(self, fg_color="transparent")
        toggle_row.pack(pady=(0, 8))
        self.zoom_detection_btn = ctk.CTkButton(
            toggle_row, text="🔍 Zoom Detection", height=28, fg_color=colors["panel_alt"],
            hover_color=colors["accent_dim"], command=self._toggle_zoom_detection,
        )
        self.zoom_detection_btn.pack(side="left", padx=3)
        self.event_region_btn = ctk.CTkButton(
            toggle_row, text="▭ Show Event Region", height=28, fg_color=colors["panel_alt"],
            hover_color=colors["accent_dim"], command=self._toggle_event_region,
        )
        self.event_region_btn.pack(side="left", padx=3)

        nav_row1 = ctk.CTkFrame(self, fg_color="transparent")
        nav_row1.pack(pady=(0, 4))
        ctk.CTkButton(
            nav_row1, text="◀ Previous Frame", width=130, fg_color=colors["panel_alt"],
            hover_color=colors["accent_dim"], command=lambda: self._navigate_frame(-1),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            nav_row1, text="Next Frame ▶", width=130, fg_color=colors["panel_alt"],
            hover_color=colors["accent_dim"], command=lambda: self._navigate_frame(1),
        ).pack(side="left", padx=3)

        nav_row2 = ctk.CTkFrame(self, fg_color="transparent")
        nav_row2.pack(pady=(0, 14))
        ctk.CTkButton(
            nav_row2, text="◀ Previous Event", width=120, fg_color=colors["panel_alt"],
            hover_color=colors["accent_dim"], command=lambda: self.on_navigate_event(-1),
        ).pack(side="left", padx=3)
        self.play_button = ctk.CTkButton(
            nav_row2, text="▶ Play", width=70, fg_color=colors["accent"],
            hover_color=colors["accent_hover"], command=self._toggle_play,
        )
        self.play_button.pack(side="left", padx=3)
        ctk.CTkButton(
            nav_row2, text="Next Event ▶", width=120, fg_color=colors["panel_alt"],
            hover_color=colors["accent_dim"], command=lambda: self.on_navigate_event(1),
        ).pack(side="left", padx=3)

    # ------------------------------------------------------------------
    def set_video(self, video_path: Optional[str]) -> None:
        self.video_path = video_path

    def show_detection(self, events: List[Event], index: Optional[int], empty_message: Optional[str] = None) -> None:
        self._stop_play()
        self.events = events
        self.current_index = index
        self.frame_offset = 0
        self._empty_message = empty_message or "No detection selected"
        self._render()

    def _current_event(self) -> Optional[Event]:
        if self.current_index is None or not (0 <= self.current_index < len(self.events)):
            return None
        return self.events[self.current_index]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self) -> None:
        event = self._current_event()
        if event is None or not self.video_path:
            self.image_label.configure(image=None, text=getattr(self, "_empty_message", "No detection selected"))
            self.marker_index_label.configure(text="")
            self.timestamp_label.configure(text="--:--.---")
            self.detail_label.configure(text="")
            return

        self.marker_index_label.configure(text=f"MARKER {self.current_index + 1} OF {len(self.events)}")
        # preview_frame (the clearest/best-confidence frame), not frame_number (now the bookmark
        # -- the event's START, which may be a weaker partial glimpse) -- and the same frame the
        # overlay bbox below was computed against, so the box actually lines up with the image.
        frame_number = max(0, event.preview_frame + self.frame_offset)
        frame = extract_frame(self.video_path, frame_number)
        if frame is None:
            self.image_label.configure(image=None, text="Frame unavailable")
            return

        if self.zoom_detection_mode:
            pil_image = self._build_zoom_detection_image(frame, event)
        else:
            pil_image = self._build_overlay_image(frame, event)

        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image  # keep a reference so Tk doesn't garbage-collect it

        self.timestamp_label.configure(text=event.timestamp_display)
        offset_note = f"   (frame {'+' if self.frame_offset >= 0 else ''}{self.frame_offset})" if self.frame_offset else ""
        self.detail_label.configure(
            text=(
                f"Event #{event.event_number}   Skulls: {event.skull_count}   "
                f"Colours: {event.colors_display}   Confidence: {event.confidence_percent}%{offset_note}"
            )
        )

    def _build_overlay_image(self, frame_bgr, event: Event) -> Image.Image:
        pil_image = build_overlay_image(frame_bgr, event, box_mode=self.show_event_region_mode)
        base_w, base_h = pil_image.size
        self._last_base_size = (base_w, base_h)
        fit_scale = min(PREVIEW_WIDTH / base_w, PREVIEW_HEIGHT / base_h, 1.0)
        final_scale = 1.0 if self.zoom_is_native else fit_scale * self.zoom_factor
        new_size = (max(1, int(base_w * final_scale)), max(1, int(base_h * final_scale)))
        return pil_image.resize(new_size)

    def _build_zoom_detection_image(self, frame_bgr, event: Event) -> Image.Image:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        pad = 14
        target_h = int(180 * self.zoom_factor)
        crops = []
        for skull in event.skulls:
            x, y, w, h = skull.bbox
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(pil_image.width, x + w + pad), min(pil_image.height, y + h + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = pil_image.crop((x0, y0, x1, y1))
            scale = target_h / crop.height
            crop = crop.resize((max(1, int(crop.width * scale)), target_h))
            crops.append((crop, skull.color_name))

        if not crops:
            return pil_image.resize((240, 160))

        gap = 12
        total_w = sum(c.width for c, _ in crops) + gap * (len(crops) - 1)
        combined = Image.new("RGB", (total_w, target_h + 22), _BOX_BG)
        draw = ImageDraw.Draw(combined)
        x_off = 0
        for crop, color_name in crops:
            combined.paste(crop, (x_off, 0))
            draw.rectangle([x_off, 0, x_off + crop.width - 1, target_h - 1], outline=_BOX_COLOR, width=2)
            draw.text((x_off + 4, target_h + 4), color_name, fill=(240, 240, 240))
            x_off += crop.width + gap
        return combined

    # ------------------------------------------------------------------
    # Zoom controls
    # ------------------------------------------------------------------
    def _zoom_in(self) -> None:
        if self.zoom_is_native:
            base_w, base_h = self._last_base_size
            fit_scale = min(PREVIEW_WIDTH / base_w, PREVIEW_HEIGHT / base_h, 1.0)
            self.zoom_factor = (1.0 / fit_scale) if fit_scale > 0 else 1.0
            self.zoom_is_native = False
        self.zoom_factor = min(4.0, self.zoom_factor * 1.25)
        self._render()

    def _zoom_out(self) -> None:
        if self.zoom_is_native:
            base_w, base_h = self._last_base_size
            fit_scale = min(PREVIEW_WIDTH / base_w, PREVIEW_HEIGHT / base_h, 1.0)
            self.zoom_factor = (1.0 / fit_scale) if fit_scale > 0 else 1.0
            self.zoom_is_native = False
        self.zoom_factor = max(0.25, self.zoom_factor / 1.25)
        self._render()

    def _zoom_fit(self) -> None:
        self.zoom_factor = 1.0
        self.zoom_is_native = False
        self._render()

    def _zoom_native(self) -> None:
        self.zoom_is_native = True
        self._render()

    def _toggle_zoom_detection(self) -> None:
        self.zoom_detection_mode = not self.zoom_detection_mode
        self.zoom_detection_btn.configure(
            fg_color=self.colors["accent"] if self.zoom_detection_mode else self.colors["panel_alt"]
        )
        self._render()

    def _toggle_event_region(self) -> None:
        self.show_event_region_mode = not self.show_event_region_mode
        self.event_region_btn.configure(
            fg_color=self.colors["accent"] if self.show_event_region_mode else self.colors["panel_alt"]
        )
        self._render()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _navigate_frame(self, direction: int) -> None:
        if self._current_event() is None:
            return
        self.frame_offset += direction
        self._render()

    def _toggle_play(self) -> None:
        if self._play_job is not None:
            self._stop_play()
            return
        if self._current_event() is None:
            return
        self.play_button.configure(text="■ Stop")
        self._play_offset = -2
        self._play_step()

    def _play_step(self) -> None:
        if self._current_event() is None:
            self._stop_play()
            return
        self.frame_offset = self._play_offset
        self._render()
        self._play_offset += 1
        if self._play_offset > 2:
            self._play_offset = -2
        self._play_job = self.after(220, self._play_step)

    def _stop_play(self) -> None:
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None
        self.play_button.configure(text="▶ Play")
        self.frame_offset = 0
