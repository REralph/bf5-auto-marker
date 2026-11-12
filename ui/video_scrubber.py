"""
ui/video_scrubber.py
-----------------------
Reusable frame-accurate video navigation widget (spec item 3): a draggable
timeline, frame/second stepping, play/pause, a "go to time" entry, and
keyboard shortcuts (arrow keys, spacebar). Used by the CALIBRATE page to
search for a clean skull example.

Reads and writes through an AppSession (core/session.py) instead of owning its own
cv2.VideoCapture -- every seek, step, or play tick calls session.load_frame(), and current
position is always session.current_frame,
never a locally tracked copy. This is what lets the current frame survive a
page switch: nothing here resets it, because nothing here owns it.

Reports position through `on_frame_changed(frame_number, frame_bgr)` -- the
frame is already decoded, so callers don't need a second read.

Playback prioritizes frame-accuracy over perfectly smooth motion: each tick
does a real decode rather than dropping frames, since this is a search tool
for finding one clean example, not a video player.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

from core.session import AppSession
from data.detection import parse_timecode


class VideoScrubber(ctk.CTkFrame):
    def __init__(self, master, colors: dict, on_frame_changed: Callable[[int, object], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.colors = colors
        self.on_frame_changed = on_frame_changed

        self.session: Optional[AppSession] = None
        self._playing = False
        self._play_job = None
        self._suppress_slider_event = False

        self.timeline_var = tk.DoubleVar(value=0)
        self.timeline_slider = ctk.CTkSlider(
            self, from_=0, to=1, number_of_steps=2000, variable=self.timeline_var,
            progress_color=colors["accent"], button_color=colors["accent"], button_hover_color=colors["accent_hover"],
            command=self._on_slider_moved,
        )
        self.timeline_slider.pack(fill="x", padx=4, pady=(4, 2))

        self.time_label = ctk.CTkLabel(
            self, text="00:00.000", font=("Segoe UI", 12, "bold"), text_color=colors["text_primary"]
        )
        self.time_label.pack()

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(pady=(4, 4))
        button_specs = [
            ("◀ 1 Frame", 90, lambda: self.step_frames(-1)),
            ("◀ 1 Sec", 80, lambda: self.step_seconds(-1)),
        ]
        for text, width, cmd in button_specs:
            ctk.CTkButton(
                controls, text=text, width=width, fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
                command=cmd,
            ).pack(side="left", padx=2)

        self.play_button = ctk.CTkButton(
            controls, text="▶ PLAY", width=80, fg_color=colors["accent"], hover_color=colors["accent_hover"],
            command=self.toggle_play,
        )
        self.play_button.pack(side="left", padx=2)

        for text, width, cmd in [("1 Sec ▶", 80, lambda: self.step_seconds(1)), ("1 Frame ▶", 90, lambda: self.step_frames(1))]:
            ctk.CTkButton(
                controls, text=text, width=width, fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
                command=cmd,
            ).pack(side="left", padx=2)

        goto_row = ctk.CTkFrame(self, fg_color="transparent")
        goto_row.pack(pady=(4, 4))
        ctk.CTkLabel(goto_row, text="GO TO TIME:", text_color=colors["text_secondary"]).pack(side="left", padx=(0, 6))
        self.goto_var = tk.StringVar(value="00:00.000")
        ctk.CTkEntry(goto_row, textvariable=self.goto_var, width=110).pack(side="left")
        ctk.CTkButton(
            goto_row, text="GO", width=50, fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            command=self._on_goto,
        ).pack(side="left", padx=4)

        self.bind("<Button-1>", lambda e: self.focus_set())
        self.bind("<Left>", lambda e: self.step_frames(-1))
        self.bind("<Right>", lambda e: self.step_frames(1))
        self.bind("<space>", lambda e: self.toggle_play())

    # ------------------------------------------------------------------
    @property
    def metadata(self):
        return self.session.metadata if self.session else None

    @property
    def current_frame(self) -> int:
        return self.session.current_frame if self.session else 0

    def attach_session(self, session: AppSession) -> None:
        """
        Point the scrubber at the shared VideoSession. Does NOT seek or
        reset anything -- the session already holds whatever frame is
        current (frame 0 right after a fresh load, or wherever the user
        last scrubbed to if this is just the page becoming visible again,
        spec item 15). This just brings the widget's own display in sync
        with state that already exists.
        """
        self.session = session
        self._stop_play()
        self._sync_display(notify=True)

    def set_frame(self, frame_number: int, notify: bool = True) -> None:
        if self.session is None:
            return
        self.session.load_frame(frame_number)
        self._sync_display(notify=notify)

    def step_frames(self, delta: int) -> None:
        self._stop_play()
        self.set_frame(self.current_frame + delta)

    def step_seconds(self, delta: float) -> None:
        if self.session is None:
            return
        self._stop_play()
        self.set_frame(self.current_frame + int(round(delta * (self.session.fps or 30.0))))

    def _sync_display(self, notify: bool) -> None:
        self._update_display()
        if notify and self.session is not None:
            self.on_frame_changed(self.session.current_frame, self.session.current_image)

    def _on_slider_moved(self, value: float) -> None:
        if self.session is None or self._suppress_slider_event:
            return
        self._stop_play()
        frame_number = int(round(value * max(1, self.session.total_frames - 1)))
        self.set_frame(frame_number)

    def _on_goto(self) -> None:
        if self.session is None:
            return
        seconds = parse_timecode(self.goto_var.get())
        if seconds is None:
            self._update_time_label()  # not a valid time: put the current time back so it's visibly rejected
            return
        self.set_frame(int(round(seconds * (self.session.fps or 30.0))))

    # ------------------------------------------------------------------
    def toggle_play(self) -> None:
        self._stop_play() if self._playing else self._start_play()

    def _start_play(self) -> None:
        if self.session is None:
            return
        self._playing = True
        self.play_button.configure(text="■ STOP")
        self._play_step()

    def _stop_play(self) -> None:
        self._playing = False
        self.play_button.configure(text="▶ PLAY")
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None

    def _play_step(self) -> None:
        if not self._playing or self.session is None:
            return
        if self.current_frame >= self.session.total_frames - 1:
            self._stop_play()
            return
        self.set_frame(self.current_frame + 1)
        interval_ms = max(15, int(1000 / (self.session.fps or 30.0)))
        self._play_job = self.after(interval_ms, self._play_step)

    # ------------------------------------------------------------------
    def _update_display(self) -> None:
        self._update_time_label()
        if self.session is not None and self.session.total_frames > 1:
            self._suppress_slider_event = True
            self.timeline_var.set(self.current_frame / (self.session.total_frames - 1))
            self._suppress_slider_event = False

    def _update_time_label(self) -> None:
        if self.session is None or not self.session.fps:
            self.time_label.configure(text="00:00.000")
            return
        text = self._format_timecode(self.current_frame / self.session.fps)
        self.time_label.configure(text=text)
        self.goto_var.set(text)

    @staticmethod
    def _format_timecode(seconds: float) -> str:
        total_ms = int(round(seconds * 1000))
        h, rem = divmod(total_ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if h else f"{m:02d}:{s:02d}.{ms:03d}"

    def release(self) -> None:
        self._stop_play()
