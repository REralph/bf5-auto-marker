"""
ui/pages/settings_page.py
----------------------------
SETTINGS tab: the few knobs that still mean something under the
one-detection-region, multi-reference-image model (detection thresholds,
Resolve marker/timeline settings), calibration management, and Developer
Mode. Manual colour tolerance/mode and a separate detection region override
were part of the old multi-example calibration system and no longer apply
-- CALIBRATE owns the region and reference-image library directly now.
"""

from __future__ import annotations

import os
import tkinter as tk

import customtkinter as ctk

import core.calibration_store as calibration_store
import ui.devtools as devtools
import ui.dialogs as dialogs
from data.config import MARKER_COLORS, MARKER_NAME_MAX_LENGTH, save_config
from ui.dialogs import InfoIcon

_SAVE_DELAY_MS = 400  # a slider drag fires many events; write the settings once it settles


def _open_in_explorer(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    os.startfile(path)


class SettingsPage(ctk.CTkFrame):
    def __init__(self, master, app, colors: dict):
        super().__init__(master, fg_color=colors["bg"])
        self.app = app
        self.colors = colors
        self._save_job = None
        self._save_failure_shown = False
        self._build()

    def _section(self, text: str, ref_id: str = "", pady=(18, 6)) -> ctk.CTkLabel:
        """A section heading; `ref_id` also gives it a reference code, shown right after the heading."""
        heading = ctk.CTkLabel(
            self.scroll, text=text, font=("Segoe UI", 13, "bold"), text_color=self.colors["text_secondary"]
        )
        heading.pack(anchor="w", padx=8, pady=pady)
        if ref_id:
            devtools.section_label(self.scroll, self.app, self.colors, ref_id, "SETTINGS", anchor=heading)
        return heading

    def _slider_row(self, label: str, info_text: str, from_: float, to: float, steps: int, initial: float, on_change, dev_id: str = "") -> ctk.CTkLabel:
        row = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row.pack(fill="x", padx=8)
        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left")
        ctk.CTkLabel(left, text=label, text_color=self.colors["text_primary"]).pack(side="left")
        InfoIcon(left, self.colors, label, info_text).pack(side="left", padx=(4, 0))
        value_label = ctk.CTkLabel(row, text="", text_color=self.colors["accent"], font=("Segoe UI", 13, "bold"))
        value_label.pack(side="right")
        if dev_id:
            devtools.field_label(
                row, self.app, self.colors, dev_id, lambda: value_label.cget("text"), anchor=value_label, side="left",
            )
        slider = ctk.CTkSlider(
            self.scroll, from_=from_, to=to, number_of_steps=steps, progress_color=self.colors["accent"],
            button_color=self.colors["accent"], button_hover_color=self.colors["accent_hover"], command=on_change,
        )
        slider.set(initial)
        slider.pack(fill="x", padx=8, pady=(4, 10))
        return value_label

    def _build(self) -> None:
        colors = self.colors
        config = self.app.config

        ctk.CTkLabel(
            self, text="SETTINGS", font=("Segoe UI", 22, "bold"), text_color=colors["text_primary"]
        ).pack(anchor="w", padx=28, pady=(24, 4))
        devtools.page_label(self, self.app, colors, "SETTINGS")

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self._section("DETECTION", "DETECTION")

        self.confidence_value_label = self._slider_row(
            "Detection confidence",
            "Determines how closely an object must match the skull reference image before it is "
            "considered a detection. Higher values reduce false detections but may miss some genuine skulls.",
            0.50, 0.99, 49, config.confidence_threshold, self._on_confidence_changed, dev_id="CONFIDENCE_THRESHOLD",
        )
        self.confidence_value_label.configure(text=f"{round(config.confidence_threshold*100)}%")

        self.sampling_value_label = self._slider_row(
            "Frame sampling",
            "Determines how frequently the video is analysed. Analysing more frames improves accuracy but "
            "increases processing time.",
            1, 10, 9, config.frame_sampling, self._on_sampling_changed, dev_id="FRAME_SAMPLING",
        )
        self.sampling_value_label.configure(text=f"Every {config.frame_sampling} frame(s)")

        multi_row = ctk.CTkFrame(self.scroll, fg_color="transparent")
        multi_row.pack(fill="x", padx=8, pady=(0, 12))
        self.multi_skull_var = tk.BooleanVar(value=config.multiple_skull_detection)
        ctk.CTkCheckBox(
            multi_row, text="Multiple skull detection", variable=self.multi_skull_var,
            text_color=colors["text_primary"], command=self._on_multi_skull_changed,
        ).pack(side="left")
        InfoIcon(
            multi_row, colors, "Multiple Skull Detection",
            "Allows the software to recognise several skulls appearing together as one event.",
        ).pack(side="left", padx=(4, 0))

        self._section("RESOLVE", "RESOLVE_SETTINGS")
        row = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row.pack(fill="x", padx=8)
        ctk.CTkLabel(row, text="Marker name", text_color=colors["text_primary"]).pack(side="left")
        self.marker_name_var = tk.StringVar(value=config.marker_name)
        marker_entry = ctk.CTkEntry(row, textvariable=self.marker_name_var, width=160)
        marker_entry.pack(side="right")
        devtools.field_label(
            row, self.app, colors, "MARKER_NAME", lambda: self.marker_name_var.get(), anchor=marker_entry, side="left",
        )
        self.marker_name_var.trace_add("write", lambda *_: self._on_marker_changed())

        row2 = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(10, 0))
        ctk.CTkLabel(row2, text="Marker colour", text_color=colors["text_primary"]).pack(side="left")
        self.marker_color_var = tk.StringVar(value=config.marker_color)
        ctk.CTkOptionMenu(row2, values=MARKER_COLORS, variable=self.marker_color_var, width=160).pack(side="right")
        self.marker_color_var.trace_add("write", lambda *_: self._on_marker_changed())

        self._section("CALIBRATION", "CALIBRATION_MANAGEMENT")
        ctk.CTkLabel(
            self.scroll, text="Inspect the saved detection area and reference image, or reset the calibration entirely.",
            text_color=colors["text_secondary"], wraplength=460, justify="left",
        ).pack(anchor="w", padx=8)
        devtools.dev_button(
            self.scroll, self.app, colors, "MANAGE_CALIBRATION", "SETTINGS", "CALIBRATION_MANAGEMENT",
            text="MANAGE CALIBRATION", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            text_color=colors["text_primary"], command=self._open_manage_calibration, width=180,
            pack_opts={"padx": 8, "pady": (8, 4), "anchor": "w"},
        )

        self._section("DEVELOPER MODE", "DEVELOPER_MODE")
        dev_row = ctk.CTkFrame(self.scroll, fg_color="transparent")
        dev_row.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(dev_row, text="Developer Mode", text_color=colors["text_primary"]).pack(side="left")
        InfoIcon(
            dev_row, colors, "Developer Mode",
            "Shows the extra diagnostic panels and live-value readouts used for troubleshooting "
            "(ANALYSIS evidence strip, RESOLVE diagnostics, the TEST GOLD breakdown). For the small "
            "numbered reference codes use the REF CODES switch at the top right of the window instead.",
        ).pack(side="left", padx=(4, 0))
        self.developer_mode_var = tk.BooleanVar(value=config.developer_mode)
        ctk.CTkSwitch(
            dev_row, text="", variable=self.developer_mode_var, progress_color=colors["accent"],
            command=self._on_developer_mode_changed,
        ).pack(side="right")
        ctk.CTkLabel(
            self.scroll,
            text="Extra diagnostic panels and live readouts. (Reference codes: the REF CODES switch, top right.)",
            text_color=colors["text_secondary"], font=("Segoe UI", 11), wraplength=460, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 12))

        devtools.dev_button(
            self.scroll, self.app, colors, "SAVE_SETTINGS", "SETTINGS", "SETTINGS_ACTIONS",
            text="Save Settings", fg_color=colors["accent"], hover_color=colors["accent_hover"],
            command=self._save, width=180, height=36, pack_opts={"padx": 8, "pady": 24, "anchor": "w"},
        )

    # ------------------------------------------------------------------
    # Every control applies to app.config the moment it changes (so what you see is what the next
    # analysis/send uses, whether or not you press Save Settings) and is written to disk shortly
    # after the last change.
    # ------------------------------------------------------------------
    def _on_confidence_changed(self, value: float) -> None:
        self.app.config.confidence_threshold = round(value, 2)
        self.confidence_value_label.configure(text=f"{round(value*100)}%")
        self._schedule_save()

    def _on_sampling_changed(self, value: float) -> None:
        n = max(1, int(round(value)))
        self.app.config.frame_sampling = n
        self.sampling_value_label.configure(text=f"Every {n} frame(s)")
        self._schedule_save()

    def _on_multi_skull_changed(self) -> None:
        self.app.config.multiple_skull_detection = self.multi_skull_var.get()
        self._schedule_save()

    def _on_marker_changed(self) -> None:
        self.app.config.marker_name = self.marker_name_var.get().strip()[:MARKER_NAME_MAX_LENGTH] or "Kill"
        self.app.config.marker_color = self.marker_color_var.get()
        self._schedule_save()

    def _on_developer_mode_changed(self) -> None:
        self.app.config.developer_mode = self.developer_mode_var.get()
        self.flush_pending_save(force=True)
        self.app.refresh_developer_mode()

    def _save(self) -> None:
        self._on_marker_changed()
        if self.flush_pending_save(force=True):
            dialogs.show_info(self.app.root, self.colors, "Settings Saved", "Your settings have been saved.")

    # ------------------------------------------------------------------
    def _schedule_save(self) -> None:
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(_SAVE_DELAY_MS, self.flush_pending_save)

    def flush_pending_save(self, force: bool = False) -> bool:
        """Writes the settings now if a change is waiting for its delayed save (or `force`), and
        cancels that scheduled write. With nothing pending it does nothing, so closing the window
        never rewrites an unchanged file. Returns False -- after telling the user once -- if the
        file couldn't be written; the settings stay applied in memory."""
        if self._save_job is None and not force:
            return True
        if self._save_job is not None:
            self.after_cancel(self._save_job)
            self._save_job = None
        try:
            save_config(self.app.config)
        except OSError as exc:
            if not self._save_failure_shown:
                self._save_failure_shown = True
                dialogs.show_error(
                    self.app.root, self.colors, "SETTINGS COULD NOT BE SAVED",
                    "Your settings are active for this session, but they could not be written to disk.",
                    why=[str(exc), "The drive may be full or the folder may be read-only."],
                    what_to_do="Free some disk space, then change a setting again to retry.",
                )
            return False
        self._save_failure_shown = False
        return True

    def refresh(self) -> None:
        pass

    def _open_manage_calibration(self) -> None:
        ManageCalibrationDialog(self.app.root, self.app, self.colors)


class ManageCalibrationDialog(ctk.CTkToplevel):
    """MANAGE CALIBRATION: a separate, deliberately out-of-the-way screen for inspecting or
    resetting the permanent calibration -- an everyday recovery action, not part of the
    guided workflow. Operates directly on `app.session` (the one shared AppSession), so any
    change here is immediately visible on CALIBRATE too."""

    def __init__(self, master, app, colors: dict):
        super().__init__(master)
        self.colors = colors
        self.app = app
        self._reference_tk_image = None

        self.title("Manage Calibration")
        self.configure(fg_color=colors["bg"])
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        ctk.CTkLabel(
            self, text="CURRENT CALIBRATION", font=("Segoe UI", 15, "bold"), text_color=colors["text_primary"]
        ).pack(padx=24, pady=(20, 10), anchor="w")

        session = app.session
        region_ok = bool(session.detection_region)
        ctk.CTkLabel(
            self, text=f"{'✓' if region_ok else '✕'} Skull detection area"
            + (f"  (X={session.detection_region[0]}, Y={session.detection_region[1]}, "
               f"W={session.detection_region[2]}, H={session.detection_region[3]})" if region_ok else ""),
            text_color=colors["success"] if region_ok else colors["danger"], wraplength=420, justify="left",
        ).pack(padx=24, anchor="w")

        reference_count = len(session.reference_images)
        reference_ok = reference_count > 0
        labels = ", ".join(ref.label for ref in session.reference_images)
        ctk.CTkLabel(
            self, text=f"{'✓' if reference_ok else '✕'} Skull reference image(s)"
            + (f"  ({labels})" if reference_ok else ""),
            text_color=colors["success"] if reference_ok else colors["danger"], wraplength=420, justify="left",
        ).pack(padx=24, pady=(4, 10), anchor="w")
        ctk.CTkLabel(
            self, text="Manage reference images and test footage on the CALIBRATE page.",
            text_color=colors["text_secondary"], font=("Segoe UI", 11),
        ).pack(padx=24, pady=(0, 10), anchor="w")

        devtools.dev_button(
            self, app, colors, "OPEN_CALIBRATION_FOLDER", "SETTINGS", "MANAGE_CALIBRATION",
            text="OPEN CALIBRATION FOLDER", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            text_color=colors["text_primary"], command=lambda: _open_in_explorer(calibration_store.CALIBRATION_DIR),
            pack_opts={"padx": 24, "pady": 4, "fill": "x"}, button_pack_opts={"fill": "x"},
        )
        devtools.dev_button(
            self, app, colors, "RESET_CALIBRATION", "SETTINGS", "MANAGE_CALIBRATION",
            text="RESET CALIBRATION", fg_color=colors["panel_alt"], hover_color=colors["danger"],
            text_color=colors["text_primary"], command=self._reset,
            pack_opts={"padx": 24, "pady": (4, 20), "fill": "x"}, button_pack_opts={"fill": "x"},
        )

    def _reset(self) -> None:
        confirmed = dialogs.ask_yes_no(
            self, self.colors, "Reset Calibration",
            "This clears the saved skull detection area, reference images, and test footage. Continue?",
        )
        if not confirmed:
            return
        calibrate_page = self.app.pages.get("CALIBRATE")
        if calibrate_page is not None:
            calibrate_page.release_preview()  # Windows can't delete a clip this page still has open
        failure = None
        try:
            calibration_store.reset()
        except OSError as exc:
            failure = exc  # some files may be gone, some not: keep going so memory matches the disk
        session = self.app.session
        session.detection_region = None
        session.calibration_video_width = session.calibration_video_height = 0
        session.reference_images = []
        session.bookmark_adjustment_seconds = 0.0
        if calibrate_page is not None:
            calibrate_page.on_calibration_reset()
        self.destroy()
        if failure is not None:
            dialogs.show_error(
                self.app.root, self.colors, "RESET INCOMPLETE",
                "The calibration was cleared in the app, but some files could not be deleted.",
                why=[str(failure), "Another program may have one of them open."],
                what_to_do="Close anything using the calibration folder, then reset again (or delete the files by hand).",
            )
