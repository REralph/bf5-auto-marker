"""
ui/pages/resolve_page.py
---------------------------
RESOLVE stage: normally embedded (standalone=False) as the third stage inside
ui/pages/workflow_page.py's consolidated WORKFLOW page -- standalone=True (the
default) still renders it as a full, independent page for any other caller.
A compact default view -- connection status, project/timeline,
and one SEND N MARKERS TO RESOLVE button. Clicking it sends straight away -- there is no
confirmation pop-up. The single signed "bookmark time adjustment" calibrated on the
CALIBRATE tab (0.0 uses the detected bookmark exactly) is applied exactly once at send time
(resolve_controller.compute_marker_timestamp()) and never mutates the event's own stored
bookmark; a one-line note above the button says what adjustment is in effect. RECONNECT still runs a real, full
connection attempt (run_full_diagnostic()) and a genuine post-send
verification pass (add_markers_from_detections() re-reads the timeline via
GetMarkers()) -- none of that machinery was removed, it's just Developer-
Mode-only now: the granular per-stage CONNECTION DIAGNOSTICS panel, TEST
BOOKMARK, RESOLVE DEBUG, and the lightweight Refresh Status button all live
behind Developer Mode so the default page matches the simplified workflow.
"""

from __future__ import annotations

import threading
import time

import customtkinter as ctk

import ui.devtools as devtools
from resolve.resolve_controller import (
    ConnectionStepResult, FullDiagnostic, ResolveStatus, check_connection_status, create_test_bookmark,
    run_full_diagnostic,
)
from ui.workflow_dialogs import show_resolve_help

_STATUS_POLL_INTERVAL_MS = 20_000


class ResolvePage(ctk.CTkFrame):
    def __init__(self, master, app, colors: dict, standalone: bool = True):
        # Embedded in a WORKFLOW stage card -> transparent, so it blends into the card instead of
        # painting a darker page-coloured rectangle over it.
        super().__init__(master, fg_color=colors["bg"] if standalone else "transparent")
        self.app = app
        self.colors = colors
        self.standalone = standalone
        self._last_diag = None
        self._diag_header = None
        self._debug_lines = ["No diagnostic has been run yet."]
        self._sending = False
        self._status_check_running = False
        self._build()
        self._refresh_status_async()
        self._schedule_background_poll()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        colors = self.colors
        if self.standalone:
            ctk.CTkLabel(
                self, text="RESOLVE", font=("Segoe UI", 22, "bold"), text_color=colors["text_primary"]
            ).pack(anchor="w", padx=28, pady=(24, 4))
            devtools.page_label(self, self.app, colors, "RESOLVE")

        # Embedded inside WORKFLOW's own outer scrollable page, so skip a second nested
        # scrollable frame here (that would create an awkward nested-scrollbar) -- pack directly
        # into self instead, at zero extra padding since the stage card already provides its own.
        if self.standalone:
            body = ctk.CTkScrollableFrame(self, fg_color="transparent")
            body.pack(fill="both", expand=True, padx=20, pady=(10, 10))
        else:
            body = self
        self.body = body

        self._build_connection_status(body, colors)
        self._build_send_markers(body, colors)
        self._build_dev_panel(body, colors)

    # ------------------------------------------------------------------
    # CONNECTION STATUS -- the compact default view
    # ------------------------------------------------------------------
    def _build_connection_status(self, parent, colors: dict) -> None:
        status_panel = ctk.CTkFrame(parent, fg_color=colors["panel"], corner_radius=10)
        status_panel.pack(fill="x", pady=(0, 20))
        devtools.section_label(
            status_panel, self.app, colors, "CONNECTION_STATUS", "RESOLVE",
            pack_opts={"anchor": "w", "padx": 20, "pady": (14, 0)},
        )

        inner = ctk.CTkFrame(status_panel, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=(20, 0))
        self.dot = ctk.CTkLabel(inner, text="●", font=("Segoe UI", 20), text_color=colors["danger"])
        self.dot.pack(side="left")
        self.status_label = ctk.CTkLabel(
            inner, text="Checking...", font=("Segoe UI", 16, "bold"), text_color=colors["text_primary"]
        )
        self.status_label.pack(side="left", padx=(8, 0))

        # ⓘ and RECONNECT only appear once disconnected (see _set_disconnected_controls_visible)
        # -- both are the LAST children packed in their respective parents at build time, so
        # toggling their visibility later can never reorder anything relative to a sibling.
        self.info_icon = ctk.CTkLabel(
            inner, text="ⓘ", font=("Segoe UI", 15), text_color=colors["text_secondary"], cursor="hand2"
        )
        self.info_icon.pack(side="left", padx=(12, 0))
        self.info_icon.bind(
            "<Button-1>", lambda e: show_resolve_help(self.app.root, colors, on_reconnect=self._on_reconnect)
        )

        self.connection_detail_label = ctk.CTkLabel(
            status_panel, text="", text_color=colors["text_secondary"], justify="left", anchor="w",
        )
        self.connection_detail_label.pack(anchor="w", padx=20, pady=(6, 16))

        self.reconnect_row = ctk.CTkFrame(status_panel, fg_color="transparent")
        self.reconnect_row.pack(fill="x", padx=20, pady=(0, 20))
        self.reconnect_btn = devtools.dev_button(
            self.reconnect_row, self.app, colors, "RECONNECT_TO_RESOLVE", "RESOLVE", "CONNECTION_STATUS",
            text="RECONNECT", fg_color=colors["accent"], hover_color=colors["accent_hover"],
            command=self._on_reconnect, pack_opts={"side": "left"},
        )

    def _set_disconnected_controls_visible(self, visible: bool) -> None:
        if visible:
            self.info_icon.pack(side="left", padx=(12, 0))
            self.reconnect_row.pack(fill="x", padx=20, pady=(0, 20))
        else:
            self.info_icon.pack_forget()
            self.reconnect_row.pack_forget()

    # ------------------------------------------------------------------
    # SEND MARKERS -- one button, sends immediately, then reports a genuinely verified result
    # ------------------------------------------------------------------
    def _build_send_markers(self, parent, colors: dict) -> None:
        panel = ctk.CTkFrame(parent, fg_color=colors["panel"], corner_radius=10)
        panel.pack(fill="x", pady=(0, 20))
        devtools.section_label(
            panel, self.app, colors, "SEND_MARKERS", "RESOLVE",
            pack_opts={"anchor": "w", "padx": 20, "pady": (12, 0)},
        )
        self.send_markers_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.send_markers_frame.pack(fill="x", padx=20, pady=(14, 20))
        self._render_send_idle()

    def _render_send_idle(self) -> None:
        for child in self.send_markers_frame.winfo_children():
            child.destroy()
        count = len(self.app.events)
        ctk.CTkLabel(
            self.send_markers_frame, text=f"Markers ready: {count}", font=("Segoe UI", 13, "bold"),
            text_color=self.colors["text_primary"],
        ).pack(anchor="w", pady=(0, 2))
        adjustment = self.app.session.bookmark_adjustment_seconds
        note = (
            "Bookmarks are placed at the exact detected time."
            if adjustment == 0
            else f"Bookmarks are shifted {adjustment:+.2f}s from the detected time (set in CALIBRATE)."
        )
        ctk.CTkLabel(
            self.send_markers_frame, text=note, font=("Segoe UI", 11), text_color=self.colors["text_secondary"],
        ).pack(anchor="w", pady=(0, 10))
        self.send_btn = devtools.dev_button(
            self.send_markers_frame, self.app, self.colors, "SEND_TO_RESOLVE", "RESOLVE", "SEND_MARKERS",
            text=f"SEND {count} MARKERS TO RESOLVE", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            height=42, state=("normal" if count else "disabled"), command=self.send_now, pack_opts={"anchor": "w"},
        )

    def send_now(self) -> None:
        """Sends every marker to Resolve immediately (no confirmation pop-up). Shared by this
        stage's own button and ANALYSIS' completion-screen SEND TO RESOLVE button, so the
        SENDING.../CREATED/FAILED status always renders in this stage's card. Ignored while a send
        is already running, so a double-click can't fire the same markers twice."""
        if self._sending or not self.app.events:
            return
        self._sending = True
        for child in self.send_markers_frame.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            self.send_markers_frame, text="SENDING MARKERS TO RESOLVE...", font=("Segoe UI", 13, "bold"),
            text_color=self.colors["accent"],
        ).pack(anchor="w", pady=(0, 10))
        self.app.send_markers()

    def on_send_result(self, result) -> None:
        """Called by MainWindow._poll_queue() with the MarkerSendResult from
        add_markers_from_detections() -- never reports success unless `result.created` reflects
        a genuine post-write re-read of the timeline, per the workflow's own requirement."""
        self._sending = False
        for child in self.send_markers_frame.winfo_children():
            child.destroy()

        if result.verification_error is not None:
            ctk.CTkLabel(
                self.send_markers_frame, text="SENT, BUT UNVERIFIED", font=("Segoe UI", 14, "bold"),
                text_color=self.colors["text_secondary"],
            ).pack(anchor="w", pady=(0, 4))
            ctk.CTkLabel(
                self.send_markers_frame,
                text=f"Markers were sent, but the result couldn't be confirmed: {result.verification_error}\n"
                "Check DaVinci Resolve's timeline directly.",
                text_color=self.colors["text_secondary"], wraplength=480, justify="left",
            ).pack(anchor="w", pady=(0, 10))
            ctk.CTkButton(
                self.send_markers_frame, text="RETRY", fg_color=self.colors["accent"],
                hover_color=self.colors["accent_hover"], command=self.send_now,
            ).pack(anchor="w")
        elif result.all_created:
            ctk.CTkLabel(
                self.send_markers_frame, text="✓ MARKERS CREATED", font=("Segoe UI", 14, "bold"),
                text_color=self.colors["success"],
            ).pack(anchor="w", pady=(0, 4))
            summary = f"{result.created} / {result.total} markers successfully added."
            if result.already_present:
                summary = (
                    f"{result.created} new marker(s) added; {result.already_present} already on the "
                    f"timeline from an earlier send were left as they are."
                )
            if result.nudged:
                summary += f"\n{result.nudged} marker(s) were moved a few frames later because their exact frame was already taken."
            ctk.CTkLabel(
                self.send_markers_frame, text=summary, text_color=self.colors["text_primary"], justify="left",
            ).pack(anchor="w")
        else:
            failed = len(result.failed_events)
            ctk.CTkLabel(
                self.send_markers_frame, text="MARKER VERIFICATION FAILED", font=("Segoe UI", 14, "bold"),
                text_color=self.colors["danger"],
            ).pack(anchor="w", pady=(0, 4))
            ctk.CTkLabel(
                self.send_markers_frame,
                text=f"Detected: {result.total}\nCreated: {result.created}\n{failed} marker(s) could not be created.",
                text_color=self.colors["text_primary"], justify="left",
            ).pack(anchor="w", pady=(0, 10))
            action_row = ctk.CTkFrame(self.send_markers_frame, fg_color="transparent")
            action_row.pack(anchor="w")
            ctk.CTkButton(
                action_row, text="VIEW ERRORS", fg_color=self.colors["panel_alt"], hover_color=self.colors["accent_dim"],
                text_color=self.colors["text_primary"], command=lambda: self._show_send_errors(result),
            ).pack(side="left")
            ctk.CTkButton(
                action_row, text="RETRY", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
                command=self.send_now,
            ).pack(side="left", padx=8)

    def on_send_error(self, error_message: str) -> None:
        self._sending = False
        for child in self.send_markers_frame.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            self.send_markers_frame, text="COULD NOT SEND MARKERS", font=("Segoe UI", 14, "bold"),
            text_color=self.colors["danger"],
        ).pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(
            self.send_markers_frame, text=error_message, text_color=self.colors["text_secondary"],
            wraplength=480, justify="left",
        ).pack(anchor="w", pady=(0, 10))
        ctk.CTkButton(
            self.send_markers_frame, text="RETRY", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=self.send_now,
        ).pack(anchor="w")

    def _show_send_errors(self, result) -> None:
        dialog = ctk.CTkToplevel(self.app.root)
        dialog.title("Marker Errors")
        dialog.configure(fg_color=self.colors["bg"])
        dialog.resizable(False, False)
        dialog.transient(self.app.root)
        dialog.grab_set()
        ctk.CTkLabel(
            dialog, text="MARKERS THAT COULD NOT BE CREATED", font=("Segoe UI", 14, "bold"),
            text_color=self.colors["danger"],
        ).pack(padx=20, pady=(20, 10), anchor="w")
        scroll = ctk.CTkScrollableFrame(dialog, fg_color=self.colors["panel_alt"], corner_radius=8, width=380, height=200)
        scroll.pack(padx=20, pady=(0, 16))
        for event, frame_id in result.failed_events:
            ctk.CTkLabel(
                scroll, text=f"Event #{event.event_number} — {event.hms_display} (frame {frame_id})",
                text_color=self.colors["text_primary"], anchor="w",
            ).pack(fill="x", pady=2, padx=6)
        ctk.CTkButton(
            dialog, text="Close", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
            command=dialog.destroy,
        ).pack(pady=(0, 20))

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if not self._sending:  # never swap the SENDING... status back to an idle button mid-send
            self._render_send_idle()

    def on_developer_mode_changed(self) -> None:
        """Called by MainWindow.refresh_developer_mode() so the Developer Mode panel
        shows/hides immediately even if the user is already sitting on this page."""
        self._set_debug_panel_visible(devtools.is_enabled(self.app))

    def _set_debug_panel_visible(self, visible: bool) -> None:
        if visible:
            self.debug_panel.pack(fill="x")
        else:
            self.debug_panel.pack_forget()

    # ------------------------------------------------------------------
    # Lightweight status check -- initial page load / background poll / Developer Mode's
    # Refresh Status button
    # ------------------------------------------------------------------
    # Worker threads here never touch a widget: results are handed to the main thread through
    # app.post_to_main() (calling Tk from another thread is unsafe, and fails outright once the
    # window is closing).
    def _refresh_status_async(self) -> None:
        if self._status_check_running:
            return  # a check is already in flight (a hung Resolve call must not pile up threads)
        self._status_check_running = True

        def worker() -> None:
            try:
                status = check_connection_status()
            except Exception as exc:  # noqa: BLE001 - a failed check is a status, not a crash
                status = ResolveStatus(connected=False, detail=f"Could not check DaVinci Resolve: {exc}")
            self.app.post_to_main(lambda: self._status_check_done(status))

        threading.Thread(target=worker, daemon=True).start()

    def _status_check_done(self, status) -> None:
        self._status_check_running = False
        self._apply_status(status)

    def _apply_status(self, status) -> None:
        if not self.winfo_exists():
            return
        color = self.colors["success"] if status.connected else self.colors["danger"]
        self.dot.configure(text=("●" if status.connected else "○"), text_color=color)
        self.status_label.configure(
            text="Connected" if status.connected else "Resolve Not Connected", text_color=color
        )
        self._set_disconnected_controls_visible(not status.connected)
        if status.connected:
            self.connection_detail_label.configure(
                text=f"Project: {status.project_name}\nTimeline: {status.timeline_name}",
                text_color=self.colors["text_secondary"],
            )
        else:
            self.connection_detail_label.configure(text=status.detail, text_color=self.colors["text_secondary"])

    def _schedule_background_poll(self) -> None:
        # Passive disconnect detection: re-checks status periodically without requiring the
        # user to click anything, so "Connected" doesn't silently go stale once Resolve (or
        # its project/timeline) closes. Uses the lightweight check, not the full diagnostic
        # (which also probes bookmark access), so it doesn't hammer Resolve's API.
        def poll() -> None:
            if not self.winfo_exists():
                return
            self._refresh_status_async()
            self.after(_STATUS_POLL_INTERVAL_MS, poll)

        self.after(_STATUS_POLL_INTERVAL_MS, poll)

    # ------------------------------------------------------------------
    # RECONNECT -- a real, full connection attempt with live per-stage progress (Developer
    # Mode shows every stage; the default view just sees the final status update)
    # ------------------------------------------------------------------
    def _on_reconnect(self) -> None:
        self.reconnect_btn.configure(text="CONNECTING...", state="disabled")
        self.test_bookmark_btn.configure(state="disabled")
        self.bookmark_result_label.configure(text="")
        for child in self.diag_results_frame.winfo_children():
            child.destroy()
        self._diag_header = ctk.CTkLabel(
            self.diag_results_frame, text="CONNECTING TO DAVINCI RESOLVE...",
            font=("Segoe UI", 13, "bold"), text_color=self.colors["accent"],
        )
        self._diag_header.pack(anchor="w")

        def on_step(step) -> None:
            self.app.post_to_main(lambda s=step: self._append_live_step(s))

        def worker() -> None:
            try:
                diag = run_full_diagnostic(on_step=on_step)
            except Exception as exc:  # noqa: BLE001 - always end the CONNECTING... state
                diag = FullDiagnostic(steps=[ConnectionStepResult("Resolve connection", False, f"Unexpected error: {exc}")])
            self.app.post_to_main(lambda: self._on_reconnect_done(diag))

        threading.Thread(target=worker, daemon=True).start()

    def _append_live_step(self, step) -> None:
        row = ctk.CTkFrame(self.diag_results_frame, fg_color="transparent")
        row.pack(fill="x", pady=1)
        mark = "✓" if step.passed else "✗"
        color = self.colors["success"] if step.passed else self.colors["danger"]
        ctk.CTkLabel(row, text=mark, width=20, text_color=color, font=("Segoe UI", 12, "bold")).pack(side="left")
        label_text = step.label if not (step.passed and step.detail) else f"{step.label} — {step.detail}"
        ctk.CTkLabel(row, text=label_text, text_color=self.colors["text_primary"]).pack(side="left")

    def _on_reconnect_done(self, diag) -> None:
        self._last_diag = diag
        self.reconnect_btn.configure(text="RECONNECT", state="normal")
        self.test_bookmark_btn.configure(state=("normal" if diag.connected else "disabled"))

        if self._diag_header is not None and self._diag_header.winfo_exists():
            self._diag_header.destroy()
        self._diag_header = None

        ctk.CTkLabel(
            self.diag_results_frame,
            text=f"STATUS: {'CONNECTED' if diag.connected else 'NOT CONNECTED'}",
            font=("Segoe UI", 13, "bold"),
            text_color=self.colors["success"] if diag.connected else self.colors["danger"],
        ).pack(anchor="w", pady=(8, 0))

        failure = diag.first_failure
        if failure is not None:
            ctk.CTkLabel(
                self.diag_results_frame, text="ERROR:", font=("Segoe UI", 12, "bold"),
                text_color=self.colors["danger"],
            ).pack(anchor="w", pady=(10, 2))
            ctk.CTkLabel(
                self.diag_results_frame, text=failure.detail, text_color=self.colors["text_secondary"],
                wraplength=520, justify="left",
            ).pack(anchor="w")

            action_row = ctk.CTkFrame(self.diag_results_frame, fg_color="transparent")
            action_row.pack(anchor="w", pady=(10, 0))
            ctk.CTkButton(
                action_row, text="COPY ERROR", fg_color=self.colors["panel_alt"],
                hover_color=self.colors["accent_dim"], text_color=self.colors["text_primary"],
                command=lambda: self._copy_error(diag),
            ).pack(side="left")
            ctk.CTkButton(
                action_row, text="RETRY", fg_color=self.colors["accent"], hover_color=self.colors["accent_hover"],
                command=self._on_reconnect,
            ).pack(side="left", padx=8)

        # Apply the same result directly to the persistent status area -- no need to run a
        # second, separate connection check right after this one just finished.
        color = self.colors["success"] if diag.connected else self.colors["danger"]
        self.dot.configure(text=("●" if diag.connected else "○"), text_color=color)
        self.status_label.configure(text="Connected" if diag.connected else "Resolve Not Connected", text_color=color)
        self._set_disconnected_controls_visible(not diag.connected)
        detail = (
            f"Project: {diag.project_name}\nTimeline: {diag.timeline_name}" if diag.connected
            else (failure.detail if failure is not None else "")
        )
        self.connection_detail_label.configure(text=detail, text_color=self.colors["text_secondary"])

        self._refresh_debug_panel()

    def _render_diag_idle(self) -> None:
        for child in self.diag_results_frame.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            self.diag_results_frame, text="Click RECONNECT to run a full connection test.",
            text_color=self.colors["text_secondary"],
        ).pack(anchor="w")

    def _copy_error(self, diag) -> None:
        lines = ["RESOLVE CONNECTION TEST"]
        for step in diag.steps:
            lines.append(f"{'PASS' if step.passed else 'FAIL'} - {step.label}" + (f": {step.detail}" if step.detail else ""))
        text = "\n".join(lines)
        try:
            self.app.root.clipboard_clear()
            self.app.root.clipboard_append(text)
        except Exception:  # noqa: BLE001 - clipboard access is best-effort, never worth crashing over
            pass

    # ------------------------------------------------------------------
    # TEST BOOKMARK -- creates one real, identifiable marker end-to-end (Developer Mode)
    # ------------------------------------------------------------------
    def _on_test_bookmark(self) -> None:
        self.test_bookmark_btn.configure(text="CREATING TEST MARKER...", state="disabled")
        self.bookmark_result_label.configure(text="")

        def worker() -> None:
            try:
                result = create_test_bookmark()
            except Exception as exc:  # noqa: BLE001 - always end the CREATING... state
                result = ConnectionStepResult("Test bookmark", False, f"Unexpected error: {exc}")
            self.app.post_to_main(lambda: self._on_test_bookmark_done(result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_test_bookmark_done(self, result) -> None:
        can_retry = self._last_diag is not None and self._last_diag.connected
        self.test_bookmark_btn.configure(text="TEST BOOKMARK", state=("normal" if can_retry else "disabled"))
        color = self.colors["success"] if result.passed else self.colors["danger"]
        prefix = "SUCCESS — " if result.passed else "FAILED — "
        self.bookmark_result_label.configure(text=prefix + (result.detail or ""), text_color=color)
        self._refresh_debug_panel(last_bookmark=result)

    # ------------------------------------------------------------------
    # Developer Mode panel: Refresh Status, CONNECTION DIAGNOSTICS, TEST BOOKMARK, RESOLVE DEBUG
    # ------------------------------------------------------------------
    def _build_dev_panel(self, parent, colors: dict) -> None:
        # Container packed unconditionally/immediately -- pack_forget() does not remember a
        # widget's earlier position, so a widget whose first real pack() happens later (only
        # once Developer Mode is switched on) would land after whatever real panels get built
        # in the meantime instead of staying here, right after SEND MARKERS.
        self.debug_panel_container = ctk.CTkFrame(parent, fg_color="transparent")
        self.debug_panel_container.pack(fill="x")
        self.debug_panel = ctk.CTkFrame(self.debug_panel_container, fg_color="transparent")

        devtools.dev_button(
            self.debug_panel, self.app, colors, "REFRESH_STATUS", "RESOLVE", "CONNECTION_STATUS",
            text="Refresh Status", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            text_color=colors["text_primary"], command=self._refresh_status_async,
            pack_opts={"anchor": "w", "pady": (0, 16)},
        )
        self._build_connection_diagnostics(self.debug_panel, colors)
        self._build_test_bookmark(self.debug_panel, colors)
        self._build_resolve_debug(self.debug_panel, colors)

        self._set_debug_panel_visible(devtools.is_enabled(self.app))

    def _build_connection_diagnostics(self, parent, colors: dict) -> None:
        diag_panel = ctk.CTkFrame(parent, fg_color=colors["panel"], corner_radius=10)
        diag_panel.pack(fill="x", pady=(0, 20))
        devtools.section_label(
            diag_panel, self.app, colors, "CONNECTION_DIAGNOSTICS", "RESOLVE",
            pack_opts={"anchor": "w", "padx": 20, "pady": (12, 0)},
        )
        ctk.CTkLabel(
            diag_panel, text="CONNECTION DIAGNOSTICS", font=("Segoe UI", 13, "bold"),
            text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(14, 2))
        ctk.CTkLabel(
            diag_panel,
            text="RECONNECT runs a real, full connection attempt -- Resolve running, API "
            "available, instance, project, timeline, bookmark access -- and reports exactly "
            "which stage fails, instead of one generic error.",
            text_color=colors["text_secondary"], font=("Segoe UI", 11), wraplength=520, justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 10))
        self.diag_results_frame = ctk.CTkFrame(diag_panel, fg_color="transparent")
        self.diag_results_frame.pack(fill="x", padx=20, pady=(0, 16))
        self._render_diag_idle()

    def _build_test_bookmark(self, parent, colors: dict) -> None:
        bookmark_panel = ctk.CTkFrame(parent, fg_color=colors["panel"], corner_radius=10)
        bookmark_panel.pack(fill="x", pady=(0, 20))
        devtools.section_label(
            bookmark_panel, self.app, colors, "TEST_BOOKMARK", "RESOLVE",
            pack_opts={"anchor": "w", "padx": 20, "pady": (12, 0)},
        )
        ctk.CTkLabel(
            bookmark_panel, text="TEST BOOKMARK", font=("Segoe UI", 13, "bold"),
            text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(14, 2))
        ctk.CTkLabel(
            bookmark_panel,
            text="Creates one real, clearly-labelled marker on the current timeline to confirm "
            "marker creation genuinely works end-to-end. Enabled once RECONNECT succeeds.",
            text_color=colors["text_secondary"], font=("Segoe UI", 11), wraplength=520, justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 10))
        self.test_bookmark_btn = devtools.dev_button(
            bookmark_panel, self.app, colors, "TEST_BOOKMARK", "RESOLVE", "TEST_BOOKMARK",
            text="TEST BOOKMARK", fg_color=colors["accent"], hover_color=colors["accent_hover"],
            state="disabled", command=self._on_test_bookmark, pack_opts={"anchor": "w", "padx": 20},
        )
        self.bookmark_result_label = ctk.CTkLabel(
            bookmark_panel, text="", text_color=colors["text_secondary"], wraplength=520, justify="left",
        )
        self.bookmark_result_label.pack(anchor="w", padx=20, pady=(8, 16))

    def _build_resolve_debug(self, parent, colors: dict) -> None:
        panel = ctk.CTkFrame(parent, fg_color=colors["panel"], corner_radius=10)
        panel.pack(fill="x", pady=(0, 20))
        devtools.section_label(
            panel, self.app, colors, "RESOLVE_DEBUG", "RESOLVE",
            pack_opts={"anchor": "w", "padx": 20, "pady": (12, 0)},
        )
        ctk.CTkLabel(
            panel, text="RESOLVE DEBUG", font=("Segoe UI", 13, "bold"), text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(14, 6))
        self.debug_text_label = ctk.CTkLabel(
            panel, text="No diagnostic has been run yet.", text_color=colors["text_primary"],
            justify="left", anchor="w", font=("Consolas", 11),
        )
        self.debug_text_label.pack(anchor="w", padx=20)
        ctk.CTkButton(
            panel, text="COPY DIAGNOSTICS", fg_color=colors["panel_alt"],
            hover_color=colors["accent_dim"], text_color=colors["text_primary"],
            command=self._copy_diagnostics,
        ).pack(anchor="w", padx=20, pady=(10, 16))

    def _refresh_debug_panel(self, last_bookmark=None) -> None:
        if self._last_diag is None:
            return
        diag = self._last_diag
        lines = [f"Last diagnostic run: {time.strftime('%H:%M:%S')}"]
        for step in diag.steps:
            lines.append(f"{'PASS' if step.passed else 'FAIL'}  {step.label}" + (f" — {step.detail}" if step.detail else ""))
        lines.append(f"Overall connected: {diag.connected}")
        lines.append(f"Project: {diag.project_name or '-'}")
        lines.append(f"Timeline: {diag.timeline_name or '-'}")
        if last_bookmark is not None:
            lines.append(f"Last TEST BOOKMARK: {'PASS' if last_bookmark.passed else 'FAIL'} — {last_bookmark.detail}")
        self._debug_lines = lines
        self.debug_text_label.configure(text="\n".join(lines))

    def _copy_diagnostics(self) -> None:
        text = "\n".join(self._debug_lines)
        try:
            self.app.root.clipboard_clear()
            self.app.root.clipboard_append(text)
        except Exception:  # noqa: BLE001 - clipboard access is best-effort, never worth crashing over
            pass
