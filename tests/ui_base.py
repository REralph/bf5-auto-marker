"""
tests/ui_base.py
------------------
Base class for tests that drive the real MainWindow. The window is created fully transparent (nothing
appears on screen), config/calibration/crash-log paths are redirected to a temp folder, Resolve is
faked, and every modal dialog is replaced by a recorder so a test can never block waiting for a click.
Uncaught Tk callback exceptions are collected and fail the test.
"""

from __future__ import annotations

import gc
import time
import traceback
import unittest
from unittest import mock

from tests import helpers

try:
    import customtkinter as ctk
    from tkinterdnd2 import TkinterDnD
    _TK_OK = helpers.display_available()
except Exception:  # noqa: BLE001
    _TK_OK = False


class UiTestCase(unittest.TestCase):
    """Subclasses get: self.root, self.app, self.timeline (fake Resolve), self.dialogs, self.errors."""

    @classmethod
    def setUpClass(cls):
        if not _TK_OK:
            raise unittest.SkipTest("no display available for Tk")

    def setUp(self):
        import ui.dialogs as dialogs
        import ui.main_window as main_window
        import ui.pages.resolve_page as resolve_page
        from resolve.resolve_controller import ResolveStatus

        self.state_dir = helpers.isolate_app_state(self)
        self.errors = []
        self.dialogs = mock.MagicMock(name="dialogs")
        self.dialogs.ask_yes_no.return_value = True
        for name in ("show_error", "show_info", "ask_yes_no"):
            patcher = mock.patch.object(dialogs, name, getattr(self.dialogs, name))
            patcher.start()
            self.addCleanup(patcher.stop)

        self.timeline = helpers.FakeTimeline()
        project = helpers.FakeProject("60")
        for module, name, value in (
            (main_window, "get_resolve", lambda: object()),
            (main_window, "get_current_project", lambda r: project),
            (main_window, "get_current_timeline", lambda p: self.timeline),
            (resolve_page, "check_connection_status", lambda: ResolveStatus(True, "Connected", "P", "T")),
        ):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        errors = self.errors

        class TestRoot(ctk.CTk, TkinterDnD.DnDWrapper):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.TkdndVersion = TkinterDnD._require(self)

            def report_callback_exception(self, exc, val, tb):
                errors.append("".join(traceback.format_exception(exc, val, tb)))

        ctk.set_appearance_mode("dark")
        self.root = TestRoot()
        self.root.attributes("-alpha", 0.0)
        self.app = main_window.MainWindow(self.root)
        self.pump(0.3)
        self.addCleanup(self._teardown_root)

    def _teardown_root(self):
        try:
            self.app.shutdown()   # cancel the window's own repeating timers before it goes away
        except Exception:  # noqa: BLE001
            pass
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass
        # Tk Variable objects left over from this window are finalised wherever the garbage collector
        # happens to run; if that is a later test's analysis thread, each finaliser becomes a
        # cross-thread Tcl call and the scan crawls. Collect them here, on the main thread, instead.
        gc.collect()
        self.assertEqual(self.errors, [], "uncaught exception inside a Tk callback:\n" + "\n".join(self.errors))

    # ------------------------------------------------------------------
    def pump(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.root.update()
            time.sleep(0.01)

    def wait_for(self, condition, timeout: float = 20.0, message: str = "condition") -> None:
        start = time.time()
        while not condition():
            self.root.update()
            time.sleep(0.02)
            if time.time() - start > timeout:
                self.fail(f"timed out waiting for {message}")

    @staticmethod
    def widgets(widget):
        yield widget
        for child in widget.winfo_children():
            yield from UiTestCase.widgets(child)

    def texts(self, widget):
        out = []
        for w in self.widgets(widget):
            try:
                out.append(str(w.cget("text")))
            except Exception:  # noqa: BLE001
                pass
        return out

    def toplevels(self):
        return [w for w in self.widgets(self.root) if isinstance(w, ctk.CTkToplevel) and w.winfo_exists()]

    def error_dialog_titles(self):
        return [call.args[2] for call in self.dialogs.show_error.call_args_list]
