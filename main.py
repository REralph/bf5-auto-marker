"""
main.py
-------
Entry point for BF5 Auto Marker. Builds the CustomTkinter root window and
hands control to MainWindow (ui/main_window.py). All detection logic lives
in detectors/, video handling in video/, Resolve communication in
resolve/, and shared data models/settings in data/.

Run with:
    python main.py
"""

import traceback

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD

from core.crash_log import CRASH_LOG_PATH, log_exception_text
from ui.main_window import MainWindow


class DnDCTk(ctk.CTk, TkinterDnD.DnDWrapper):
    """CTk's own root doesn't know about tkdnd -- this mixin loads the tkdnd Tcl package
    into the interpreter so VIDEO's drop zone can register as a real OS drop target."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

    def report_callback_exception(self, exc, val, tb):
        """
        Tkinter's own default here calls sys.stderr.write(...) -- but the app is normally
        launched with pythonw.exe (no console attached), where sys.stdout/sys.stderr are both
        None. That made ANY exception inside ANY widget callback anywhere in the app (a template
        library's tqdm progress bar trying to write its console output was the specific case
        that motivated this) fail a second time trying to report the first failure, instead of
        the visible error dialog it should have produced -- from the user's side, indistinguishable
        from the whole app silently doing nothing. Logging to a real file and showing a real
        dialog here never touches sys.stdout/sys.stderr, so it can't repeat that failure.
        """
        message = "".join(traceback.format_exception(exc, val, tb))
        log_exception_text(message)  # size-capped (core/crash_log.py); never raises
        try:
            import ui.dialogs as dialogs
            from ui.main_window import COLORS
            dialogs.show_error(
                self, COLORS, "UNEXPECTED ERROR",
                "Something went wrong and the current action could not complete.",
                why=[str(val)], what_to_do=f"Details were written to {CRASH_LOG_PATH}.",
            )
        except Exception:
            pass


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = DnDCTk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
