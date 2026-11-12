"""
ui/window_perf.py
-------------------
Keeps window resizing and monitor changes cheap when the app has several heavy tabs.

CustomTkinter re-scales EVERY widget in a window the moment it detects a different DPI (e.g. the
window was dragged to another monitor), after dimming the window to 15% opacity -- so the window
looks blank for as long as that takes, which grows with the total widget count (every tab, shown
or not). DeferredPageScaling changes that to: re-scale the shell and the tab on screen right away,
and re-scale each hidden tab only when it is next shown (apply_pending). Hidden tabs are never on
screen, so nothing visible changes -- they just stop adding to the blank time.

(Resizing is kept cheap separately, in MainWindow.show_page(): hidden pages are unmapped, so a
resize only re-lays-out the tab that is actually showing.)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Set

from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker


class DeferredPageScaling:
    def __init__(self, root, pages: Dict[str, object], current_page: Callable[[], Optional[object]]):
        self.root = root
        self.pages = pages
        self.current_page = current_page
        self._dirty: Set[int] = set()  # id() of hidden pages whose widgets missed a re-scale
        current = ScalingTracker.update_scaling_callbacks_for_window
        previous = getattr(current, "__self__", None)
        # If another window's DeferredPageScaling is already installed, wrap CustomTkinter's own
        # function rather than that one (windows come and go; the wrappers must not pile up).
        self._original = previous._original if isinstance(previous, DeferredPageScaling) else current

    def install(self) -> None:
        ScalingTracker.update_scaling_callbacks_for_window = self._update_for_window

    def uninstall(self) -> None:
        ScalingTracker.update_scaling_callbacks_for_window = self._original

    # ------------------------------------------------------------------
    def _scaling_args(self):
        window_scaling = ScalingTracker.window_dpi_scaling_dict[self.root]
        return window_scaling * ScalingTracker.widget_scaling, window_scaling * ScalingTracker.window_scaling

    def _page_ids(self) -> Set[int]:
        return {id(p) for p in self.pages.values()}

    @staticmethod
    def _page_of(callback, page_ids: Set[int]) -> Optional[object]:
        widget = getattr(callback, "__self__", None)
        while widget is not None and id(widget) not in page_ids:
            widget = getattr(widget, "master", None)
        return widget

    def _update_for_window(self, window) -> None:
        if window is not self.root or not ScalingTracker.window_widgets_dict.get(window):
            return self._original(window)
        args = self._scaling_args()
        showing = self.current_page()
        page_ids = self._page_ids()
        saved_placement = self._unmap(showing)
        try:
            for callback in list(ScalingTracker.window_widgets_dict[window]):
                page = self._page_of(callback, page_ids)
                if page is not None and page is not showing:
                    self._dirty.add(id(page))
                    continue
                callback(*args)
        finally:
            self._remap(showing, saved_placement)

    # Re-scaling a visible tab widget by widget makes Tk re-lay-out and repaint it after nearly every
    # widget; taking it off screen while that happens and putting it back once is ~1.7x faster.
    # CTk's own place_forget() must be the one that unmaps it: it also clears the "last geometry
    # call" CTk replays from _set_scaling(), which would otherwise re-place (and re-map) the page
    # in the middle of the loop. The saved placement is restored raw, so a frozen pixel size (see
    # ResizeFreezer) comes back exactly rather than being DPI-scaled a second time.
    _PLACE_KEYS = ("x", "y", "relx", "rely", "width", "height", "relwidth", "relheight", "anchor", "bordermode")

    def _unmap(self, page):
        if page is None or page.winfo_manager() != "place":
            return None
        info = page.place_info()
        options = []
        for key in self._PLACE_KEYS:
            options += ["-" + key, info.get(key, "")]
        page.place_forget()
        return options

    @staticmethod
    def _remap(page, options) -> None:
        if page is not None and options is not None and page.winfo_exists():
            page.tk.call("place", "configure", page._w, *options)  # exactly as it was, incl. a frozen size

    def apply_pending(self, page) -> bool:
        """Re-scale a page that was hidden when the DPI last changed. Call right after it is shown.
        Returns True if anything had to be re-applied."""
        if id(page) not in self._dirty:
            return False
        self._dirty.discard(id(page))
        args = self._scaling_args()
        page_ids = self._page_ids()
        for callback in list(ScalingTracker.window_widgets_dict.get(self.root, [])):
            if self._page_of(callback, page_ids) is page:
                callback(*args)
        return True


class ResizeFreezer:
    """While the window edge is being dragged, keep the visible page at its last size instead of
    re-flowing every widget on every size step (each step costs ~100+ ms of Tk layout and repaint on
    the heavy tabs). Nothing goes blank: the page stays on screen as it was, the newly exposed
    strip shows the plain background, and once the size has stopped changing for SETTLE_MS the page
    is released to fill the window and re-flows exactly once. Moves that don't change the window
    size (dragging it around, or between monitors of the same size) never trigger it."""

    SETTLE_MS = 90

    def __init__(self, root, current_page: Callable[[], Optional[object]]):
        self.root = root
        self.current_page = current_page
        self._last_size = None
        self._frozen_page = None
        self._job = None
        root.bind("<Configure>", self._on_configure, add="+")

    @staticmethod
    def _place(page, *options) -> None:
        # Raw Tk call: CTk's own place() would rescale width/height by the DPI factor a second time.
        page.tk.call("place", "configure", page._w, *options)

    def _on_configure(self, event) -> None:
        if event.widget is not self.root:
            return  # "." is in every widget's bindtags, so Configure of any child lands here too
        size = (event.width, event.height)
        if size == self._last_size:
            return
        first_event = self._last_size is None
        self._last_size = size
        if first_event:
            return
        page = self.current_page()
        if page is None or not page.winfo_ismapped():
            return
        if self._frozen_page is None:
            width, height = page.winfo_width(), page.winfo_height()
            if width > 1 and height > 1:
                self._place(page, "-relwidth", "", "-relheight", "", "-width", width, "-height", height)
                self._frozen_page = page
        if self._job is not None:
            self.root.after_cancel(self._job)
        self._job = self.root.after(self.SETTLE_MS, self.thaw)

    def thaw(self) -> None:
        """Release the frozen page so it fills the window again (one re-flow). Safe to call any time."""
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except Exception:  # noqa: BLE001 - a stale/finished job id is harmless
                pass
            self._job = None
        page, self._frozen_page = self._frozen_page, None
        if page is not None and page.winfo_exists():
            self._place(page, "-width", "", "-height", "", "-relwidth", 1, "-relheight", 1)
