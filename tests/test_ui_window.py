"""Window resize / monitor-change behaviour (ui/window_perf.py): only the visible page is laid out, the
page is held still while an edge is being dragged, and a DPI change re-scales hidden pages lazily.
The monitor's DPI is mocked; the real CustomTkinter DPI tracker code path is what runs."""

import time
import unittest
from unittest import mock

import customtkinter as ctk
from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker

from tests.ui_base import UiTestCase


def labels(test, page):
    return [w for w in test.widgets(page) if isinstance(w, ctk.CTkLabel)]


def scale(widget):
    return round(widget._get_widget_scaling(), 3)


class OnlyTheVisiblePageIsLaidOut(UiTestCase):
    def test_exactly_one_page_is_mapped_and_it_fills_the_window(self):
        for name in ("WORKFLOW", "CALIBRATE", "SETTINGS", "PREVIEW", "WORKFLOW"):
            self.app.show_page(name)
            self.pump(0.15)
            mapped = [n for n, p in self.app.pages.items() if p.winfo_ismapped()]
            self.assertEqual(mapped, [name])
            page, content = self.app.pages[name], self.app.content_frame
            self.assertLessEqual(abs(page.winfo_width() - content.winfo_width()), 1)
            self.assertLessEqual(abs(page.winfo_height() - content.winfo_height()), 1)

    def test_nav_highlight_follows_the_page(self):
        self.app.show_page("CALIBRATE")
        from ui.main_window import COLORS
        self.assertEqual(self.app.nav_buttons["CALIBRATE"].cget("fg_color"), COLORS["accent"])
        self.assertNotEqual(self.app.nav_buttons["WORKFLOW"].cget("fg_color"), COLORS["accent"])


class ResizeFreeze(UiTestCase):
    def setUp(self):
        super().setUp()
        self.page = self.app.pages["WORKFLOW"]
        self.freezer = self.app._resize_freezer
        self.pump(0.3)

    def info(self):
        return self.page.place_info()

    def test_the_page_is_held_at_its_size_while_the_window_size_keeps_changing(self):
        width, height = self.page.winfo_width(), self.page.winfo_height()
        self.assertEqual(float(self.info()["relwidth"]), 1.0)
        self.root.geometry("1100x760")
        self.root.update()
        self.assertIs(self.freezer._frozen_page, self.page)
        self.assertEqual((int(float(self.info()["width"])), int(float(self.info()["height"]))), (width, height))
        self.root.geometry("1180x800")
        self.root.update()
        self.root.geometry("1250x830")
        self.root.update()
        self.assertEqual(int(float(self.info()["width"])), width, "one freeze for the whole drag, no re-flow in between")

    def test_it_is_released_and_fills_the_window_once_the_size_settles(self):
        self.root.geometry("1250x830")
        self.root.update()
        self.pump(0.4)
        self.assertIsNone(self.freezer._frozen_page)
        self.assertEqual(float(self.info()["relwidth"]), 1.0)
        self.assertIn(str(self.info().get("width", "")), ("", "0"))
        self.assertLessEqual(abs(self.page.winfo_width() - self.app.content_frame.winfo_width()), 1)

    def test_moving_the_window_without_resizing_never_freezes_anything(self):
        self.pump(0.2)
        self.root.geometry(f"+{self.root.winfo_x() + 30}+{self.root.winfo_y() + 20}")
        self.root.update()
        self.pump(0.15)
        self.assertIsNone(self.freezer._frozen_page)

    def test_switching_pages_mid_drag_releases_the_old_page_first(self):
        self.root.geometry("1180x790")
        self.root.update()
        self.assertIsNotNone(self.freezer._frozen_page)
        self.app.show_page("CALIBRATE")
        self.pump(0.3)
        self.assertIsNone(self.freezer._frozen_page)
        calibrate, content = self.app.pages["CALIBRATE"], self.app.content_frame
        self.assertEqual(float(calibrate.place_info()["relwidth"]), 1.0)
        self.assertLessEqual(abs(calibrate.winfo_width() - content.winfo_width()), 1)

    def test_content_reflows_to_the_new_width_after_a_big_resize(self):
        self.root.geometry("1500x900")
        self.root.update()
        self.pump(0.4)
        self.assertGreater(self.page.video_section.master.winfo_width(), 1250)


class MonitorChange(UiTestCase):
    """Simulates dragging the window to a monitor with a different scale factor."""

    def setUp(self):
        super().setUp()
        self.dpi = {"value": 1.0}
        patcher = mock.patch.object(ScalingTracker, "get_window_dpi_scaling", classmethod(lambda cls, window: self.dpi["value"]))
        patcher.start()
        self.addCleanup(patcher.stop)
        real_attributes = self.root.attributes
        # CustomTkinter dims the window to 15% during a re-scale; keep the invisible test window invisible
        self.root.attributes = lambda *a, **k: None if (a and a[0] == "-alpha") else real_attributes(*a, **k)
        self.app.show_page("WORKFLOW")
        self.pump(0.3)

    def change_monitor_to(self, value):
        self.dpi["value"] = value
        started = time.perf_counter()
        ScalingTracker.check_dpi_scaling()
        return time.perf_counter() - started

    def test_the_visible_page_and_the_shell_rescale_at_once_hidden_pages_wait(self):
        workflow, calibrate = self.app.pages["WORKFLOW"], self.app.pages["CALIBRATE"]
        self.change_monitor_to(1.5)
        self.assertTrue(all(scale(w) == 1.5 for w in labels(self, workflow)))
        shell = [w for w in labels(self, self.root) if not any(w in set(self.widgets(p)) for p in self.app.pages.values())]
        self.assertTrue(shell)
        self.assertTrue(all(scale(w) == 1.5 for w in shell))
        self.assertTrue(all(scale(w) == 1.0 for w in labels(self, calibrate)), "a hidden page is not re-scaled yet")
        self.root.update()  # Tk maps a re-placed widget on its next idle cycle
        self.assertTrue(workflow.winfo_ismapped(), "and the visible page is back on screen afterwards")
        self.assertEqual(float(workflow.place_info()["relwidth"]), 1.0)

    def test_a_deferred_page_is_brought_up_to_date_when_it_is_shown(self):
        self.change_monitor_to(1.5)
        self.app.show_page("CALIBRATE")
        self.pump(0.3)
        self.assertTrue(all(scale(w) == 1.5 for w in labels(self, self.app.pages["CALIBRATE"])))

    def test_several_changes_in_a_row_leave_every_page_consistent(self):
        for value in (1.5, 1.0, 1.25):
            self.change_monitor_to(value)
            self.pump(0.2)
        for name in ("CALIBRATE", "SETTINGS", "PREVIEW", "WORKFLOW"):
            self.app.show_page(name)
            self.pump(0.2)
            self.assertTrue(all(scale(w) == 1.25 for w in labels(self, self.app.pages[name])), name)

    def test_widgets_rebuilt_on_a_deferred_page_do_not_break_the_catch_up(self):
        self.app.show_page("SETTINGS")
        self.pump(0.2)
        self.change_monitor_to(1.25)
        calibrate = self.app.pages["CALIBRATE"]
        calibrate._refresh_footage_list()
        calibrate._refresh_reference_list()
        self.app.show_page("CALIBRATE")
        self.pump(0.3)
        self.assertTrue(all(scale(w) == 1.25 for w in labels(self, calibrate)))

    def test_a_monitor_change_during_a_resize_keeps_the_frozen_size(self):
        page, freezer = self.app.pages["WORKFLOW"], self.app._resize_freezer
        self.root.geometry("1220x800")
        self.root.update()
        frozen_width = int(float(page.place_info()["width"]))
        self.change_monitor_to(1.25)            # no event pumping: the settle timer cannot have fired yet
        info = page.place_info()
        self.assertIs(freezer._frozen_page, page)
        self.assertEqual(int(float(info["width"])), frozen_width)
        self.pump(0.4)
        self.assertIsNone(freezer._frozen_page)
        self.assertLessEqual(abs(page.winfo_width() - self.app.content_frame.winfo_width()), 1)

    def test_a_second_window_does_not_stack_wrappers_on_the_tracker(self):
        from ui.window_perf import DeferredPageScaling
        first = ScalingTracker.update_scaling_callbacks_for_window.__self__
        self.assertIsInstance(first, DeferredPageScaling)
        second = DeferredPageScaling(self.root, self.app.pages, lambda: None)
        self.assertIs(second._original, first._original, "wraps CustomTkinter's own function, not the other wrapper")


if __name__ == "__main__":
    unittest.main()
