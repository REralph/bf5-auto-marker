"""The CALIBRATE page on synthetic test footage (a skull with the green training box drawn around it):
AUTO-DETECT / calibration test running off the UI thread, the library staying consistent with the
session, the bookmark adjustment, and RESET CALIBRATION with a preview clip still open."""

import os
import shutil
import tempfile
import time
import unittest

import cv2

import core.calibration_store as cs
from core.session import AppSession
from tests import helpers
from tests.ui_base import UiTestCase

SKULL_CENTRE = (480, 270)
_cache = {}


def training_clip() -> str:
    """960x540, 90 frames: a gold skull with a pure-green rectangle around it on frames 20-69."""
    if "path" not in _cache:
        directory = tempfile.mkdtemp(prefix="bf5_cal_clip_")
        _cache["dir"] = directory
        frames = []
        for i in range(90):
            frame = helpers.noise_background((540, 960, 3), i)
            if 20 <= i < 70:
                helpers.draw_skull(frame, *SKULL_CENTRE, 60, helpers.GOLD_BGR, background=(5, 5, 5))
                cx, cy = SKULL_CENTRE
                cv2.rectangle(frame, (cx - 48, cy - 52), (cx + 48, cy + 58), (0, 255, 0), 4)
            frames.append(frame)
        _cache["path"] = helpers.write_clip(os.path.join(directory, "Gold Skull.avi"), frames, fps=30.0)
    return _cache["path"]


def tearDownModule():
    if "dir" in _cache:
        shutil.rmtree(_cache["dir"], ignore_errors=True)


class CalibratePageCase(UiTestCase):
    def setUp(self):
        super().setUp()
        self.page = self.app.pages["CALIBRATE"]
        self.app.show_page("CALIBRATE")
        self.pump(0.2)

    def add_library(self):
        reference = os.path.join(self.state_dir, "Gold Skull.png")
        cv2.imwrite(reference, helpers.make_reference_image(helpers.GOLD_BGR))
        cs.add_reference_image(reference)
        cs.add_test_footage(training_clip())
        self.page._refresh_all_statuses()

    def run_background_action(self, start, timeout=120):
        start()
        self.assertTrue(self.page._busy)
        self.wait_for(lambda: not self.page._busy, timeout, "the calibration action to finish")
        self.pump(0.1)


class BackgroundRunner(CalibratePageCase):
    def test_the_window_stays_responsive_while_a_long_action_runs(self):
        ticks = []
        applied = []
        label = self.page.search_area_result_label

        def slow():
            time.sleep(0.8)
            return "done"

        self.page._run_background(label, "Working", slow, applied.append)
        self.assertTrue(self.page._busy)
        self.assertIn("Working", label.cget("text"))
        self.assertTrue(all(b.cget("state") == "disabled" for b in self.page._heavy_buttons))

        def tick():
            ticks.append(time.time())
            if self.page._busy:
                self.root.after(50, tick)

        self.root.after(50, tick)
        self.wait_for(lambda: not self.page._busy, 10, "background action")
        self.assertGreaterEqual(len(ticks), 5, "the event loop kept running while the worker computed")
        self.assertEqual(applied, ["done"])
        self.assertTrue(all(b.cget("state") == "normal" for b in self.page._heavy_buttons))

    def test_a_second_long_action_is_refused_while_one_runs(self):
        label = self.page.search_area_result_label
        ran = []
        self.page._run_background(label, "First", lambda: time.sleep(0.4) or 1, ran.append)
        self.page._run_background(label, "Second", lambda: 2, ran.append)
        self.assertEqual(self.dialogs.show_info.call_args.args[2], "Please Wait")
        self.wait_for(lambda: not self.page._busy, 10, "first action")
        self.assertEqual(ran, [1])

    def test_expected_and_unexpected_failures_are_shown_not_raised(self):
        from ui.pages.calibrate_page import CalibrationError
        label = self.page.search_area_result_label

        def expected():
            raise CalibrationError("Add a clip first.")

        self.page._run_background(label, "x", expected, lambda r: None)
        self.wait_for(lambda: not self.page._busy, 10, "expected failure")
        self.assertEqual(label.cget("text"), "Add a clip first.")

        def unexpected():
            raise ValueError("something odd")

        self.page._run_background(label, "x", unexpected, lambda r: None)
        self.wait_for(lambda: not self.page._busy, 10, "unexpected failure")
        self.assertIn("CALIBRATION ERROR", label.cget("text"))
        self.assertIn("something odd", label.cget("text"))
        self.assertIn("something odd", self.page._error_trace)
        self.assertTrue(all(b.cget("state") == "normal" for b in self.page._heavy_buttons), "buttons come back after a failure")

    def test_library_changes_are_refused_while_busy(self):
        label = self.page.search_area_result_label
        self.page._run_background(label, "x", lambda: time.sleep(0.4), lambda r: None)
        for action in (self.page._on_upload_reference_image, self.page._on_add_test_video, self.page._on_save_calibration,
                       lambda: self.page._on_remove_reference_image("x.png")):
            self.dialogs.show_info.reset_mock()
            action()
            self.assertEqual(self.dialogs.show_info.call_args.args[2], "Please Wait")
        self.wait_for(lambda: not self.page._busy, 10, "background action")


class SyntheticAutoDetect(CalibratePageCase):
    def test_auto_detect_learns_the_region_and_the_real_size_and_colour(self):
        self.add_library()
        self.run_background_action(self.page._on_auto_detect)
        text = self.page.search_area_result_label.cget("text")
        self.assertIn("✓ SEARCH AREA DETECTED", text, text)
        self.assertIn("Learned from: Gold Skull.avi", text)

        session = self.app.session
        x, y, w, h = session.detection_region
        self.assertTrue(x < SKULL_CENTRE[0] < x + w and y < SKULL_CENTRE[1] < y + h, "the search area covers the skull")
        self.assertEqual((session.calibration_video_width, session.calibration_video_height), (960, 540))
        gold = session.reference_images[0]
        self.assertEqual(gold.label, "Gold Skull")
        self.assertAlmostEqual(gold.scale, 0.95, delta=0.12)
        self.assertIsNotNone(gold.footage, "the size/colour of the skull in the user's own footage was learned")
        self.assertGreater(gold.footage.lab_b, 180)
        self.assertIn("learned from your footage", text)

    def test_calibration_test_then_save_then_reload(self):
        self.add_library()
        self.run_background_action(self.page._on_auto_detect)
        self.run_background_action(self.page._on_run_calibration_test)
        text = self.page.calibration_test_result_label.cget("text")
        self.assertIn("CALIBRATION PASSED", text, text)
        self.assertIn("Detected as Gold Skull", text)

        self.page._on_save_calibration()
        self.assertEqual(self.dialogs.show_info.call_args.args[2], "Calibration Saved")
        reloaded = AppSession()
        cs.load_into(reloaded)
        self.assertEqual(reloaded.detection_region, self.app.session.detection_region)
        self.assertIsNotNone(reloaded.reference_images[0].footage)

    def test_gold_skull_test_passes_and_jumps_to_the_best_frame(self):
        self.add_library()
        self.run_background_action(self.page._on_auto_detect)
        self.run_background_action(self.page._on_run_gold_skull_test)
        text = self.page.gold_test_result_label.cget("text")
        self.assertIn("Gold skull detected: YES", text, text)
        self.assertIn("✓ PASS", text)
        best = self.page._last_gold_test.best_frame_index
        self.assertTrue(20 <= best < 70)
        self.assertEqual(self.page.test_session.current_frame, best)

    def test_auto_detect_without_footage_or_references_explains_what_to_do(self):
        self.run_background_action(self.page._on_auto_detect)
        self.assertIn("No test footage", self.page.search_area_result_label.cget("text"))
        cs.add_test_footage(training_clip())
        self.run_background_action(self.page._on_auto_detect)
        self.assertIn("Upload at least one skull reference", self.page.search_area_result_label.cget("text"))


class LibraryConsistency(CalibratePageCase):
    def test_removing_a_reference_stops_it_being_used_at_once(self):
        self.add_library()
        self.run_background_action(self.page._on_auto_detect)
        self.assertEqual(len(self.app.session.reference_images), 1)
        self.page._on_remove_reference_image("Gold Skull.png")
        self.assertEqual(self.app.session.reference_images, [], "not left in memory until the next restart")
        self.assertFalse(self.app.session.is_calibrated())
        self.assertFalse(cs.list_reference_images())

    def test_changing_the_references_shows_a_rerun_hint_until_auto_detect(self):
        self.add_library()
        self.run_background_action(self.page._on_auto_detect)
        extra = os.path.join(self.state_dir, "White.png")
        cv2.imwrite(extra, helpers.make_reference_image(helpers.WHITE_BGR))
        cs.add_reference_image(extra)
        self.page._library_changed = True
        self.page._refresh_all_statuses()
        self.assertIn("re-run AUTO-DETECT", self.page.reference_section.status_label.cget("text"))
        self.run_background_action(self.page._on_auto_detect)
        self.assertNotIn("re-run", self.page.reference_section.status_label.cget("text"))

    def test_file_failures_are_reported_plainly(self):
        from unittest import mock
        image = os.path.join(self.state_dir, "Ref.png")
        cv2.imwrite(image, helpers.make_reference_image(helpers.GOLD_BGR))
        with mock.patch("ui.pages.calibrate_page.filedialog.askopenfilename", return_value=image), \
                mock.patch("core.calibration_store.add_reference_image", side_effect=OSError("There is not enough space on the disk")):
            self.page._on_upload_reference_image()
        self.assertIn("FILE OPERATION FAILED", self.error_dialog_titles())
        self.assertEqual(cs.list_reference_images(), [])

    def test_unreadable_reference_image_is_refused(self):
        from unittest import mock
        bad = os.path.join(self.state_dir, "notanimage.png")
        helpers.write_text(bad, "not an image")
        with mock.patch("ui.pages.calibrate_page.filedialog.askopenfilename", return_value=bad):
            self.page._on_upload_reference_image()
        self.assertIn("Image Could Not Be Used", self.error_dialog_titles())
        self.assertEqual(cs.list_reference_images(), [])

    def test_reference_thumbnails_show_the_real_colour_and_what_the_file_contains(self):
        self.add_library()
        rows = " ".join(self.texts(self.page.reference_list_frame))
        self.assertIn("Gold / Yellow", rows)
        self.assertIn("colour information available", rows)
        import numpy as np
        rgb = np.array(self.page._reference_thumb_images[0]._light_image.convert("RGB")).reshape(-1, 3)
        skull = rgb[rgb.sum(axis=1) > 200]
        self.assertGreater(skull[:, 0].mean(), 200)
        self.assertLess(skull[:, 2].mean(), 60, "the thumbnail is yellow, not a white-looking grayscale rendering")


class BookmarkAdjustmentEntry(CalibratePageCase):
    def test_only_real_numbers_in_range_are_applied(self):
        session = self.app.session
        self.page.bookmark_adjustment_var.set("-0.25")
        self.assertEqual(session.bookmark_adjustment_seconds, -0.25)
        for bad in ("nan", "inf", "-inf", "9999", "abc", "-"):
            self.page.bookmark_adjustment_var.set(bad)
            self.assertEqual(session.bookmark_adjustment_seconds, -0.25, f"{bad!r} must not replace a valid value")
            self.assertEqual(self.page.bookmark_entry.cget("border_color"), self.colors_danger())
        self.page.bookmark_adjustment_var.set("0.3")
        self.assertEqual(session.bookmark_adjustment_seconds, 0.3)
        self.assertNotEqual(self.page.bookmark_entry.cget("border_color"), self.colors_danger())

    def colors_danger(self):
        from ui.main_window import COLORS
        return COLORS["danger"]

    def test_test_button_previews_the_move_and_rejects_garbage(self):
        cs.add_test_footage(training_clip())
        self.page._refresh_all_statuses()
        self.page._load_preview_clip("Gold Skull.avi")
        self.page.scrubber.set_frame(45)
        self.page.bookmark_adjustment_var.set("0.5")
        self.page._on_test_bookmark_adjustment()
        text = self.page.bookmark_test_result_label.cget("text")
        self.assertIn("Original: 00:00:01.500", text)
        self.assertIn("New position: 00:00:02.000", text)
        self.page.bookmark_adjustment_var.set("nan")
        self.page._on_test_bookmark_adjustment()
        self.assertIn("between -600 and 600", self.page.bookmark_test_result_label.cget("text"))

    def test_unsaved_marker_until_the_calibration_is_saved(self):
        self.add_library()
        self.run_background_action(self.page._on_auto_detect)
        self.page.bookmark_adjustment_var.set("0.4")
        self.assertIn("not saved yet", self.page.bookmark_unsaved_label.cget("text"))
        self.page._on_save_calibration()
        self.assertEqual(self.page.bookmark_unsaved_label.cget("text"), "")
        reloaded = AppSession()
        cs.load_into(reloaded)
        self.assertEqual(reloaded.bookmark_adjustment_seconds, 0.4)
        self.page.bookmark_adjustment_var.set("0")
        self.assertIn("not saved yet", self.page.bookmark_unsaved_label.cget("text"))


class ResetCalibration(CalibratePageCase):
    def test_reset_works_even_while_a_preview_clip_is_open(self):
        from ui.pages.settings_page import ManageCalibrationDialog
        from ui.main_window import COLORS
        self.add_library()
        self.run_background_action(self.page._on_auto_detect)
        self.page.bookmark_adjustment_var.set("0.4")
        self.page._load_preview_clip("Gold Skull.avi")
        self.assertTrue(self.page.test_session.is_loaded, "the clip is open, so Windows would refuse to delete it")

        dialog = ManageCalibrationDialog(self.root, self.app, COLORS)
        self.pump(0.1)
        dialog._reset()
        self.pump(0.2)

        self.assertEqual(cs.list_test_footage(), [], "the open clip could be deleted")
        self.assertEqual(cs.list_reference_images(), [])
        self.assertFalse(os.path.exists(cs.MANIFEST_PATH))
        session = self.app.session
        self.assertEqual((session.detection_region, session.reference_images, session.bookmark_adjustment_seconds), (None, [], 0.0))
        self.assertEqual(self.page.bookmark_adjustment_var.get(), "0")
        self.assertFalse(self.page.test_session.is_loaded)
        self.assertIn("No reference images yet", " ".join(self.texts(self.page.reference_list_frame)))
        self.assertNotIn("RESET INCOMPLETE", self.error_dialog_titles())

    def test_reset_reports_files_it_could_not_delete_but_still_clears_the_app(self):
        from unittest import mock
        from ui.pages.settings_page import ManageCalibrationDialog
        from ui.main_window import COLORS
        self.add_library()
        dialog = ManageCalibrationDialog(self.root, self.app, COLORS)
        with mock.patch("core.calibration_store.reset", side_effect=OSError("Could not remove: Gold Skull.png: in use")):
            dialog._reset()
        self.pump(0.2)
        self.assertIn("RESET INCOMPLETE", self.error_dialog_titles())
        self.assertEqual(self.app.session.reference_images, [])

    def test_showing_the_page_again_resyncs_it_with_the_library(self):
        self.add_library()
        self.app.show_page("WORKFLOW")
        cs.remove_reference_image("Gold Skull.png")      # changed behind the page's back
        self.app.show_page("CALIBRATE")
        self.pump(0.2)
        self.assertIn("No reference images yet", " ".join(self.texts(self.page.reference_list_frame)))


if __name__ == "__main__":
    unittest.main()
