"""calibration/ persistence: round trips, tolerance of damaged files, safe reset. Runs entirely in a
temp folder -- never the real calibration."""

import os
import unittest
from unittest import mock

import cv2

import core.calibration_store as cs
from core.session import AppSession
from tests import helpers


class CalibrationStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = helpers.isolate_app_state(self)
        os.makedirs(cs.REFERENCES_DIR, exist_ok=True)
        os.makedirs(cs.TEST_FOOTAGE_DIR, exist_ok=True)

    def _add_reference(self, name="Gold Skull.png", color=helpers.GOLD_BGR):
        path = os.path.join(cs.REFERENCES_DIR, name)
        cv2.imwrite(path, helpers.make_reference_image(color))
        return path

    def _session(self):
        session = AppSession()
        cs.load_into(session)
        return session

    # ------------------------------------------------------------------ round trips
    def test_nothing_saved_loads_as_not_found(self):
        session = AppSession()
        self.assertFalse(cs.load_into(session))
        self.assertIsNone(session.detection_region)
        self.assertEqual(session.bookmark_adjustment_seconds, 0.0)
        self.assertFalse(session.is_calibrated())

    def test_region_references_scales_and_adjustment_round_trip(self):
        self._add_reference("Gold Skull.png")
        self._add_reference("White.png", helpers.WHITE_BGR)
        session = self._session()
        session.set_detection_region((930, 666, 57, 51), 1920, 1080)
        session.reference_images[0].scale = 1.25
        session.reference_images[1].scale = 0.13
        session.bookmark_adjustment_seconds = -0.25
        cs.save_region(session)

        loaded = self._session()
        self.assertEqual(loaded.detection_region, (930, 666, 57, 51))
        self.assertEqual((loaded.calibration_video_width, loaded.calibration_video_height), (1920, 1080))
        self.assertEqual({r.label: r.scale for r in loaded.reference_images}, {"Gold Skull": 1.25, "White": 0.13})
        self.assertEqual(loaded.bookmark_adjustment_seconds, -0.25)
        self.assertTrue(loaded.is_calibrated())

    def test_references_keep_their_colour_alongside_the_grayscale_template(self):
        self._add_reference("Gold Skull.png")
        session = self._session()
        ref = session.reference_images[0]
        self.assertEqual(ref.image.ndim, 2, "the legacy template the original detector matches is grayscale")
        self.assertIsNotNone(ref.profile)
        self.assertEqual(ref.profile.signature.description, "Gold / Yellow")
        self.assertTrue(ref.profile.signature.chromatic)

    def test_learned_footage_appearance_round_trips(self):
        from detectors.colour_matcher import FootageObservation
        self._add_reference("Gold Skull.png")
        session = self._session()
        session.set_detection_region((1, 2, 30, 40), 1920, 1080)
        session.reference_images[0].footage = FootageObservation(scale=1.2, lab_l=246, lab_a=102, lab_b=220, confidence=0.89)
        cs.save_region(session)
        footage = self._session().reference_images[0].footage
        self.assertEqual((footage.scale, footage.lab_a, footage.lab_b), (1.2, 102, 220))

    def test_saved_file_is_valid_json_with_no_temp_leftovers(self):
        session = self._session()
        session.set_detection_region((1, 2, 3, 4), 100, 100)
        cs.save_region(session)
        self.assertEqual([n for n in os.listdir(cs.CALIBRATION_DIR) if n.endswith(".tmp")], [])
        self.assertIn('"detection_region"', helpers.read_text(cs.MANIFEST_PATH))

    # ------------------------------------------------------------------ damaged / hand-edited files
    def test_corrupt_manifest_is_tolerated(self):
        helpers.write_text(cs.MANIFEST_PATH, "{ this is not json")
        self._add_reference()
        session = self._session()
        self.assertIsNone(session.detection_region)
        self.assertEqual(len(session.reference_images), 1)

    def test_invalid_values_are_sanitised_rather_than_trusted(self):
        import json
        self._add_reference("Gold Skull.png")
        helpers.write_text(cs.MANIFEST_PATH, json.dumps({
            "detection_region": [10, 20, 0, 50],          # zero width
            "calibration_video_width": "wide",
            "bookmark_adjustment_seconds": "NaN",
            "reference_scales": {"Gold Skull": "big"},
            "reference_footage": {"Gold Skull": {"scale": "x"}},
        }))
        session = self._session()
        self.assertIsNone(session.detection_region)
        self.assertEqual(session.bookmark_adjustment_seconds, 0.0)
        self.assertEqual(session.reference_images[0].scale, 1.0)
        self.assertIsNone(session.reference_images[0].footage)

    def test_non_finite_and_absurd_bookmark_adjustments_are_clamped_or_rejected(self):
        import json
        for value, expected in ((float("nan"), 0.0), (float("inf"), 0.0), (99999, 600.0), (-99999, -600.0), (0.3, 0.3)):
            helpers.write_text(cs.MANIFEST_PATH, json.dumps({"bookmark_adjustment_seconds": value}))
            self.assertEqual(self._session().bookmark_adjustment_seconds, expected, value)

    def test_region_with_float_coordinates_loads_as_ints(self):
        import json
        helpers.write_text(cs.MANIFEST_PATH, json.dumps({"detection_region": [930.0, 666.4, 57.0, 51.0]}))
        self.assertEqual(self._session().detection_region, (930, 666, 57, 51))

    # ------------------------------------------------------------------ library operations
    def test_adding_a_reference_never_overwrites_a_different_file_with_the_same_name(self):
        source_dir = helpers.make_temp_dir(self)
        a = os.path.join(source_dir, "skull.png")
        cv2.imwrite(a, helpers.make_reference_image(helpers.GOLD_BGR))
        first = cs.add_reference_image(a)
        second = cs.add_reference_image(a)
        self.assertEqual(first, "skull.png")
        self.assertNotEqual(first, second)
        self.assertEqual(len(cs.list_reference_images()), 2)

    def test_reset_removes_everything(self):
        self._add_reference()
        helpers.write_text(os.path.join(cs.TEST_FOOTAGE_DIR, "clip.mp4"), "x")
        session = self._session()
        session.set_detection_region((1, 2, 3, 4), 100, 100)
        cs.save_region(session)
        cs.reset()
        self.assertEqual(cs.list_reference_images(), [])
        self.assertEqual(cs.list_test_footage(), [])
        self.assertFalse(os.path.exists(cs.MANIFEST_PATH))

    def test_reset_attempts_every_deletion_and_reports_what_failed(self):
        self._add_reference("a.png")
        self._add_reference("b.png")
        helpers.write_text(os.path.join(cs.TEST_FOOTAGE_DIR, "clip.mp4"), "x")
        real_remove = os.remove

        def flaky_remove(path):
            if os.path.basename(path) == "a.png":
                raise PermissionError("in use by another process")
            real_remove(path)

        with mock.patch("os.remove", side_effect=flaky_remove):
            with self.assertRaises(OSError) as raised:
                cs.reset()
        self.assertIn("a.png", str(raised.exception))
        self.assertEqual(cs.list_reference_images(), ["a.png"], "only the locked file is left")
        self.assertEqual(cs.list_test_footage(), [], "the rest was still removed")


if __name__ == "__main__":
    unittest.main()
