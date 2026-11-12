"""Event model, timestamp parsing, result files, and AppSession's calibration maths."""

import csv
import json
import os
import unittest
from unittest import mock

import numpy as np

from core.session import AppSession, ReferenceImage, build_colour_reference_set, scaled_region
from data import export
from data.detection import Event, SkullSighting, format_seconds, parse_timecode
from tests import helpers
from video.video_probe import seconds_to_timecode


def make_event(number=1, seconds=272.5, skulls=1, colour="White", symbol="White Skull"):
    return Event(
        event_number=number, timestamp_seconds=seconds, frame_number=int(seconds * 60),
        timecode=seconds_to_timecode(seconds, 60),
        skulls=[SkullSighting((1, 2, 3, 4), 0.9, colour, 1.0, symbol) for _ in range(skulls)],
        event_end_frame=int(seconds * 60) + 100, event_end_seconds=seconds + 100 / 60,
        confirmation_frame=int(seconds * 60) + 130, confirmation_seconds=seconds + 130 / 60,
    )


class ParseTimecode(unittest.TestCase):
    def test_valid_forms(self):
        self.assertEqual(parse_timecode("1:23"), 83.0)
        self.assertEqual(parse_timecode("01:23:45"), 5025.0)
        self.assertEqual(parse_timecode("0:00.5"), 0.5)
        self.assertEqual(parse_timecode("  90 "), 90.0)

    def test_everything_else_is_rejected(self):
        for bad in ("", "abc", "nan", "NaN", "inf", "-inf", "-5", "1:-2", "1:2:3:4", "::", "1:", ":30", "1e999"):
            self.assertIsNone(parse_timecode(bad), repr(bad))

    def test_format_round_trip(self):
        self.assertEqual(format_seconds(272.5), "00:04:32.500")
        self.assertEqual(parse_timecode("4:32.5"), 272.5)


class EventModel(unittest.TestCase):
    def test_bookmark_is_the_start_not_the_confirmation(self):
        event = make_event()
        self.assertEqual(event.bookmark_seconds, event.timestamp_seconds)
        self.assertLess(event.timestamp_seconds, event.event_end_seconds)
        self.assertLess(event.event_end_seconds, event.confirmation_seconds)

    def test_preview_frame_falls_back_to_the_bookmark_frame(self):
        event = make_event()
        self.assertEqual(event.preview_frame, event.frame_number)
        event.best_frame = event.frame_number + 40
        self.assertEqual(event.preview_frame, event.frame_number + 40)

    def test_colours_and_counts(self):
        event = make_event(skulls=3)
        self.assertEqual(event.skull_count, 3)
        self.assertEqual(event.colors, ["White"])
        self.assertEqual(event.marker_label, "KILL — 3 SKULLS")
        self.assertEqual(make_event(skulls=1).marker_label, "KILL — 1 SKULL")


class ResultFiles(unittest.TestCase):
    def setUp(self):
        self.dir = helpers.make_temp_dir(self)
        self.events = [make_event(1, 100.0, colour="White"), make_event(2, 200.5, colour="Gold", symbol="Gold Skull")]

    def test_csv(self):
        path = export.write_csv(self.events, self.dir)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["event_number"] for r in rows], ["1", "2"])
        self.assertEqual(rows[1]["colours"], "Gold")
        self.assertEqual(rows[1]["timestamp_seconds"], "200.500")
        self.assertEqual(rows[1]["frame_number"], str(int(200.5 * 60)))

    def test_json_and_txt(self):
        data = json.loads(helpers.read_text(export.write_json(self.events, self.dir)))
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["timecode"], self.events[0].timecode)
        self.assertEqual(data[1]["colours"], ["Gold"])
        text = helpers.read_text(export.write_txt(self.events, self.dir))
        self.assertIn("Total events: 2", text)

    def test_summary(self):
        summary = export.summarize_events(self.events + [make_event(3, 300.0, skulls=3)])
        self.assertEqual((summary.total_events, summary.total_kills, summary.multi_kill_events), (3, 5, 1))
        self.assertEqual(summary.highest_event_skulls, 3)
        self.assertEqual(export.summarize_events([]).total_events, 0)

    def test_output_dir_is_created_beside_the_video(self):
        video = os.path.join(self.dir, "my clip.mkv")
        out = export.get_output_dir(video)
        self.assertTrue(os.path.isdir(out))
        self.assertEqual(os.path.basename(out), "my clip_BF5AutoMarker_Output")

    def test_logger_writes_and_never_raises_afterwards(self):
        logger = export.AnalysisLogger(self.dir)
        logger.log("first")
        self.assertIn("first", helpers.read_text(logger.log_path))
        with mock.patch("builtins.open", side_effect=OSError("no space left on device")):
            logger.log("second")      # must not raise: a full disk can't abort an analysis
            logger.log("third")
        self.assertIsNotNone(logger.write_error)
        self.assertNotIn("second", helpers.read_text(logger.log_path))

    def test_logger_creation_failure_is_an_oserror_the_caller_can_handle(self):
        with self.assertRaises(OSError):
            export.AnalysisLogger(os.path.join(self.dir, "does", "not", "exist"))


class SessionScaling(unittest.TestCase):
    def test_region_is_scaled_to_the_video_resolution(self):
        region, mismatch = scaled_region((930, 666, 57, 51), 1920, 1080, 2560, 1440)
        self.assertEqual(region, (1240, 888, 76, 68))
        self.assertFalse(mismatch)

    def test_same_resolution_and_unknown_calibration_resolution_are_unchanged(self):
        self.assertEqual(scaled_region((1, 2, 3, 4), 1920, 1080, 1920, 1080), ((1, 2, 3, 4), False))
        self.assertEqual(scaled_region((1, 2, 3, 4), 0, 0, 2560, 1440), ((1, 2, 3, 4), False))
        self.assertEqual(scaled_region(None, 1920, 1080, 2560, 1440), (None, False))

    def test_aspect_ratio_mismatch_is_flagged(self):
        _, mismatch = scaled_region((10, 10, 20, 20), 1920, 1080, 1080, 1920)
        self.assertTrue(mismatch)

    def test_oversized_references_are_named(self):
        session = AppSession()
        session.set_detection_region((0, 0, 20, 20), 1920, 1080)
        session.reference_images = [
            ReferenceImage("small", np.zeros((10, 10), np.uint8), 1.0),
            ReferenceImage("huge", np.zeros((50, 50), np.uint8), 1.0),
            ReferenceImage("scaled-down", np.zeros((50, 50), np.uint8), 0.2),
        ]
        self.assertEqual(session.oversized_reference_labels(), ["huge"])

    def test_is_calibrated_needs_region_and_reference(self):
        session = AppSession()
        self.assertFalse(session.is_calibrated())
        session.set_detection_region((0, 0, 20, 20), 1920, 1080)
        self.assertFalse(session.is_calibrated())
        session.reference_images = [ReferenceImage("x", np.zeros((5, 5), np.uint8))]
        self.assertTrue(session.is_calibrated())

    def test_colour_size_band_grows_with_the_video_resolution(self):
        from detectors.reference_profile import build_reference_profile
        profile = build_reference_profile("Gold", helpers.make_reference_image(helpers.GOLD_BGR))
        refs = [ReferenceImage("Gold", np.zeros((5, 5), np.uint8), scale=1.25, profile=profile)]
        same = build_colour_reference_set(refs, 1080, 1080).scale_bands["Gold"]
        bigger = build_colour_reference_set(refs, 1080, 1440).scale_bands["Gold"]
        self.assertAlmostEqual(bigger[3] / same[3], 1440 / 1080)  # upper full-credit bound scales with the video
        self.assertEqual(bigger[2], same[2], "never searches smaller than the calibrated size on a bigger video")

    def test_achromatic_references_get_no_colour_matcher(self):
        from detectors.reference_profile import build_reference_profile
        white = build_reference_profile("White", helpers.make_reference_image(helpers.WHITE_BGR))
        refs = [ReferenceImage("White", np.zeros((5, 5), np.uint8), profile=white)]
        self.assertIsNone(build_colour_reference_set(refs, 1080, 1440))


if __name__ == "__main__":
    unittest.main()
