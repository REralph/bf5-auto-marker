"""Marker placement and the post-send verification against Resolve's real marker list."""

import unittest

from data.detection import Event, SkullSighting
from resolve.resolve_controller import MAX_NUDGE_FRAMES, add_markers_from_detections, compute_marker_timestamp
from tests.helpers import FakeProject, FakeTimeline

FPS = 60


def event(number, seconds, skulls=1):
    return Event(
        event_number=number, timestamp_seconds=seconds, frame_number=int(seconds * FPS), timecode="",
        skulls=[SkullSighting((0, 0, 10, 10), 0.9, "White", 1.0, "White Skull") for _ in range(skulls)],
    )


def send(timeline, events, adjustment=0.0, name="Kill", colour="Red"):
    return add_markers_from_detections(timeline, FakeProject(str(FPS)), events, name, colour, adjustment)


class ComputeMarkerTimestamp(unittest.TestCase):
    def test_zero_is_the_exact_detected_time(self):
        self.assertEqual(compute_marker_timestamp(event(1, 272.5)), 272.5)

    def test_signed_adjustment_positive_later_negative_earlier(self):
        self.assertAlmostEqual(compute_marker_timestamp(event(1, 272.5), 0.3), 272.8)
        self.assertAlmostEqual(compute_marker_timestamp(event(1, 272.5), -0.25), 272.25)

    def test_never_negative(self):
        self.assertEqual(compute_marker_timestamp(event(1, 1.0), -5.0), 0.0)

    def test_does_not_modify_the_event(self):
        e = event(1, 10.0)
        compute_marker_timestamp(e, -3.0)
        self.assertEqual(e.timestamp_seconds, 10.0)


class SendMarkers(unittest.TestCase):
    def test_markers_land_on_the_exact_frames(self):
        timeline = FakeTimeline()
        result = send(timeline, [event(1, 100.0), event(2, 200.0), event(3, 300.5)])
        self.assertTrue(result.all_created)
        self.assertEqual((result.total, result.created, result.already_present, result.nudged), (3, 3, 0, 0))
        self.assertEqual(sorted(timeline.markers), [6000, 12000, 18030])

    def test_adjustment_is_applied_exactly_once(self):
        timeline = FakeTimeline()
        send(timeline, [event(1, 100.0)], adjustment=0.5)
        self.assertEqual(sorted(timeline.markers), [6030])

    def test_marker_label_and_note_describe_the_event(self):
        timeline = FakeTimeline()
        send(timeline, [event(1, 10.0, skulls=3)], name="kill", colour="Cyan")
        marker = timeline.markers[600]
        self.assertEqual(marker["name"], "KILL — 3 SKULLS")
        self.assertEqual(marker["color"], "Cyan")
        self.assertIn("Skulls: 3", marker["note"])

    def test_events_clamped_to_the_same_frame_all_get_a_marker(self):
        # a large negative adjustment clamps every early event to frame 0; Resolve allows one marker per frame
        timeline = FakeTimeline()
        result = send(timeline, [event(1, 2.0), event(2, 4.0), event(3, 6.0)], adjustment=-30.0)
        self.assertTrue(result.all_created, "no event may be silently lost or double counted")
        self.assertEqual(result.created, 3)
        self.assertEqual(sorted(timeline.markers), [0, 1, 2])
        self.assertEqual(result.nudged, 2)

    def test_a_different_marker_already_on_the_frame_is_not_mistaken_for_ours(self):
        timeline = FakeTimeline(existing={6000: {"name": "MY OWN NOTE", "color": "Blue", "note": "", "duration": 1}})
        result = send(timeline, [event(1, 100.0)])
        self.assertTrue(result.all_created)
        self.assertEqual(result.created, 1)
        self.assertEqual(result.nudged, 1)
        self.assertEqual(timeline.markers[6000]["name"], "MY OWN NOTE", "the user's marker is untouched")
        self.assertIn(6001, timeline.markers)

    def test_sending_the_same_events_again_does_not_duplicate_markers(self):
        timeline = FakeTimeline()
        events = [event(1, 100.0), event(2, 200.0)]
        send(timeline, events)
        second = send(timeline, events)
        self.assertEqual(len(timeline.markers), 2, "no duplicate markers")
        self.assertEqual((second.created, second.already_present, second.nudged), (0, 2, 0))
        self.assertTrue(second.all_created)

    def test_one_existing_marker_only_stands_in_for_one_event(self):
        timeline = FakeTimeline()
        send(timeline, [event(1, 100.0)])
        result = send(timeline, [event(1, 100.0), event(2, 100.0)])  # two events, same frame, one already placed
        self.assertEqual(result.already_present, 1)
        self.assertEqual(result.created, 1)
        self.assertEqual(len(timeline.markers), 2)

    def test_markers_that_resolve_refuses_are_reported_as_failed(self):
        timeline = FakeTimeline(fail_add=True)
        result = send(timeline, [event(1, 100.0), event(2, 200.0)])
        self.assertFalse(result.all_created)
        self.assertEqual(result.created, 0)
        self.assertEqual(len(result.failed_events), 2)

    def test_a_frame_that_cannot_be_nudged_clear_is_a_failure_not_a_fake_success(self):
        crowded = {6000 + i: {"name": "x", "color": "Blue", "note": "", "duration": 1} for i in range(MAX_NUDGE_FRAMES + 5)}
        result = send(FakeTimeline(existing=crowded), [event(1, 100.0)])
        self.assertFalse(result.all_created)
        self.assertEqual(len(result.failed_events), 1)

    def test_resolve_going_away_before_verification_is_its_own_state(self):
        timeline = FakeTimeline(fail_get_after=2)  # first read (existing markers) works, the verification read fails
        result = send(timeline, [event(1, 100.0)])
        self.assertIsNotNone(result.verification_error)
        self.assertFalse(result.all_created)
        self.assertEqual(result.failed_events, [], "unverified is not the same as failed")

    def test_unreadable_existing_markers_do_not_stop_the_send(self):
        timeline = FakeTimeline(fail_get_after=1)  # the very first GetMarkers() raises
        result = send(timeline, [event(1, 100.0)])
        self.assertIsNotNone(result.verification_error)  # ...and so does the verification read
        self.assertEqual(timeline.add_calls, [6000], "the marker was still sent")

    def test_empty_event_list(self):
        result = send(FakeTimeline(), [])
        self.assertTrue(result.all_created)
        self.assertEqual(result.total, 0)

    def test_string_frame_keys_from_resolve_are_understood(self):
        class StringKeyTimeline(FakeTimeline):
            def GetMarkers(self):
                return {str(k): v for k, v in self.markers.items()}
        timeline = StringKeyTimeline()
        self.assertTrue(send(timeline, [event(1, 100.0)]).all_created)
        self.assertEqual(send(timeline, [event(1, 100.0)]).already_present, 1)

    def test_source_events_are_never_modified(self):
        events = [event(1, 100.0)]
        send(FakeTimeline(), events, adjustment=-2.0)
        self.assertEqual(events[0].timestamp_seconds, 100.0)


if __name__ == "__main__":
    unittest.main()
