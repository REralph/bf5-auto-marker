"""
data/export.py
----------------
Writes the three output files into a folder created beside the source
video, plus computes the analysis summary shown after a scan finishes
(spec item 21). Never touches the original video file itself.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
from dataclasses import dataclass
from typing import List, Optional

from data.detection import Event


def get_output_dir(video_path: str) -> str:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(video_dir, f"{video_stem}_BF5AutoMarker_Output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def write_csv(events: List[Event], output_dir: str) -> str:
    # `timestamp_seconds`/`frame_number` keep their existing column names for anything already
    # parsing this file by header -- they now hold the event's bookmark (start) time rather than
    # its best-frame time, per Event's own docstring, but the column identity is unchanged.
    csv_path = os.path.join(output_dir, "kill_timestamps.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_number", "timestamp_seconds", "timecode", "frame_number", "skull_count", "colours",
                "confidence", "type", "event_end_seconds", "confirmation_seconds",
            ]
        )
        for e in events:
            writer.writerow(
                [
                    e.event_number,
                    f"{e.timestamp_seconds:.3f}",
                    e.timecode,
                    e.frame_number,
                    e.skull_count,
                    ";".join(e.colors),
                    f"{e.confidence:.2f}",
                    e.detection_type,
                    f"{e.event_end_seconds:.3f}" if e.event_end_seconds is not None else "",
                    f"{e.confirmation_seconds:.3f}" if e.confirmation_seconds is not None else "",
                ]
            )
    return csv_path


def write_txt(events: List[Event], output_dir: str) -> str:
    txt_path = os.path.join(output_dir, "kill_timestamps.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BF5 Auto Marker - Detected Events\n")
        f.write("=" * 44 + "\n")
        f.write(f"Total events: {len(events)}    Total kills: {sum(e.skull_count for e in events)}\n\n")
        for e in events:
            f.write(
                f"#{e.event_number:<4} {e.timestamp_display:<14} "
                f"frame {e.frame_number:<8} {e.marker_label:<18} colours: {e.colors_display:<20} "
                f"confidence {e.confidence_percent:>3}%"
                f"{'  (ended ' + e.event_end_display + ')' if e.event_end_seconds is not None else ''}\n"
            )
    return txt_path


def write_json(events: List[Event], output_dir: str) -> str:
    """detections.json -- the machine-readable counterpart to kill_timestamps.csv/.txt, with
    per-event detection detail (skull count, confidence, colours, frame) for anything that
    wants to consume this programmatically rather than parse the CSV."""
    json_path = os.path.join(output_dir, "detections.json")
    data = [
        {
            "event_number": e.event_number,
            "timestamp_seconds": round(e.timestamp_seconds, 3),
            "timecode": e.timecode,
            "frame_number": e.frame_number,
            "skull_count": e.skull_count,
            "colours": e.colors,
            "confidence_percent": e.confidence_percent,
            "detection_type": e.detection_type,
            "event_end_seconds": round(e.event_end_seconds, 3) if e.event_end_seconds is not None else None,
            "event_end_frame": e.event_end_frame,
            "confirmation_seconds": round(e.confirmation_seconds, 3) if e.confirmation_seconds is not None else None,
            "confirmation_frame": e.confirmation_frame,
        }
        for e in events
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return json_path


class AnalysisLogger:
    """Append-only per-run log the user can open directly in Notepad.

    Creating the log can raise OSError (the caller reports that). Writing to it afterwards never
    raises: a full disk must not be able to abort an analysis that is otherwise working -- the first
    write failure is kept in `write_error` and later lines are dropped."""

    def __init__(self, output_dir: str):
        self.log_path = os.path.join(output_dir, "analysis_log.txt")
        self.write_error: Optional[OSError] = None
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"BF5 Auto Marker - Analysis Log\nStarted: {datetime.datetime.now().isoformat()}\n")
            f.write("-" * 50 + "\n")

    def log(self, message: str) -> None:
        if self.write_error is not None:
            return
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except OSError as exc:
            self.write_error = exc


@dataclass
class AnalysisSummary:
    total_events: int
    total_kills: int
    single_kill_events: int
    multi_kill_events: int
    highest_event_skulls: int
    colors_detected: List[str]
    average_confidence_percent: int


def summarize_events(events: List[Event]) -> AnalysisSummary:
    """Everything shown on the ANALYSIS COMPLETE summary screen (spec item 21)."""
    if not events:
        return AnalysisSummary(0, 0, 0, 0, 0, [], 0)

    total_kills = sum(e.skull_count for e in events)
    single_kill = sum(1 for e in events if e.skull_count == 1)
    multi_kill = sum(1 for e in events if e.skull_count > 1)
    highest = max(e.skull_count for e in events)

    colors_seen: List[str] = []
    for e in events:
        for c in e.colors:
            base_name = c.replace(" (new)", "")
            if base_name not in colors_seen:
                colors_seen.append(base_name)

    avg_confidence = round(sum(e.confidence for e in events) / len(events) * 100)

    return AnalysisSummary(
        total_events=len(events),
        total_kills=total_kills,
        single_kill_events=single_kill,
        multi_kill_events=multi_kill,
        highest_event_skulls=highest,
        colors_detected=colors_seen,
        average_confidence_percent=avg_confidence,
    )
