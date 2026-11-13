# BF5 Auto Marker: user guide

The detailed guide. For an overview, see the [README](../README.md).

## Quick start

1. **Open it** -- double-click **`run.bat`** (or a Desktop shortcut to it); it runs
   `.venv\Scripts\pythonw.exe main.py` with no console window.
2. **Calibrate once** (CALIBRATE tab) -- upload a clean skull image per colour, add a short test clip or two, press
   **AUTO-DETECT FROM TEST FOOTAGE**, then **SAVE CALIBRATION**. It is remembered for every future video.
3. **Every recording** (WORKFLOW tab) -- drag the video in → **ANALYSE VIDEO** → with Resolve open on a timeline,
   **SEND N MARKERS TO RESOLVE**.

## The app at a glance

Three tabs in the nav bar; the everyday path is one page:

```
WORKFLOW:   01 VIDEO  →  02 ANALYSIS  →  03 RESOLVE
```

| Tab | What it does |
|---|---|
| **WORKFLOW** | **01 VIDEO**: drag a recording in (or SELECT VIDEO). **02 ANALYSIS**: ANALYSE VIDEO shows progress, a live preview of the latest detection and a MARKERS FOUND counter; when it finishes, REVIEW MARKERS opens PREVIEW and SEND TO RESOLVE sends. **03 RESOLVE**: connection status, "Markers ready: N" and one **SEND N MARKERS TO RESOLVE** button. |
| **CALIBRATE** | Teach the app what the skull looks like and where it appears (done once, remembered permanently), plus the bookmark time adjustment. |
| **SETTINGS** | Detection confidence, frame sampling, multiple-skull detection, marker name/colour, calibration management, Developer Mode. |
| **PREVIEW** *(via REVIEW MARKERS)* | Step through the detected events with box overlays and zoom, add a marker by hand, delete one. |

The stage strip at the top of WORKFLOW only ever reflects real state (video loaded, analysis finished). The **REF CODES**
switch at the top right of the window shows click-to-copy reference numbers (see "Reference codes" below).

## Calibration is permanent, not per-video

Everything lives in `calibration/` beside the app and is reused automatically for every video:

```
calibration/
    calibration.json      detection region + the resolution it was drawn at, each reference's learned
                          scale and appearance, the bookmark time adjustment
    references/           one or more skull reference images (e.g. White and Gold) -- copied in, never moved
    test_footage/         1-3 short clips, each showing one skull appearance inside a drawn green box
```

1. **SKULL REFERENCE** -- upload a clean image of each skull variant (white, gold, ...). The background
   is removed automatically; the skull's **colour is kept** (the thumbnail shows the real colour and
   what the app read from the file, e.g. "Gold / Yellow · colour information available").
2. **TEST FOOTAGE** -- add the short clips. The green box is *training information only*: it tells the
   app where and when the skull really is. Real gameplay never needs one.
3. **SEARCH AREA** -- **AUTO-DETECT FROM TEST FOOTAGE** learns the search region, each reference's real
   on-screen size, and the colour of a coloured skull as it appears in *your* recordings.
   **FIND SKULL POSITION** and **SELECT SEARCH AREA** are manual fallbacks.
4. **CALIBRATION TEST** / **TEST GOLD SKULL** -- run the real detector against your clips and report what it
   finds, including the per-signal evidence in Developer Mode. These (and AUTO-DETECT) run on a
   background thread: the window stays responsive and the buttons are disabled until they finish.
5. **SAVE CALIBRATION** -- writes it to `calibration/`. Changes you make before saving are held in memory;
   a "not saved yet" note appears next to the bookmark adjustment until you save.

If a video's resolution differs from the one you calibrated on, the search region is scaled
proportionally (and you're warned if the aspect ratio differs a lot).

### Bookmark time adjustment

At the top of CALIBRATE: a signed number of seconds applied to **every** marker, **once**, when you send
to Resolve. **Negative = earlier, positive = later, 0 = the exact detected moment** (the default). **TEST**
shows what a value does to the currently previewed frame before you commit to a whole batch. It is saved
with the calibration, so you only find the right value once. (Limit: ±600 s; anything that isn't a real
number is rejected.)

## How detection works

One pass over the video. For each *sampled* frame (every Nth, see SETTINGS):

```
crop to the calibrated region
  → grayscale multi-template matching against every reference        (shape; the original detector)
  → colour-aware matching for coloured references such as gold       (structure + colour + edge + luminance,
                                                                       size, aspect, position)
  → merge candidates, name each skull from its own colour evidence
  → require the neighbouring frames to hold up too                   (rejects single-frame flashes)
  → event state machine: an event starts when the skull is confirmed, ends once it has been gone for a
    short grace period
```

Each event keeps four distinct moments -- never confuse them:

* **start** -- when the skull first appeared. **This is the bookmark** sent to Resolve.
* **end** -- the last frame it was still seen.
* **confirmation** -- when enough evidence existed to confirm (always after the end).
* **best frame** -- the clearest frame, used only for preview thumbnails.

Colour is a supporting signal for coloured skulls, never an exact-RGB test, and the global confidence
threshold is never lowered to make something "work".

## Result files

Every completed run writes to `<video name>_BF5AutoMarker_Output/` beside the video:
`kill_timestamps.csv`, `kill_timestamps.txt`, `detections.json`, `analysis_log.txt`. The original video is
never modified. If those files can't be written (a full drive, a read-only folder) the app says so and
carries on -- the markers are still found and can be sent to Resolve.

## Sending to Resolve

Click **SEND N MARKERS TO RESOLVE** -- it sends immediately (no pop-up). The adjustment currently in
effect is shown above the button. The result is **verified by re-reading the timeline**, never assumed:

* **✓ MARKERS CREATED** appears only when every marker is really there.
* Sending the same events again doesn't duplicate anything ("already on the timeline").
* Resolve allows one marker per frame: if two markers would land on the same frame, or one of *your own*
  markers is already there, the new marker moves forward to the next free frame (and you're told).
* Anything that couldn't be created shows **MARKER VERIFICATION FAILED** with **VIEW ERRORS** and **RETRY**.
* If Resolve isn't reachable you get the specific reason and **RETRY**.

Marker positions are timeline-relative frame offsets (0 = the timeline's first frame), per Resolve's own
scripting documentation.

### One-time Resolve setup (Windows)

1. `DaVinci Resolve → Preferences → System → General → External scripting using` → **Local**
   (the free edition has no scripting API).
2. Non-default install location:
   ```powershell
   $env:RESOLVE_SCRIPT_API = "D:\Path\To\DaVinci Resolve\Support\Developer\Scripting"
   $env:RESOLVE_SCRIPT_LIB = "D:\Path\To\DaVinci Resolve\fusionscript.dll"
   python main.py
   ```
3. Open Resolve with a project and a timeline before sending.

## Settings and saving

Every control applies immediately and is written to `config.json` shortly after you stop changing it
(and when you close the window); **Save Settings** writes straight away. `config.json` and
`calibration.json` are written atomically, so a full disk or a crash can't leave a half-written file; if a
write fails you get a clear message and the setting stays active for the session. A damaged or hand-edited
file never stops the app from starting (bad values fall back to defaults, numbers are clamped to the slider
ranges).

## Reference codes (REF CODES switch, top right)

For telling Claude exactly what to change: flip **REF CODES** at the top right of the window and every page,
section, tab, button and value gets a small faint number -- `[07]`, `[23]`, … -- drawn *over* the UI.

* **Click a code to copy it.** What lands on the clipboard is the number plus where it is, e.g.
  `REF: 23` / `BUTTON: UPLOAD_REFERENCE` / `PAGE: CALIBRATE` / `SECTION: SKULL_REFERENCE` (value codes also copy the
  value as it is right then). Paste it with a note like "this section needs to change". Hover a code to see the
  same text first.
* **It never moves anything.** The codes are placed on top of the layout, not packed into it, so switching them on
  or off leaves every widget exactly where it was (a test measures this on every page).
* **Shaded, not loud.** Each code is a muted shade of whatever it sits on -- soft grey on the dark panels, a darker
  orange on the orange buttons -- and brightens slightly when you hover it.
* **It turns itself off after two hours.** The time left shows beside the switch (`1h 59m` … `<1m`). Switching it
  off and on again restarts the two hours. It is deliberately never saved, so the app always starts with it off.

Numbers are handed out in the order the pages are built, so they can shift when the app changes; the copied text
names the element too, so a pasted code stays unambiguous.

## Developer Mode (SETTINGS → Developer Mode)

Off by default, purely additive, and separate from the reference codes: extra ANALYSIS fields (detection state,
active-event start, rejected candidates and why, the strongest near-miss with its per-signal scores -- a strip of
live-value chips you can click to copy), the reference profile of each calibration image, and on the RESOLVE stage the
granular connection diagnostics, **TEST BOOKMARK** and **RESOLVE DEBUG**.

## Project structure

```
BF5_Auto_Marker/
├── main.py                        entry point (drag-and-drop root window, crash reporting)
├── core/
│   ├── session.py                 AppSession -- the one shared state object (video, calibration, events)
│   ├── calibration_store.py       calibration/ persistence, validation, migrations from older layouts
│   └── crash_log.py               size-capped crash_log.txt
├── video/
│   ├── video_probe.py             metadata, frame extraction
│   └── video_scanner.py           the scan loop: region crop, detection, neighbour check, event grouping
├── detectors/
│   ├── skull_detector.py          grayscale multi-template matcher (the original detector)
│   ├── reference_profile.py       a reference as colour + mask + grayscale/edge/contrast/binary/inverted views
│   ├── colour_matcher.py          colour-aware matching for coloured references (gold), footage learning
│   ├── combined_detector.py       both matchers in one per-frame call; names each skull by its colour
│   └── green_box.py               finds the green training box in calibration test footage
├── resolve/resolve_controller.py  connection diagnostics, marker placement + verification
├── data/
│   ├── config.py                  AppConfig -- validated settings, atomic save
│   ├── safe_io.py                 atomic JSON writes, tolerant reads, number sanitising
│   ├── detection.py               Event / SkullSighting, timestamp parsing + formatting
│   ├── colour_label.py            names a skull's colour ("White", "Gold") for labels
│   ├── export.py                  CSV / TXT / JSON writers, the analysis log
│   └── highlights.py              keep-range maths for a future auto-highlights feature
├── ui/
│   ├── main_window.py             shell: nav, page switching, worker→UI queue, analysis/send controllers
│   ├── window_perf.py             cheap resizing and monitor changes (see below)
│   ├── pages/                     workflow_page (+ video/analyse/resolve stages), calibrate_page,
│   │                              preview_page, settings_page
│   └── …                          scrubber, timelines, preview panels, dialogs; devtools.py = the reference-code chips
├── tests/                         the automated test suite (see below)
├── assets/                        app icon (skull_template.png is a legacy file nothing loads)
├── calibration/                   your calibration (created by the app)
└── requirements.txt
```

## Installation & running

`setup.bat` and `run.bat` do this for you. By hand:

```bash
cd BF5_Auto_Marker
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Tests

A standard-library `unittest` suite -- no extra dependencies:

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

(≈ 5-6 minutes; the UI tests drive the real window, which is created fully transparent so nothing flashes on
screen. Run one file with e.g. `python -m unittest tests.test_resolve_markers`.)

It covers settings/persistence (incl. full-disk and damaged-file cases), calibration storage, the Resolve marker
logic against a fake timeline, the event model and result files, reference profiles, the colour matcher, the
combined detector, the whole scan loop on generated clips (one event per appearance, bookmark = start, flash
rejected, dropouts tolerated, cancellation), and the real `MainWindow` end to end -- analysis through the real
thread/queue, PREVIEW, sending, every failure path, the CALIBRATE page on synthetic test footage, the window
resize/monitor-change behaviour, and the REF CODES switch (appears/disappears, never moves a widget, copies the right
text, two-hour auto-off).

Safety: tests **never** touch your real `config.json`, `calibration/` or `crash_log.txt` (paths are redirected to a
temporary folder), never write result folders beside real videos, and never talk to a real Resolve (a fake
timeline stands in).

## Performance notes

* Only the tab you're looking at is laid out; hidden tabs are unmapped, so resizing costs one tab's worth of work.
* While you drag a window edge the current tab holds its size and re-flows once the edge stops moving.
* On a monitor change (different scaling) only the visible tab is re-scaled immediately; the others catch up when
  you open them.
* The live preview decodes a frame only when the latest event actually changes.
* Calibration actions that decode whole clips run off the UI thread.

## Not implemented (deliberate placeholders)

* `apply_color_grade()` (`resolve/resolve_controller.py`) -- needs your specific PowerGrade or LUT.
* `apply_auto_highlights()` -- the keep-range *calculation* (`data/highlights.py`) exists; automatically cutting a
  timeline does not, because a bad automatic cut costs far more than an ignorable wrong marker.

Neither is reachable from the UI.

## Troubleshooting

**A skull isn't detected / too many false detections** -- check CALIBRATE: AUTO-DETECT FROM TEST FOOTAGE, then
CALIBRATION TEST and TEST GOLD SKULL. In Developer Mode the ANALYSIS stage shows the strongest near-miss and the
per-signal scores. Adjust Detection confidence in SETTINGS only as a last resort -- raising it too far drops the real
skulls, lowering it brings back false markers.

**Markers land slightly before/after the skull** -- set the bookmark time adjustment on CALIBRATE (use TEST), then
SAVE CALIBRATION.

**Resolve won't connect** -- RECONNECT runs a real step-by-step attempt (Resolve running → API → instance → project →
timeline → bookmark access) and says which stage failed. Developer Mode adds TEST BOOKMARK.

**"Result files not saved"** -- the drive is full or the folder is read-only. The markers are unaffected; free some space
and re-run if you want the files. (`crash_log.txt` is capped at 512 KB; the previous one is kept as
`crash_log.previous.txt`.)

**Analysis is slow** -- make sure a detection region is calibrated; a larger frame sampling number (every Nth frame) is
faster.

**Video won't open** -- some `.mkv`/`.avi` codecs aren't supported by this OpenCV build; re-export to `.mp4` (H.264).

## Assumptions the app makes

* **Windows + DaVinci Resolve Studio** with *External scripting using → Local* switched on (the free edition has no
  scripting API). Everything else runs without Resolve; only sending needs it.
* **Marker position = seconds into the analysed video.** Markers are placed at that offset from the timeline's first
  frame, so the clip is assumed to sit at the start of the timeline. If it sits later or is trimmed, use the bookmark
  time adjustment (or move the clip) -- the app does not read the clip's position on the timeline.
* **The kill-confirmed skull appears in one fixed region of the screen**, at about the same size in every clip of the
  same resolution (other resolutions are scaled proportionally).
* **White and gold skulls are the two variants.** Another colour works the same way if you upload a reference for it,
  but only white and gold have been checked on real footage.
* **Sampling every Nth frame is enough:** a skull must stay on screen long enough to be seen on neighbouring sampled
  frames (the neighbour check is what rejects one-frame flashes).

## Known limits

* The automated tests use generated clips and a fake Resolve timeline. Real gameplay footage was checked by hand
  during development and Resolve through actual use -- not by the suite -- so after a Resolve update, or a game
  HUD/skin change, re-run **CALIBRATION TEST** and send a handful of markers before trusting a big batch.
* Resolve allows one marker per frame, so when two markers would share a frame the later one moves to the next free
  frame (you're told).
* Reference codes are numbered in build order, so the numbers can change when the app changes (the copied text always
  names the element as well).
* There is no undo for "RESET CALIBRATION"; it asks first and removes the saved region, references and test footage.
* The project folder is not under version control and `calibration/` is about 70 MB (mostly test footage). Copy the
  whole `BF5_Auto_Marker` folder before big changes.

## Maintenance and next steps

* **Back up** `calibration/` (and `config.json`) -- it's the only thing that takes effort to recreate.
* **After changing code,** run the test suite (see Tests); a failing test names exactly which behaviour moved.
* **If detection drifts** (new HUD, new skull skin, different recording resolution): add a fresh reference and test clip,
  AUTO-DETECT again, SAVE. Don't lower the confidence threshold to compensate.
* **Optional ideas:** a marker-offset check that reads the clip's position on the timeline; per-kill-type marker
  colours (white vs gold); batch-analysing a folder of recordings; a stable, hand-written list of reference codes
  instead of build-order numbers; automatic highlight cutting (`data/highlights.py` has the maths).
