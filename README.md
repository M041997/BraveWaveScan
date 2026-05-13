# eeg-muse

Live EEG + heart-rate tools for a Muse 2 headset on Linux, using
[muselsl](https://github.com/alexandrebarachant/muse-lsl) to bridge Bluetooth →
LSL, plus a small collection of PyQt6/pyqtgraph apps for visualization,
recording, and analysis.

## Quick start

Everything assumes a Python venv at `./.venv`. To set up from scratch:

```bash
python3 -m venv .venv
.venv/bin/pip install muselsl pylsl numpy scipy pandas matplotlib pyqt6 pyqtgraph bleak
```

**Always start the LSL stream first** (on this machine, the default backend
fails — use `bleak`):

```bash
.venv/bin/muselsl stream --backend bleak
```

Wait until you see "Streaming...", then in another terminal launch any of the
tools below.

## What's in here

| File | What it does |
| --- | --- |
| [viewer.py](viewer.py) | Live filtered EEG traces + per-band power bars + signal-quality dots |
| [recorder.py](recorder.py) | GUI to record sessions to CSV with inline event labels |
| [focus_balloon.py](focus_balloon.py) | Biofeedback game — concentrate to inflate a balloon, pop it |
| [session_inspector.py](session_inspector.py) | Interactive scrubber for a saved session (traces, spectrogram, quality timeline) |
| [segment_bands.py](segment_bands.py) | Per-segment band-power summary + PNG chart for a saved session |
| [band_report.py](band_report.py) | Quick 5-second relative-band-power snapshot to terminal |
| [check_signal.py](check_signal.py) | 5-second per-channel signal-quality check (variance / rail saturation) |
| [heart_rate.py](heart_rate.py) | Live PPG-based heart-rate viewer (needs `--ppg` on muselsl) |
| [brain_view.py](brain_view.py), [brain3d.py](brain3d.py) | 3D brain visualization (uses `brain.stl` / `brain2.stl`) |

## Typical workflow

### 1. Check signal quality before recording

```bash
.venv/bin/python viewer.py
```

Look at the top bar — each channel (TP9 / AF7 / AF8 / TP10) has a colored dot
with a live `HF/LF` number:

- **green (< 0.7)** — clean cortical signal
- **yellow (0.7–2.5)** — marginal
- **red (> 2.5)** — facial-muscle artifact (EMG) is dominating; **reseat the
  pad on that side**

Forehead pads (AF7/AF8) are the most artifact-prone. Press the headset firmly
on your forehead, dampen the pads if dry, and give it 30 seconds to settle
before recording. The classic sign of bad contact is huge fast oscillations
in the trace and a `HF/LF` over 2.

### 2. Record a session

```bash
.venv/bin/python recorder.py
```

- Hit **START RECORDING**. EEG (and PPG if available) write to
  `~/eeg-muse/sessions/<timestamp>/eeg.csv`.
- While recording, type a label in the combo box (or pick a quick label) and
  press Enter to log a timestamped event to `events.csv`. Use this every time
  you change mental state — "eyes_closed", "reading", "balloon", etc.
- Optional: fill in the free-text notes field at any point — it gets saved to
  `meta.json` on stop.
- Hit **STOP RECORDING**. The session dir now contains:
  ```
  eeg.csv      (timestamp, tp9, af7, af8, tp10, aux)
  ppg.csv      (timestamp, ambient, ir, red)   [only if --ppg was on]
  events.csv   (timestamp, elapsed_s, label)
  meta.json    (start, end, duration_s, sample counts, event count, notes)
  ```

If you want PPG (heart rate, HRV), start muselsl with `--ppg`:
```bash
.venv/bin/muselsl stream --backend bleak --ppg
```

### 3. Inspect a recorded session

```bash
.venv/bin/python session_inspector.py            # latest session
.venv/bin/python session_inspector.py <dir>      # specific session
```

Interactive scrubber. Click anywhere on the top overview to jump to that
moment; status bar shows current segment and per-channel HF/LF verdict.
Spectrogram channel is selectable from the dropdown.

For a static PNG report (per-segment band power):

```bash
.venv/bin/python segment_bands.py
```

Saves `band_summary.png` into the session dir.

### 4. Play the focus balloon

```bash
.venv/bin/python focus_balloon.py
```

20-second warmup (the balloon is hidden — your baseline is being learned
from a rolling 60-second window of your own signal). After warmup, the
balloon inflates as your frontal beta / (alpha + theta) rises above your
recent baseline; hold it pegged at full size for ~1.2 seconds to pop it.

The two dots top-right show AF7/AF8 quality in real time. **Channels with
bad quality are automatically weighted out of the focus score** — if AF7 is
contaminated you'll see it go red and the balloon will only react to AF8.
If both forehead pads are bad, the status bar says "no clean forehead
signal — reseat AF7/AF8 pads" and the balloon won't move.

Live z-scores are printed to the terminal so you can tune thresholds
(`Z_LO`, `Z_HI`, `POP_HOLD_SEC` at the top of `focus_balloon.py`) to your
own dynamic range.

## Signal-quality heuristic

A few tools (`viewer.py`, `focus_balloon.py`, `session_inspector.py`) share
the same EMG-contamination estimator:

```
HF/LF = power(15-45 Hz) / power(1-15 Hz)
```

Healthy frontal cortical EEG sits well below 1.0. Anything above ~2.5 is
dominated by facial-muscle electromyographic (EMG) activity bleeding into
the higher frequencies — usually a bad electrode seal. The 0.7 / 2.5 GOOD
/ BAD thresholds are calibrated against typical Muse 2 behavior; they map
linearly to a `[0, 1]` quality weight in between.

This caught a real problem in the first labeled session on this machine:
AF7 read HF/LF ≈ 3.15 throughout, meaning its "beta" was almost entirely
muscle artifact rather than cortical activity. Without channel-quality
weighting, the focus balloon's score was being driven by jaw tension.

## Channel reference (Muse 2)

| Index | Name | Location |
| --- | --- | --- |
| 0 | TP9  | over the left mastoid (behind/below ear) |
| 1 | AF7  | left forehead |
| 2 | AF8  | right forehead |
| 3 | TP10 | over the right mastoid |
| 4 | AUX  | not used by the Muse 2; ignore |

Sampling rate is **256 Hz** nominal. The EEG values are already in
microvolts (uV). Filtering used everywhere is a 4th-order Butterworth
bandpass at 1.0–45.0 Hz (`scipy.signal.butter` → `sosfiltfilt`).

## Session data layout

```
~/eeg-muse/sessions/
  2026-04-30_06-41-32/
    eeg.csv         # (timestamp, tp9, af7, af8, tp10, aux) at 256 Hz
    ppg.csv         # (timestamp, ambient, ir, red) at 64 Hz   [if --ppg]
    events.csv      # (timestamp, elapsed_s, label)             [if any]
    meta.json       # start/end/duration/counts/notes/segments
    band_summary.png  # if segment_bands.py was run
```

`timestamp` in every CSV is the LSL timestamp (seconds, monotonic), so all
streams in a single session share a common clock and can be joined directly.

If you have a session without per-event labels but you remember roughly
when each segment was, you can hand-edit `meta.json` to add a `segments`
array of the form:

```json
"segments": [
  {"label": "eyes_closed", "start_s": 0, "end_s": 120,
   "description": "eyes closed, resting"},
  ...
]
```

…and `session_inspector.py` will color the overview accordingly.

## Troubleshooting

**"No EEG stream found."**
You haven't started `muselsl stream --backend bleak`, or it crashed. Power
the Muse on, then run muselsl in another terminal and wait for "Streaming...".

**`ValueError: Unexpected error when scanning: SetDiscoveryFilter success`**
The default `bluetoothctl` backend in muselsl is broken on this machine.
Use `--backend bleak`.

**Balloon barely reacts / pops constantly.**
Two knobs at the top of `focus_balloon.py`:
- `Z_HI` (default 2.2) — the z-score that fully inflates the balloon. Lower
  it if popping is too hard; raise it if it's too easy.
- `POP_HOLD_SEC` (default 1.2) — how long you must hold at full size to pop.

The terminal prints live z-scores every second — use those numbers to pick
sensible values for your own focus dynamics rather than guessing.

**One forehead channel reads HF/LF ≈ 3+ no matter what.**
That pad isn't making good contact. Try: (a) cleaning the silver pad with
a damp finger, (b) wetting your forehead lightly where the pad sits, (c)
adjusting the headband tension, (d) sliding the headset slightly up/down.
You can verify in [viewer.py](viewer.py) — the affected channel's trace
will look like dense fast scribble rather than a smooth slow wave.

**`recorder.py` crashes on STOP.**
Old versions overrode `threading.Thread._stop` with an Event, which collided
with Python's internal cleanup. Already patched in this repo (the event is
named `_stop_event`). If you see this on an old checkout, pull and retry.
The raw EEG/PPG CSVs are flushed each chunk, so a stop-time crash does not
lose data — only the `meta.json` write is skipped.
