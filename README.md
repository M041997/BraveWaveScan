# Brave Wave Scan

A live EEG experience that turns Muse 2 signals into a cinematic, real-time 3D
cortical visualization. It is designed to feel at home in an exhibition booth,
on a large display, or in a recorded neurotechnology demo.

## Flagship experience

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python brain3d.py
```

The app looks for an LSL EEG stream for two seconds. If no stream is available,
it automatically starts in a clearly labeled synthetic demo mode. You can skip
discovery and launch immediately with:

```bash
.venv/bin/python brain3d.py --demo
```

Place an anatomical `brain.stl` next to `brain3d.py` to use it privately. Without
one, the viewer creates a procedural cortical form, so no personal scan data is
required.

Controls:

- Drag to rotate and scroll to zoom
- `Space` pauses the visualization
- `D` toggles between demo data and an available live LSL stream
- `R` resets the camera

## Live Muse 2 setup

Start the EEG stream first, then launch the viewer:

```bash
.venv/bin/muselsl stream
.venv/bin/python brain3d.py
```

The dashboard shows relative delta, theta, alpha, beta, and gamma power; four
electrode contact estimates; focus and calm presentation indices; a rolling
trace; surface activity; and animated frontal-to-posterior signal paths.

> The visualizations and derived indices are experiential, not medical or
> diagnostic instruments.

The repository also includes focused tools for raw traces (`viewer.py`), a 2D
top-down view (`brain_view.py`), a band report (`band_report.py`), signal contact
checks (`check_signal.py`), and Muse PPG heart rate (`heart_rate.py`).
