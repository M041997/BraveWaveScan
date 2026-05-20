#!/usr/bin/env python3
"""Live Muse 2 viewer.

Left column: filtered EEG trace per channel (channel name labels what you're seeing,
            colors are neutral — they carry no info).
Right column: power per brainwave BAND. Each band gets a fixed semantic color,
            shared across all 4 channels. Same color = same band, everywhere.
"""
import argparse
import sys
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, welch

WINDOW_SEC = 5.0
PSD_WINDOW_SEC = 2.0
BLINK_WINDOW_SEC = 0.35
MAX_BLINK_HOLD_SEC = 0.35
EEG_CHANNELS = ["TP9 (left ear)", "AF7 (left forehead)",
                "AF8 (right forehead)", "TP10 (right ear)"]
BANDS = [
    ("delta", 1, 4,    (0.20, 0.10, 0.55)),  # indigo  — deep sleep
    ("theta", 4, 8,    (0.10, 0.45, 0.75)),  # blue    — drowsy / meditation
    ("alpha", 8, 13,   (0.10, 0.75, 0.55)),  # green   — relaxed wakeful
    ("beta",  13, 30,  (0.95, 0.70, 0.10)),  # amber   — focused
    ("gamma", 30, 45,  (0.90, 0.20, 0.20)),  # red     — peak attention
]
BANDPASS = (1.0, 45.0)
TRACE_COLOR = "#4ec9b0"  # neutral cyan for all traces

# EMG-quality heuristic: HF/LF = power(15-45Hz) / power(1-15Hz)
# < 0.7 = clean cortical signal; > 2.5 = facial-muscle contamination
HF_LF_GOOD = 0.7
HF_LF_BAD = 2.5

def parse_args():
    parser = argparse.ArgumentParser(description="Live Muse 2 EEG viewer.")
    parser.add_argument(
        "--ignore-blinks",
        action="store_true",
        help="Hold waveforms and readouts steady during likely blink artifacts.",
    )
    return parser.parse_args()


def is_blink(filt, srate):
    blink_len = min(filt.shape[1], int(BLINK_WINDOW_SEC * srate))
    if blink_len <= 0:
        return False

    recent = filt[:, -blink_len:]
    front = recent[1:3]
    temporal = recent[[0, 3]]

    # Blinks are slow, large swings that should hit AF7/AF8 harder than TP9/TP10.
    # Avoid treating rhythmic pulse/contact drift as blink just because it moves
    # all channels together.
    front_ptp = np.ptp(front, axis=1)
    temporal_ptp = np.ptp(temporal, axis=1)
    front_common_ptp = float(np.ptp(front.mean(axis=0)))
    temporal_common_ptp = max(float(np.ptp(temporal.mean(axis=0))), 1.0)
    front_slope = float(np.max(np.abs(np.diff(front.mean(axis=0))))) * srate

    frontal_blink = (
        front_common_ptp > 220.0
        and front_common_ptp > temporal_common_ptp * 1.35
        and bool(np.all(front_ptp > 140.0))
        and bool(np.any(front_ptp > temporal_ptp * 1.25))
        and front_slope > 1200.0
    )
    return frontal_blink


def main():
    args = parse_args()
    print("Resolving Muse EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=10)
    if not streams:
        print("No EEG stream found. Run muselsl stream first.")
        sys.exit(1)
    inlet = StreamInlet(streams[0], max_chunklen=12)
    info = inlet.info()
    srate = int(info.nominal_srate())
    n_eeg = 4
    buf_len = int(WINDOW_SEC * srate)
    psd_len = int(PSD_WINDOW_SEC * srate)
    buf = np.zeros((n_eeg, buf_len), dtype=np.float32)
    clean_buf = np.zeros((n_eeg, buf_len), dtype=np.float32)
    sos = butter(4, BANDPASS, btype="bandpass", fs=srate, output="sos")

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Muse 2 EEG — live (filtered + bands)")
    win.resize(1500, 800)
    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    # top bar: bands legend + per-channel signal quality dots
    top_bar = QtWidgets.QWidget()
    top_bar.setStyleSheet("background:#0a0a0a;")
    top_layout = QtWidgets.QHBoxLayout(top_bar)
    top_layout.setContentsMargins(8, 4, 8, 4)

    legend = QtWidgets.QLabel()
    legend_html = "<span style='color:#888'>bands: </span>"
    for name, lo, hi, (r, g, b) in BANDS:
        color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        legend_html += (f"<span style='color:{color}; font-weight:600'>"
                        f"{name}</span><span style='color:#666'> ({lo}-{hi}Hz)  </span>")
    legend.setText(legend_html)
    top_layout.addWidget(legend)
    top_layout.addStretch(1)

    # quality readouts per channel
    artifact_label = QtWidgets.QLabel()
    artifact_label.setText(
        "<span style='color:#666'>blink filter: "
        f"{'on' if args.ignore_blinks else 'off'}</span>")
    top_layout.addWidget(artifact_label)

    quality_label = QtWidgets.QLabel()
    quality_label.setText("<span style='color:#888'>signal quality: collecting...</span>")
    top_layout.addWidget(quality_label)
    root.addWidget(top_bar)

    glw = pg.GraphicsLayoutWidget()
    glw.setBackground("#0a0a0a")
    root.addWidget(glw)

    band_x = np.arange(len(BANDS))
    x = np.arange(buf_len) / srate
    trace_curves, band_bars = [], []

    for i, name in enumerate(EEG_CHANNELS):
        # left: filtered trace
        p = glw.addPlot(row=i, col=0)
        p.setLabel("left", name, units="uV")
        p.setYRange(-100, 100)
        p.showGrid(x=False, y=True, alpha=0.2)
        if i < n_eeg - 1:
            p.hideAxis("bottom")
        else:
            p.setLabel("bottom", "time (s, rolling)")
        trace_curves.append(p.plot(x, buf[i], pen=pg.mkPen(TRACE_COLOR, width=1)))

        # right: per-band bars; each band has its own color
        bp = glw.addPlot(row=i, col=1)
        bp.setLabel("left", "power (dB)")
        # Muse units are not calibrated volts, so live band power can run above
        # 40 dB when contact is noisy. Keep the full bar visible for diagnosis.
        bp.setYRange(-10, 60)
        bp.showGrid(x=False, y=True, alpha=0.2)
        # one BarGraphItem per band so we can color each independently
        bars = []
        for j, (_, _, _, (r, g, b)) in enumerate(BANDS):
            bar = pg.BarGraphItem(x=[j], height=[0], width=0.7,
                                  brush=pg.mkBrush(int(r*255), int(g*255), int(b*255)))
            bp.addItem(bar)
            bars.append(bar)
        band_bars.append(bars)
        ax = bp.getAxis("bottom")
        ax.setTicks([[(j, BANDS[j][0]) for j in range(len(BANDS))]])
        if i < n_eeg - 1:
            bp.hideAxis("bottom")

    glw.ci.layout.setColumnStretchFactor(0, 3)
    glw.ci.layout.setColumnStretchFactor(1, 1)
    blink_state = {"accepted": 0, "held": 0, "last_filt": None}

    def update():
        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=256)
        if not chunk:
            return
        arr = np.asarray(chunk, dtype=np.float32).T[:n_eeg]
        n = arr.shape[1]
        buf[:, :-n] = buf[:, n:]
        buf[:, -n:] = arr
        raw_filt = sosfiltfilt(sos, buf, axis=1).astype(np.float32)

        can_ignore = (
            args.ignore_blinks
            and blink_state["accepted"] >= psd_len
            and blink_state["held"] < int(MAX_BLINK_HOLD_SEC * srate)
        )
        if can_ignore and is_blink(raw_filt, srate):
            blink_state["held"] += n
            artifact_label.setText(
                "<span style='color:#e0c040; font-weight:600'>blink ignored</span>")
            if blink_state["last_filt"] is not None:
                for i, c in enumerate(trace_curves):
                    c.setData(x, blink_state["last_filt"][i])
            return
        else:
            clean_buf[:, :-n] = clean_buf[:, n:]
            clean_buf[:, -n:] = arr
            blink_state["accepted"] += n
            blink_state["held"] = 0
            if args.ignore_blinks:
                artifact_label.setText(
                    "<span style='color:#3acb6b'>blink filter: on</span>")

        display_buf = clean_buf if args.ignore_blinks else buf
        filt = sosfiltfilt(sos, display_buf, axis=1).astype(np.float32)
        blink_state["last_filt"] = filt
        for i, c in enumerate(trace_curves):
            c.setData(x, filt[i])

        seg = filt[:, -psd_len:]
        f, psd = welch(seg, fs=srate, nperseg=min(256, psd_len), axis=1)
        for i in range(n_eeg):
            for j, (_, lo, hi, _) in enumerate(BANDS):
                m = (f >= lo) & (f < hi)
                power = np.trapezoid(psd[i, m], f[m]) if m.any() else 1e-12
                h = 10 * np.log10(max(power, 1e-12))
                band_bars[i][j].setOpts(height=[h])

        # signal-quality dots — HF/LF ratio per channel
        lf_m = (f >= 1) & (f < 15)
        hf_m = (f >= 15) & (f < 45)
        ratios = []
        for i in range(n_eeg):
            lf = float(np.trapezoid(psd[i, lf_m], f[lf_m]))
            hf = float(np.trapezoid(psd[i, hf_m], f[hf_m]))
            ratios.append(hf / max(lf, 1e-12))
        ch_short = ["TP9", "AF7", "AF8", "TP10"]
        parts = ["<span style='color:#888'>signal: </span>"]
        for i, (name, r) in enumerate(zip(ch_short, ratios)):
            if r < HF_LF_GOOD:
                col = "#3acb6b"; tag = ""
            elif r > HF_LF_BAD:
                col = "#e64a4a"; tag = " EMG!"
            else:
                col = "#e0c040"; tag = ""
            parts.append(
                f"<span style='color:{col}; font-weight:600'>● {name}</span>"
                f"<span style='color:#666'> {r:.2f}{tag}  </span>")
        quality_label.setText("".join(parts))

    win.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
