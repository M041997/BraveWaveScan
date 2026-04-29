#!/usr/bin/env python3
"""Live Muse 2 viewer.

Left column: filtered EEG trace per channel (channel name labels what you're seeing,
            colors are neutral — they carry no info).
Right column: power per brainwave BAND. Each band gets a fixed semantic color,
            shared across all 4 channels. Same color = same band, everywhere.
"""
import sys
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, welch

WINDOW_SEC = 5.0
PSD_WINDOW_SEC = 2.0
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

def main():
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

    # legend bar — fixed band colors, shown once
    legend = QtWidgets.QLabel()
    legend.setStyleSheet("background:#0a0a0a; padding:8px;")
    legend_html = "<span style='color:#888'>brainwave bands: </span>"
    for name, lo, hi, (r, g, b) in BANDS:
        color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        legend_html += (f"<span style='color:{color}; font-weight:600'>"
                        f"{name}</span><span style='color:#666'> ({lo}-{hi}Hz)  </span>")
    legend.setText(legend_html)
    legend.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    root.addWidget(legend)

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
        bp.setYRange(0, 40)
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

    def update():
        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=256)
        if not chunk:
            return
        arr = np.asarray(chunk, dtype=np.float32).T[:n_eeg]
        n = arr.shape[1]
        buf[:, :-n] = buf[:, n:]
        buf[:, -n:] = arr
        filt = sosfiltfilt(sos, buf, axis=1).astype(np.float32)
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

    win.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
