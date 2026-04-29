#!/usr/bin/env python3
"""Live heart rate from Muse 2 PPG. Bandpass 0.7-3 Hz + peak detection."""
import sys
import numpy as np
from collections import deque
from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, find_peaks

WINDOW_SEC = 10.0
PPG_CH = 1  # IR channel (cleanest pulse on Muse 2: 0=ambient, 1=IR, 2=red)


class HeartRate(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse 2 — Heart Rate")
        self.resize(900, 500)

        print("Resolving PPG stream...")
        streams = resolve_byprop("type", "PPG", timeout=10)
        if not streams:
            print("No PPG stream. Run muselsl with --ppg.")
            sys.exit(1)
        self.inlet = StreamInlet(streams[0])
        self.srate = int(self.inlet.info().nominal_srate())  # 64 Hz
        self.buf_len = int(WINDOW_SEC * self.srate)
        self.buf = np.zeros(self.buf_len, dtype=np.float32)
        # bandpass 0.7-3 Hz = 42-180 BPM
        self.sos = butter(3, (0.7, 3.0), btype="bandpass", fs=self.srate, output="sos")
        self.bpm_history = deque(maxlen=15)

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background:#0a0a0a;")
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # big BPM display + heart pulse indicator
        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self.bpm_label = QtWidgets.QLabel("--")
        self.bpm_label.setStyleSheet(
            "color:#ff4060; font-size:72pt; font-weight:800;")
        self.bpm_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.bpm_label, 2)

        right = QtWidgets.QVBoxLayout()
        top.addLayout(right, 1)
        self.bpm_unit = QtWidgets.QLabel("BPM")
        self.bpm_unit.setStyleSheet("color:#888; font-size:18pt;")
        self.bpm_unit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.bpm_unit)
        self.heart_label = QtWidgets.QLabel("♥")
        self.heart_label.setStyleSheet("color:#ff4060; font-size:54pt;")
        self.heart_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.heart_label)
        self.status = QtWidgets.QLabel("waiting for pulse...")
        self.status.setStyleSheet("color:#666; font-size:10pt;")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.status)

        # PPG trace
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0a0a0a")
        self.plot.setLabel("left", "PPG (IR)")
        self.plot.setLabel("bottom", "time (s, rolling)")
        self.plot.showGrid(x=False, y=True, alpha=0.2)
        self.x = np.arange(self.buf_len) / self.srate
        self.curve = self.plot.plot(self.x, self.buf, pen=pg.mkPen("#4ec9b0", width=1.5))
        self.peak_scatter = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush("#ff4060"), size=10, symbol="t1")
        self.plot.addItem(self.peak_scatter)
        layout.addWidget(self.plot, stretch=2)

        # pulse animation timer (briefly enlarge heart on each detected beat)
        self._pulse_anim = QtCore.QTimer(self)
        self._pulse_anim.setSingleShot(True)
        self._pulse_anim.timeout.connect(self._restore_heart)

    def _pulse(self):
        self.heart_label.setStyleSheet("color:#ff90a0; font-size:72pt;")
        self._pulse_anim.start(120)

    def _restore_heart(self):
        self.heart_label.setStyleSheet("color:#ff4060; font-size:54pt;")

    def update_data(self):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=64)
        if not chunk:
            return
        arr = np.asarray(chunk, dtype=np.float32)
        ppg = arr[:, PPG_CH]
        n = len(ppg)
        if n == 0:
            return
        self.buf[:-n] = self.buf[n:]
        self.buf[-n:] = ppg

        if np.std(self.buf) < 1.0:
            self.status.setText("no signal — finger sensor not lit?")
            return

        filt = sosfiltfilt(self.sos, self.buf).astype(np.float32)
        self.curve.setData(self.x, filt)

        # peak detection on filtered signal
        # min distance ~0.4s (=upper bound of 150 BPM)
        min_dist = int(0.4 * self.srate)
        # require some prominence relative to local std
        prom = max(filt.std() * 0.4, 1.0)
        peaks, _ = find_peaks(filt, distance=min_dist, prominence=prom)

        if len(peaks) >= 2:
            intervals_sec = np.diff(peaks) / self.srate
            # use median for robustness against missed/extra peaks
            ibi = np.median(intervals_sec)
            bpm = 60.0 / ibi
            if 35 < bpm < 200:
                self.bpm_history.append(bpm)
                smoothed = float(np.median(self.bpm_history))
                self.bpm_label.setText(f"{smoothed:.0f}")
                self.status.setText(f"{len(peaks)} beats in {WINDOW_SEC:.0f}s window")
            self.peak_scatter.setData(self.x[peaks], filt[peaks])
            # pulse the heart on most recent peak if it was very recent
            if (len(self.buf) - peaks[-1]) / self.srate < 0.25:
                self._pulse()
        else:
            self.status.setText("not enough peaks yet — wait...")


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = HeartRate()
    w.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(w.update_data)
    timer.start(100)  # 10 Hz UI updates
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
