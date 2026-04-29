#!/usr/bin/env python3
"""Top-down brain projection with live band-power glow at Muse electrode sites.

Keys: 1=delta 2=theta 3=alpha 4=beta 5=gamma  f=flip front/back  r=rotate 90deg
"""
import sys
import numpy as np
import trimesh
from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, welch

STL_PATH = "brain.stl"
PSD_WINDOW_SEC = 2.0
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {
    "1": ("delta", 1, 4),
    "2": ("theta", 4, 8),
    "3": ("alpha", 8, 13),
    "4": ("beta", 13, 30),
    "5": ("gamma", 30, 45),
}
# electrode positions in normalized head coords: (-1..1 left-right, -1..1 back..front)
ELECTRODE_XY = {
    "AF7":  (-0.45,  0.85),
    "AF8":  ( 0.45,  0.85),
    "TP9":  (-0.85, -0.40),
    "TP10": ( 0.85, -0.40),
}

class BrainView(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse 2 — Brain Activity (top-down)")
        self.resize(900, 950)

        self.current_band_key = "3"  # alpha
        self.flip = False
        self.rotation = 0  # 0, 90, 180, 270 deg

        # LSL
        print("Resolving Muse EEG stream...")
        streams = resolve_byprop("type", "EEG", timeout=10)
        if not streams:
            print("No EEG stream found. Run `muselsl stream` first.")
            sys.exit(1)
        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        self.srate = int(self.inlet.info().nominal_srate())
        self.psd_len = int(PSD_WINDOW_SEC * self.srate)
        self.buf = np.zeros((4, self.psd_len * 3), dtype=np.float32)
        self.sos = butter(4, (1.0, 45.0), btype="bandpass", fs=self.srate, output="sos")

        # mesh — project vertices to 2D and normalize to [-1, 1]
        print(f"Loading {STL_PATH}...")
        mesh = trimesh.load(STL_PATH)
        v = mesh.vertices.copy()
        v -= v.mean(axis=0)
        # extents: pick the two longest axes for projection (top-down)
        ext = mesh.extents
        idx = np.argsort(ext)[::-1][:2]  # two largest axes
        proj = v[:, idx]
        scale = np.abs(proj).max()
        proj /= scale
        # subsample for render speed
        if len(proj) > 30000:
            sel = np.random.choice(len(proj), 30000, replace=False)
            proj = proj[sel]
        self.brain_xy = proj.astype(np.float32)

        # plot
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.label = QtWidgets.QLabel("alpha (8-13 Hz)")
        self.label.setStyleSheet("color: #ddd; font-size: 14pt; padding: 4px;")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0a0a0a")
        self.plot.setAspectLocked(True)
        self.plot.setXRange(-1.2, 1.2)
        self.plot.setYRange(-1.2, 1.2)
        self.plot.hideAxis("left"); self.plot.hideAxis("bottom")
        layout.addWidget(self.plot)

        # brain silhouette (faint scatter)
        self.brain_scatter = pg.ScatterPlotItem(
            pos=self.transformed_brain(), pen=None,
            brush=pg.mkBrush(120, 160, 220, 30), size=2)
        self.plot.addItem(self.brain_scatter)

        # electrode glow markers — order matches EEG_CHANNELS for power lookup
        self.electrode_items = {}
        for name in EEG_CHANNELS:
            x, y = ELECTRODE_XY[name]
            outer = pg.ScatterPlotItem(pos=[(x, y)], size=80, pen=None,
                                       brush=pg.mkBrush(255, 255, 0, 0))
            inner = pg.ScatterPlotItem(pos=[(x, y)], size=20, pen=pg.mkPen("#fff", width=1.5),
                                       brush=pg.mkBrush(255, 255, 255, 200))
            text = pg.TextItem(name, anchor=(0.5, -0.6), color="#fff")
            text.setPos(x, y)
            self.plot.addItem(outer); self.plot.addItem(inner); self.plot.addItem(text)
            self.electrode_items[name] = (outer, inner, text)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_data)
        timer.start(80)

    def transformed_brain(self):
        xy = self.brain_xy.copy()
        if self.flip:
            xy[:, 1] = -xy[:, 1]
        theta = np.deg2rad(self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return xy @ rot.T

    def transformed_electrodes(self):
        out = {}
        theta = np.deg2rad(self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        for name, (x, y) in ELECTRODE_XY.items():
            if self.flip:
                y = -y
            out[name] = (c * x - s * y, s * x + c * y)
        return out

    def keyPressEvent(self, e):
        key = e.text()
        if key in BANDS:
            self.current_band_key = key
            name, lo, hi = BANDS[key]
            self.label.setText(f"{name} ({lo}-{hi} Hz)")
        elif key == "f":
            self.flip = not self.flip
            self.refresh_static()
        elif key == "r":
            self.rotation = (self.rotation + 90) % 360
            self.refresh_static()

    def refresh_static(self):
        self.brain_scatter.setData(pos=self.transformed_brain())
        for name, (x, y) in self.transformed_electrodes().items():
            outer, inner, text = self.electrode_items[name]
            outer.setData(pos=[(x, y)])
            inner.setData(pos=[(x, y)])
            text.setPos(x, y)

    def update_data(self):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=256)
        if not chunk:
            return
        arr = np.asarray(chunk, dtype=np.float32).T[:4]
        n = arr.shape[1]
        self.buf[:, :-n] = self.buf[:, n:]
        self.buf[:, -n:] = arr
        filt = sosfiltfilt(self.sos, self.buf, axis=1).astype(np.float32)
        seg = filt[:, -self.psd_len:]
        f, psd = welch(seg, fs=self.srate, nperseg=min(256, self.psd_len), axis=1)
        _, lo, hi = BANDS[self.current_band_key]
        mask = (f >= lo) & (f < hi)
        powers = np.array([np.trapezoid(psd[i, mask], f[mask]) if mask.any() else 1e-12
                           for i in range(4)])
        # log-scale, normalize across electrodes for relative display
        db = 10 * np.log10(np.maximum(powers, 1e-12))
        rng = max(db.max() - db.min(), 1e-3)
        norm = (db - db.min()) / rng  # 0..1

        for i, name in enumerate(EEG_CHANNELS):
            outer, _, _ = self.electrode_items[name]
            alpha = int(60 + 180 * norm[i])
            size = 60 + 80 * norm[i]
            color = pg.colormap.get("inferno").map(norm[i], mode="byte")
            outer.setData(size=size,
                          brush=pg.mkBrush(int(color[0]), int(color[1]), int(color[2]), alpha))


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = BrainView()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
