#!/usr/bin/env python3
"""Interactive inspector for a recorded session.

Loads ~/eeg-muse/sessions/<dir>/{eeg.csv, meta.json}, precomputes filtered
traces, per-channel spectrograms, and the EMG-quality (HF/LF) timeline,
then lets you scrub through the session to investigate specific moments.

Usage:
  session_inspector.py                      # latest session
  session_inspector.py <session_dir>
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, spectrogram, welch

CHANNELS = ["tp9", "af7", "af8", "tp10"]
CHANNEL_LABELS = ["TP9 (left ear)", "AF7 (left fwd)",
                  "AF8 (right fwd)", "TP10 (right ear)"]
BANDS = [
    ("delta", 1, 4,    (0.20, 0.10, 0.55)),
    ("theta", 4, 8,    (0.10, 0.45, 0.75)),
    ("alpha", 8, 13,   (0.10, 0.75, 0.55)),
    ("beta",  13, 30,  (0.95, 0.70, 0.10)),
    ("gamma", 30, 45,  (0.90, 0.20, 0.20)),
]
BANDPASS = (1.0, 45.0)
SR = 256
WIN_SEC = 5.0          # scrubber's time-window width for trace + bars
SPEC_NPERSEG = 256     # 1s at 256Hz
SPEC_HOP = 128         # 0.5s — => 2 Hz quality-time resolution
HF_LF_GOOD = 0.7
HF_LF_BAD = 2.5

SEG_PALETTE = ["#3b6ea5", "#7c4ea0", "#c89c4d", "#4d9275", "#b85a4d",
               "#5d6e87", "#8c5d3b"]


def session_dir():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    sessions = sorted((Path.home() / "eeg-muse/sessions").iterdir())
    if not sessions:
        print("no sessions found")
        sys.exit(1)
    return sessions[-1]


def quality_color(weight):
    """Maps weight in [0,1] to a Qt color: red→yellow→green."""
    if weight < 0.5:
        r, g, b = 230, int(80 + 240 * weight), 80
    else:
        r, g, b = int(230 - 220 * (weight - 0.5)), 200, 80
    return QtGui.QColor(r, g, b)


class Session:
    """Loads + precomputes everything we need from a session dir."""
    def __init__(self, d: Path):
        self.dir = d
        meta_path = d / "meta.json"
        self.meta = json.load(open(meta_path)) if meta_path.exists() else {}
        # events.csv (written by recorder.py): (timestamp, elapsed_s, label)
        ev_path = d / "events.csv"
        if ev_path.exists():
            try:
                edf = pd.read_csv(ev_path)
                self.events = list(zip(
                    edf["elapsed_s"].astype(float).tolist(),
                    edf["label"].astype(str).tolist()))
            except Exception:
                self.events = []
        else:
            self.events = []
        df = pd.read_csv(d / "eeg.csv")
        self.t = df["timestamp"].to_numpy()
        self.t_rel = self.t - self.t[0]
        self.duration = float(self.t_rel[-1])
        self.raw = df[CHANNELS].to_numpy(dtype=np.float32).T  # (4, N)
        sos = butter(4, BANDPASS, btype="bandpass", fs=SR, output="sos")
        self.filt = sosfiltfilt(sos, self.raw, axis=1).astype(np.float32)
        self.n_samples = self.filt.shape[1]
        # one spectrogram per channel; reuse for display + quality timeline
        self.specs = []
        self.spec_t = None
        self.spec_f = None
        self.quality = None
        for i in range(4):
            f, t, Sxx = spectrogram(
                self.filt[i], fs=SR, nperseg=SPEC_NPERSEG,
                noverlap=SPEC_NPERSEG - SPEC_HOP)
            band = (f >= 1) & (f <= 45)
            self.spec_f = f[band]
            self.spec_t = t
            self.specs.append(10 * np.log10(np.maximum(Sxx[band], 1e-12)))
            if self.quality is None:
                self.quality = np.zeros((4, len(t)), dtype=np.float32)
            lf_mask = (f >= 1) & (f < 15)
            hf_mask = (f >= 15) & (f < 45)
            lf = np.trapezoid(Sxx[lf_mask], f[lf_mask], axis=0)
            hf = np.trapezoid(Sxx[hf_mask], f[hf_mask], axis=0)
            self.quality[i] = hf / np.maximum(lf, 1e-12)
        self.specs = np.stack(self.specs)


class Inspector(QtWidgets.QMainWindow):
    def __init__(self, sess: Session):
        super().__init__()
        self.sess = sess
        self.setWindowTitle(f"Session inspector — {sess.dir.name}")
        self.resize(1500, 950)
        pg.setConfigOptions(antialias=False, useOpenGL=False, background="#0a0a0a")

        self.cursor_t = 0.0       # seconds from start
        self.spec_channel = 1     # AF7 by default

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # header info
        head = QtWidgets.QHBoxLayout()
        info = QtWidgets.QLabel(self._header_text())
        info.setStyleSheet("color:#cfd6e4; font-size:11pt;")
        head.addWidget(info)
        head.addStretch(1)
        ch_lbl = QtWidgets.QLabel("spectrogram channel:")
        ch_lbl.setStyleSheet("color:#7b86a0;")
        head.addWidget(ch_lbl)
        self.ch_combo = QtWidgets.QComboBox()
        for lbl in CHANNEL_LABELS:
            self.ch_combo.addItem(lbl)
        self.ch_combo.setCurrentIndex(self.spec_channel)
        self.ch_combo.currentIndexChanged.connect(self._set_spec_channel)
        head.addWidget(self.ch_combo)
        v.addLayout(head)

        # main plot grid
        self.glw = pg.GraphicsLayoutWidget()
        v.addWidget(self.glw, 1)

        self._build_overview()
        self._build_trace_plots()
        self._build_spec_plot()
        self._build_band_bars()
        # row stretch: overview compact, traces / spec / bands roomy
        layout = self.glw.ci.layout
        layout.setRowStretchFactor(0, 2)   # overview
        for r in range(1, 5):
            layout.setRowStretchFactor(r, 3)
        layout.setColumnStretchFactor(0, 2)
        layout.setColumnStretchFactor(1, 2)

        # status footer
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(
            "color:#cfd6e4; font-size:10pt; padding:4px;")
        v.addWidget(self.status)

        self._update_cursor(0.0)

    def _header_text(self):
        m = self.sess.meta
        start = m.get("start", "")
        dur = m.get("duration_s", self.sess.duration)
        n_seg = len(m.get("segments", []))
        n_ev = len(self.sess.events)
        return (f"<b>{self.sess.dir.name}</b>  "
                f"<span style='color:#7b86a0'>start={start} · "
                f"duration={dur:.1f}s · {n_seg} segments · "
                f"{n_ev} events · "
                f"{self.sess.n_samples:,} samples</span>")

    # ---------- overview row (full session): segments + quality strips ----
    def _build_overview(self):
        ov = self.glw.addPlot(row=0, col=0, colspan=2)
        ov.setLabel("bottom", "session time (s)")
        ov.setLabel("left", "channel quality (HF/LF)")
        ov.setYRange(0, 4)
        ov.showGrid(x=True, y=True, alpha=0.15)
        ov.setMouseEnabled(x=True, y=False)

        # color background by segment
        for i, seg in enumerate(self.sess.meta.get("segments", [])):
            color = QtGui.QColor(SEG_PALETTE[i % len(SEG_PALETTE)])
            color.setAlphaF(0.25)
            r = pg.LinearRegionItem(
                values=(seg["start_s"], seg["end_s"]),
                brush=pg.mkBrush(color),
                pen=pg.mkPen(color, width=0),
                movable=False,
            )
            r.setZValue(-10)
            ov.addItem(r)
            txt = pg.TextItem(seg["label"], color="#cfd6e4", anchor=(0, 0))
            txt.setPos(seg["start_s"] + 0.5, 3.6)
            ov.addItem(txt)

        # quality bands threshold lines
        for thr, c in [(HF_LF_GOOD, "#3a8b50"), (HF_LF_BAD, "#a04040")]:
            line = pg.InfiniteLine(angle=0, pos=thr,
                                   pen=pg.mkPen(c, width=1, style=QtCore.Qt.PenStyle.DashLine))
            ov.addItem(line)

        # plot quality lines per channel
        ch_pens = ["#7da7d8", "#e87878", "#7dd8a8", "#d8c878"]
        for i in range(4):
            q = np.clip(self.sess.quality[i], 0, 4)
            ov.plot(self.sess.spec_t, q, pen=pg.mkPen(ch_pens[i], width=1.2),
                    name=CHANNEL_LABELS[i])

        # legend
        leg = ov.addLegend(offset=(10, 10))
        for i in range(4):
            dummy = pg.PlotDataItem(pen=pg.mkPen(ch_pens[i], width=2))
            leg.addItem(dummy, CHANNEL_LABELS[i])

        # event markers (vertical dashed lines + labels)
        for elapsed, label in self.sess.events:
            line = pg.InfiniteLine(
                angle=90, pos=elapsed,
                pen=pg.mkPen("#cfd6e4", width=1,
                             style=QtCore.Qt.PenStyle.DashLine))
            ov.addItem(line)
            txt = pg.TextItem(label, color="#cfd6e4", anchor=(0, 1))
            txt.setPos(elapsed + 0.5, 3.95)
            ov.addItem(txt)

        # cursor (vertical line) — draggable
        self.cursor_line = pg.InfiniteLine(angle=90, pos=0,
                                           pen=pg.mkPen("#ffd24a", width=2),
                                           movable=True)
        self.cursor_line.sigPositionChanged.connect(self._cursor_dragged)
        ov.addItem(self.cursor_line)

        # window-shaded region around cursor (visual only)
        self.window_region = pg.LinearRegionItem(
            values=(0, WIN_SEC),
            brush=pg.mkBrush(255, 210, 74, 35),
            pen=pg.mkPen(0, 0, 0, 0),
            movable=False,
        )
        self.window_region.setZValue(-5)
        ov.addItem(self.window_region)

        ov.scene().sigMouseClicked.connect(self._overview_clicked)
        self.ov = ov

    # ---------- 4-row trace stack (current window) ------------------------
    def _build_trace_plots(self):
        self.trace_plots = []
        self.trace_curves = []
        for i in range(4):
            p = self.glw.addPlot(row=1 + i, col=0)
            p.setLabel("left", CHANNEL_LABELS[i], units="uV")
            p.setYRange(-100, 100)
            p.showGrid(x=False, y=True, alpha=0.2)
            if i < 3:
                p.hideAxis("bottom")
            else:
                p.setLabel("bottom", "time (s, window)")
            curve = p.plot([], [], pen=pg.mkPen("#4ec9b0", width=1))
            self.trace_plots.append(p)
            self.trace_curves.append(curve)

    # ---------- spectrogram (selected channel, full session) --------------
    def _build_spec_plot(self):
        self.spec_plot = self.glw.addPlot(row=1, col=1, rowspan=2)
        self.spec_plot.setLabel("left", "frequency (Hz)")
        self.spec_plot.setLabel("bottom", "session time (s)")
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        cmap = pg.colormap.get("inferno")
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.spec_img.setLookupTable(lut)
        self.spec_cursor = pg.InfiniteLine(angle=90, pos=0,
                                           pen=pg.mkPen("#ffd24a", width=1.5))
        self.spec_plot.addItem(self.spec_cursor)
        self._render_spectrogram()

    def _render_spectrogram(self):
        S = self.sess.specs[self.spec_channel]
        # transpose so x=time, y=freq
        self.spec_img.setImage(S.T, autoLevels=False)
        # set proper coordinate system
        t = self.sess.spec_t
        f = self.sess.spec_f
        x0, x1 = float(t[0]), float(t[-1])
        y0, y1 = float(f[0]), float(f[-1])
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        self.spec_img.setRect(rect)
        # decent contrast
        lo = float(np.percentile(S, 5))
        hi = float(np.percentile(S, 99))
        self.spec_img.setLevels((lo, hi))
        self.spec_plot.setYRange(1, 45)
        self.spec_plot.setXRange(0, self.sess.duration)
        self.spec_plot.setTitle(
            f"Spectrogram — {CHANNEL_LABELS[self.spec_channel]} (dB)",
            color="#cfd6e4")

    # ---------- band-power bars at cursor (one combined plot) -----------
    def _build_band_bars(self):
        bp = self.glw.addPlot(row=3, col=1, rowspan=2)
        bp.setLabel("left", "power (dB)")
        bp.setTitle("band power @ cursor — colored by channel",
                    color="#cfd6e4")
        bp.setYRange(0, 40)
        bp.showGrid(x=False, y=True, alpha=0.2)
        ch_colors = ["#7da7d8", "#e87878", "#7dd8a8", "#d8c878"]
        self.bars = [[None] * len(BANDS) for _ in range(4)]
        group_w = 0.18
        for ch in range(4):
            color = QtGui.QColor(ch_colors[ch])
            for j in range(len(BANDS)):
                xpos = j + (ch - 1.5) * group_w
                bar = pg.BarGraphItem(
                    x=[xpos], height=[0], width=group_w * 0.9,
                    brush=pg.mkBrush(color))
                bp.addItem(bar)
                self.bars[ch][j] = bar
        ax = bp.getAxis("bottom")
        ax.setTicks([[(j, BANDS[j][0]) for j in range(len(BANDS))]])
        # legend for channel→color
        leg = bp.addLegend(offset=(-10, 10))
        for ch in range(4):
            dummy = pg.BarGraphItem(
                x=[0], height=[0], width=0,
                brush=pg.mkBrush(QtGui.QColor(ch_colors[ch])))
            leg.addItem(dummy, CHANNELS[ch].upper())
        self.bar_plot = bp

    # ---------- interaction --------------------------------------------------
    def _overview_clicked(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = ev.scenePos()
        if not self.ov.sceneBoundingRect().contains(pos):
            return
        x = self.ov.vb.mapSceneToView(pos).x()
        self._update_cursor(x)

    def _cursor_dragged(self):
        x = float(self.cursor_line.value())
        self._update_cursor(x, from_cursor=True)

    def _set_spec_channel(self, idx):
        self.spec_channel = int(idx)
        self._render_spectrogram()

    def _update_cursor(self, t, from_cursor=False):
        t = float(np.clip(t, 0.0, self.sess.duration))
        self.cursor_t = t
        if not from_cursor:
            self.cursor_line.blockSignals(True)
            self.cursor_line.setValue(t)
            self.cursor_line.blockSignals(False)
        # update window region around cursor
        half = WIN_SEC / 2
        self.window_region.setRegion((max(0, t - half),
                                      min(self.sess.duration, t + half)))
        self.spec_cursor.setValue(t)
        self._update_traces(t)
        self._update_bars(t)
        self._update_status(t)

    # ---------- per-cursor data updates -------------------------------------
    def _update_traces(self, t):
        half = WIN_SEC / 2
        t0 = max(0.0, t - half)
        t1 = min(self.sess.duration, t + half)
        i0 = int(t0 * SR)
        i1 = max(i0 + 1, int(t1 * SR))
        x = self.sess.t_rel[i0:i1] - t0
        for ch in range(4):
            y = self.sess.filt[ch, i0:i1]
            self.trace_curves[ch].setData(x, y)
            self.trace_plots[ch].setXRange(0, t1 - t0)

    def _update_bars(self, t):
        half = WIN_SEC / 2
        i0 = max(0, int((t - half) * SR))
        i1 = min(self.sess.n_samples, int((t + half) * SR))
        if i1 - i0 < SR:
            return
        seg = self.sess.filt[:, i0:i1]
        f, psd = welch(seg, fs=SR, nperseg=min(512, i1 - i0), axis=1)
        for ch in range(4):
            for j, (_, lo, hi, _) in enumerate(BANDS):
                m = (f >= lo) & (f < hi)
                p = float(np.trapezoid(psd[ch, m], f[m])) if m.any() else 1e-12
                self.bars[ch][j].setOpts(height=[10 * np.log10(max(p, 1e-12))])

    def _update_status(self, t):
        # which segment
        seg_label = "(unlabeled)"
        for s in self.sess.meta.get("segments", []):
            if s["start_s"] <= t < s["end_s"]:
                seg_label = s["label"]
                break
        # nearest event within ±10s
        nearest_event = ""
        if self.sess.events:
            elapsed_arr = np.array([e[0] for e in self.sess.events])
            idx = int(np.argmin(np.abs(elapsed_arr - t)))
            dt = t - self.sess.events[idx][0]
            if abs(dt) <= 10.0:
                arrow = "←" if dt > 0 else "→"
                nearest_event = (f"   event {arrow} "
                                 f"<b>{self.sess.events[idx][1]}</b> "
                                 f"({dt:+.1f}s)")
        # current quality at this t
        ti = int(np.searchsorted(self.sess.spec_t, t))
        ti = max(0, min(ti, len(self.sess.spec_t) - 1))
        parts = []
        for ch in range(4):
            q = float(self.sess.quality[ch, ti])
            verdict = ("GOOD" if q < HF_LF_GOOD else
                       "BAD"  if q > HF_LF_BAD  else "marg")
            parts.append(f"{CHANNELS[ch].upper()}={q:.2f} {verdict}")
        self.status.setText(
            f"t={t:6.2f}s   segment: <b>{seg_label}</b>{nearest_event}   "
            + "   ".join(parts))


def main():
    d = session_dir()
    print(f"loading {d}...")
    sess = Session(d)
    print(f"  duration: {sess.duration:.1f}s  "
          f"samples: {sess.n_samples:,}  "
          f"segments: {len(sess.meta.get('segments', []))}")
    app = QtWidgets.QApplication(sys.argv)
    w = Inspector(sess)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
