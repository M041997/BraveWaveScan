#!/usr/bin/env python3
"""Brave Wave Scan — cinematic live 3D EEG dashboard.

Runs against a Muse/LSL EEG stream when one is available and automatically falls
back to a synthetic signal for booth demos. A private ``brain.stl`` can be placed
beside this file; otherwise a procedural cortical form is generated.

Controls: drag rotate · wheel zoom · space pause · D demo/live · R reset camera
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import trimesh
from PyQt6 import QtCore, QtGui, QtWidgets
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, sosfiltfilt, welch


APP_NAME = "BRAVE WAVE SCAN"
STL_PATH = Path(__file__).with_name("brain.stl")
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = [
    ("DELTA", 1, 4, "RESTORE", (0.35, 0.25, 1.00)),
    ("THETA", 4, 8, "IMAGINE", (0.08, 0.55, 1.00)),
    ("ALPHA", 8, 13, "SETTLE", (0.05, 0.95, 0.78)),
    ("BETA", 13, 30, "FOCUS", (1.00, 0.72, 0.12)),
    ("GAMMA", 30, 45, "INTEGRATE", (1.00, 0.22, 0.50)),
]

# x=left/right, y=posterior/anterior, z=inferior/superior
ELECTRODE_XYZ = {
    "AF7": (-0.38, 0.77, 0.48),
    "AF8": (0.38, 0.77, 0.48),
    "TP9": (-0.82, -0.35, 0.02),
    "TP10": (0.82, -0.35, 0.02),
}

BG = "#05080f"
PANEL = "#0b111d"
PANEL_2 = "#0e1726"
TEXT = "#f1f6ff"
MUTED = "#738098"
CYAN = "#27f2d0"


def rgba_css(rgb: tuple[float, float, float], alpha: int = 255) -> str:
    r, g, b = (int(c * 255) for c in rgb)
    return f"rgba({r},{g},{b},{alpha})"


class SignalSource:
    """Small adapter that keeps hardware and exhibition demo data interchangeable."""

    def __init__(self, force_demo: bool = False):
        self.inlet = None
        self.srate = 256
        self.demo = force_demo
        self.started = time.monotonic()
        self.buf = np.zeros((4, self.srate * 6), dtype=np.float32)
        self.sos = butter(4, (1.0, 45.0), btype="bandpass", fs=self.srate, output="sos")
        if not force_demo:
            streams = resolve_byprop("type", "EEG", timeout=2.0)
            if streams:
                self.inlet = StreamInlet(streams[0], max_chunklen=12)
                self.srate = int(self.inlet.info().nominal_srate()) or 256
                self.buf = np.zeros((4, self.srate * 6), dtype=np.float32)
                self.sos = butter(4, (1.0, 45.0), btype="bandpass", fs=self.srate, output="sos")
                self.demo = False
            else:
                self.demo = True

    def toggle(self) -> None:
        # Toggling away from live is always safe. Returning to live is attempted
        # without ever interrupting the visual experience.
        if not self.demo:
            self.demo = True
            return
        streams = resolve_byprop("type", "EEG", timeout=0.25)
        if streams:
            self.inlet = StreamInlet(streams[0], max_chunklen=12)
            self.demo = False

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return band power (4x5), quality (4), and a display trace."""
        if self.demo:
            return self._demo_sample()
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=256)
        if not chunk:
            return None
        arr = np.asarray(chunk, dtype=np.float32).T[:4]
        n = min(arr.shape[1], self.buf.shape[1])
        self.buf[:, :-n] = self.buf[:, n:]
        self.buf[:, -n:] = arr[:, -n:]
        filt = sosfiltfilt(self.sos, self.buf, axis=1).astype(np.float32)
        segment = filt[:, -2 * self.srate:]
        f, psd = welch(segment, fs=self.srate, nperseg=min(256, segment.shape[1]), axis=1)
        powers = np.zeros((4, 5), dtype=np.float32)
        for j, (_, lo, hi, _, _) in enumerate(BANDS):
            mask = (f >= lo) & (f < hi)
            if mask.any():
                powers[:, j] = np.trapezoid(psd[:, mask], f[mask], axis=1)
        raw = self.buf[:, -2 * self.srate:]
        std = raw.std(axis=1)
        rails = (np.abs(raw) > 100).mean(axis=1)
        quality = np.clip((std - 4) / 8, 0, 1) * np.clip((180 - std) / 100, 0, 1)
        quality *= np.clip(1 - (rails - 0.12) * 3, 0, 1)
        return np.maximum(powers, 1e-12), quality.astype(np.float32), filt.mean(axis=0)[-256:]

    def _demo_sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = time.monotonic() - self.started
        base = np.array([0.22, 0.44, 0.92, 0.61, 0.27], dtype=np.float32)
        phase = np.arange(4, dtype=np.float32)[:, None] * 0.7
        band_phase = np.arange(5, dtype=np.float32)[None, :] * 0.95
        movement = 0.16 * np.sin(t * 0.72 + phase + band_phase)
        alpha_breath = np.zeros((4, 5), dtype=np.float32)
        alpha_breath[:, 2] = 0.18 * (0.5 + 0.5 * np.sin(t * 0.45 + np.arange(4)))
        powers = np.clip(base[None, :] + movement + alpha_breath, 0.03, None)
        quality = np.clip(0.91 + 0.06 * np.sin(t * 0.35 + np.arange(4) * 1.4), 0, 1)
        x = np.linspace(0, 1, 256, dtype=np.float32)
        trace = (18 * np.sin(2 * np.pi * 10 * x + t * 2.2)
                 + 7 * np.sin(2 * np.pi * 6 * x - t)
                 + 3 * np.sin(2 * np.pi * 23 * x + t * 0.4))
        return powers.astype(np.float32), quality.astype(np.float32), trace.astype(np.float32)


class MetricCard(QtWidgets.QFrame):
    def __init__(self, label: str, color: str):
        super().__init__()
        self.setObjectName("metricCard")
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(14, 11, 14, 11)
        box.setSpacing(1)
        title = QtWidgets.QLabel(label)
        title.setObjectName("eyebrow")
        self.value = QtWidgets.QLabel("--")
        self.value.setStyleSheet(f"color:{color};font-size:24px;font-weight:700;")
        box.addWidget(title)
        box.addWidget(self.value)


class BrainDashboard(QtWidgets.QMainWindow):
    def __init__(self, force_demo: bool = False):
        super().__init__()
        self.source = SignalSource(force_demo)
        self.paused = False
        self.frame = 0
        self.quality_smooth = np.ones(4, dtype=np.float32)
        self.setWindowTitle(f"{APP_NAME} — Real-Time EEG")
        self.resize(1520, 940)
        self.setMinimumSize(1120, 720)
        self._build_shell()
        self._build_brain()
        self._build_sidebar()
        self._apply_theme()
        self._refresh_mode()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)

    def _build_shell(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(22, 16, 22, 18)
        outer.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        brand = QtWidgets.QVBoxLayout()
        brand.setSpacing(0)
        eyebrow = QtWidgets.QLabel("NEURAL TELEMETRY / 01")
        eyebrow.setObjectName("eyebrowAccent")
        name = QtWidgets.QLabel(APP_NAME)
        name.setObjectName("brand")
        brand.addWidget(eyebrow)
        brand.addWidget(name)
        header.addLayout(brand)
        header.addStretch()
        self.clock = QtWidgets.QLabel()
        self.clock.setObjectName("mono")
        self.mode_pill = QtWidgets.QLabel()
        self.mode_pill.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.mode_pill.setMinimumWidth(126)
        header.addWidget(self.clock)
        header.addSpacing(18)
        header.addWidget(self.mode_pill)
        outer.addLayout(header)

        self.body = QtWidgets.QHBoxLayout()
        self.body.setSpacing(14)
        outer.addLayout(self.body, 1)

        footer = QtWidgets.QHBoxLayout()
        self.footer_status = QtWidgets.QLabel("●  PIPELINE NOMINAL")
        self.footer_status.setObjectName("status")
        footer.addWidget(self.footer_status)
        footer.addStretch()
        hint = QtWidgets.QLabel("DRAG ROTATE   ·   SCROLL ZOOM   ·   SPACE PAUSE   ·   D SOURCE   ·   R RESET")
        hint.setObjectName("mono")
        footer.addWidget(hint)
        outer.addLayout(footer)

    def _load_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if STL_PATH.exists():
            mesh = trimesh.load(STL_PATH, force="mesh")
            v = mesh.vertices.astype(np.float32)
            v -= v.mean(axis=0)
            ext = np.ptp(v, axis=0)
            ap, lr, si = np.argsort(ext)[::-1]
            v = v[:, [lr, ap, si]]
            v /= np.abs(v).max() + 1e-9
            return v, mesh.faces.astype(np.uint32)
        mesh = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
        v = mesh.vertices.astype(np.float32)
        v *= np.array([0.77, 1.0, 0.68], dtype=np.float32)
        # A restrained anatomical suggestion: flattened underside, frontal lift,
        # and shallow medial separation without pretending to be a medical model.
        v[:, 2] = np.maximum(v[:, 2], -0.48)
        v[:, 2] += 0.06 * np.maximum(v[:, 1], 0)
        v[:, 0] += 0.025 * np.sign(v[:, 0]) * (1 - np.abs(v[:, 1]))
        return v, mesh.faces.astype(np.uint32)

    def _build_brain(self) -> None:
        stage = QtWidgets.QFrame()
        stage.setObjectName("stage")
        stage_layout = QtWidgets.QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor(BG)
        self.gl_view.setCameraPosition(distance=2.75, elevation=22, azimuth=-88)
        stage_layout.addWidget(self.gl_view)
        self.body.addWidget(stage, 3)

        self.verts, self.faces = self._load_mesh()
        base = np.tile(np.array([0.06, 0.12, 0.18, 1.0], dtype=np.float32), (len(self.verts), 1))
        self.mesh_item = gl.GLMeshItem(
            vertexes=self.verts, faces=self.faces, vertexColors=base,
            smooth=True, drawEdges=False, shader="shaded")
        self.gl_view.addItem(self.mesh_item)

        # Sparse wireframe creates the technical scan/readout texture.
        wire = gl.GLMeshItem(
            vertexes=self.verts, faces=self.faces, smooth=False,
            drawFaces=False, drawEdges=True, edgeColor=(0.10, 0.72, 0.72, 0.08))
        self.gl_view.addItem(wire)

        grid = gl.GLGridItem()
        grid.setSize(3.4, 3.4)
        grid.setSpacing(0.2, 0.2)
        grid.translate(0, 0, -0.57)
        grid.setColor((35, 88, 105, 38))
        self.gl_view.addItem(grid)

        self.weights, self.snapped = self._surface_weights()
        self.electrodes = []
        for p in self.snapped:
            dot = gl.GLScatterPlotItem(pos=np.array([p * 1.035]), size=10,
                                       color=np.array([[0.15, 1.0, 0.82, 1.0]]), pxMode=True)
            self.gl_view.addItem(dot)
            self.electrodes.append(dot)

        self.flow_paths = [self._arc(self.snapped[1], self.snapped[0]),
                           self._arc(self.snapped[2], self.snapped[3])]
        self.flow_lines = []
        for path in self.flow_paths:
            line = gl.GLLinePlotItem(pos=path, color=(0.1, 0.95, 0.8, 0.28), width=2,
                                     antialias=True, mode="line_strip")
            self.gl_view.addItem(line)
            self.flow_lines.append(line)

        self.hero_title = QtWidgets.QLabel("CORTICAL SIGNAL FLOW", self.gl_view)
        self.hero_title.setObjectName("heroTitle")
        self.hero_sub = QtWidgets.QLabel("FRONTAL  →  OCCIPITAL  /  REAL-TIME BAND POWER", self.gl_view)
        self.hero_sub.setObjectName("heroSub")
        self.hero_title.move(22, 20)
        self.hero_sub.move(24, 58)
        self.hero_title.adjustSize()
        self.hero_sub.adjustSize()

    def _surface_weights(self) -> tuple[np.ndarray, np.ndarray]:
        ideal = np.array([ELECTRODE_XYZ[name] for name in EEG_CHANNELS], dtype=np.float32)
        snapped = np.empty_like(ideal)
        for i, p in enumerate(ideal):
            snapped[i] = self.verts[np.argmin(((self.verts - p) ** 2).sum(axis=1))]
        dist2 = ((self.verts[None, :, :] - snapped[:, None, :]) ** 2).sum(axis=-1)
        return np.exp(-dist2 / 0.24).astype(np.float32), snapped

    @staticmethod
    def _arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        t = np.linspace(0, 1, 70, dtype=np.float32)[:, None]
        path = (1 - t) * a + t * b
        path[:, 2] += 0.22 * np.sin(np.pi * t[:, 0])
        return path * 1.045

    def _build_sidebar(self) -> None:
        side = QtWidgets.QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(355)
        box = QtWidgets.QVBoxLayout(side)
        box.setContentsMargins(20, 19, 20, 19)
        box.setSpacing(13)

        heading = QtWidgets.QLabel("LIVE SPECTRUM")
        heading.setObjectName("section")
        box.addWidget(heading)
        self.band_bars = []
        self.band_values = []
        for name, lo, hi, state, rgb in BANDS:
            row = QtWidgets.QVBoxLayout()
            top = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"{name}  <span style='color:{MUTED}'>{lo}–{hi} HZ</span>")
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            value = QtWidgets.QLabel("0%")
            value.setObjectName("monoBright")
            top.addWidget(label)
            top.addStretch()
            top.addWidget(value)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(
                "QProgressBar{border:0;background:#172235;border-radius:3px;}"
                f"QProgressBar::chunk{{background:{rgba_css(rgb)};border-radius:3px;}}")
            row.addLayout(top)
            row.addWidget(bar)
            box.addLayout(row)
            self.band_bars.append(bar)
            self.band_values.append(value)

        metrics = QtWidgets.QHBoxLayout()
        metrics.setSpacing(8)
        self.focus_card = MetricCard("FOCUS INDEX", "#ffbd32")
        self.calm_card = MetricCard("CALM INDEX", CYAN)
        metrics.addWidget(self.focus_card)
        metrics.addWidget(self.calm_card)
        box.addLayout(metrics)

        channel_heading = QtWidgets.QLabel("ELECTRODE CONTACT")
        channel_heading.setObjectName("section")
        box.addWidget(channel_heading)
        self.channel_dots = []
        self.channel_values = []
        for channel in EEG_CHANNELS:
            row = QtWidgets.QHBoxLayout()
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet(f"color:{CYAN};font-size:13px;")
            label = QtWidgets.QLabel(channel)
            value = QtWidgets.QLabel("--")
            value.setObjectName("mono")
            row.addWidget(dot)
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            box.addLayout(row)
            self.channel_dots.append(dot)
            self.channel_values.append(value)

        self.trace = pg.PlotWidget()
        self.trace.setFixedHeight(94)
        self.trace.setBackground(PANEL_2)
        self.trace.hideAxis("left")
        self.trace.hideAxis("bottom")
        self.trace.setMouseEnabled(x=False, y=False)
        self.trace.setYRange(-45, 45)
        self.trace_curve = self.trace.plot(pen=pg.mkPen(CYAN, width=1.2))
        box.addWidget(self.trace)
        self.dominant = QtWidgets.QLabel("DOMINANT  —")
        self.dominant.setObjectName("dominant")
        box.addWidget(self.dominant)
        box.addStretch()
        self.body.addWidget(side)

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TEXT}; font-family:'DejaVu Sans'; }}
            QFrame#stage, QFrame#sidebar {{ background:{PANEL}; border:1px solid #1b293c; border-radius:10px; }}
            QFrame#metricCard {{ background:{PANEL_2}; border:1px solid #1b293c; border-radius:7px; }}
            QLabel#brand {{ font-size:25px; font-weight:700; letter-spacing:3px; }}
            QLabel#eyebrow, QLabel#mono {{ color:{MUTED}; font-family:'DejaVu Sans Mono'; font-size:10px; letter-spacing:1px; }}
            QLabel#eyebrowAccent {{ color:{CYAN}; font-family:'DejaVu Sans Mono'; font-size:9px; letter-spacing:2px; }}
            QLabel#monoBright {{ color:#d9e6f8; font-family:'DejaVu Sans Mono'; font-size:10px; }}
            QLabel#section {{ color:#aab8cc; font-size:10px; font-weight:700; letter-spacing:2px; padding-top:3px; }}
            QLabel#heroTitle {{ color:white; background:transparent; font-size:21px; font-weight:700; letter-spacing:2px; }}
            QLabel#heroSub {{ color:{CYAN}; background:transparent; font-family:'DejaVu Sans Mono'; font-size:9px; letter-spacing:1px; }}
            QLabel#status {{ color:{CYAN}; font-family:'DejaVu Sans Mono'; font-size:10px; letter-spacing:1px; }}
            QLabel#dominant {{ background:#101d2d; border-left:3px solid {CYAN}; padding:10px; font-size:11px; font-weight:700; letter-spacing:1px; }}
        """)

    def _refresh_mode(self) -> None:
        if self.source.demo:
            self.mode_pill.setText("●  DEMO SIGNAL")
            self.mode_pill.setStyleSheet(
                "color:#ffbd32;background:#241d0d;border:1px solid #594516;"
                "border-radius:13px;padding:6px;font:700 10px 'DejaVu Sans Mono';")
            self.footer_status.setText("●  SYNTHETIC EEG · HARDWARE READY")
        else:
            self.mode_pill.setText("●  LIVE EEG")
            self.mode_pill.setStyleSheet(
                f"color:{CYAN};background:#09221f;border:1px solid #155c52;"
                "border-radius:13px;padding:6px;font:700 10px 'DejaVu Sans Mono';")
            self.footer_status.setText("●  LSL STREAM LOCKED · PIPELINE NOMINAL")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space:
            self.paused = not self.paused
            self.footer_status.setText("Ⅱ  VISUALIZATION PAUSED" if self.paused else "●  PIPELINE NOMINAL")
        elif key == QtCore.Qt.Key.Key_D:
            self.source.toggle()
            self._refresh_mode()
        elif key == QtCore.Qt.Key.Key_R:
            self.gl_view.setCameraPosition(distance=2.75, elevation=22, azimuth=-88)
        else:
            super().keyPressEvent(event)

    def update_data(self) -> None:
        self.clock.setText(QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd  hh:mm:ss"))
        if self.paused:
            return
        packet = self.source.sample()
        if packet is None:
            return
        powers, quality, trace = packet
        self.quality_smooth = 0.88 * self.quality_smooth + 0.12 * quality

        # Relative power gives viewers an immediately legible 100% composition.
        band_total = powers.mean(axis=0)
        relative = band_total / (band_total.sum() + 1e-12)
        for i, value in enumerate(relative):
            pct = int(round(float(value) * 100))
            self.band_bars[i].setValue(pct)
            self.band_values[i].setText(f"{pct:02d}%")

        dominant = int(np.argmax(relative))
        name, _, _, state, rgb = BANDS[dominant]
        self.dominant.setText(f"DOMINANT  {name}  /  {state}")
        self.dominant.setStyleSheet(
            f"background:#101d2d;border-left:3px solid {rgba_css(rgb)};padding:10px;"
            "font-size:11px;font-weight:700;letter-spacing:1px;")
        focus = np.clip((relative[3] + relative[4]) * 175, 0, 100)
        calm = np.clip((relative[2] + 0.45 * relative[1]) * 150, 0, 100)
        self.focus_card.value.setText(f"{focus:02.0f}")
        self.calm_card.value.setText(f"{calm:02.0f}")

        for i, q in enumerate(self.quality_smooth):
            pct = int(float(q) * 100)
            color = CYAN if pct >= 75 else "#ffbd32" if pct >= 45 else "#ff456a"
            self.channel_dots[i].setStyleSheet(f"color:{color};font-size:13px;")
            self.channel_values[i].setText(f"{pct}%")
            self.electrodes[i].setData(
                pos=np.array([self.snapped[i] * 1.035]), size=8 + 6 * float(q),
                color=np.array([[1 - q, q, 0.55, 1.0]], dtype=np.float32))

        self.trace_curve.setData(trace)
        self._update_brain(powers, self.quality_smooth)
        self._update_flow(relative)
        self.frame += 1

    def _update_brain(self, powers: np.ndarray, quality: np.ndarray) -> None:
        logp = np.log10(np.maximum(powers, 1e-12))
        logp -= logp.min(axis=1, keepdims=True)
        norm = logp / (logp.max(axis=1, keepdims=True) + 1e-9)
        weighted = self.weights * quality[:, None]
        weighted /= weighted.sum(axis=0, keepdims=True) + 1e-9
        vertex_band = weighted.T @ norm
        exp = np.exp((vertex_band - vertex_band.max(axis=1, keepdims=True)) * 2.4)
        proportions = exp / exp.sum(axis=1, keepdims=True)
        palette = np.array([band[4] for band in BANDS], dtype=np.float32)
        active = proportions @ palette
        coverage = (self.weights * quality[:, None]).sum(axis=0)
        coverage /= coverage.max() + 1e-9
        pulse = 0.92 + 0.08 * np.sin(self.frame * 0.12)
        light = np.clip((0.13 + 0.87 * np.sqrt(coverage)) * pulse, 0, 1)
        floor = np.array([0.025, 0.055, 0.09], dtype=np.float32)
        rgb = floor + active * light[:, None] * 0.92
        rgba = np.column_stack([np.clip(rgb, 0, 1), np.ones(len(rgb), dtype=np.float32)])
        self.mesh_item.setMeshData(vertexes=self.verts, faces=self.faces,
                                   vertexColors=rgba.astype(np.float32))

    def _update_flow(self, relative: np.ndarray) -> None:
        energy = float(np.clip(relative[2] + relative[3] + relative[4], 0, 1))
        for offset, (line, path) in enumerate(zip(self.flow_lines, self.flow_paths)):
            colors = np.tile(np.array([0.08, 0.95, 0.78, 0.12], dtype=np.float32), (len(path), 1))
            center = (self.frame * 2 + offset * 27) % len(path)
            distance = np.minimum(np.abs(np.arange(len(path)) - center),
                                  len(path) - np.abs(np.arange(len(path)) - center))
            glow = np.exp(-(distance ** 2) / 24.0)
            colors[:, 0] += 0.65 * glow
            colors[:, 2] += 0.18 * glow
            colors[:, 3] = np.clip(0.08 + energy * 0.28 + glow * 0.72, 0, 1)
            line.setData(pos=path, color=colors, width=2.0, mode="line_strip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="skip hardware discovery")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    pg.setConfigOptions(antialias=True)
    window = BrainDashboard(force_demo=args.demo)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
