#!/usr/bin/env python3
"""3D brain that lights up by live EEG band power.

The mesh surface is colored: cool=quiet, hot=active. Activity = your selected
brainwave band, smeared from electrode positions across nearby vertices.

Keys: 1=delta 2=theta 3=alpha 4=beta 5=gamma
      f=flip front/back  r=rotate 90deg  s=swap L/R
"""
import sys
import numpy as np
import trimesh
from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.signal import butter, sosfiltfilt, welch

STL_PATH = "brain.stl"
PSD_WINDOW_SEC = 2.0
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {
    "1": ("delta", 1, 4,    "deep sleep / unconscious"),
    "2": ("theta", 4, 8,    "drowsy / meditation"),
    "3": ("alpha", 8, 13,   "relaxed wakeful / eyes closed"),
    "4": ("beta",  13, 30,  "focused / alert thinking"),
    "5": ("gamma", 30, 45,  "peak attention / cognition"),
}
# semantic band colors (matplotlib viridis-ish progression, low-freq cool → high-freq warm)
BAND_COLORS = {
    "delta": (0.20, 0.10, 0.55),   # indigo
    "theta": (0.10, 0.45, 0.75),   # blue
    "alpha": (0.10, 0.75, 0.55),   # green
    "beta":  (0.95, 0.70, 0.10),   # amber
    "gamma": (0.90, 0.20, 0.20),   # red
}
# normalized 3D electrode positions (x = L→R, y = back→front, z = bottom→top)
ELECTRODE_XYZ = {
    "AF7":  (-0.45,  0.85,  0.50),
    "AF8":  ( 0.45,  0.85,  0.50),
    "TP9":  (-0.85, -0.40,  0.00),
    "TP10": ( 0.85, -0.40,  0.00),
}
ELECTRODE_SIGMA = 0.45  # how far each electrode's influence smears


class Brain3D(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse 2 — Brain Activity 3D")
        self.resize(1100, 950)

        self.current_key = "3"
        self.flip_y = False
        self.swap_x = False
        self.rotation_deg = 0

        # LSL
        print("Resolving Muse EEG stream...")
        streams = resolve_byprop("type", "EEG", timeout=10)
        if not streams:
            print("No EEG stream. Run muselsl stream first.")
            sys.exit(1)
        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        self.srate = int(self.inlet.info().nominal_srate())
        self.psd_len = int(PSD_WINDOW_SEC * self.srate)
        self.buf = np.zeros((4, self.psd_len * 3), dtype=np.float32)
        self.sos = butter(4, (1.0, 45.0), btype="bandpass", fs=self.srate, output="sos")

        # Mesh
        print(f"Loading {STL_PATH}...")
        mesh = trimesh.load(STL_PATH)
        v = mesh.vertices.astype(np.float32)
        v -= v.mean(axis=0)
        # axis order: largest extent = anterior-posterior (front-back) = y,
        #             second = left-right = x, smallest = superior-inferior = z
        ext = mesh.extents
        ap_axis, lr_axis, si_axis = np.argsort(ext)[::-1]
        reorder = np.array([lr_axis, ap_axis, si_axis])
        v = v[:, reorder]
        v /= np.abs(v).max()
        self.verts_orig = v
        self.faces = mesh.faces.astype(np.uint32)

        # GL view fills the whole window; labels float on top as overlays
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=2.4, elevation=30, azimuth=-90)
        self.gl_view.setBackgroundColor("#000000")
        self.setCentralWidget(self.gl_view)

        # overlay labels parented to the GL widget
        self.title = QtWidgets.QLabel(self.gl_view)
        self.title.setStyleSheet(
            "color:#fff; font-size:16pt; font-weight:700; "
            "background: rgba(0,0,0,140); padding:6px 12px; border-radius:4px;")
        self.title.move(14, 10)
        self.title.adjustSize()

        self.subtitle = QtWidgets.QLabel(self.gl_view)
        self.subtitle.setStyleSheet(
            "color:#bbb; font-size:10pt; "
            "background: rgba(0,0,0,140); padding:4px 10px; border-radius:4px;")
        self.subtitle.move(14, 50)
        self.subtitle.adjustSize()

        self.hint = QtWidgets.QLabel(
            "1-5 band  ·  drag rotate  ·  scroll zoom  ·  f flip  ·  s swap L/R  ·  r rotate",
            self.gl_view)
        self.hint.setStyleSheet(
            "color:#777; font-size:8pt; background: rgba(0,0,0,120); "
            "padding:3px 8px; border-radius:3px;")

        # mesh item — vertex colors will be updated each frame
        self.mesh_item = gl.GLMeshItem(
            vertexes=self.verts_orig, faces=self.faces,
            vertexColors=np.tile([0.18, 0.22, 0.30, 1.0], (len(self.verts_orig), 1)).astype(np.float32),
            smooth=True, drawEdges=False, shader="shaded")
        self.gl_view.addItem(self.mesh_item)

        # electrode marker spheres
        self.elec_items = {}
        for name in EEG_CHANNELS:
            md = gl.MeshData.sphere(rows=8, cols=8, radius=0.04)
            sph = gl.GLMeshItem(meshdata=md, smooth=True, color=(1, 1, 1, 1), shader="shaded")
            self.gl_view.addItem(sph)
            self.elec_items[name] = sph

        # band-color array used to blend per-vertex colors
        self.band_colors_arr = np.array(
            [BAND_COLORS[BANDS[k][0]] for k in BANDS], dtype=np.float32)

        # band legend strip — 5 colored chips at the top-right
        self.legend_chips = []
        for j, k in enumerate(BANDS):
            name = BANDS[k][0]
            r, g, b = BAND_COLORS[name]
            chip = QtWidgets.QLabel(name, self.gl_view)
            chip.setStyleSheet(
                f"color:#fff; font-size:9pt; font-weight:600; "
                f"background: rgba({int(r*255)},{int(g*255)},{int(b*255)},220); "
                f"padding:3px 10px; border-radius:3px;")
            chip.adjustSize()
            self.legend_chips.append(chip)

        # Per-vertex distance-weight matrix (4, n_verts) — recompute when orientation changes
        self.weights = None
        self.weights_norm = None
        self.update_weights()
        self.update_label()

    # -------- orientation helpers --------
    def transform_pos(self, pos):
        x, y, z = pos
        if self.flip_y:
            y = -y
        if self.swap_x:
            x = -x
        theta = np.deg2rad(self.rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        return (c * x - s * y, s * x + c * y, z)

    def current_verts(self):
        v = self.verts_orig.copy()
        if self.flip_y:
            v[:, 1] = -v[:, 1]
        if self.swap_x:
            v[:, 0] = -v[:, 0]
        theta = np.deg2rad(self.rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return v @ rot.T

    def update_weights(self):
        verts = self.current_verts()
        # snap each electrode to its nearest brain vertex so the gaussian falloff
        # actually lands on the surface instead of in empty space outside the mesh
        ideal = np.array(
            [self.transform_pos(ELECTRODE_XYZ[n]) for n in EEG_CHANNELS],
            dtype=np.float32)
        snapped = np.empty_like(ideal)
        for i, p in enumerate(ideal):
            d = ((verts - p) ** 2).sum(axis=1)
            snapped[i] = verts[np.argmin(d)]
        # (4, n_verts) gaussian
        d2 = ((verts[None, :, :] - snapped[:, None, :]) ** 2).sum(axis=-1)
        self.weights = np.exp(-d2 / (ELECTRODE_SIGMA ** 2)).astype(np.float32)
        col_sum = self.weights.sum(axis=0, keepdims=True) + 1e-9
        self.weights_norm = (self.weights / col_sum).astype(np.float32)
        # remember snapped positions so the spheres draw on the surface too
        self._snapped_pos = snapped
        for i, name in enumerate(EEG_CHANNELS):
            sph = self.elec_items.get(name)
            if sph is not None:
                sph.resetTransform()
                # nudge slightly outside the surface so the sphere is visible
                p = snapped[i] * 1.06
                sph.translate(float(p[0]), float(p[1]), float(p[2]))
        # update mesh + electrode positions
        self.mesh_item.setMeshData(
            vertexes=verts, faces=self.faces,
            vertexColors=self.mesh_item.opts.get("meshdata").vertexColors()
                if self.mesh_item.opts.get("meshdata") is not None else
                np.tile([0.18, 0.22, 0.30, 1.0], (len(verts), 1)).astype(np.float32))

    # -------- input --------
    def keyPressEvent(self, e):
        k = e.text()
        if k in BANDS:
            self.current_key = k
            self.update_label()
        elif k == "f":
            self.flip_y = not self.flip_y; self.update_weights()
        elif k == "s":
            self.swap_x = not self.swap_x; self.update_weights()
        elif k == "r":
            self.rotation_deg = (self.rotation_deg + 90) % 360
            self.update_weights()

    def update_label(self):
        self.title.setText("ALL BANDS")
        self.title.setStyleSheet(
            "color:#fff; font-size:16pt; font-weight:700; "
            "background: rgba(0,0,0,140); padding:6px 12px; border-radius:4px;")
        self.subtitle.setText("each region colored by its dominant brainwave")
        self.title.adjustSize()
        self.subtitle.adjustSize()
        self.hint.adjustSize()
        self.reposition_overlays()

    def reposition_overlays(self):
        if hasattr(self, "hint"):
            self.hint.move(14, self.gl_view.height() - self.hint.height() - 10)
        if hasattr(self, "legend_chips") and self.legend_chips:
            x = self.gl_view.width() - 14
            y = 12
            for chip in reversed(self.legend_chips):
                x -= chip.width() + 6
                chip.move(x, y)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.reposition_overlays()

    # -------- per-frame --------
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

        # per-electrode signal quality 0..1, smoothed over time to stop flicker
        raw_seg = self.buf[:, -self.psd_len:]
        raw_std = raw_seg.std(axis=1)
        rail_pct = (np.abs(raw_seg) > 100).mean(axis=1)
        # std_score: 0 if flat (<5 µV) or huge (>180 µV); 1 in the 10–80 µV sweet spot
        std_score = np.clip((raw_std - 4) / 6, 0, 1) * np.clip((180 - raw_std) / 100, 0, 1)
        # rail_score: tolerate up to ~15% railing (real EEG with blinks can spike)
        rail_score = np.clip(1 - (rail_pct - 0.15) * 3, 0, 1)
        quality_inst = (std_score * rail_score).astype(np.float32)
        if not hasattr(self, "quality_smooth"):
            self.quality_smooth = quality_inst.copy()
        # EMA smoothing so the spheres don't strobe between frames
        self.quality_smooth = (0.85 * self.quality_smooth + 0.15 * quality_inst).astype(np.float32)
        quality = self.quality_smooth
        for i, name in enumerate(EEG_CHANNELS):
            q = float(quality[i])
            self.elec_items[name].setColor((1.0 - q, q, 0.1, 1.0))

        # power per electrode per band — shape (4, 5)
        n_bands = len(BANDS)
        powers = np.zeros((4, n_bands), dtype=np.float32)
        for j, key in enumerate(BANDS):
            _, lo, hi, _ = BANDS[key]
            m = (f >= lo) & (f < hi)
            if m.any():
                powers[:, j] = np.trapezoid(psd[:, m], f[m], axis=1)
        powers = np.maximum(powers, 1e-12)
        # log scale, normalize PER ELECTRODE so each contributes its band shape
        # not its absolute level — this prevents one railed electrode from
        # drowning out clean ones.
        db = 10 * np.log10(powers)
        db = db - db.min(axis=1, keepdims=True)
        db_norm = db / (db.max(axis=1, keepdims=True) + 1e-9)

        # gate each electrode by its signal quality before blending
        # bad/off electrodes contribute ~nothing to the per-vertex band shape
        weighted = self.weights * quality[:, None]
        col_sum = weighted.sum(axis=0, keepdims=True) + 1e-9
        weights_norm_q = (weighted / col_sum).astype(np.float32)

        # per-vertex per-band power: (V, 5) = weights_norm_q.T @ db_norm
        vert_band = weights_norm_q.T @ db_norm  # (V, 5)

        # softmax across bands per vertex; lower T = more honest blending
        T = 2.0
        e = np.exp((vert_band - vert_band.max(axis=1, keepdims=True)) * T)
        prop = e / e.sum(axis=1, keepdims=True)  # (V, 5)

        # blend the band colors by softmaxed proportion
        color = prop @ self.band_colors_arr  # (V, 3)

        # brightness from QUALITY-WEIGHTED coverage. Vertices near a dead
        # electrode go dark, not gray-blue floor; only good electrodes light up
        # their region.
        coverage = (self.weights * quality[:, None]).sum(axis=0)
        max_cov = coverage.max() + 1e-9
        cov_norm = coverage / max_cov
        brightness = 0.05 + 0.95 * (cov_norm ** 0.5)

        floor = np.array([0.10, 0.12, 0.18], dtype=np.float32)
        rgb = floor[None, :] * (1 - brightness[:, None]) + color * brightness[:, None]
        rgba = np.empty((len(rgb), 4), dtype=np.float32)
        rgba[:, :3] = np.clip(rgb, 0, 1)
        rgba[:, 3] = 1.0
        self.mesh_item.setMeshData(
            vertexes=self.current_verts(), faces=self.faces,
            vertexColors=rgba)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = Brain3D()
    w.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(w.update_data)
    timer.start(100)  # 10 fps — vertex color upload is the bottleneck
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
