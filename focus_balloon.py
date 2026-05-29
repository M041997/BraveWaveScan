#!/usr/bin/env python3
"""Focus balloon — concentrate to inflate and pop a virtual balloon.

Reads live EEG from Muse 2 via LSL, computes a focus score from frontal
(AF7+AF8) beta power relative to alpha+theta, and inflates a balloon you pop
by sustaining intense focus.

Phases:
  1. CALIBRATION (~10s): sit still, breathe normally — establishes baseline.
  2. PLAY: balloon size tracks your focus z-score. Hold it pegged at max
     for ~0.8s and it pops. Respawns automatically.
"""
import sys
import math
import random
import time
import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtWidgets, QtCore, QtGui
from scipy.signal import butter, sosfiltfilt, welch

WINDOW_SEC = 4.0
PSD_WINDOW_SEC = 2.0
BANDPASS = (1.0, 45.0)
N_EEG = 4
FRONTAL_IDX = (1, 2)          # AF7, AF8
CALIBRATION_SEC = 10.0
EMA_ALPHA = 0.18              # focus-score smoothing (higher = snappier)
BALLOON_MIN_R = 35
BALLOON_MAX_R = 230
POP_HOLD_SEC = 1.2            # time at full size required to pop
RESPAWN_SEC = 1.4             # delay after pop before next balloon

# z-score range mapped onto balloon size — tuned to typical user dynamics
Z_LO, Z_HI = -0.5, 2.2

# EMG-contamination heuristic on a forehead channel:
#   HF/LF = power(15-45Hz) / power(1-15Hz)
# Healthy frontal cortical signal sits well below 1.0; heavy EMG (jaw, scalp
# muscles, bad contact) pushes it above 2.0. We linearly map this into a
# quality weight in [0, 1] and use it to weight per-channel contributions.
HF_LF_GOOD = 0.7   # weight = 1.0 below this
HF_LF_BAD = 2.5    # weight = 0.0 above this
QUALITY_EMA = 0.15 # smoothing on per-channel HF/LF ratio

FOCUS_SESSIONS_DIR = Path.home() / "eeg-muse" / "focus_sessions"
LOG_HEADER = [
    "lsl_ts", "wall_ts", "elapsed_s", "phase",
    "z", "ema_log_ratio", "baseline_median", "baseline_mad",
    "w_af7", "w_af8", "hflf_af7", "hflf_af8",
    "balloon_pct", "hold_pct", "alive", "event",
]

BALLOON_COLORS = [
    (0.95, 0.30, 0.35),  # red
    (0.30, 0.65, 0.95),  # blue
    (0.40, 0.85, 0.45),  # green
    (0.95, 0.75, 0.25),  # yellow
    (0.75, 0.40, 0.90),  # purple
    (0.95, 0.55, 0.25),  # orange
]


def band_power(psd, f, lo, hi):
    m = (f >= lo) & (f < hi)
    if not m.any():
        return 1e-12
    return float(np.trapezoid(psd[m], f[m]))


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "r", "color")

    def __init__(self, x, y, color):
        ang = random.uniform(0, 2 * math.pi)
        speed = random.uniform(180, 480)
        self.x = x
        self.y = y
        self.vx = math.cos(ang) * speed
        self.vy = math.sin(ang) * speed
        self.life = 1.0
        self.r = random.uniform(3, 7)
        self.color = color

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 600 * dt  # gravity
        self.vx *= 0.98
        self.life -= dt / 0.9


class BalloonWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(700, 600)
        self.focus_z = 0.0          # smoothed display value
        self.target_z = 0.0
        self.calibrating = True
        self.calib_remaining = CALIBRATION_SEC
        self.status_text = "Calibrating — sit still and breathe normally..."

        self.balloon_radius = BALLOON_MIN_R
        self.balloon_color = random.choice(BALLOON_COLORS)
        self.alive = True
        self.full_hold = 0.0        # seconds at >=95% size
        self.respawn_timer = 0.0    # counts down after a pop
        self.particles = []
        self.pop_count = 0
        self.last_t = time.monotonic()

        # bobbing offset
        self.bob_phase = 0.0

        # per-channel signal quality: list of (label, weight, hf_lf_ratio)
        self.channel_quality = []
        self.no_signal = False

    def set_focus(self, z, calibrating, calib_remaining, no_signal=False):
        self.target_z = z
        self.calibrating = calibrating
        self.calib_remaining = calib_remaining
        self.no_signal = no_signal
        if calibrating:
            self.status_text = (f"Calibrating — sit still and breathe normally "
                                f"({calib_remaining:0.0f}s)")
        elif no_signal:
            self.status_text = ("⚠ no clean forehead signal — "
                                "reseat AF7/AF8 pads")
        else:
            self.status_text = f"Focus z={z:+.2f}    Pops: {self.pop_count}"

    def set_channel_quality(self, channels):
        """channels: list of (label, weight in [0,1], hf_lf raw ratio)."""
        self.channel_quality = list(channels)

    def tick(self):
        now = time.monotonic()
        dt = min(0.1, now - self.last_t)
        self.last_t = now
        self.bob_phase += dt

        # ease displayed z toward target
        self.focus_z += (self.target_z - self.focus_z) * 0.25

        # particles
        if self.particles:
            for p in self.particles:
                p.step(dt)
            self.particles = [p for p in self.particles if p.life > 0]

        if self.calibrating or self.no_signal:
            self.balloon_radius = BALLOON_MIN_R
            self.full_hold = 0.0
        elif not self.alive:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0 and not self.particles:
                self.alive = True
                self.balloon_color = random.choice(BALLOON_COLORS)
                self.balloon_radius = BALLOON_MIN_R
                self.full_hold = 0.0
        else:
            # map z → radius
            t = (self.focus_z - Z_LO) / (Z_HI - Z_LO)
            t = max(0.0, min(1.0, t))
            target_r = BALLOON_MIN_R + t * (BALLOON_MAX_R - BALLOON_MIN_R)
            self.balloon_radius += (target_r - self.balloon_radius) * 0.15

            # pop check
            if self.balloon_radius >= 0.95 * BALLOON_MAX_R:
                self.full_hold += dt
                if self.full_hold >= POP_HOLD_SEC:
                    self._pop()
            else:
                self.full_hold = max(0.0, self.full_hold - dt * 0.5)

        self.update()

    def _pop(self):
        cx, cy = self._balloon_center()
        r, g, b = self.balloon_color
        color = QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))
        for _ in range(60):
            self.particles.append(Particle(cx, cy, color))
        self.alive = False
        self.pop_count += 1
        self.respawn_timer = RESPAWN_SEC

    def _balloon_center(self):
        w, h = self.width(), self.height()
        bob = math.sin(self.bob_phase * 1.6) * 8
        cx = w / 2
        cy = h / 2 - 30 + bob
        return cx, cy

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # background
        bg = QtGui.QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0.0, QtGui.QColor("#0b1220"))
        bg.setColorAt(1.0, QtGui.QColor("#050810"))
        p.fillRect(self.rect(), bg)

        # focus meter (left side)
        self._draw_meter(p, 30, 80, 22, h - 200)

        # status text top
        p.setPen(QtGui.QColor("#cfd6e4"))
        f = QtGui.QFont("Helvetica", 13)
        p.setFont(f)
        p.drawText(QtCore.QRect(0, 16, w, 28),
                   QtCore.Qt.AlignmentFlag.AlignCenter, self.status_text)
        if not self.calibrating:
            small = QtGui.QFont("Helvetica", 10)
            p.setFont(small)
            p.setPen(QtGui.QColor("#7b86a0"))
            hint = ("focus harder → balloon grows.  hold at max to pop.  "
                    f"hold gauge: {self.full_hold/POP_HOLD_SEC*100:.0f}%")
            p.drawText(QtCore.QRect(0, 44, w, 20),
                       QtCore.Qt.AlignmentFlag.AlignCenter, hint)

        # per-channel quality dots (top-right)
        if self.channel_quality:
            self._draw_channel_quality(p, w - 180, 20)

        # balloon
        if self.alive and not self.calibrating and not self.no_signal:
            self._draw_balloon(p)

        # particles
        for part in self.particles:
            a = max(0.0, min(1.0, part.life))
            c = QtGui.QColor(part.color)
            c.setAlphaF(a)
            p.setBrush(c)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawEllipse(QtCore.QPointF(part.x, part.y), part.r, part.r)

        p.end()

    def _draw_meter(self, p, x, y, w, h):
        p.setPen(QtGui.QPen(QtGui.QColor("#2a3346"), 1))
        p.setBrush(QtGui.QColor("#11182a"))
        p.drawRoundedRect(QtCore.QRectF(x, y, w, h), 4, 4)
        # fill
        t = (self.focus_z - Z_LO) / (Z_HI - Z_LO)
        t = max(0.0, min(1.0, t))
        fh = t * h
        grad = QtGui.QLinearGradient(0, y + h, 0, y)
        grad.setColorAt(0.0, QtGui.QColor(40, 130, 200))
        grad.setColorAt(0.6, QtGui.QColor(230, 200, 60))
        grad.setColorAt(1.0, QtGui.QColor(230, 80, 80))
        p.setBrush(grad)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(QtCore.QRectF(x, y + h - fh, w, fh), 3, 3)
        # label
        p.setPen(QtGui.QColor("#7b86a0"))
        p.setFont(QtGui.QFont("Helvetica", 9))
        p.drawText(QtCore.QRectF(x - 10, y + h + 6, w + 20, 16),
                   QtCore.Qt.AlignmentFlag.AlignCenter, "focus")

    def _draw_channel_quality(self, p, x, y):
        p.setFont(QtGui.QFont("Helvetica", 9))
        p.setPen(QtGui.QColor("#7b86a0"))
        p.drawText(QtCore.QRectF(x, y, 160, 14),
                   QtCore.Qt.AlignmentFlag.AlignLeft, "channel quality")
        for i, (label, weight, hf_lf) in enumerate(self.channel_quality):
            cx = x + 10 + i * 75
            cy = y + 32
            # weight → color: red(0) → yellow(0.5) → green(1)
            if weight < 0.5:
                r = 230; g = int(80 + 240 * weight); b = 80
            else:
                r = int(230 - 220 * (weight - 0.5)); g = 200; b = 80
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(r, g, b))
            p.drawEllipse(QtCore.QPointF(cx, cy), 7, 7)
            p.setPen(QtGui.QColor("#cfd6e4"))
            p.setFont(QtGui.QFont("Helvetica", 9, QtGui.QFont.Weight.Bold))
            p.drawText(QtCore.QRectF(cx + 12, cy - 8, 60, 16),
                       QtCore.Qt.AlignmentFlag.AlignLeft, label)
            p.setFont(QtGui.QFont("Helvetica", 8))
            p.setPen(QtGui.QColor("#7b86a0"))
            p.drawText(QtCore.QRectF(cx - 14, cy + 10, 60, 14),
                       QtCore.Qt.AlignmentFlag.AlignLeft,
                       f"hf/lf={hf_lf:.2f}")

    def _draw_balloon(self, p):
        cx, cy = self._balloon_center()
        r = self.balloon_radius
        rx = r * 0.9
        ry = r
        # string
        anchor_x, anchor_y = cx, cy + ry + 12
        p.setPen(QtGui.QPen(QtGui.QColor("#9aa3b8"), 1.4))
        path = QtGui.QPainterPath()
        path.moveTo(anchor_x, anchor_y)
        end_y = self.height() - 30
        cp1 = QtCore.QPointF(anchor_x - 30, anchor_y + (end_y - anchor_y) * 0.4)
        cp2 = QtCore.QPointF(anchor_x + 25, anchor_y + (end_y - anchor_y) * 0.75)
        path.cubicTo(cp1, cp2, QtCore.QPointF(anchor_x + 5, end_y))
        p.drawPath(path)
        # body — radial gradient for shading
        cr, cg, cb = self.balloon_color
        body = QtGui.QColor(int(cr * 255), int(cg * 255), int(cb * 255))
        light = QtGui.QColor(min(255, int(cr * 255) + 80),
                             min(255, int(cg * 255) + 80),
                             min(255, int(cb * 255) + 80))
        rg = QtGui.QRadialGradient(cx - rx * 0.35, cy - ry * 0.35, rx * 1.5)
        rg.setColorAt(0.0, light)
        rg.setColorAt(1.0, body)
        p.setBrush(rg)
        p.setPen(QtGui.QPen(QtGui.QColor(int(cr * 180), int(cg * 180),
                                         int(cb * 180)), 1.5))
        p.drawEllipse(QtCore.QPointF(cx, cy), rx, ry)
        # knot triangle
        knot = QtGui.QPolygonF([
            QtCore.QPointF(cx - 8, cy + ry),
            QtCore.QPointF(cx + 8, cy + ry),
            QtCore.QPointF(cx, cy + ry + 12),
        ])
        p.setBrush(body.darker(125))
        p.drawPolygon(knot)
        # specular highlight
        hi = QtGui.QColor(255, 255, 255, 110)
        p.setBrush(hi)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(QtCore.QPointF(cx - rx * 0.35, cy - ry * 0.4),
                      rx * 0.18, ry * 0.10)


class FocusBalloon(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Focus Balloon — Muse 2")
        self.resize(900, 720)
        self.canvas = BalloonWidget()
        self.setCentralWidget(self.canvas)

        # connect to Muse
        print("Resolving Muse EEG stream...")
        streams = resolve_byprop("type", "EEG", timeout=10)
        if not streams:
            print("No EEG stream found. Run `muselsl stream` first.")
            sys.exit(1)
        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        self.srate = int(self.inlet.info().nominal_srate())
        self.buf_len = int(WINDOW_SEC * self.srate)
        self.psd_len = int(PSD_WINDOW_SEC * self.srate)
        self.buf = np.zeros((N_EEG, self.buf_len), dtype=np.float32)
        self.sos = butter(4, BANDPASS, btype="bandpass",
                          fs=self.srate, output="sos")
        self.samples_seen = 0
        self.warmup_target = self.psd_len  # need this many before computing

        # focus state — rolling self-calibrating baseline
        self.ema_log_ratio = None
        self.log_window = deque(maxlen=int(60 * 20))  # ~60s at ~20 Hz
        self.warmup_sec = 20.0
        self.start_t = None
        self._z_log_t = 0.0
        self._z_min = float("inf")
        self._z_max = float("-inf")
        # per-channel EMG-quality smoothing: maps frontal channel idx -> EMA HF/LF ratio
        self.hf_lf_ema = {}

        # session logging — focus_log.csv joinable to any concurrent
        # recorder.py eeg.csv via the LSL timestamp column
        ts_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = FOCUS_SESSIONS_DIR / ts_stamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.session_dir / "focus_log.csv"
        self.log_fh = open(self.log_path, "w", newline="")
        self.log_writer = csv.writer(self.log_fh)
        self.log_writer.writerow(LOG_HEADER)
        self.log_fh.flush()
        self._last_pop_count = 0
        self._wall_start = time.time()
        meta = {
            "start": datetime.fromtimestamp(self._wall_start).isoformat(),
            "config": {
                "Z_LO": Z_LO, "Z_HI": Z_HI,
                "POP_HOLD_SEC": POP_HOLD_SEC,
                "HF_LF_GOOD": HF_LF_GOOD, "HF_LF_BAD": HF_LF_BAD,
                "warmup_sec": 20.0,
                "frontal_channels": ["AF7", "AF8"],
            },
        }
        with open(self.session_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"logging session to: {self.session_dir}")

        # timers
        self.eeg_timer = QtCore.QTimer(self)
        self.eeg_timer.timeout.connect(self.pull_and_score)
        self.eeg_timer.start(50)

        self.draw_timer = QtCore.QTimer(self)
        self.draw_timer.timeout.connect(self.canvas.tick)
        self.draw_timer.start(16)  # ~60 fps

    def _fnum(self, v):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return ""
        return f"{v:.6f}"

    def _log_row(self, lsl_ts, phase, *, z=None, ema=None,
                 base_med=None, base_mad=None, weights=None, qsmooth=None,
                 event=""):
        elapsed = time.monotonic() - (self.start_t or time.monotonic())
        balloon_pct = 100.0 * (self.canvas.balloon_radius - BALLOON_MIN_R) \
            / max(BALLOON_MAX_R - BALLOON_MIN_R, 1)
        hold_pct = 100.0 * self.canvas.full_hold / POP_HOLD_SEC
        w_af7 = weights[0] if weights else None
        w_af8 = weights[1] if weights else None
        h_af7 = qsmooth[0] if qsmooth else None
        h_af8 = qsmooth[1] if qsmooth else None
        self.log_writer.writerow([
            self._fnum(lsl_ts),
            self._fnum(time.time()),
            f"{elapsed:.3f}",
            phase,
            self._fnum(z), self._fnum(ema),
            self._fnum(base_med), self._fnum(base_mad),
            self._fnum(w_af7), self._fnum(w_af8),
            self._fnum(h_af7), self._fnum(h_af8),
            f"{max(0, balloon_pct):.2f}",
            f"{min(100, max(0, hold_pct)):.2f}",
            int(self.canvas.alive),
            event,
        ])
        self.log_fh.flush()

    def _check_pop_event(self):
        c = self.canvas.pop_count
        if c > self._last_pop_count:
            self._last_pop_count = c
            return "pop"
        return ""

    def closeEvent(self, ev):
        try:
            if self.log_fh and not self.log_fh.closed:
                self.log_fh.close()
            # update meta with end + total pops
            meta_path = self.session_dir / "meta.json"
            if meta_path.exists():
                meta = json.load(open(meta_path))
                meta["end"] = datetime.now().isoformat()
                meta["duration_s"] = round(time.time() - self._wall_start, 2)
                meta["total_pops"] = self.canvas.pop_count
                json.dump(meta, open(meta_path, "w"), indent=2)
        except Exception:
            pass
        super().closeEvent(ev)

    def pull_and_score(self):
        chunk, ts_list = self.inlet.pull_chunk(timeout=0.0, max_samples=256)
        if not chunk:
            return
        lsl_ts = float(ts_list[-1]) if ts_list else float("nan")
        arr = np.asarray(chunk, dtype=np.float32).T[:N_EEG]
        n = arr.shape[1]
        self.buf[:, :-n] = self.buf[:, n:]
        self.buf[:, -n:] = arr
        self.samples_seen += n
        if self.samples_seen < self.warmup_target:
            self._log_row(lsl_ts, "buffering")
            return

        filt = sosfiltfilt(self.sos, self.buf, axis=1).astype(np.float32)
        seg = filt[:, -self.psd_len:]
        f, psd = welch(seg, fs=self.srate,
                       nperseg=min(256, self.psd_len), axis=1)

        # per-channel band powers + EMG-quality estimate (frontal only —
        # cognitive focus shows up there, but those channels are also where
        # facial-muscle EMG contaminates worst)
        ch_bp = {}
        weights = []
        quality_smoothed = []
        for i in FRONTAL_IDX:
            theta = band_power(psd[i], f, 4, 8)
            alpha = band_power(psd[i], f, 8, 13)
            beta_p = band_power(psd[i], f, 13, 30)
            lf = band_power(psd[i], f, 1, 15)
            hf = band_power(psd[i], f, 15, 45)
            hf_lf = hf / max(lf, 1e-12)
            prev = self.hf_lf_ema.get(i, hf_lf)
            sm = QUALITY_EMA * hf_lf + (1 - QUALITY_EMA) * prev
            self.hf_lf_ema[i] = sm
            w = max(0.0, min(1.0,
                             (HF_LF_BAD - sm) / (HF_LF_BAD - HF_LF_GOOD)))
            ch_bp[i] = (theta, alpha, beta_p)
            weights.append(w)
            quality_smoothed.append(sm)

        total_w = sum(weights)
        # surface per-channel quality to the UI
        self.canvas.set_channel_quality(
            [("AF7", weights[0], quality_smoothed[0]),
             ("AF8", weights[1], quality_smoothed[1])])

        if total_w < 0.05:
            # both forehead channels look like EMG — refuse to score
            self.canvas.set_focus(0.0, calibrating=False,
                                  calib_remaining=0.0,
                                  no_signal=True)
            self._log_row(lsl_ts, "no_signal",
                          weights=weights, qsmooth=quality_smoothed,
                          event=self._check_pop_event())
            return

        beta = sum(weights[k] * ch_bp[FRONTAL_IDX[k]][2]
                   for k in range(len(FRONTAL_IDX))) / total_w
        a_w = sum(weights[k] * ch_bp[FRONTAL_IDX[k]][1]
                  for k in range(len(FRONTAL_IDX))) / total_w
        t_w = sum(weights[k] * ch_bp[FRONTAL_IDX[k]][0]
                  for k in range(len(FRONTAL_IDX))) / total_w
        ratio = beta / max(a_w + t_w, 1e-12)
        log_ratio = math.log(max(ratio, 1e-12))

        # smooth raw log-ratio
        if self.ema_log_ratio is None:
            self.ema_log_ratio = log_ratio
        else:
            self.ema_log_ratio = (EMA_ALPHA * log_ratio +
                                  (1 - EMA_ALPHA) * self.ema_log_ratio)

        # rolling self-calibrating baseline
        self.log_window.append(self.ema_log_ratio)
        if self.start_t is None:
            self.start_t = time.monotonic()
        elapsed = time.monotonic() - self.start_t
        if elapsed < self.warmup_sec:
            remaining = self.warmup_sec - elapsed
            self.canvas.set_focus(0.0, calibrating=True,
                                  calib_remaining=remaining)
            self._log_row(lsl_ts, "warmup",
                          ema=self.ema_log_ratio,
                          weights=weights, qsmooth=quality_smoothed)
            return

        # robust stats over the rolling window — median + scaled MAD
        arr = np.asarray(self.log_window)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        baseline_std = max(1.4826 * mad, 0.05)
        z = (self.ema_log_ratio - med) / baseline_std
        z = max(-3.0, min(6.0, z))
        self.canvas.set_focus(z, calibrating=False, calib_remaining=0.0)
        self._log_row(lsl_ts, "play",
                      z=z, ema=self.ema_log_ratio,
                      base_med=med, base_mad=mad,
                      weights=weights, qsmooth=quality_smoothed,
                      event=self._check_pop_event())
        # log live z range every ~1s so we can tune
        self._z_min = min(self._z_min, z)
        self._z_max = max(self._z_max, z)
        now = time.monotonic()
        if now - self._z_log_t > 1.0:
            print(f"z={z:+.2f}  rolling min={self._z_min:+.2f} "
                  f"max={self._z_max:+.2f}", flush=True)
            self._z_log_t = now
            # decay rolling extremes so they reflect recent state
            self._z_min += 0.2
            self._z_max -= 0.2


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FocusBalloon()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
