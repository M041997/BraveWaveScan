#!/usr/bin/env python3
"""Session recorder for Muse 2: writes EEG + PPG to per-session CSV files.

Each session creates ~/eeg-muse/sessions/YYYY-MM-DD_HH-MM-SS/ containing:
  - eeg.csv        (timestamp, tp9, af7, af8, tp10, aux)
  - ppg.csv        (timestamp, ambient, ir, red)
  - meta.json      (start, end, duration_s, notes, sample counts)
"""
import sys
import os
import csv
import json
import time
import threading
from datetime import datetime
from pathlib import Path

from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtWidgets, QtCore

SESSIONS_DIR = Path.home() / "eeg-muse" / "sessions"
EEG_HEADER = ["timestamp", "tp9", "af7", "af8", "tp10", "aux"]
PPG_HEADER = ["timestamp", "ambient", "ir", "red"]
EVENTS_HEADER = ["timestamp", "elapsed_s", "label"]
QUICK_LABELS = ["eyes_open", "eyes_closed", "focused", "relaxed",
                "reading", "balloon", "meditation", "music", "idle"]


class StreamWriter(threading.Thread):
    """Pulls from one LSL inlet, writes timestamped samples to a CSV until stopped."""
    def __init__(self, inlet, path, header):
        super().__init__(daemon=True)
        self.inlet = inlet
        self.path = path
        self.header = header
        self._stop_event = threading.Event()
        self.sample_count = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            n_data_cols = len(self.header) - 1
            while not self._stop_event.is_set():
                chunk, ts = self.inlet.pull_chunk(timeout=0.2, max_samples=256)
                if not chunk:
                    continue
                for sample, t in zip(chunk, ts):
                    row = [f"{t:.6f}"] + [f"{v:.4f}" for v in sample[:n_data_cols]]
                    w.writerow(row)
                self.sample_count += len(chunk)
                f.flush()  # crash-safety: data on disk every chunk


class Recorder(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse 2 — Session Recorder")
        self.resize(560, 360)

        self.eeg_inlet = None
        self.ppg_inlet = None
        self.eeg_writer = None
        self.ppg_writer = None
        self.session_dir = None
        self.start_time = None

        central = QtWidgets.QWidget()
        central.setStyleSheet("background:#0a0a0a;")
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)

        # status / streams
        self.streams_label = QtWidgets.QLabel("checking streams...")
        self.streams_label.setStyleSheet("color:#aaa; font-size:11pt;")
        layout.addWidget(self.streams_label)

        # big elapsed-time display
        self.elapsed_label = QtWidgets.QLabel("00:00")
        self.elapsed_label.setStyleSheet(
            "color:#fff; font-size:54pt; font-weight:700;")
        self.elapsed_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.elapsed_label)

        # sample counters
        self.counts_label = QtWidgets.QLabel("EEG: 0 samples    PPG: 0 samples")
        self.counts_label.setStyleSheet("color:#888; font-size:10pt;")
        self.counts_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.counts_label)

        # notes field
        self.notes = QtWidgets.QLineEdit()
        self.notes.setPlaceholderText(
            "session notes (e.g. 'eyes closed 5min, then math problems')")
        self.notes.setStyleSheet(
            "color:#ddd; background:#1a1a1a; padding:8px; "
            "border:1px solid #333; border-radius:4px; font-size:10pt;")
        layout.addWidget(self.notes)

        # record button
        self.record_btn = QtWidgets.QPushButton("● START RECORDING")
        self.record_btn.setStyleSheet(self._btn_style_idle())
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)

        # inline event labeling — quick-pick combo + free text + Mark button
        events_row = QtWidgets.QHBoxLayout()
        self.label_combo = QtWidgets.QComboBox()
        self.label_combo.setEditable(True)
        for lbl in QUICK_LABELS:
            self.label_combo.addItem(lbl)
        self.label_combo.setCurrentText("")
        self.label_combo.setStyleSheet(
            "color:#ddd; background:#1a1a1a; padding:6px; "
            "border:1px solid #333; border-radius:4px; font-size:10pt;")
        self.label_combo.lineEdit().returnPressed.connect(self.mark_event)
        events_row.addWidget(self.label_combo, 1)
        self.mark_btn = QtWidgets.QPushButton("⌅ Mark Event")
        self.mark_btn.setStyleSheet(
            "background:#2a4d6e; color:#fff; padding:8px 14px; "
            "border-radius:4px; border:none; font-size:10pt; font-weight:600;")
        self.mark_btn.clicked.connect(self.mark_event)
        self.mark_btn.setEnabled(False)
        events_row.addWidget(self.mark_btn)
        layout.addLayout(events_row)

        # recent events list (last 6)
        self.events_log = QtWidgets.QLabel("")
        self.events_log.setStyleSheet(
            "color:#888; font-size:9pt; padding:4px 8px; "
            "background:#0e0e0e; border-radius:4px;")
        self.events_log.setWordWrap(True)
        self.events_log.setMinimumHeight(70)
        self.events_log.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.events_log)

        # session path display
        self.path_label = QtWidgets.QLabel("")
        self.path_label.setStyleSheet("color:#666; font-size:9pt;")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        # event log state
        self._events = []          # list of (elapsed_s, label) for UI
        self._events_csv = None    # open file handle while recording

        # 1Hz UI tick
        self.tick = QtCore.QTimer(self)
        self.tick.timeout.connect(self.refresh_ui)
        self.tick.start(250)

        # connect to streams once on startup
        QtCore.QTimer.singleShot(100, self.connect_streams)

    def _btn_style_idle(self):
        return ("background:#5a1f2a; color:#fff; font-size:14pt; font-weight:700;"
                "padding:14px; border-radius:6px; border:none;")

    def _btn_style_recording(self):
        return ("background:#a02030; color:#fff; font-size:14pt; font-weight:700;"
                "padding:14px; border-radius:6px; border:none;")

    def connect_streams(self):
        eeg = resolve_byprop("type", "EEG", timeout=4)
        ppg = resolve_byprop("type", "PPG", timeout=2)
        msg_parts = []
        if eeg:
            self.eeg_inlet = StreamInlet(eeg[0], max_chunklen=12)
            msg_parts.append(f"EEG ({int(eeg[0].nominal_srate())} Hz)")
        else:
            msg_parts.append("EEG: not found")
        if ppg:
            self.ppg_inlet = StreamInlet(ppg[0], max_chunklen=12)
            msg_parts.append(f"PPG ({int(ppg[0].nominal_srate())} Hz)")
        else:
            msg_parts.append("PPG: not found")
        ok = (eeg is not None) and len(eeg) > 0
        color = "#6cc" if ok else "#c66"
        self.streams_label.setStyleSheet(f"color:{color}; font-size:11pt;")
        self.streams_label.setText("  •  ".join(msg_parts))
        self.record_btn.setEnabled(ok)

    def toggle_recording(self):
        if self.eeg_writer is None:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = SESSIONS_DIR / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.eeg_writer = StreamWriter(
            self.eeg_inlet, self.session_dir / "eeg.csv", EEG_HEADER)
        self.eeg_writer.start()
        if self.ppg_inlet is not None:
            self.ppg_writer = StreamWriter(
                self.ppg_inlet, self.session_dir / "ppg.csv", PPG_HEADER)
            self.ppg_writer.start()
        # events file
        self._events = []
        self._events_csv = open(self.session_dir / "events.csv", "w",
                                newline="")
        self._events_writer = csv.writer(self._events_csv)
        self._events_writer.writerow(EVENTS_HEADER)
        self._events_csv.flush()
        self.mark_btn.setEnabled(True)
        self.record_btn.setText("■ STOP RECORDING")
        self.record_btn.setStyleSheet(self._btn_style_recording())
        self.path_label.setText(f"writing to: {self.session_dir}")
        self.notes.setEnabled(True)
        self._render_events()

    def stop_recording(self):
        elapsed = time.time() - self.start_time
        eeg_n = self.eeg_writer.sample_count if self.eeg_writer else 0
        ppg_n = self.ppg_writer.sample_count if self.ppg_writer else 0
        if self.eeg_writer:
            self.eeg_writer.stop(); self.eeg_writer.join(timeout=2)
        if self.ppg_writer:
            self.ppg_writer.stop(); self.ppg_writer.join(timeout=2)
        n_events = len(self._events)
        if self._events_csv is not None:
            self._events_csv.close()
            self._events_csv = None
        self.mark_btn.setEnabled(False)

        meta = {
            "start": datetime.fromtimestamp(self.start_time).isoformat(),
            "end": datetime.now().isoformat(),
            "duration_s": round(elapsed, 2),
            "eeg_samples": eeg_n,
            "ppg_samples": ppg_n,
            "events": n_events,
            "notes": self.notes.text(),
        }
        with open(self.session_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.eeg_writer = None
        self.ppg_writer = None
        self.record_btn.setText("● START RECORDING")
        self.record_btn.setStyleSheet(self._btn_style_idle())
        self.path_label.setText(
            f"saved {eeg_n + ppg_n:,} samples + {n_events} events to "
            f"{self.session_dir.name}/")

    def mark_event(self):
        if self._events_csv is None or self.start_time is None:
            return
        label = self.label_combo.currentText().strip()
        if not label:
            return
        elapsed = time.time() - self.start_time
        ts = time.time()
        self._events_writer.writerow([f"{ts:.6f}", f"{elapsed:.3f}", label])
        self._events_csv.flush()
        self._events.append((elapsed, label))
        self._render_events()
        self.label_combo.setCurrentText("")

    def _render_events(self):
        if not self._events:
            self.events_log.setText(
                "<span style='color:#666'>events: type a label "
                "and press Enter (or click Mark Event) to timestamp the moment</span>")
            return
        # show last 6
        rows = []
        for elapsed, label in self._events[-6:]:
            mm, ss = divmod(int(elapsed), 60)
            rows.append(f"<span style='color:#bbb'>"
                        f"{mm:02d}:{ss:02d}</span> "
                        f"<span style='color:#7dd8a8'>{label}</span>")
        self.events_log.setText(
            f"<span style='color:#888'>{len(self._events)} events  </span>"
            + "  ·  ".join(rows))

    def refresh_ui(self):
        if self.start_time is not None and self.eeg_writer is not None:
            elapsed = int(time.time() - self.start_time)
            mm, ss = divmod(elapsed, 60)
            self.elapsed_label.setText(f"{mm:02d}:{ss:02d}")
            eeg_n = self.eeg_writer.sample_count if self.eeg_writer else 0
            ppg_n = self.ppg_writer.sample_count if self.ppg_writer else 0
            self.counts_label.setText(
                f"EEG: {eeg_n:,} samples    PPG: {ppg_n:,} samples")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Recorder()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
