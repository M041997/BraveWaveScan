#!/usr/bin/env python3
"""Record Muse EEG/PPG + microphone audio while marking speech events.

Press Space to mark a stutter/block/repetition at the exact moment it happens.
Sessions are saved under ~/eeg-muse/sessions/YYYY-MM-DD_HH-MM-SS_speech/.
"""
import csv
import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from pylsl import StreamInlet, resolve_byprop
from PyQt6 import QtCore, QtWidgets

SESSIONS_DIR = Path.home() / "eeg-muse" / "sessions"
EEG_HEADER = ["timestamp", "tp9", "af7", "af8", "tp10", "aux"]
PPG_HEADER = ["timestamp", "ambient", "ir", "red"]
EVENTS_HEADER = ["timestamp", "elapsed_s", "label"]
AUDIO_LEVEL_HEADER = ["timestamp", "elapsed_s", "rms", "peak"]

READING_TEXT = (
    "The morning train moved slowly through the quiet city. I watched the "
    "windows flash with sunlight and tried to describe each passing sound: "
    "the soft brakes, the low hum, and the quick rhythm of people stepping "
    "onto the platform. Today I am reading at a steady pace, noticing each "
    "word clearly, and marking any moment where my speech catches, repeats, "
    "or blocks."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Record speech events with Muse EEG.")
    parser.add_argument(
        "--audio-device",
        help="Input audio device id or name substring, e.g. 11 or Samson.",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_audio_device(selector):
    if selector is None:
        return None, sd.query_devices(kind="input")
    try:
        device_id = int(selector)
        return device_id, sd.query_devices(device_id, kind="input")
    except ValueError:
        needle = selector.lower()
        for idx, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0 and needle in device["name"].lower():
                return idx, sd.query_devices(idx, kind="input")
    raise ValueError(f"No input audio device matching {selector!r}")


class StreamWriter(threading.Thread):
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
            writer = csv.writer(f)
            writer.writerow(self.header)
            n_data_cols = len(self.header) - 1
            while not self._stop_event.is_set():
                chunk, timestamps = self.inlet.pull_chunk(timeout=0.2, max_samples=256)
                if not chunk:
                    continue
                for sample, timestamp in zip(chunk, timestamps):
                    row = [f"{timestamp:.6f}"] + [
                        f"{value:.4f}" for value in sample[:n_data_cols]
                    ]
                    writer.writerow(row)
                self.sample_count += len(chunk)
                f.flush()


class AudioWriter(threading.Thread):
    def __init__(self, wav_path, levels_path, samplerate, channels, start_time):
        super().__init__(daemon=True)
        self.wav_path = wav_path
        self.levels_path = levels_path
        self.samplerate = samplerate
        self.channels = channels
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self.frame_count = 0
        self.start_time = start_time

    def stop(self):
        self._stop_event.set()

    def push(self, data, timestamp):
        self.queue.put((data.copy(), timestamp))

    def run(self):
        with sf.SoundFile(
            self.wav_path,
            mode="w",
            samplerate=self.samplerate,
            channels=self.channels,
            subtype="PCM_16",
        ) as wav, open(self.levels_path, "w", newline="") as levels_file:
            levels = csv.writer(levels_file)
            levels.writerow(AUDIO_LEVEL_HEADER)
            while not self._stop_event.is_set() or not self.queue.empty():
                try:
                    data, timestamp = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                wav.write(data)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
                peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
                elapsed = timestamp - self.start_time
                levels.writerow([f"{timestamp:.6f}", f"{elapsed:.3f}", f"{rms:.6f}", f"{peak:.6f}"])
                self.frame_count += len(data)
                levels_file.flush()


class SpeechMarker(QtWidgets.QMainWindow):
    def __init__(self, audio_device_selector=None):
        super().__init__()
        self.setWindowTitle("Muse Speech Marker")
        self.resize(760, 620)

        self.eeg_inlet = None
        self.ppg_inlet = None
        self.eeg_writer = None
        self.ppg_writer = None
        self.audio_writer = None
        self.audio_stream = None
        self.session_dir = None
        self.start_time = None
        self.audio_device_selector = audio_device_selector
        self.audio_device_id = None
        self.audio_device_info = None
        self.events = []
        self.events_file = None
        self.events_writer = None

        central = QtWidgets.QWidget()
        central.setStyleSheet("background:#0a0a0a; color:#ddd;")
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.status = QtWidgets.QLabel("checking EEG/PPG streams and microphone...")
        self.status.setStyleSheet("color:#aaa; font-size:11pt;")
        layout.addWidget(self.status)

        self.elapsed = QtWidgets.QLabel("00:00")
        self.elapsed.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.elapsed.setStyleSheet("font-size:52pt; font-weight:800; color:#fff;")
        layout.addWidget(self.elapsed)

        self.reading = QtWidgets.QTextEdit()
        self.reading.setPlainText(READING_TEXT)
        self.reading.setReadOnly(True)
        self.reading.setStyleSheet(
            "background:#111; color:#f0f0f0; border:1px solid #333; "
            "border-radius:4px; padding:10px; font-size:15pt; line-height:1.3;"
        )
        layout.addWidget(self.reading, 1)

        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("START SESSION")
        self.start_btn.setStyleSheet(
            "background:#225a3a; color:#fff; padding:12px; border:none; "
            "border-radius:5px; font-size:13pt; font-weight:700;"
        )
        self.start_btn.clicked.connect(self.toggle_recording)
        row.addWidget(self.start_btn)

        self.mark_btn = QtWidgets.QPushButton("MARK STUTTER  [Space]")
        self.mark_btn.setEnabled(False)
        self.mark_btn.setStyleSheet(
            "background:#6e2a42; color:#fff; padding:12px; border:none; "
            "border-radius:5px; font-size:13pt; font-weight:700;"
        )
        self.mark_btn.clicked.connect(lambda: self.mark_event("stutter"))
        row.addWidget(self.mark_btn)
        layout.addLayout(row)

        self.counts = QtWidgets.QLabel("EEG: 0  PPG: 0  audio: 0.0s  events: 0")
        self.counts.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.counts.setStyleSheet("color:#888; font-size:10pt;")
        layout.addWidget(self.counts)

        self.events_log = QtWidgets.QLabel("events will appear here")
        self.events_log.setWordWrap(True)
        self.events_log.setMinimumHeight(70)
        self.events_log.setStyleSheet(
            "background:#0f0f0f; color:#888; padding:8px; border-radius:4px;"
        )
        layout.addWidget(self.events_log)

        self.path_label = QtWidgets.QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color:#666; font-size:9pt;")
        layout.addWidget(self.path_label)

        self.tick = QtCore.QTimer(self)
        self.tick.timeout.connect(self.refresh_ui)
        self.tick.start(250)

        QtCore.QTimer.singleShot(100, self.connect_inputs)

    def connect_inputs(self):
        eeg = resolve_byprop("type", "EEG", timeout=4)
        ppg = resolve_byprop("type", "PPG", timeout=2)
        parts = []
        if eeg:
            self.eeg_inlet = StreamInlet(eeg[0], max_chunklen=12)
            parts.append(f"EEG {int(eeg[0].nominal_srate())} Hz")
        else:
            parts.append("EEG not found")
        if ppg:
            self.ppg_inlet = StreamInlet(ppg[0], max_chunklen=12)
            parts.append(f"PPG {int(ppg[0].nominal_srate())} Hz")
        else:
            parts.append("PPG not found")
        try:
            self.audio_device_id, self.audio_device_info = resolve_audio_device(
                self.audio_device_selector)
            parts.append(f"mic: {self.audio_device_info['name']}")
            mic_ok = True
        except Exception as exc:
            parts.append(f"mic error: {exc}")
            mic_ok = False
        ok = self.eeg_inlet is not None and mic_ok
        self.status.setText("  |  ".join(parts))
        self.status.setStyleSheet(f"color:{'#7dd8a8' if ok else '#e06c75'}; font-size:11pt;")
        self.start_btn.setEnabled(ok)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space and self.start_time is not None:
            self.mark_event("stutter")
        else:
            super().keyPressEvent(event)

    def toggle_recording(self):
        if self.start_time is None:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_speech")
        self.session_dir = SESSIONS_DIR / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.events = []

        self.eeg_writer = StreamWriter(self.eeg_inlet, self.session_dir / "eeg.csv", EEG_HEADER)
        self.eeg_writer.start()
        if self.ppg_inlet is not None:
            self.ppg_writer = StreamWriter(self.ppg_inlet, self.session_dir / "ppg.csv", PPG_HEADER)
            self.ppg_writer.start()

        self.events_file = open(self.session_dir / "events.csv", "w", newline="")
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow(EVENTS_HEADER)
        self.events_file.flush()

        samplerate = 44100
        channels = 1
        self.audio_writer = AudioWriter(
            self.session_dir / "speech.wav",
            self.session_dir / "audio_levels.csv",
            samplerate,
            channels,
            self.start_time,
        )
        self.audio_writer.start()

        def callback(indata, frames, callback_time, status):
            if status:
                print(status, file=sys.stderr)
            self.audio_writer.push(indata, time.time())

        self.audio_stream = sd.InputStream(
            samplerate=samplerate,
            device=self.audio_device_id,
            channels=channels,
            dtype="float32",
            callback=callback,
        )
        self.audio_stream.start()

        self.start_btn.setText("STOP SESSION")
        self.start_btn.setStyleSheet(
            "background:#8a2430; color:#fff; padding:12px; border:none; "
            "border-radius:5px; font-size:13pt; font-weight:700;"
        )
        self.mark_btn.setEnabled(True)
        self.path_label.setText(f"writing to: {self.session_dir}")
        self.render_events()

    def stop_recording(self):
        elapsed = time.time() - self.start_time
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        if self.audio_writer is not None:
            self.audio_writer.stop()
            self.audio_writer.join(timeout=3)
        if self.eeg_writer is not None:
            self.eeg_writer.stop()
            self.eeg_writer.join(timeout=2)
        if self.ppg_writer is not None:
            self.ppg_writer.stop()
            self.ppg_writer.join(timeout=2)
        if self.events_file is not None:
            self.events_file.close()
            self.events_file = None

        meta = {
            "kind": "speech_marker",
            "start": datetime.fromtimestamp(self.start_time).isoformat(),
            "end": datetime.now().isoformat(),
            "duration_s": round(elapsed, 2),
            "eeg_samples": self.eeg_writer.sample_count if self.eeg_writer else 0,
            "ppg_samples": self.ppg_writer.sample_count if self.ppg_writer else 0,
            "audio_samples": self.audio_writer.frame_count if self.audio_writer else 0,
            "audio_samplerate": self.audio_writer.samplerate if self.audio_writer else None,
            "audio_device": self.audio_device_info["name"] if self.audio_device_info else None,
            "events": len(self.events),
            "reading_text": self.reading.toPlainText(),
        }
        with open(self.session_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        saved_dir = self.session_dir
        self.start_time = None
        self.eeg_writer = None
        self.ppg_writer = None
        self.audio_writer = None
        self.start_btn.setText("START SESSION")
        self.start_btn.setStyleSheet(
            "background:#225a3a; color:#fff; padding:12px; border:none; "
            "border-radius:5px; font-size:13pt; font-weight:700;"
        )
        self.mark_btn.setEnabled(False)
        self.path_label.setText(f"saved session: {saved_dir}")

    def mark_event(self, label):
        if self.start_time is None or self.events_writer is None:
            return
        timestamp = time.time()
        elapsed = timestamp - self.start_time
        self.events_writer.writerow([f"{timestamp:.6f}", f"{elapsed:.3f}", label])
        self.events_file.flush()
        self.events.append((elapsed, label))
        self.render_events()

    def render_events(self):
        if not self.events:
            self.events_log.setText("Press Space the moment a stutter, block, or repetition happens.")
            return
        rows = []
        for elapsed, label in self.events[-8:]:
            mm, ss = divmod(int(elapsed), 60)
            rows.append(f"<span style='color:#bbb'>{mm:02d}:{ss:02d}</span> "
                        f"<span style='color:#ff8fab'>{label}</span>")
        self.events_log.setText(f"{len(self.events)} events  |  " + "  ".join(rows))

    def refresh_ui(self):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        mm, ss = divmod(int(elapsed), 60)
        self.elapsed.setText(f"{mm:02d}:{ss:02d}")
        eeg_n = self.eeg_writer.sample_count if self.eeg_writer else 0
        ppg_n = self.ppg_writer.sample_count if self.ppg_writer else 0
        audio_s = (
            self.audio_writer.frame_count / self.audio_writer.samplerate
            if self.audio_writer and self.audio_writer.samplerate else 0.0
        )
        self.counts.setText(
            f"EEG: {eeg_n:,}  PPG: {ppg_n:,}  audio: {audio_s:.1f}s  "
            f"events: {len(self.events)}"
        )


def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = SpeechMarker(args.audio_device)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
