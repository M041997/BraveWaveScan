#!/usr/bin/env python3
"""Post-session speech review: transcribe and mark speech-event candidates."""
import argparse
import csv
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

SESSIONS_DIR = Path.home() / "eeg-muse" / "sessions"
EEG_CHANNELS = ["tp9", "af7", "af8", "tp10"]
FILLERS = {
    "um", "uh", "erm", "hmm", "mm", "ah", "like", "you know", "i mean",
    "well", "so",
}
BANDS = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]


def latest_speech_session():
    sessions = sorted(SESSIONS_DIR.glob("*_speech"))
    if not sessions:
        raise SystemExit(f"No speech sessions found in {SESSIONS_DIR}")
    return sessions[-1]


def load_levels(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "timestamp": float(row["timestamp"]),
                "elapsed_s": float(row["elapsed_s"]),
                "rms": float(row["rms"]),
                "peak": float(row["peak"]),
            })
    return rows


def load_events(path):
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "source": "manual",
                "label": row["label"],
                "start_s": float(row["elapsed_s"]),
                "end_s": float(row["elapsed_s"]),
                "text": row["label"],
            })
    return rows


def load_eeg(path, session_start):
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            timestamp = float(row["timestamp"])
            rows.append([
                timestamp - session_start,
                float(row["tp9"]),
                float(row["af7"]),
                float(row["af8"]),
                float(row["tp10"]),
            ])
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def session_start_epoch(session):
    meta_path = session / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if "start" in meta:
            return datetime.fromisoformat(meta["start"]).timestamp()
    eeg_path = session / "eeg.csv"
    if eeg_path.exists():
        with open(eeg_path, newline="") as f:
            first = next(csv.DictReader(f), None)
        if first:
            return float(first["timestamp"])
    return 0.0


def detect_pauses(levels, min_pause_s=0.55):
    if not levels:
        return [], 0.0
    rms = np.array([row["rms"] for row in levels], dtype=np.float64)
    floor = float(np.percentile(rms, 20))
    active = float(np.percentile(rms, 90))
    threshold = max(floor * 2.5, active * 0.08, 0.003)

    pauses = []
    start = None
    for row in levels:
        quiet = row["rms"] < threshold
        if quiet and start is None:
            start = row
        elif not quiet and start is not None:
            duration = row["elapsed_s"] - start["elapsed_s"]
            if duration >= min_pause_s:
                pauses.append((start["elapsed_s"], row["elapsed_s"], duration))
            start = None
    if start is not None:
        end = levels[-1]
        duration = end["elapsed_s"] - start["elapsed_s"]
        if duration >= min_pause_s:
            pauses.append((start["elapsed_s"], end["elapsed_s"], duration))
    return pauses, threshold


def normalize_words(text):
    return re.findall(r"[a-z']+", text.lower())


def word_spans(row):
    matches = list(re.finditer(r"[A-Za-z']+", row["text"]))
    if not matches:
        return []
    duration = max(row["end"] - row["start"], 0.001)
    text_len = max(len(row["text"]), 1)
    spans = []
    for match in matches:
        start = row["start"] + duration * (match.start() / text_len)
        end = row["start"] + duration * (match.end() / text_len)
        spans.append({
            "word": match.group(0).lower(),
            "start": start,
            "end": end,
            "index": len(spans),
        })
    return spans


def all_word_spans(transcript):
    spans = []
    for row in transcript:
        spans.extend(word_spans(row))
    return spans


def detect_repetitions(transcript):
    candidates = []
    spans = all_word_spans(transcript)
    if len(spans) < 2:
        return candidates

    stop_repeats = {"i", "a", "the", "to", "and", "or", "but"}
    for i in range(1, len(spans)):
        prev = spans[i - 1]
        cur = spans[i]
        gap = cur["start"] - prev["end"]
        if cur["word"] == prev["word"] and len(cur["word"]) > 1:
            label = "nearby_repetition_candidate" if gap > 0.35 else "repetition_candidate"
            if label == "nearby_repetition_candidate" and cur["word"] in stop_repeats:
                continue
            if gap <= 1.5:
                candidates.append({
                    "source": "transcript",
                    "label": label,
                    "start_s": prev["start"],
                    "end_s": cur["end"],
                    "text": f"{prev['word']} {cur['word']}",
                })

    words = [span["word"] for span in spans]
    for n in (2, 3, 4):
        for i in range(n, len(words) - n + 1):
            if words[i - n:i] == words[i:i + n]:
                candidates.append({
                    "source": "transcript",
                    "label": "phrase_repetition_candidate",
                    "start_s": spans[i - n]["start"],
                    "end_s": spans[i + n - 1]["end"],
                    "text": " ".join(words[i:i + n]),
                })
    return candidates


def detect_fillers(transcript):
    candidates = []
    for row in transcript:
        spans = word_spans(row)
        words = [span["word"] for span in spans]
        for i, span in enumerate(spans):
            if span["word"] in FILLERS:
                candidates.append({
                    "source": "transcript",
                    "label": "filler_candidate",
                    "start_s": span["start"],
                    "end_s": span["end"],
                    "text": span["word"],
                })
            if i + 1 < len(words):
                bigram = f"{words[i]} {words[i + 1]}"
                if bigram in FILLERS:
                    candidates.append({
                        "source": "transcript",
                        "label": "filler_candidate",
                        "start_s": spans[i]["start"],
                        "end_s": spans[i + 1]["end"],
                        "text": bigram,
                    })
    return candidates


def detect_restarts(transcript):
    candidates = []
    spans = all_word_spans(transcript)
    words = [span["word"] for span in spans]
    for n in (2, 3, 4):
        for i in range(len(words) - n):
            phrase = words[i:i + n]
            if len(set(phrase) & {"i", "a", "the", "to", "and"}) == len(set(phrase)):
                continue
            for j in range(i + n, min(len(words) - n + 1, i + n + 8)):
                if phrase == words[j:j + n]:
                    candidates.append({
                        "source": "transcript",
                        "label": "restart_revision_candidate",
                        "start_s": spans[i]["start"],
                        "end_s": spans[j + n - 1]["end"],
                        "text": f"{' '.join(phrase)} ... {' '.join(words[j:j + n])}",
                    })
                    break
    return candidates


def detect_possible_blocks(pauses, transcript):
    candidates = []
    if len(transcript) < 2:
        return candidates
    for pause_start, pause_end, duration in pauses:
        if duration < 0.7:
            continue
        prev_row = None
        next_row = None
        for row in transcript:
            if row["end"] <= pause_start and pause_start - row["end"] <= 0.8:
                prev_row = row
            if row["start"] >= pause_end and row["start"] - pause_end <= 0.8:
                next_row = row
                break
        if prev_row is None or next_row is None:
            continue
        prev_text = prev_row["text"].strip()
        next_text = next_row["text"].strip()
        prev_open = not prev_text.endswith((".", "?", "!"))
        next_lower = next_text[:1].islower() or normalize_words(next_text)[:1] in (["and"], ["but"], ["or"], ["so"])
        if prev_open or next_lower:
            candidates.append({
                "source": "audio",
                "label": "possible_block_candidate",
                "start_s": pause_start,
                "end_s": pause_end,
                "text": f"silent gap inside thought: {duration:.2f}s",
            })
    return candidates


def detect_prolongations(transcript):
    candidates = []
    repeated_letter = re.compile(r"\b[a-z]*([a-z])\1{2,}[a-z]*\b", re.I)
    hyphenated_sound = re.compile(r"\b([a-z]{1,3}-){1,}[a-z]+\b", re.I)
    for row in transcript:
        matches = repeated_letter.findall(row["text"]) or hyphenated_sound.findall(row["text"])
        if not matches:
            continue
        candidates.append({
            "source": "transcript",
            "label": "prolongation_candidate",
            "start_s": row["start"],
            "end_s": row["end"],
            "text": row["text"],
        })
    return candidates


def estimate_srate(times):
    if len(times) < 3:
        return 256.0
    dt = np.diff(times)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 256.0
    return float(1.0 / np.median(dt))


def eeg_features(eeg, start_s, end_s, pad_s=0.75):
    features = {}
    if eeg is None:
        return features

    lo = max(0.0, start_s - pad_s)
    hi = max(lo + 0.1, end_s + pad_s)
    mask = (eeg[:, 0] >= lo) & (eeg[:, 0] <= hi)
    seg = eeg[mask]
    if len(seg) < 16:
        return features

    fs = estimate_srate(seg[:, 0])
    samples = seg[:, 1:5].T
    features["eeg_window_start_s"] = lo
    features["eeg_window_end_s"] = hi
    features["eeg_samples"] = len(seg)

    freqs, psd = welch(samples, fs=fs, nperseg=min(256, samples.shape[1]), axis=1)
    lf_mask = (freqs >= 1) & (freqs < 15)
    hf_mask = (freqs >= 15) & (freqs < 45)

    for idx, channel in enumerate(EEG_CHANNELS):
        signal = samples[idx]
        features[f"{channel}_mean"] = float(np.mean(signal))
        features[f"{channel}_std"] = float(np.std(signal))
        features[f"{channel}_ptp"] = float(np.ptp(signal))
        lf = float(np.trapezoid(psd[idx, lf_mask], freqs[lf_mask]))
        hf = float(np.trapezoid(psd[idx, hf_mask], freqs[hf_mask]))
        features[f"{channel}_hf_lf"] = hf / max(lf, 1e-12)
        for band, band_lo, band_hi in BANDS:
            band_mask = (freqs >= band_lo) & (freqs < band_hi)
            power = float(np.trapezoid(psd[idx, band_mask], freqs[band_mask]))
            features[f"{channel}_{band}_db"] = 10 * np.log10(max(power, 1e-12))
    return features


def write_brain_marks(session, marks, eeg):
    feature_names = [
        "eeg_window_start_s", "eeg_window_end_s", "eeg_samples",
    ]
    for channel in EEG_CHANNELS:
        feature_names.extend([
            f"{channel}_mean",
            f"{channel}_std",
            f"{channel}_ptp",
            f"{channel}_hf_lf",
        ])
        feature_names.extend([f"{channel}_{band}_db" for band, _, _ in BANDS])

    path = session / "brain_scan_marks.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source", "label", "start_s", "end_s", "duration_s", "text",
                *feature_names,
            ],
        )
        writer.writeheader()
        for mark in marks:
            features = eeg_features(eeg, mark["start_s"], mark["end_s"])
            row = {
                "source": mark["source"],
                "label": mark["label"],
                "start_s": f"{mark['start_s']:.3f}",
                "end_s": f"{mark['end_s']:.3f}",
                "duration_s": f"{max(0.0, mark['end_s'] - mark['start_s']):.3f}",
                "text": mark.get("text", ""),
            }
            for name in feature_names:
                value = features.get(name, "")
                row[name] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(row)
    return path


def transcribe_with_faster_whisper(wav_path, model_name):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav_path), vad_filter=True, word_timestamps=True)
    rows = []
    for segment in segments:
        rows.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        })
    return rows, f"faster-whisper:{model_name}"


def transcribe_with_whisper(wav_path, model_name):
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(str(wav_path), fp16=False)
    rows = []
    for segment in result.get("segments", []):
        rows.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment["text"].strip(),
        })
    return rows, f"whisper:{model_name}"


def transcribe(wav_path, model_name):
    if importlib.util.find_spec("faster_whisper") is not None:
        return transcribe_with_faster_whisper(wav_path, model_name)
    if importlib.util.find_spec("whisper") is not None:
        return transcribe_with_whisper(wav_path, model_name)
    return [], "none"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Review a speech marker session.")
    parser.add_argument("session", nargs="?", help="Session directory; defaults to latest *_speech session.")
    parser.add_argument("--model", default="base", help="Whisper model name if whisper/faster-whisper is installed.")
    parser.add_argument("--min-pause", type=float, default=0.55, help="Minimum quiet gap to mark as a pause.")
    args = parser.parse_args()

    session = Path(args.session).expanduser() if args.session else latest_speech_session()
    wav_path = session / "speech.wav"
    levels_path = session / "audio_levels.csv"
    if not wav_path.exists():
        raise SystemExit(f"Missing {wav_path}")
    if not levels_path.exists():
        raise SystemExit(f"Missing {levels_path}")

    audio_info = sf.info(wav_path)
    levels = load_levels(levels_path)
    pauses, threshold = detect_pauses(levels, min_pause_s=args.min_pause)
    write_csv(
        session / "pause_candidates.csv",
        ["start_s", "end_s", "duration_s"],
        [[f"{s:.3f}", f"{e:.3f}", f"{d:.3f}"] for s, e, d in pauses],
    )

    transcript, engine = transcribe(wav_path, args.model)
    if transcript:
        write_csv(
            session / "transcript.csv",
            ["start_s", "end_s", "text"],
            [[f"{r['start']:.3f}", f"{r['end']:.3f}", r["text"]] for r in transcript],
        )
    repetition_candidates = detect_repetitions(transcript)
    filler_candidates = detect_fillers(transcript)
    restart_candidates = detect_restarts(transcript)
    possible_block_candidates = detect_possible_blocks(pauses, transcript)
    prolongation_candidates = detect_prolongations(transcript)
    speech_event_candidates = [
        *repetition_candidates,
        *restart_candidates,
        *filler_candidates,
        *possible_block_candidates,
        *prolongation_candidates,
    ]
    speech_event_candidates.sort(key=lambda row: (row["start_s"], row["end_s"], row["label"]))
    write_csv(
        session / "speech_event_candidates.csv",
        ["label", "start_s", "end_s", "duration_s", "text"],
        [[
            event["label"],
            f"{event['start_s']:.3f}",
            f"{event['end_s']:.3f}",
            f"{max(0.0, event['end_s'] - event['start_s']):.3f}",
            event["text"],
        ] for event in speech_event_candidates],
    )

    marks = []
    marks.extend(load_events(session / "events.csv"))
    marks.extend({
        "source": "audio",
        "label": "pause_candidate",
        "start_s": start,
        "end_s": end,
        "text": f"quiet pause {duration:.2f}s",
    } for start, end, duration in pauses)
    marks.extend({
        "source": "transcript",
        "label": "speech_segment",
        "start_s": row["start"],
        "end_s": row["end"],
        "text": row["text"],
    } for row in transcript)
    marks.extend(speech_event_candidates)
    marks.sort(key=lambda row: (row["start_s"], row["end_s"], row["source"]))

    eeg = load_eeg(session / "eeg.csv", session_start_epoch(session))
    brain_marks_path = write_brain_marks(session, marks, eeg)

    review = [
        "# Speech Review",
        "",
        f"Session: `{session}`",
        f"Audio: {audio_info.duration:.2f}s at {audio_info.samplerate} Hz",
        f"Transcription engine: `{engine}`",
        f"Pause threshold RMS: `{threshold:.6f}`",
        f"Neutral pause candidates: `{len(pauses)}`",
        f"Repetition candidates: `{len(repetition_candidates)}`",
        f"Restart/revision candidates: `{len(restart_candidates)}`",
        f"Filler candidates: `{len(filler_candidates)}`",
        f"Possible block candidates: `{len(possible_block_candidates)}`",
        f"Prolongation candidates: `{len(prolongation_candidates)}`",
        f"Brain scan marks: `{brain_marks_path.name}`",
        "",
        "## Speech Event Candidates",
    ]
    if speech_event_candidates:
        for event in speech_event_candidates:
            review.append(
                f"- {event['label']} {event['start_s']:.2f}s - "
                f"{event['end_s']:.2f}s: {event['text']}")
    else:
        review.append("- none")
    review.extend([
        "",
        "## Neutral Pause Candidates",
    ]
    )
    if pauses:
        for start, end, duration in pauses[:50]:
            review.append(f"- {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
    else:
        review.append("- none")
    review.append("")
    review.append("## Transcript")
    if transcript:
        for row in transcript:
            review.append(f"- {row['start']:.2f}s - {row['end']:.2f}s: {row['text']}")
    else:
        review.append("- No local Whisper engine is installed, so only pause review was generated.")
    (session / "speech_review.md").write_text("\n".join(review) + "\n")

    print(f"Wrote {session / 'speech_review.md'}")
    print(f"Wrote {session / 'pause_candidates.csv'}")
    print(f"Wrote {session / 'speech_event_candidates.csv'}")
    print(f"Wrote {brain_marks_path}")
    if transcript:
        print(f"Wrote {session / 'transcript.csv'}")
    else:
        print("No local transcription engine found; install whisper or faster-whisper to add transcript.csv.")


if __name__ == "__main__":
    main()
