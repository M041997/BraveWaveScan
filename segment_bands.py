#!/usr/bin/env python3
"""Sanity-check a labeled session: compute relative band power per (segment, channel)
and render a grouped bar chart. Eyes-closed should show a big alpha bump on posterior
(TP9/TP10) channels — that's the "Berger effect" and is the classic sanity check
that the EEG signal is real and electrodes are placed correctly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

CHANNELS = ["tp9", "af7", "af8", "tp10"]
CHANNEL_LABELS = ["TP9 (left ear)", "AF7 (left fwd)", "AF8 (right fwd)", "TP10 (right ear)"]
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("beta", 13, 30), ("gamma", 30, 45)]
BAND_COLORS = ["#3320aa", "#1c83c8", "#1ec38c", "#f2b020", "#e63838"]

SR = 256  # Muse 2 nominal


def session_dir():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    sessions = sorted((Path.home() / "eeg-muse/sessions").iterdir())
    return sessions[-1]


def main():
    d = session_dir()
    print(f"session: {d}")
    meta = json.load(open(d / "meta.json"))
    df = pd.read_csv(d / "eeg.csv")
    t0 = df["timestamp"].iloc[0]
    df["t_s"] = df["timestamp"] - t0  # seconds from start

    sos = butter(4, (1.0, 45.0), btype="bandpass", fs=SR, output="sos")

    rows = []  # one row per (segment, channel, band)
    for seg in meta["segments"]:
        s, e = seg["start_s"], seg["end_s"]
        sub = df[(df["t_s"] >= s) & (df["t_s"] < e)]
        if len(sub) < SR * 2:
            print(f"  {seg['label']:>20}  too short, skipped")
            continue
        x = sub[CHANNELS].to_numpy(dtype=np.float32).T  # (4, N)
        x = sosfiltfilt(sos, x, axis=1)
        f, psd = welch(x, fs=SR, nperseg=512, axis=1)
        for ch_i, ch in enumerate(CHANNELS):
            total = float(np.trapezoid(psd[ch_i, (f >= 1) & (f < 45)],
                                       f[(f >= 1) & (f < 45)]))
            for b_name, lo, hi in BANDS:
                m = (f >= lo) & (f < hi)
                p = float(np.trapezoid(psd[ch_i, m], f[m]))
                rows.append({
                    "segment": seg["label"],
                    "channel": ch,
                    "band": b_name,
                    "abs_uv2": p,
                    "rel_pct": 100.0 * p / max(total, 1e-12),
                })
        print(f"  {seg['label']:>20}  n={len(sub):>6}  ({(e-s):.0f}s)")

    R = pd.DataFrame(rows)

    # printed summary: relative alpha per (channel, segment) — the Berger check
    print("\nrelative alpha % (eyes_closed should be highest, esp. on TP9/TP10):")
    pivot = (R[R["band"] == "alpha"]
             .pivot(index="segment", columns="channel", values="rel_pct")
             .reindex([s["label"] for s in meta["segments"]]))
    print(pivot.round(1).to_string())

    # also: beta % per segment (the focus_balloon driver)
    print("\nrelative beta % (frontal AF7/AF8 — balloon_popping should be highest):")
    pivot_b = (R[R["band"] == "beta"]
               .pivot(index="segment", columns="channel", values="rel_pct")
               .reindex([s["label"] for s in meta["segments"]]))
    print(pivot_b.round(1).to_string())

    # plot: 4 subplots (one per channel), grouped bars (segment x band)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    seg_labels = [s["label"] for s in meta["segments"]]
    n_seg = len(seg_labels)
    n_band = len(BANDS)
    width = 0.85 / n_band
    x_seg = np.arange(n_seg)
    for ch_i, ax in enumerate(axes.flat):
        ch = CHANNELS[ch_i]
        for b_i, (b_name, _, _) in enumerate(BANDS):
            heights = []
            for seg in seg_labels:
                row = R[(R["segment"] == seg) & (R["channel"] == ch) &
                        (R["band"] == b_name)]
                heights.append(row["rel_pct"].iloc[0] if len(row) else 0)
            ax.bar(x_seg + (b_i - n_band/2) * width + width/2, heights,
                   width=width, color=BAND_COLORS[b_i], label=b_name)
        ax.set_title(CHANNEL_LABELS[ch_i])
        ax.set_xticks(x_seg)
        ax.set_xticklabels(seg_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("% of band power 1-45Hz")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(ncol=5, loc="upper center",
                      bbox_to_anchor=(1.0, 1.18), frameon=False)
    fig.suptitle(f"Band power per segment — {d.name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = d / "band_summary.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved: {out}")


if __name__ == "__main__":
    main()
