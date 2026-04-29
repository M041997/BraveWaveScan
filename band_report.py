"""Sample 5s, report relative band power per channel + relaxation index."""
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, sosfiltfilt, welch
import time

CHANNELS = ["TP9 ", "AF7 ", "AF8 ", "TP10"]
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("beta", 13, 30), ("gamma", 30, 45)]

inlet = StreamInlet(resolve_byprop("type", "EEG", timeout=5)[0])
srate = int(inlet.info().nominal_srate())
sos = butter(4, (1.0, 45.0), btype="bandpass", fs=srate, output="sos")

samples = []
t0 = time.time()
while time.time() - t0 < 5:
    chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=256)
    if chunk: samples.extend(chunk)
arr = np.asarray(samples).T[:4]
filt = sosfiltfilt(sos, arr, axis=1)
f, psd = welch(filt, fs=srate, nperseg=512, axis=1)

# (n_ch, n_bands) absolute power
P = np.zeros((4, len(BANDS)))
for j, (_, lo, hi) in enumerate(BANDS):
    m = (f >= lo) & (f < hi)
    P[:, j] = np.trapezoid(psd[:, m], f[m], axis=1)

# relative band power per channel (each row sums to 100%)
rel = 100 * P / P.sum(axis=1, keepdims=True)

print(f"\nrelative band power (% of channel total)")
print(f"{'ch':<6}" + "".join(f"{n:>8}" for n, _, _ in BANDS))
for i, name in enumerate(CHANNELS):
    print(f"{name:<6}" + "".join(f"{rel[i,j]:>7.1f}%" for j in range(len(BANDS))))

# alpha/beta ratio: classic "relaxation" indicator (higher = more relaxed)
ab = P[:, 2] / (P[:, 3] + 1e-12)
print(f"\nalpha/beta ratio (relaxation indicator, higher = more relaxed):")
for i, name in enumerate(CHANNELS):
    bar = "█" * int(min(ab[i], 5) * 6)
    print(f"  {name}: {ab[i]:>5.2f}  {bar}")

# left/right asymmetry on alpha
back_lr = P[0, 2] / (P[3, 2] + 1e-12)  # TP9 alpha / TP10 alpha
front_lr = P[1, 2] / (P[2, 2] + 1e-12)  # AF7 alpha / AF8 alpha
print(f"\nL/R alpha asymmetry (1.0 = symmetric):")
print(f"  back  (TP9/TP10):  {back_lr:.2f}")
print(f"  front (AF7/AF8):   {front_lr:.2f}")
