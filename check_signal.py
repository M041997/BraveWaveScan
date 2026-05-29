"""Quick signal-quality check: 5s sample, per-channel stats + EMG contamination.

Combines two heuristics so this catches both failure modes:
  - flatness / rail saturation  (no contact or full electrode lift-off)
  - HF/LF power ratio           (facial-muscle EMG bleeding into beta)
"""
import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch

CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]
HF_LF_GOOD = 0.7
HF_LF_BAD = 2.5

streams = resolve_byprop("type", "EEG", timeout=5)
inlet = StreamInlet(streams[0])
srate = int(inlet.info().nominal_srate())
print(f"sampling 5s @ {srate} Hz...")
samples = []
t0 = time.time()
while time.time() - t0 < 5:
    chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=256)
    if chunk:
        samples.extend(chunk)
arr = np.asarray(samples).T  # (n_ch, n_samp)
print(f"got {arr.shape[1]} samples ({arr.shape[1]/srate:.1f}s)\n")

print(f"{'ch':<6}{'mean':>9}{'std':>8}{'%>±100':>9}{'HF/LF':>9}  verdict")
for i, name in enumerate(CHANNELS[:arr.shape[0]]):
    ch = arr[i]
    pct_rail = 100 * (np.abs(ch) > 100).mean()
    f, psd = welch(ch, fs=srate, nperseg=min(512, len(ch)))
    lf_m = (f >= 1) & (f < 15)
    hf_m = (f >= 15) & (f < 45)
    lf = float(np.trapezoid(psd[lf_m], f[lf_m]))
    hf = float(np.trapezoid(psd[hf_m], f[hf_m]))
    hf_lf = hf / max(lf, 1e-12)
    # ordered verdicts: hardest failures first
    if np.std(ch) < 1.5:
        verdict = "FLAT (no contact)"
    elif pct_rail > 30:
        verdict = "RAILED (bad contact)"
    elif hf_lf > HF_LF_BAD:
        verdict = "EMG-CONTAMINATED (reseat)"
    elif pct_rail > 5:
        verdict = "noisy"
    elif hf_lf > HF_LF_GOOD:
        verdict = "marginal EMG"
    elif np.std(ch) > 80:
        verdict = "very noisy"
    else:
        verdict = "OK"
    print(f"{name:<6}{np.mean(ch):>9.1f}{np.std(ch):>8.1f}"
          f"{pct_rail:>8.1f}%{hf_lf:>9.2f}  {verdict}")
