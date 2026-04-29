"""Quick signal-quality check: 5s sample, per-channel stats."""
import numpy as np
from pylsl import StreamInlet, resolve_byprop

CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]
streams = resolve_byprop("type", "EEG", timeout=5)
inlet = StreamInlet(streams[0])
srate = int(inlet.info().nominal_srate())
samples = []
print(f"sampling 5s @ {srate} Hz...")
import time
t0 = time.time()
while time.time() - t0 < 5:
    chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=256)
    if chunk:
        samples.extend(chunk)
arr = np.asarray(samples).T  # (n_ch, n_samp)
print(f"got {arr.shape[1]} samples ({arr.shape[1]/srate:.1f}s)\n")
print(f"{'ch':<6}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'%>±100':>10}  verdict")
for i, name in enumerate(CHANNELS[:arr.shape[0]]):
    ch = arr[i]
    pct_rail = 100 * (np.abs(ch) > 100).mean()
    verdict = "OK"
    if np.std(ch) < 1.5:                    verdict = "FLAT (no contact)"
    elif pct_rail > 30:                     verdict = "RAILED (bad contact)"
    elif pct_rail > 5:                      verdict = "noisy"
    elif np.std(ch) > 80:                   verdict = "very noisy"
    print(f"{name:<6}{np.mean(ch):>10.1f}{np.std(ch):>10.1f}{np.min(ch):>10.1f}{np.max(ch):>10.1f}{pct_rail:>9.1f}%  {verdict}")
