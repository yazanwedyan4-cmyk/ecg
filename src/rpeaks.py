
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from preprocessing import preprocess

def pan_tompkins(x, fs=360, k=0.3, first=0.1):
    """Simplified Pan-Tompkins: band-pass 5-15Hz, derivative^2, moving-window
    integration, then per-10s peak picking with a median-based threshold
    (robust to large PVCs)."""
    f = sosfiltfilt(butter(2, [5, 15], 'bandpass', fs=fs, output='sos'), x)
    w = int(0.15 * fs)
    mwi = np.convolve(np.gradient(f) ** 2, np.ones(w) / w, mode='same')
    cand = []
    for s in range(0, len(mwi), 10 * fs):
        seg = mwi[s:s + 10 * fs]
        p, pr = find_peaks(seg, distance=int(0.2 * fs), height=first * seg.max())
        if len(p):
            h = pr['peak_heights']
            cand += list(p[h >= k * np.median(h)] + s)
    xb, out, hw = preprocess(x, fs), [], int(0.1 * fs)
    for p in cand:
        lo, hi = max(0, p - hw), min(len(xb), p + hw)
        out.append(lo + np.argmax(np.abs(xb[lo:hi])))
    return np.unique(out)
