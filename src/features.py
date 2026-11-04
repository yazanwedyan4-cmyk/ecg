
import numpy as np
from scipy.stats import skew, kurtosis

FS = 360
R0 = 90   # موقع الـ R-peak داخل نافذة النبضة (pre=90)

def qrs_width(b):
    base = np.median(b)
    half = 0.5 * abs(b[R0] - base)
    dev = np.abs(b - base) >= half
    l = r = R0
    while l > 0 and dev[l - 1]: l -= 1
    while r < len(b) - 1 and dev[r + 1]: r += 1
    return (r - l + 1) / FS

def beat_features(b):
    base = np.mean(b[R0 - 30:R0 - 15])           # قبل الـ QRS (تقريب PR segment)
    st = np.mean(b[R0 + 30:R0 + 45]) - base      # بعد الـ QRS (تقريب ST)
    return [qrs_width(b), b[R0], st, b.std(), skew(b), kurtosis(b),
            b.max(), b.min()]

FEATURE_NAMES = ['pre_rr', 'post_rr', 'rr_ratio', 'local_rr',
                 'qrs_width', 'r_amp', 'st_dev', 'std', 'skew', 'kurt',
                 'max', 'min']

def build_features(beats, rr):
    F = np.array([beat_features(b) for b in beats])
    return np.hstack([rr, F])
