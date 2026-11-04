
import numpy as np, wfdb
from scipy.signal import butter, sosfiltfilt

FS = 360
AAMI = {**dict.fromkeys(['N','L','R','e','j'], 'N'),
        **dict.fromkeys(['A','a','J','S'], 'S'),
        **dict.fromkeys(['V','E'], 'V'),
        'F': 'F',
        **dict.fromkeys(['/','f','Q'], 'Q')}
CLASSES = ['N', 'S', 'V', 'F', 'Q']

DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

def bandpass(x, fs=FS, lo=0.5, hi=40, order=3):
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def preprocess(x, fs=FS):
    return zscore(bandpass(x, fs))

def process_record(rec, path, pre=90, post=110, ch=0):
    r = wfdb.rdrecord(f'{path}/{rec}')
    a = wfdb.rdann(f'{path}/{rec}', 'atr')
    sig = preprocess(r.p_signal[:, ch], r.fs)

    keep = [(s, AAMI[sym]) for s, sym in zip(a.sample, a.symbol) if sym in AAMI]
    pos = np.array([s for s, _ in keep]); lab = [l for _, l in keep]

    beats, y, rr = [], [], []
    for i in range(1, len(pos) - 1):
        if pos[i] - pre < 0 or pos[i] + post > len(sig):
            continue
        pre_rr, post_rr = pos[i] - pos[i-1], pos[i+1] - pos[i]
        local = np.mean(np.diff(pos[max(0, i-10):i+1]))
        beats.append(sig[pos[i]-pre : pos[i]+post])
        rr.append([pre_rr/r.fs, post_rr/r.fs, pre_rr/post_rr, local/r.fs])
        y.append(CLASSES.index(lab[i]))
    return np.array(beats), np.array(rr), np.array(y)

def build_split(records, path):
    B, R, Y, ID = [], [], [], []
    for rec in records:
        b, rr, y = process_record(str(rec), path)
        B.append(b); R.append(rr); Y.append(y); ID.append(np.full(len(y), rec))
    return (np.concatenate(B), np.concatenate(R),
            np.concatenate(Y), np.concatenate(ID))
