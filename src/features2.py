import numpy as np, wfdb
from preprocessing import preprocess, AAMI
from features import beat_features

PRE, POST, R0 = 90, 110, 90

def _leads(r):
    names = [n.upper() for n in r.sig_name]
    a = names.index('MLII') if 'MLII' in names else 0
    return a, (1 - a if len(names) > 1 else a)

def process_record2(rec, path):
    r = wfdb.rdrecord(f'{path}/{rec}'); an = wfdb.rdann(f'{path}/{rec}', 'atr')
    ia, ib = _leads(r)
    sa = preprocess(r.p_signal[:, ia], r.fs); sb = preprocess(r.p_signal[:, ib], r.fs)
    keep = [(s, AAMI[sym]) for s, sym in zip(an.sample, an.symbol) if sym in AAMI]
    pos = np.array([s for s, _ in keep]); lab = [l for _, l in keep]
    rr_all = np.diff(pos)
    CL = ['N', 'S', 'V', 'F', 'Q']
    BA, BB, RR, Y = [], [], [], []
    for i in range(1, len(pos) - 1):
        if pos[i] - PRE < 0 or pos[i] + POST > len(sa):
            continue
        pre_rr, post_rr = rr_all[i-1], rr_all[i]
        local = np.mean(rr_all[max(0, i-10):i])
        hist = rr_all[max(0, i-21):i-1]            # causal: previous intervals only
        med = np.median(hist) if len(hist) else pre_rr
        RR.append([pre_rr/r.fs, post_rr/r.fs, pre_rr/post_rr, local/r.fs,
                   pre_rr/med, post_rr/med, local/med])
        BA.append(sa[pos[i]-PRE:pos[i]+POST]); BB.append(sb[pos[i]-PRE:pos[i]+POST])
        Y.append(CL.index(lab[i]))
    return np.array(BA), np.array(BB), np.array(RR), np.array(Y)

def build_split2(records, path):
    out = [process_record2(str(rec), path) for rec in records]
    return [np.concatenate([o[k] for o in out]) for k in range(4)]

def seg_feats(b):
    p, q, t = b[R0-60:R0-25], b[R0-15:R0+20], b[R0+25:R0+90]
    return [p.mean(), np.ptp(p), np.sum(q**2), t.mean(), np.ptp(t)]

def morph(B):
    return np.array([beat_features(b) + seg_feats(b) for b in B])
