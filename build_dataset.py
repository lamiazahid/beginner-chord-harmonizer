import os, json, random
import numpy as np
import pandas as pd
import pretty_midi
from vocab import *

BASE = "POP909-Dataset/POP909"
BARS_PER_WINDOW = 8
STRIDE_BARS = 4
SLOTS_PER_WINDOW = BARS_PER_WINDOW * 2
MIN_NOTES = 8
MAX_NOTES = 192
MAX_NOCHORD_FRAC = 0.5


def key_at(keys, t):
    for s, e, k in keys:
        if s <= t < e:
            return k
    return min(keys, key=lambda x: min(abs(t - x[0]), abs(t - x[1])))[2]


def build_song(sid, split):
    b = pd.read_csv(f"{BASE}/{sid}/beat_midi.txt", sep=r"\s+", header=None, names=["t", "c2", "c3"])
    beat_t = b.t.values
    downs = np.where(b.c3.values == 1)[0]
    if len(downs) < BARS_PER_WINDOW + 1:
        return None, "too_short"

    midi = pretty_midi.PrettyMIDI(f"{BASE}/{sid}/{sid}.mid")
    mel = next((i for i in midi.instruments if i.name == "MELODY"), None)
    if mel is None:
        return None, "no_melody"
    notes = sorted(mel.notes, key=lambda n: n.start)

    ch = pd.read_csv(f"{BASE}/{sid}/chord_midi.txt", sep="\t", header=None, names=["s", "e", "c"])
    cs, ce, cl = ch.s.values, ch.e.values, ch.c.values

    kd = pd.read_csv(f"{BASE}/{sid}/key_audio.txt", sep="\t", header=None, names=["s", "e", "k"])
    keys = list(zip(kd.s, kd.e, kd.k))

    rows = np.arange(len(beat_t))
    out = []
    sb = 0
    while sb + BARS_PER_WINDOW < len(downs):
        if not np.all(np.diff(downs[sb:sb + BARS_PER_WINDOW + 1]) == 4):
            sb += 1
            continue
        start = sb
        sb += STRIDE_BARS
        row0 = downs[start]
        t0, t1 = beat_t[row0], beat_t[downs[start + BARS_PER_WINDOW]]
        tonic, mode = parse_key(key_at(keys, t0))

        # best capo for this window's key, so Tier-1-shape constraints
        # can be reconstructed later from (tonic, mode, capo) alone
        capo, _ = best_capo(tonic, mode)

        mtoks = []
        for n in notes:
            if not (t0 <= n.start < t1):
                continue
            b0 = np.interp(n.start, beat_t, rows) - row0
            b1 = np.interp(n.end, beat_t, rows) - row0
            t16 = min(int(round(b0 * 4)), 127)
            dur = max(0, min(int(round((b1 - b0) * 4)), 16))
            mtoks.append([(n.pitch - tonic) % 12, n.pitch // 12, dur, t16])
        if len(mtoks) < MIN_NOTES or len(mtoks) > MAX_NOTES:
            continue

        ctoks = []
        for s in range(SLOTS_PER_WINDOW):
            a, z = beat_t[row0 + 2 * s], beat_t[row0 + 2 * s + 2]
            ov = np.minimum(ce, z) - np.maximum(cs, a)
            i = int(np.argmax(ov))
            tok = chord_to_token(cl[i], tonic) if ov[i] > 0 else NOCHORD
            ctoks.append(NOCHORD if tok is None else tok)
        if sum(t == NOCHORD for t in ctoks) / len(ctoks) > MAX_NOCHORD_FRAC:
            continue

        out.append({"song": sid, "split": split, "tonic": int(tonic), "mode": 0 if mode == "maj" else 1,
                    "capo": int(capo), "notes": mtoks, "chords": ctoks})
    return out, "ok"


def main():
    sids = sorted(s for s in os.listdir(BASE) if s.isdigit())

    groups = [[s] for s in sids if s not in ("017", "018")] + [["017", "018"]]
    random.Random(0).shuffle(groups)
    n_test = n_val = int(0.1 * len(groups))
    split_of = {}
    for i, g in enumerate(groups):
        sp = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for s in g:
            split_of[s] = sp

    data, reasons = [], {}
    for k, sid in enumerate(sids):
        ex, why = build_song(sid, split_of[sid])
        reasons[why] = reasons.get(why, 0) + 1
        if ex:
            data.extend(ex)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(sids)} songs done")

    with open("dataset.json", "w") as f:
        json.dump(data, f)

    print("\nsongs:", reasons)
    print("windows total:", len(data))
    for sp in ["train", "val", "test"]:
        print(f"  {sp}: {sum(d['split'] == sp for d in data)} windows")
    nn = [len(d["notes"]) for d in data]
    print("notes per window: mean", round(float(np.mean(nn)), 1), "max", max(nn))
    print("minor-key windows:", f"{np.mean([d['mode'] for d in data]):.1%}")
    from collections import Counter
    cc = Counter(d["capo"] for d in data)
    print("capo distribution across windows:", dict(sorted(cc.items())))
    print("saved dataset.json")


if __name__ == "__main__":
    main()