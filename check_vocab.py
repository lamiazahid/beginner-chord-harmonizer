import os
import pandas as pd
from collections import Counter
from vocab import *

base = "POP909-Dataset/POP909"
modes = Counter()
unknown = Counter()
n_multi_key = 0
in_tier1 = 0
total = 0

for sid in sorted(s for s in os.listdir(base) if s.isdigit()):
    k = pd.read_csv(f"{base}/{sid}/key_audio.txt", sep="\t", header=None,
                    names=["s", "e", "k"])
    if len(k) > 1:
        n_multi_key += 1
    # use the key that lasts longest in the song
    k["dur"] = k.e - k.s
    key = k.sort_values("dur", ascending=False).iloc[0].k
    tonic, mode = parse_key(key)
    modes[mode] += 1

    ch = pd.read_csv(f"{base}/{sid}/chord_midi.txt", sep="\t", header=None,
                     names=["s", "e", "c"])
    for c in ch.c:
        t = chord_to_token(c, tonic)
        if t is None:
            unknown[c] += 1
            continue
        if t == NOCHORD:
            continue
        total += 1
        if token_to_chord(t) in TIER1[mode]:
            in_tier1 += 1

print("key modes:", dict(modes))
print("songs with more than one key segment:", n_multi_key)
print("unknown chord qualities:", dict(unknown))
print("chord vocab size:", CHORD_VOCAB_SIZE)
print(f"chords inside Tier 1: {in_tier1}/{total} = {in_tier1/total:.1%}")

