"""
Measures the real payoff of capo support: for every song, what fraction of
its ACTUAL chords (weighted by how often they occur, not just structurally)
are beginner-playable with no capo, vs. with the best capo.
"""
import os
import pandas as pd
from collections import Counter
from vocab import *

BASE = "POP909-Dataset/POP909"

no_capo_hits = 0
capo_hits = 0
total = 0
capo_used = Counter()

for sid in sorted(s for s in os.listdir(BASE) if s.isdigit()):
    kd = pd.read_csv(f"{BASE}/{sid}/key_audio.txt", sep="\t", header=None, names=["s","e","k"])
    kd["dur"] = kd.e - kd.s
    key = kd.sort_values("dur", ascending=False).iloc[0].k
    tonic, mode = parse_key(key)
    if mode not in TIER1:
        continue

    k, playable_with_capo = best_capo(tonic, mode)
    playable_no_capo = {rc for rc in TIER1[mode]
                         if (NOTES[(tonic + rc[0]) % 12], rc[1]) in EASY_SHAPES}
    capo_used[k] += 1

    ch = pd.read_csv(f"{BASE}/{sid}/chord_midi.txt", sep="\t", header=None, names=["s","e","c"])
    for c in ch.c:
        tok = chord_to_token(c, tonic)
        if tok is None or tok == NOCHORD:
            continue
        rc = token_to_chord(tok)
        total += 1
        if rc in playable_no_capo:
            no_capo_hits += 1
        if rc in playable_with_capo:
            capo_hits += 1

print(f"Total chord instances checked: {total}")
print(f"Playable at capo 0 only:        {no_capo_hits}/{total} = {no_capo_hits/total:.1%}")
print(f"Playable with best capo (0-7):  {capo_hits}/{total} = {capo_hits/total:.1%}")
print(f"\nCapo positions chosen across {sum(capo_used.values())} songs:")
for k in sorted(capo_used):
    print(f"  capo {k}: {capo_used[k]} songs")