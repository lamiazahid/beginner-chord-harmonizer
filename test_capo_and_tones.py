from vocab import *

# --- Sanity check 1: reproduce the C -> E example exactly ---
tonic_pc = NOTES.index("C")  # song in C major, tonic = C
melody_pcs = {NOTES.index("C"), NOTES.index("E")}

print("Chord-tone overlap for a C->E melody fragment, key of C major:")
for rel_root, quality in TIER1["maj"]:
    score = chord_tone_overlap(tonic_pc, rel_root, quality, melody_pcs)
    abs_root = NOTES[(tonic_pc + rel_root) % 12]
    print(f"  {abs_root:3} {quality:3}  overlap = {score:.2f}")

# --- Sanity check 2: best capo for a few real keys ---
print("\nBest capo per key (maj):")
for name in ["C", "C#", "D", "F#", "G"]:
    tonic_pc = NOTES.index(name)
    k, playable = best_capo(tonic_pc, "maj")
    print(f"  {name:3} maj  ->  capo {k}, {len(playable)}/7 diatonic chords playable")