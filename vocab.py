import re

# Note names -> pitch class numbers (C=0 ... B=11), including enharmonic spellings
NOTE_TO_PC = {"C":0,"B#":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"Fb":4,
              "E#":5,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,
              "Bb":10,"B":11,"Cb":11}

# Every chord type in the dataset gets folded into one of 4 basic qualities
QUALITIES = ["maj", "min", "dim", "aug"]
QUALITY_MAP = {
    "maj": "maj", "maj7": "maj", "maj6": "maj", "7": "maj", "sus2": "maj", "sus4": "maj",
    "min": "min", "min7": "min", "min6": "min", "minmaj7": "min",
    "dim": "dim", "dim7": "dim", "hdim7": "dim",
    "aug": "aug",
}

# Special tokens
PAD, BOS, EOS, NOCHORD = 0, 1, 2, 3
N_SPECIAL = 4
CHORD_VOCAB_SIZE = N_SPECIAL + 12 * len(QUALITIES)   # = 52


def parse_key(key_str):
    """'Gb:maj' -> (tonic pitch class, 'maj')"""
    root, mode = key_str.split(":")
    return NOTE_TO_PC[root], mode


def chord_to_token(raw, tonic_pc):
    """Raw label like 'F#:maj7/5' -> token ID, relative to the song's tonic."""
    if raw == "N" or ":" not in raw:
        return NOCHORD
    root, quality = raw.split("/")[0].split(":", 1)
    quality = re.sub(r"\(.*\)", "", quality)
    rel_root = (NOTE_TO_PC[root] - tonic_pc) % 12
    q = QUALITY_MAP.get(quality)
    if q is None:
        return None
    return N_SPECIAL + rel_root * len(QUALITIES) + QUALITIES.index(q)


def token_to_chord(tok):
    """Token ID -> (relative root, quality)."""
    if tok < N_SPECIAL:
        return ("special", tok)
    i = tok - N_SPECIAL
    return (i // len(QUALITIES), QUALITIES[i % len(QUALITIES)])


# Tier 1 = the 7 diatonic triads of each key, as (relative root, quality).
# This is the table your constrained decoding will use later.
TIER1 = {
    "maj": {(0,"maj"), (2,"min"), (4,"min"), (5,"maj"), (7,"maj"), (9,"min"), (11,"dim")},
    "min": {(0,"min"), (2,"dim"), (3,"maj"), (5,"min"), (7,"min"), (8,"maj"), (10,"maj")},
}
# ---------------------------------------------------------------------------
# Capo-aware beginner tier (Option B)
# ---------------------------------------------------------------------------

NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# The open/easy shapes your Music Society confirmed. No diminished or augmented
# shape exists in open position on guitar, so those are never reachable here,
# with or without a capo.
EASY_SHAPES = {
    ("C", "maj"), ("D", "maj"), ("D", "min"), ("A", "maj"),
    ("A", "min"), ("G", "maj"), ("E", "min"), ("E", "maj"),
}

MAX_CAPO = 7  # frets above this are impractical on most guitars


def best_capo(tonic_pc, mode):
    """
    For a song in the given key, search capo positions 0-7 and find the one
    that lets the MOST diatonic chords be played with an easy open shape.
    Returns (capo_fret, set of (rel_root, quality) diatonic chords playable at that capo).
    """
    diatonic = TIER1[mode]  # existing table: {(rel_root, quality), ...}
    best_k, best_set = 0, set()
    for k in range(MAX_CAPO + 1):
        playable = set()
        for rel_root, quality in diatonic:
            abs_root_pc = (tonic_pc + rel_root) % 12
            shape_pc = (abs_root_pc - k) % 12          # the shape you'd actually play
            if (NOTES[shape_pc], quality) in EASY_SHAPES:
                playable.add((rel_root, quality))
        if len(playable) > len(best_set):
            best_k, best_set = k, playable
    return best_k, best_set


def triad_pitch_classes(tonic_pc, rel_root, quality):
    """Absolute pitch classes (0-11) in a chord's triad, given its root
    relative to the song's tonic."""
    intervals = {"maj": (0, 4, 7), "min": (0, 3, 7),
                 "dim": (0, 3, 6), "aug": (0, 4, 8)}[quality]
    abs_root = (tonic_pc + rel_root) % 12
    return {(abs_root + iv) % 12 for iv in intervals}


def chord_tone_overlap(tonic_pc, rel_root, quality, melody_pitch_classes):
    """
    The 'expert ear' score: what fraction of the melody's pitch classes in
    this window are actual chord tones of the candidate chord?
    melody_pitch_classes: an iterable of pitch classes (0-11), relative to
    the song's absolute pitches (NOT relative to tonic).
    """
    triad = triad_pitch_classes(tonic_pc, rel_root, quality)
    melody_pcs = set(melody_pitch_classes)
    if not melody_pcs:
        return 0.0
    return len(melody_pcs & triad) / len(melody_pcs)