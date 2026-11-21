"""
Extract song-level feature vectors from POP909 for similarity analysis.
"""

import os
import re
import pretty_midi
import pandas as pd
import numpy as np
from collections import Counter

BASE_DIR = "POP909-Dataset/POP909"

QUALITY_MAP = {
    "maj": "maj", "maj7": "maj", "maj6": "maj", "7": "maj",
    "min": "min", "min7": "min", "min6": "min",
    "dim": "dim", "dim7": "dim", "hdim7": "dim",
    "aug": "aug",
    "sus2": "maj", "sus4": "maj",
}

def clean_chord(raw):
    if raw == "N" or pd.isna(raw):
        return "N"
    raw = raw.split("/")[0]
    if ":" not in raw:
        return "N"
    root, quality = raw.split(":", 1)
    quality = re.sub(r"\(.*\)", "", quality)
    simplified_quality = QUALITY_MAP.get(quality, "maj")
    return f"{root}:{simplified_quality}"


def load_song(song_id):
    base = f"{BASE_DIR}/{song_id}"
    midi = pretty_midi.PrettyMIDI(f"{base}/{song_id}.mid")

    melody_track = next((i for i in midi.instruments if i.name == "MELODY"), None)
    if melody_track is None or len(melody_track.notes) == 0:
        return None

    chords_df = pd.read_csv(f"{base}/chord_midi.txt", sep="\t", header=None,
                             names=["start", "end", "chord"])
    chords_df["chord_clean"] = chords_df["chord"].apply(clean_chord)

    key_path = f"{base}/key_audio.txt"
    key_label = None
    if os.path.exists(key_path) and os.path.getsize(key_path) > 0:
        key_df = pd.read_csv(key_path, sep="\t", header=None,
                              names=["start", "end", "key"])
        if len(key_df):
            key_label = key_df.iloc[0]["key"]

    return melody_track, chords_df, key_label


def find_chord_at(time, chords_df):
    row = chords_df[(chords_df.start <= time) & (chords_df.end > time)]
    return row.chord_clean.values[0] if len(row) else "N"


def extract_features(song_id):
    loaded = load_song(song_id)
    if loaded is None:
        return None
    melody_track, chords_df, key_label = loaded

    notes = sorted(melody_track.notes, key=lambda n: n.start)
    if len(notes) < 2:
        return None

    chord_seq = [find_chord_at(n.start, chords_df) for n in notes]

    qualities = [c.split(":")[1] if ":" in c else "N" for c in chord_seq]
    chord_hist = Counter(qualities)
    total_c = sum(chord_hist.values())
    chord_hist_norm = {f"chord_{q}": chord_hist.get(q, 0) / total_c
                        for q in ["maj", "min", "dim", "aug", "N"]}

    changes = sum(1 for a, b in zip(chord_seq, chord_seq[1:]) if a != b)
    chord_change_rate = changes / max(1, len(chord_seq) - 1)
    chord_coverage = len(set(chord_seq))

    pitches = [n.pitch for n in notes]
    intervals = np.diff(pitches)
    abs_intervals = np.abs(intervals)
    interval_bins = {
        "interval_unison": np.mean(abs_intervals == 0) if len(abs_intervals) else 0,
        "interval_step": np.mean((abs_intervals >= 1) & (abs_intervals <= 2)) if len(abs_intervals) else 0,
        "interval_leap_small": np.mean((abs_intervals >= 3) & (abs_intervals <= 7)) if len(abs_intervals) else 0,
        "interval_leap_large": np.mean(abs_intervals >= 8) if len(abs_intervals) else 0,
    }

    durations = [n.end - n.start for n in notes]
    span = notes[-1].end - notes[0].start
    row = {
        "song_id": song_id,
        "key": key_label,
        "num_melody_notes": len(notes),
        "melody_range": max(pitches) - min(pitches),
        "melody_mean_pitch": float(np.mean(pitches)),
        "note_density_per_sec": len(notes) / span if span > 0 else 0,
        "mean_note_duration": float(np.mean(durations)),
        "chord_coverage": chord_coverage,
        "chord_change_rate": chord_change_rate,
        **chord_hist_norm,
        **interval_bins,
    }
    return row


def main():
    song_ids = sorted([s for s in os.listdir(BASE_DIR) if s.isdigit()])
    print(f"Found {len(song_ids)} songs. Extracting features...")

    rows = []
    failed = []
    for i, sid in enumerate(song_ids):
        try:
            row = extract_features(sid)
            if row is not None:
                rows.append(row)
            else:
                failed.append(sid)
        except Exception as e:
            failed.append(sid)
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(song_ids)}")

    df = pd.DataFrame(rows)
    df.to_csv("songs_features.csv", index=False)
    print(f"\nDone. {len(df)} songs written to songs_features.csv")
    print(f"{len(failed)} songs failed/skipped: {failed[:10]}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
