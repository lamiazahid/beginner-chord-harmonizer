import pretty_midi
import pandas as pd
import os

base_dir = "POP909-Dataset/POP909"
song_ids = sorted([s for s in os.listdir(base_dir) if s.isdigit()])
print(f"{len(song_ids)} songs found")

# Test on one song first
song_id = song_ids[0]
base = f"{base_dir}/{song_id}"

midi = pretty_midi.PrettyMIDI(f"{base}/{song_id}.mid")
for inst in midi.instruments:
    print(inst.name, len(inst.notes), "notes")

melody_track = next(i for i in midi.instruments if i.name == "MELODY")
chords = pd.read_csv(f"{base}/chord_midi.txt", sep="\t", header=None,
                      names=["start", "end", "chord"])

def find_chord_at(time, chords_df):
    row = chords_df[(chords_df.start <= time) & (chords_df.end > time)]
    return row.chord.values[0] if len(row) else "N"

print("\nFirst 10 melody notes aligned to chords:")
for note in melody_track.notes[:10]:
    chord = find_chord_at(note.start, chords)
    print(f"pitch={note.pitch} time={note.start:.2f} chord={chord}")

