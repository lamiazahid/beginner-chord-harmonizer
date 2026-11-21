import json
import torch
from torch.utils.data import Dataset, DataLoader


class ChordDataset(Dataset):
    def __init__(self, json_path, split):
        with open(json_path, "r") as f:
            data = json.load(f)

        # Keep only the requested split
        self.data = [x for x in data if x["split"] == split]

        print(f"{split}: {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Notes:
        # [relative_pitch, octave, duration, position]
        notes = torch.tensor(item["notes"], dtype=torch.long)

        # Chords: 16 target chord tokens
        chords = torch.tensor(item["chords"], dtype=torch.long)

        # Song mode:
        # 0 = major
        # 1 = minor
        mode = torch.tensor(item["mode"], dtype=torch.long)

        return {
            "notes": notes,
            "chords": chords,
            "mode": mode,
            "song": item["song"]
        }


def collate_fn(batch):
    """
    Turn a list of examples with different numbers of notes
    into one padded batch.
    """

    batch_size = len(batch)

    # Find longest note sequence in this batch
    max_notes = max(len(x["notes"]) for x in batch)

    # 4 features per note:
    # relative pitch, octave, duration, position
    notes = torch.zeros(
        batch_size,
        max_notes,
        4,
        dtype=torch.long
    )

    # True = padding
    # False = real note
    note_padding_mask = torch.ones(
        batch_size,
        max_notes,
        dtype=torch.bool
    )

    # Every example has 16 chord targets
    chords = torch.stack([
        x["chords"] for x in batch
    ])

    modes = torch.stack([
        x["mode"] for x in batch
    ])

    songs = [
        x["song"] for x in batch
    ]

    for i, item in enumerate(batch):
        n = len(item["notes"])

        notes[i, :n] = item["notes"]

        # These positions contain real notes
        note_padding_mask[i, :n] = False

    return {
        "notes": notes,
        "note_padding_mask": note_padding_mask,
        "chords": chords,
        "mode": modes,
        "songs": songs
    }


if __name__ == "__main__":

    train_dataset = ChordDataset("dataset.json", "train")
    val_dataset = ChordDataset("dataset.json", "val")
    test_dataset = ChordDataset("dataset.json", "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(train_loader))

    print("\n--- Batch shapes ---")
    print("notes:", batch["notes"].shape)
    print("note_padding_mask:", batch["note_padding_mask"].shape)
    print("chords:", batch["chords"].shape)
    print("mode:", batch["mode"].shape)

    print("\n--- First example ---")
    print("song:", batch["songs"][0])
    print("notes:")
    print(batch["notes"][0])
    print("padding mask:")
    print(batch["note_padding_mask"][0])
    print("chords:")
    print(batch["chords"][0])
    print("mode:", batch["mode"][0].item())
