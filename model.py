"""
Melody -> Chord Transformer.

Encoder reads the melody (one token per note: relative pitch class, octave,
duration, position), prefixed with two conditioning tokens (mode, capo) so
the model is aware, from the very first layer, of which key-type and which
capo-adjusted "easy shape" set it should be leaning toward.

Decoder autoregressively predicts one chord per half-bar (16 chords per
8-bar window), cross-attending to the melody encoding.
"""

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vocab import CHORD_VOCAB_SIZE, PAD, BOS, EOS, NOCHORD

# --- melody feature vocab sizes (from the real data ranges we just checked) ---
N_PITCH_CLASS = 12          # 0-11, relative to tonic
N_OCTAVE = 11                # 0-10, generous headroom around the observed 3-8
N_DURATION = 17              # 0-16 sixteenth-notes
N_POSITION = 128             # 0-127 sixteenth-note slots in an 8-bar window
N_CHORD_SLOTS = 16           # one chord per half-bar, 8 bars -> 16 slots

MAX_MELODY_LEN = 192         # matches build_dataset.py's MAX_NOTES


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChordDataset(Dataset):
    def __init__(self, path, split):
        with open(path) as f:
            data = json.load(f)
        self.data = [d for d in data if d["split"] == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "notes": d["notes"],              # list of [pc, octave, duration, position]
            "chords": d["chords"],            # list of 16 chord token ids
            "mode": d["mode"],                # 0 = major, 1 = minor
            "capo": d["capo"],                # 0-7
        }


def collate_fn(batch):
    B = len(batch)
    max_len = max(len(b["notes"]) for b in batch)

    pc = torch.zeros(B, max_len, dtype=torch.long)
    oct_ = torch.zeros(B, max_len, dtype=torch.long)
    dur = torch.zeros(B, max_len, dtype=torch.long)
    pos = torch.zeros(B, max_len, dtype=torch.long)
    enc_pad_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = PAD (ignore)

    for i, b in enumerate(batch):
        n = len(b["notes"])
        for j, (p, o, du, t16) in enumerate(b["notes"]):
            pc[i, j] = p
            oct_[i, j] = o
            dur[i, j] = du
            pos[i, j] = t16
        enc_pad_mask[i, :n] = False

    mode = torch.tensor([b["mode"] for b in batch], dtype=torch.long)
    capo = torch.tensor([b["capo"] for b in batch], dtype=torch.long)

    chords = torch.tensor([b["chords"] for b in batch], dtype=torch.long)  # (B, 16)
    decoder_in = torch.cat([torch.full((B, 1), BOS, dtype=torch.long), chords[:, :-1]], dim=1)
    decoder_target = chords

    return {
        "pc": pc, "octave": oct_, "duration": dur, "position": pos,
        "enc_pad_mask": enc_pad_mask,
        "mode": mode, "capo": capo,
        "decoder_in": decoder_in, "decoder_target": decoder_target,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MelodyToChordTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_enc_layers=3, n_dec_layers=3,
                 d_ff=256, dropout=0.15):
        super().__init__()
        self.d_model = d_model

        # --- melody note embeddings: one small table per feature, summed ---
        self.pc_emb = nn.Embedding(N_PITCH_CLASS, d_model)
        self.octave_emb = nn.Embedding(N_OCTAVE, d_model)
        self.duration_emb = nn.Embedding(N_DURATION, d_model)
        self.position_emb = nn.Embedding(N_POSITION, d_model)  # doubles as positional encoding

        # --- conditioning tokens, prepended to the encoder sequence ---
        self.mode_emb = nn.Embedding(2, d_model)     # major / minor
        self.capo_emb = nn.Embedding(8, d_model)     # capo fret 0-7

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers, enable_nested_tensor=False)

        # --- chord decoder ---
        self.chord_emb = nn.Embedding(CHORD_VOCAB_SIZE, d_model, padding_idx=PAD)
        self.chord_pos_emb = nn.Embedding(N_CHORD_SLOTS + 1, d_model)  # +1 for the BOS slot

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        self.output_proj = nn.Linear(d_model, CHORD_VOCAB_SIZE)

    def encode(self, pc, octave, duration, position, enc_pad_mask, mode, capo):
        B = pc.size(0)
        note_emb = (self.pc_emb(pc) + self.octave_emb(octave)
                    + self.duration_emb(duration) + self.position_emb(position))

        cond = torch.stack([self.mode_emb(mode), self.capo_emb(capo)], dim=1)  # (B, 2, d)
        cond_pad = torch.zeros(B, 2, dtype=torch.bool, device=pc.device)       # never padded

        enc_in = torch.cat([cond, note_emb], dim=1)                 # (B, 2+L, d)
        pad_mask = torch.cat([cond_pad, enc_pad_mask], dim=1)       # (B, 2+L)

        memory = self.encoder(enc_in, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def decode(self, decoder_in, memory, memory_pad_mask):
        B, T = decoder_in.shape
        positions = torch.arange(T, device=decoder_in.device).unsqueeze(0).expand(B, T)
        dec_emb = self.chord_emb(decoder_in) + self.chord_pos_emb(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(decoder_in.device)

        out = self.decoder(
            tgt=dec_emb, memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.output_proj(out)  # (B, T, CHORD_VOCAB_SIZE)

    def forward(self, batch):
        memory, pad_mask = self.encode(
            batch["pc"], batch["octave"], batch["duration"], batch["position"],
            batch["enc_pad_mask"], batch["mode"], batch["capo"],
        )
        logits = self.decode(batch["decoder_in"], memory, pad_mask)
        return logits


if __name__ == "__main__":
    # quick self-test: load a real batch, run a forward + backward pass
    ds = ChordDataset("dataset.json", split="train")
    print(f"train examples: {len(ds)}")
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = MelodyToChordTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}, parameters: {n_params:,}")

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    logits = model(batch)
    print("logits shape:", logits.shape, "(expect: batch, 16, 52)")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    loss = criterion(logits.reshape(-1, CHORD_VOCAB_SIZE), batch["decoder_target"].reshape(-1))
    print("initial loss:", loss.item(), "(expect roughly ln(52) =", 3.95, "for an untrained model)")

    loss.backward()
    print("backward pass OK - gradients computed")

    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("all parameters received gradients:", has_grad)