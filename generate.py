"""
Inference with constrained decoding.

The trained model proposes chords; this script GUARANTEES every chord it
actually outputs is both diatonic to the key (TIER1) and playable with the
song's best capo (EASY_SHAPES / best_capo). Among whatever survives that
filter, it also nudges the pick toward whichever chord's notes best match
the melody actually sounding in that half-bar -- the same "expert ear"
chord-tone-overlap mechanism from the C->E example.
"""

import json
import random
import torch
import torch.nn.functional as F

from model import MelodyToChordTransformer, collate_fn
from vocab import (CHORD_VOCAB_SIZE, PAD, BOS, NOCHORD, N_SPECIAL, QUALITIES,
                    NOTES, TIER1, best_capo, token_to_chord, chord_tone_overlap)

CHECKPOINT_PATH = "best_model.pt"
OVERLAP_WEIGHT = 0.6   # how strongly chord-tone fit nudges the pick, vs. the model's own preference


def chord_token_for(rel_root, quality):
    return N_SPECIAL + rel_root * len(QUALITIES) + QUALITIES.index(quality)


def allowed_token_set(tonic_pc, mode_str):
    """
    The hard guarantee: every token the model is even ALLOWED to say.

    IMPORTANT: this depends on the song's actual tonic, not just its mode.
    A capo can only move a shape UP by 0-7 frets (MAX_CAPO), it can't go
    negative or wrap around, so which diatonic chords land on an open shape
    is genuinely different depending on how far the tonic sits from C/A --
    not just "major songs behave like C major songs." Always pass the real
    tonic_pc here, never a placeholder.
    """
    capo_fret, playable = best_capo(tonic_pc, mode_str)
    allowed = {chord_token_for(r, q) for (r, q) in playable}
    allowed.add(NOCHORD)
    return allowed, capo_fret


def chord_name(rel_root, quality, tonic_pc):
    abs_root = NOTES[(tonic_pc + rel_root) % 12]
    label = {"maj": "", "min": "m", "dim": "dim", "aug": "aug"}[quality]
    return f"{abs_root}{label}"


@torch.no_grad()
def generate(model, window, device):
    """
    window: one dict from dataset.json (has 'notes', 'tonic', 'mode', 'capo').
    Returns a list of 16 (token, chord_name, was_constrained) tuples.
    """
    mode_str = "maj" if window["mode"] == 0 else "min"
    tonic_pc = window["tonic"]
    allowed, capo_fret = allowed_token_set(tonic_pc, mode_str)
    allowed_list = sorted(allowed)

    batch = collate_fn([{"notes": window["notes"], "chords": [NOCHORD] * 16,
                          "mode": window["mode"], "capo": window["capo"]}])
    batch = {k: v.to(device) for k, v in batch.items()}

    memory, pad_mask = model.encode(batch["pc"], batch["octave"], batch["duration"],
                                     batch["position"], batch["enc_pad_mask"],
                                     batch["mode"], batch["capo"])

    # group the window's melody notes by which of the 16 half-bar slots they fall in,
    # so we can score chord-tone overlap per slot (position is in 16ths, 8 per half-bar)
    notes_by_slot = {s: [] for s in range(16)}
    for pc, octave, dur, pos16 in window["notes"]:
        slot = min(pos16 // 8, 15)
        notes_by_slot[slot].append(pc)  # already tonic-relative, matches chord_tone_overlap's expectation

    decoder_in = torch.full((1, 1), BOS, dtype=torch.long, device=device)
    results = []

    for slot in range(16):
        logits = model.decode(decoder_in, memory, pad_mask)
        last_logits = logits[0, -1]  # (CHORD_VOCAB_SIZE,)
        log_probs = F.log_softmax(last_logits, dim=-1)

        best_tok, best_score = None, -float("inf")
        for tok in allowed_list:
            score = log_probs[tok].item()
            if tok >= N_SPECIAL:
                rel_root, quality = token_to_chord(tok)
                overlap = chord_tone_overlap(0, rel_root, quality, notes_by_slot[slot])
                score = (1 - OVERLAP_WEIGHT) * score + OVERLAP_WEIGHT * overlap
            if score > best_score:
                best_tok, best_score = tok, score

        if best_tok >= N_SPECIAL:
            rel_root, quality = token_to_chord(best_tok)
            name = chord_name(rel_root, quality, tonic_pc)
        else:
            name = "N"
        results.append((best_tok, name, True))  # True: every output is constraint-checked by construction

        decoder_in = torch.cat([decoder_in, torch.tensor([[best_tok]], device=device)], dim=1)

    return results, capo_fret


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = MelodyToChordTransformer().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.3f})")

    data = json.load(open("dataset.json"))
    test_windows = [d for d in data if d["split"] == "test"]
    random.seed(2)
    window = random.choice(test_windows)
    print(f"melody notes: {len(window['notes'])}")

    print(f"\nsong {window['song']}, mode {'major' if window['mode']==0 else 'minor'}, "
          f"{len(window['notes'])} melody notes")

    results, capo_fret = generate(model, window, device)
    print(f"best capo for this key: {capo_fret}\n")
    print("original chords: ", [c for c in window["chords"]])
    print("generated chords:", [t for t, n, c in results])
    print("generated names: ", [n for t, n, c in results])

    # verify the guarantee held: every generated chord token is in the allowed set
    allowed, _ = allowed_token_set(window["tonic"], "maj" if window["mode"] == 0 else "min")
    all_ok = all(t in allowed for t, n, c in results)
    print(f"\nall generated chords are diatonic + capo-easy: {all_ok}")


if __name__ == "__main__":
    main()