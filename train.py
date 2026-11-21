"""
Training loop for the melody -> chord Transformer.

Tracks train/val loss and next-chord accuracy each epoch, saves the
best-validation-loss checkpoint to best_model.pt, and stops early if
validation stops improving.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ChordDataset, collate_fn, MelodyToChordTransformer
from vocab import PAD, CHORD_VOCAB_SIZE

# --- hyperparameters ---
BATCH_SIZE = 64
LR = 3e-4
MAX_EPOCHS = 60
PATIENCE = 8           # stop if val loss hasn't improved in this many epochs
CHECKPOINT_PATH = "best_model.pt"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Pass optimizer=None for eval (no weight updates)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            logits = model(batch)
            loss = criterion(logits.reshape(-1, CHORD_VOCAB_SIZE),
                              batch["decoder_target"].reshape(-1))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=-1)
        mask = batch["decoder_target"] != PAD
        total_correct += (preds[mask] == batch["decoder_target"][mask]).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens


def main():
    device = get_device()
    print(f"device: {device}")

    train_ds = ChordDataset("dataset.json", split="train")
    val_ds = ChordDataset("dataset.json", split="val")
    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MelodyToChordTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        dt = time.time() - t0

        print(f"epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.1%} "
              f"| val loss {val_loss:.3f} acc {val_acc:.1%} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch},
                       CHECKPOINT_PATH)
            print(f"  -> saved new best checkpoint (val loss {val_loss:.3f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\nno improvement for {PATIENCE} epochs, stopping early.")
                break

    print(f"\nbest val loss: {best_val_loss:.3f}, checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()