import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

from data_utils import make_tiny_datasets
from tiny_transformer import TinyTransformerClassifier


def create_pad_mask(batch_ids, pad_idx=0):
    # batch_ids: (batch, seq_len)
    return batch_ids.eq(pad_idx)  # True where pad


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Train", leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        pad_mask = create_pad_mask(x).to(device)

        optimizer.zero_grad()
        logits = model(x, src_key_padding_mask=pad_mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return avg_loss, acc, prec, rec, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc=desc, leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        pad_mask = create_pad_mask(x).to(device)

        logits = model(x, src_key_padding_mask=pad_mask)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1, cm


def plot_curves(history, out_dir="plots_tiny"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Tiny Transformer Loss")
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Tiny Transformer Accuracy")
    plt.savefig(os.path.join(out_dir, "accuracy.png"))
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, val_ds, test_ds, vocab = make_tiny_datasets("spam.csv", max_len=40, max_vocab_size=2000)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = TinyTransformerClassifier(
        vocab_size=len(vocab),
        d_model=48,
        nhead=3,
        num_layers=2,
        dim_feedforward=96,
        num_classes=2,
        max_len=40,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 10
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate(
            model, val_loader, criterion, device, desc="Val"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # minimal logging
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )
    print("=== Tiny Transformer Test Metrics ===")
    print(f"loss={test_loss:.4f}, acc={test_acc:.4f}, prec={test_prec:.4f}, "
          f"rec={test_rec:.4f}, f1={test_f1:.4f}")
    print("Confusion matrix:\n", test_cm)

    # Save model and history
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "history": history,
    }, "tiny_transformer_checkpoint.pt")

    plot_curves(history)


if __name__ == "__main__":
    main()
