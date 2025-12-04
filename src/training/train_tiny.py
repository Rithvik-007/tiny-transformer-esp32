import os
import re
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.data_utils import make_tiny_datasets
from src.models.tiny_transformer import TinyTransformerClassifier


# ---------- Run management: experiments/tiny_transformer/runX ----------

def start_new_run(exp_dir="experiments/tiny_transformer"):
    os.makedirs(exp_dir, exist_ok=True)
    existing = [d for d in os.listdir(exp_dir) if d.startswith("run")]
    if not existing:
        run_id = 1
    else:
        nums = []
        for d in existing:
            m = re.match(r"run(\d+)", d)
            if m:
                nums.append(int(m.group(1)))
        run_id = max(nums) + 1 if nums else 1
    run_dir = os.path.join(exp_dir, f"run{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


# ---------- Utility ----------

def create_pad_mask(batch_ids, pad_idx=0):
    # batch_ids: (batch, seq_len)
    return batch_ids.eq(pad_idx)  # True where pad


def train_one_epoch(model, loader, optimizer, criterion, device, desc="Train"):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc=desc, leave=False)
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


def plot_curves(history_all, out_dir):
    epochs = range(1, len(history_all["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history_all["train_loss"], label="Train")
    plt.plot(epochs, history_all["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Tiny Transformer Loss (baseline + fine-tune)")
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history_all["train_acc"], label="Train")
    plt.plot(epochs, history_all["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Tiny Transformer Accuracy (baseline + fine-tune)")
    plt.savefig(os.path.join(out_dir, "accuracy.png"))
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----- Data -----
    train_ds, val_ds, test_ds, vocab = make_tiny_datasets(
        "data/spam.csv",
        max_len=40,
        max_vocab_size=2000,
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    # ----- Model from scratch -----
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

    # ------ Run management ------
    run_dir, run_id = start_new_run("experiments/tiny_transformer")
    print(f"Starting Tiny Transformer run{run_id} -> {run_dir}")

    # =========================
    # BASELINE TRAINING PHASE
    # =========================
    optimizer_base = torch.optim.Adam(model.parameters(), lr=3e-4)
    num_epochs_base = 5  # short baseline phase

    history_base = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    # combined history (baseline + fine-tune)
    history_all = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, num_epochs_base + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer_base, criterion, device, desc="Train(Base)"
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate(
            model, val_loader, criterion, device, desc="Val(Base)"
        )

        history_base["train_loss"].append(train_loss)
        history_base["val_loss"].append(val_loss)
        history_base["train_acc"].append(train_acc)
        history_base["val_acc"].append(val_acc)

        history_all["train_loss"].append(train_loss)
        history_all["val_loss"].append(val_loss)
        history_all["train_acc"].append(train_acc)
        history_all["val_acc"].append(val_acc)

        print(
            f"[Baseline] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    # =========================
    # FINE-TUNING PHASE
    # =========================
    # load best baseline weights as starting point
    if best_state is not None:
        model.load_state_dict(best_state)

    # smaller LR for fine-tuning
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs_ft = 10

    history_ft = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    for epoch in range(1, num_epochs_ft + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer_ft, criterion, device, desc="Train(FT)"
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate(
            model, val_loader, criterion, device, desc="Val(FT)"
        )

        history_ft["train_loss"].append(train_loss)
        history_ft["val_loss"].append(val_loss)
        history_ft["train_acc"].append(train_acc)
        history_ft["val_acc"].append(val_acc)

        history_all["train_loss"].append(train_loss)
        history_all["val_loss"].append(val_loss)
        history_all["train_acc"].append(train_acc)
        history_all["val_acc"].append(val_acc)

        print(
            f"[Fine-tune] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    # load best weights over both phases
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )
    print("=== Tiny Transformer Test Metrics (best over baseline + fine-tune) ===")
    print(f"loss={test_loss:.4f}, acc={test_acc:.4f}, prec={test_prec:.4f}, "
          f"rec={test_rec:.4f}, f1={test_f1:.4f}")
    print("Confusion matrix:\n", test_cm)

    # Save model + histories for this run
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "history_base": history_base,
        "history_ft": history_ft,
        "history_all": history_all,
        "test_metrics": {
            "loss": test_loss,
            "acc": test_acc,
            "prec": test_prec,
            "rec": test_rec,
            "f1": test_f1,
            "cm": test_cm,
        },
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Plots into run dir
    plot_curves(history_all, run_dir)


if __name__ == "__main__":
    main()
