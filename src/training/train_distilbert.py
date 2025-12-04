import os
import re
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import DistilBertForSequenceClassification
from src.data_utils import make_distilbert_datasets


# ---------- Run management: experiments/distilbert/runX ----------

def start_new_run(exp_dir="experiments/distilbert"):
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


# ---------- Training / eval loops ----------

def train_epoch_bert(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Train(BERT)", leave=False)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        optimizer.zero_grad()
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return avg_loss, acc, prec, rec, f1


@torch.no_grad()
def eval_epoch_bert(model, loader, device, desc="Val(BERT)"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc=desc, leave=False)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1, cm


# ---------- Freeze / unfreeze helpers ----------

def freeze_all_but_classifier(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_last_n_layers(model, n=2):
    # DistilBERT encoder layers: transformer.layer.0 ... 5
    for name, param in model.named_parameters():
        param.requires_grad = False

    for i in range(6 - n, 6):
        for name, param in model.named_parameters():
            if f"transformer.layer.{i}." in name:
                param.requires_grad = True

    # always train classifier
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True


# ---------- Plotting ----------

def plot_bert_history(history_base, history_ft, out_dir):
    epochs_base = range(1, len(history_base["val_acc"]) + 1)
    epochs_ft = range(len(history_base["val_acc"]) + 1,
                      len(history_base["val_acc"]) + len(history_ft["val_acc"]) + 1)

    plt.figure()
    plt.plot(epochs_base, history_base["val_acc"], label="Baseline val_acc")
    plt.plot(epochs_ft, history_ft["val_acc"], label="Fine-tune val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("DistilBERT Baseline vs Fine-tune Accuracy")
    plt.savefig(os.path.join(out_dir, "bert_val_acc.png"))
    plt.close()


# ---------- main ----------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, val_ds, test_ds = make_distilbert_datasets("data/spam.csv", max_len=64)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    run_dir, run_id = start_new_run("experiments/distilbert")
    print(f"Starting DistilBERT run{run_id} -> {run_dir}")

    # ---------- Load pretrained DistilBERT ----------
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    ).to(device)

    # ---------- Baseline: freeze encoder, train classifier only ----------
    freeze_all_but_classifier(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4
    )

    num_epochs_base = 2  # keep small for speed
    history_base = {"val_acc": []}

    for epoch in range(1, num_epochs_base + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch_bert(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = eval_epoch_bert(
            model, val_loader, device, desc="Val(BERT)"
        )
        history_base["val_acc"].append(val_acc)

        # minimal logging
        print(
            f"[Baseline] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

    # ---------- Fine-tune: unfreeze last 2 layers + classifier ----------
    unfreeze_last_n_layers(model, n=2)
    optimizer_ft = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5
    )
    num_epochs_ft = 2
    history_ft = {"val_acc": []}

    for epoch in range(1, num_epochs_ft + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch_bert(
            model, train_loader, optimizer_ft, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = eval_epoch_bert(
            model, val_loader, device, desc="Val(BERT-FT)"
        )
        history_ft["val_acc"].append(val_acc)

        print(
            f"[Fine-tune] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

    # ---------- Final test (fine-tuned) ----------
    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = eval_epoch_bert(
        model, test_loader, device, desc="Test(BERT)"
    )
    print("=== DistilBERT Test Metrics (fine-tuned) ===")
    print(f"loss={test_loss:.4f}, acc={test_acc:.4f}, "
          f"prec={test_prec:.4f}, rec={test_rec:.4f}, f1={test_f1:.4f}")
    print("Confusion matrix:\n", test_cm)

    # ---------- Save checkpoint for this run ----------
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save({
        "model_state": model.state_dict(),
        "history_base": history_base,
        "history_ft": history_ft,
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

    # ---------- Plots ----------
    plot_bert_history(history_base, history_ft, run_dir)


if __name__ == "__main__":
    main()
