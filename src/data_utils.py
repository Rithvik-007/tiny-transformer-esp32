import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

# ---------- Simple tokenizer/vocab for tiny transformer ----------

def build_vocab(texts, max_vocab_size=2000, min_freq=2):
    counter = Counter()
    for t in texts:
        for tok in t.lower().split():
            counter[tok] += 1

    # reserve:
    # 0: PAD, 1: UNK, 2: CLS
    vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode_text(text, vocab, max_len=40):
    tokens = text.lower().split()
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    # add CLS at start
    ids = [vocab["<CLS>"]] + ids
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


class TinyTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=40):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x_ids = torch.tensor(
            encode_text(self.texts[idx], self.vocab, self.max_len),
            dtype=torch.long
        )
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_ids, y


# ---------- DistilBERT dataset ----------
from transformers import DistilBertTokenizerFast

class DistilBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name="distilbert-base-uncased", max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------- Load spam dataset and split ----------

def load_spam_data(csv_path="spam.csv", test_size=0.15, val_size=0.15, random_state=42):
    # dataset uses latin-1 encoding
    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})

    # map labels: ham -> 0, spam -> 1
    label_map = {"ham": 0, "spam": 1}
    df = df[df["label"].isin(label_map.keys())]
    df["label_id"] = df["label"].map(label_map)

    texts = df["text"].tolist()
    labels = df["label_id"].tolist()

    # first split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    # then split temp into train/val
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, random_state=random_state, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_tiny_datasets(csv_path="spam.csv", max_len=40, max_vocab_size=2000):
    X_train, y_train, X_val, y_val, X_test, y_test = load_spam_data(csv_path)
    vocab = build_vocab(X_train, max_vocab_size=max_vocab_size)

    train_ds = TinyTextDataset(X_train, y_train, vocab, max_len=max_len)
    val_ds = TinyTextDataset(X_val, y_val, vocab, max_len=max_len)
    test_ds = TinyTextDataset(X_test, y_test, vocab, max_len=max_len)
    return train_ds, val_ds, test_ds, vocab


def make_distilbert_datasets(csv_path="spam.csv", max_len=64):
    X_train, y_train, X_val, y_val, X_test, y_test = load_spam_data(csv_path)
    train_ds = DistilBertDataset(X_train, y_train, max_len=max_len)
    val_ds = DistilBertDataset(X_val, y_val, max_len=max_len)
    test_ds = DistilBertDataset(X_test, y_test, max_len=max_len)
    return train_ds, val_ds, test_ds
