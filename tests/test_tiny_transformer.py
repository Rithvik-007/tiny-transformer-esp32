import torch
from src.data_utils import make_tiny_datasets
from src.models.tiny_transformer import TinyTransformerClassifier

train_ds, val_ds, test_ds, vocab = make_tiny_datasets("data/spam.csv", max_len=40, max_vocab_size=2000)

x, y = train_ds[0]
x = x.unsqueeze(0)  # batch size 1

model = TinyTransformerClassifier(
    vocab_size=len(vocab),
    d_model=48,
    nhead=3,
    num_layers=2,
    dim_feedforward=96,
    num_classes=2,
    max_len=40,
    dropout=0.1,
)

with torch.no_grad():
    logits = model(x)
print("Logits shape:", logits.shape)
print("Logits:", logits)
