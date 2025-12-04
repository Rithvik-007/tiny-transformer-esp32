from src.data_utils import make_tiny_datasets, make_distilbert_datasets

print("Testing tiny datasets...")
train_ds, val_ds, test_ds, vocab = make_tiny_datasets("data/spam.csv", max_len=40, max_vocab_size=2000)
print("Tiny dataset sizes:", len(train_ds), len(val_ds), len(test_ds))
print("Vocab size:", len(vocab))

x0, y0 = train_ds[0]
print("Sample tiny input shape:", x0.shape)
print("Sample tiny label:", y0)

print("\nTesting DistilBERT datasets (this will download tokenizer)...")
train_b, val_b, test_b = make_distilbert_datasets("data/spam.csv", max_len=64)
print("BERT dataset sizes:", len(train_b), len(val_b), len(test_b))
item = train_b[0]
print("Keys in BERT batch:", item.keys())
print("input_ids shape:", item["input_ids"].shape)
print("label:", item["labels"])
