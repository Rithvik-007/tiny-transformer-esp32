import torch
import matplotlib.pyplot as plt
import os


def main():
    tiny_ckpt = torch.load("tiny_transformer_checkpoint.pt", map_location="cpu")
    bert_ckpt = torch.load("distilbert_checkpoint.pt", map_location="cpu")

    tiny_hist = tiny_ckpt["history"]
    bert_base = bert_ckpt["history_base"]
    bert_ft = bert_ckpt["history_ft"]

    os.makedirs("plots_compare", exist_ok=True)

    # Compare validation accuracy (Tiny vs DistilBERT FT)
    plt.figure()
    plt.plot(tiny_hist["val_acc"], label="Tiny Transformer val_acc")
    plt.plot(
        range(len(tiny_hist["val_acc"])),
        [bert_ft["val_acc"][-1]] * len(tiny_hist["val_acc"]),
        "--",
        label="DistilBERT FT test_acc (flat line)"
    )
    plt.xlabel("Epoch (Tiny)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Tiny Transformer vs DistilBERT (accuracy)")
    plt.savefig("plots_compare/tiny_vs_bert.png")
    plt.close()


if __name__ == "__main__":
    main()
