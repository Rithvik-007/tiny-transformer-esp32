import os
import torch
import matplotlib.pyplot as plt


def load_tiny(run_id=2):
    path = os.path.join("experiments", "tiny_transformer", f"run{run_id}", "checkpoint.pt")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["test_metrics"]


def load_bert(run_id=1):
    path = os.path.join("experiments", "distilbert", f"run{run_id}", "checkpoint.pt")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["test_metrics"]


def print_metrics(name, m):
    print(f"\n=== {name} Test Metrics ===")
    print(f"Accuracy : {m['acc']:.4f}")
    print(f"Precision: {m['prec']:.4f}")
    print(f"Recall   : {m['rec']:.4f}")
    print(f"F1       : {m['f1']:.4f}")
    print("Confusion matrix:")
    print(m["cm"])


def plot_bar_comparison(tiny_m, bert_m, out_dir="experiments/comparison"):
    os.makedirs(out_dir, exist_ok=True)

    metrics = ["acc", "prec", "rec", "f1"]
    tiny_vals = [tiny_m[k] for k in metrics]
    bert_vals = [bert_m[k] for k in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], tiny_vals, width, label="TinyTransformer")
    plt.bar([i + width/2 for i in x], bert_vals, width, label="DistilBERT")

    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1"])
    plt.ylabel("Score")
    plt.ylim(0.7, 1.0)
    plt.title("Tiny Transformer vs DistilBERT (Test Metrics)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "metrics_bar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved comparison plot to {out_path}")


def main():
    tiny_metrics = load_tiny(run_id=2)   # your fine-tuned tiny run
    bert_metrics = load_bert(run_id=1)   # your first DistilBERT run

    print_metrics("Tiny Transformer (run2)", tiny_metrics)
    print_metrics("DistilBERT (run1)", bert_metrics)

    plot_bar_comparison(tiny_metrics, bert_metrics)


if __name__ == "__main__":
    main()
