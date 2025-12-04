import os
import torch
from src.models.tiny_transformer import TinyTransformerClassifier
from src.data_utils import make_tiny_datasets


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_memory_estimates(num_params):
    # rough estimates
    bytes_fp32 = num_params * 4          # 32-bit float
    bytes_int8 = num_params * 1          # 8-bit quantized
    bytes_int4 = num_params * 0.5        # 4-bit quantized (theoretical)

    def to_mb(x): return x / (1024 * 1024)

    print("\n=== Memory Estimates ===")
    print(f"FP32 params:  ~{to_mb(bytes_fp32):.3f} MB")
    print(f"INT8 params:  ~{to_mb(bytes_int8):.3f} MB")
    print(f"INT4 params:  ~{to_mb(bytes_int4):.3f} MB (theoretical, post-quantization)")


def main():
    # we reuse the vocab so stats match the real model
    train_ds, val_ds, test_ds, vocab = make_tiny_datasets(
        "data/spam.csv", max_len=40, max_vocab_size=2000
    )

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

    total_params, trainable_params = count_parameters(model)

    print("=== Tiny Transformer Model Stats ===")
    print(f"Vocab size        : {len(vocab)}")
    print(f"Total parameters  : {total_params:,}")
    print(f"Trainable params  : {trainable_params:,}")

    print_memory_estimates(total_params)

    # quick breakdown by component (optional, but nice for report)
    print("\n=== Parameter Breakdown (by module) ===")
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        if mod_params == 0:
            continue
        print(f"{name:20s}: {mod_params:8d} params")


if __name__ == "__main__":
    main()
