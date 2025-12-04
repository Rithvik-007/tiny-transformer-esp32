# Tiny Transformer SMS Spam Classifier (ESP32-C3 Compatible)

This project builds a **tiny Transformer model from scratch** and trains it on the SMS Spam dataset.  
The goal is to create a **lightweight, fully custom Transformer** that can be deployed on **resource-constrained hardware** such as the **ESP32-C3**.  

A pretrained **DistilBERT** model is also trained and fine-tuned as a **benchmark** to compare against the tiny model.

---

## 🚀 What This Project Includes

### **1. Custom Tiny Transformer (Built From Scratch)**
- Implemented using raw PyTorch (no HuggingFace).
- Custom tokenizer and vocabulary builder.
- Embedding → Positional Encoding → Multi-Head Attention → Feedforward → Classifier.
- Extremely small architecture (≈134k parameters).
- Memory footprint:
  - **FP32:** ~0.51 MB  
  - **INT8:** ~0.128 MB (MCU-friendly)  
  - **INT4:** ~0.064 MB (post-quantization theoretical)

Suitable for microcontrollers like **ESP32-C3** after quantization.

---

### **2. DistilBERT Benchmark**
- Loaded from HuggingFace (`distilbert-base-uncased`).
- Baseline (classifier-only training) + fine-tuning (last 2 encoder layers).
- Used only for comparison — not deployable on MCU.

---

## 📦 Project Features

- Clean dataset preprocessing pipeline for both models.
- Baseline + fine-tune training for tiny Transformer.
- DistilBERT baseline + fine-tune for comparison.
- Experiment logging with `run1`, `run2`, ... folders.
- Training curves (loss/accuracy) saved as images.
- Final test results saved per run.
- Parameter count + memory usage analysis.

---

## 📊 Final Results (Test Set)

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|---------:|----------:|-------:|---------:|
| **Tiny Transformer (run2)** | 0.9665 | 0.9200 | 0.8214 | 0.8679 |
| **DistilBERT (run1)**       | 0.9844 | 0.9381 | 0.9464 | 0.9422 |

- DistilBERT performs better overall (expected).
- Tiny Transformer still achieves **~96–97% accuracy** with **extremely small size**.
- Tiny model is suitable for **microcontroller deployment**, unlike DistilBERT.

---

## 🧪 How to Run

### Train the Tiny Transformer
```bash
python -m src.training.train_tiny
