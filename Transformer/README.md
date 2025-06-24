# Vanilla Transformer

> **An end-to-end implementation of the original "Attention Is All You Need" architecture**, built entirely from first principles in PyTorch, and trained on English↔French data from OPUS100.

---

## 🚀 Features

- Encoder–Decoder with multi-head self- and cross-attention  
- **Bilingual dataset** loader, tokenizers trained with WordLevel 
- **Interactive REPL** for one-off translation queries  

---

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model   | 384   |
| d_ff      | 1024  |
| N (layers)| 6     |
| h (heads) | 4     |
| seq_len   | 128   |
| dropout   | 0.1   |

---

**Training Summary**

> Training the model for four epochs required approximately 8 hours with GPUx2, and debugging/fixing implementation issues took an additional 20 hours.  
