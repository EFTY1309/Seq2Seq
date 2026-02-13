# Seq2Seq Code Generation - Quick Start

## 🚀 Run Everything (Choose ONE method)

### **Option 1: Docker (Recommended)**

```bash
docker-compose up all
```

### **Option 2: Single Script**

```bash
# Windows
.\run_all.ps1

# Linux/Mac
./run_all.sh
```

### **Option 3: Manual Commands**

```bash
pip install -r requirements.txt
python train.py --config config_quick.yaml --model all
python evaluate.py --model all --split test
python visualize_attention.py --num_examples 5 --summary
```

**Training time:** ~10-15 minutes

---

## 📂 Results Location

| Output             | Folder                      |
| ------------------ | --------------------------- |
| Trained models     | `checkpoints/`              |
| Evaluation metrics | `results/`                  |
| Attention heatmaps | `visualizations/attention/` |

---

## 📊 What You'll Get

**Models Trained:**

- Vanilla RNN (baseline)
- LSTM (improved memory)
- Attention LSTM (best performance)

**Metrics:**

- BLEU scores (code similarity)
- Token accuracy
- Example predictions

**Visualizations:**

- 5 attention heatmaps showing what the model focuses on

---

## ⚡ Troubleshooting

**Problem:** `ModuleNotFoundError`  
**Solution:** `pip install -r requirements.txt`

**Problem:** Training too slow  
**Solution:** Already using fast config (1000 examples)

**Problem:** "Checkpoint not found"  
**Solution:** Train models first (Step 2)

---

**Done!** Check `results/` and `visualizations/` folders for outputs.
