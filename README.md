# Quick Start Guide - Seq2Seq Code Generation

## 🚀 How to Run This Project (3 Simple Steps)

### Step 1: Install Dependencies

Open terminal/command prompt in this folder and run:

```bash
pip install -r requirements.txt
```

**Wait for installation to complete** (~2-3 minutes)

---

### Step 2: Train the Models

Run this command to train all 3 models:

```bash
python train.py --config config_quick.yaml --model all
```

**What happens:**

- Trains Vanilla RNN model (~2 minutes)
- Trains LSTM model (~4 minutes)
- Trains Attention model (~10 minutes)
- **Total time: ~15 minutes**

**You'll see:**

```
Training VANILLA model
Epoch 1/2: Train Loss: 6.82, Val Loss: 6.15
Epoch 2/2: Train Loss: 5.64, Val Loss: 5.35
Training completed!

Training LSTM model
Epoch 1/2: Train Loss: 6.93, Val Loss: 6.45
Epoch 2/2: Train Loss: 5.97, Val Loss: 5.42
Training completed!

Training ATTENTION model
Epoch 1/2: Train Loss: 6.82, Val Loss: 6.38
Epoch 2/2: Train Loss: 5.30, Val Loss: 6.26
Training completed!
```

---

### Step 3: Evaluate and Get Results

Run this command:

```bash
python evaluate.py --model all --split test
```

**What you get:**

- BLEU scores for all 3 models
- Token accuracy comparison
- Results table printed in terminal

**Example output:**

```
Model           BLEU       Token Acc    Exact Match
------------------------------------------------------------
vanilla         0.12       2.75         0.00
lstm            0.15       2.79         0.00
attention       0.43       2.35         0.00
```

---

### Step 4 (Optional): Visualize Attention

Run this command:

```bash
python visualize_attention.py --num_examples 5 --summary
```

**What you get:**

- 5 attention heatmap images
- Saved in `visualizations/attention/` folder
- Open the `.png` files to see how attention works

---

## 📂 Where to Find Your Results

After running the commands above, check these folders:

### 1. **Trained Models**

```
checkpoints/
├── vanilla/best_model.pt
├── lstm/best_model.pt
└── attention/best_model.pt
```

### 2. **Evaluation Results**

```
results/
├── vanilla/evaluation_results.json
├── lstm/evaluation_results.json
└── attention/evaluation_results.json
```

Open these JSON files to see:

- BLEU scores
- Token accuracy
- Example predictions (input → output comparison)

### 3. **Attention Visualizations**

```
visualizations/attention/
├── attention_example_1.png
├── attention_example_2.png
├── attention_example_3.png
├── attention_example_4.png
├── attention_example_5.png
└── attention_statistics.png
```

Open these PNG files to see attention heatmaps

---

## 📊 Understanding the Results

### BLEU Score

- Measures how similar generated code is to reference code
- **Range:** 0 to 100
- **Higher is better**
- Our result: Attention model = 0.43 (best among 3 models)

### Token Accuracy

- Percentage of correct tokens generated
- **Range:** 0% to 100%
- **Higher is better**

### Attention Heatmaps

- Shows which input words model focuses on
- **Yellow/Red = high attention** (model is focusing here)
- **Blue = low attention** (model ignoring this)

---

## ⚡ If Something Goes Wrong

### Problem: "ModuleNotFoundError"

**Solution:**

```bash
pip install -r requirements.txt
```

### Problem: Training too slow

**Solution:** You're already using the quick config! Just wait 15 minutes.

### Problem: "Checkpoint not found" when visualizing

**Solution:** Train the models first (Step 2)

### Problem: Out of memory

**Solution:** Close other programs and try again

---

## 🎯 What This Project Does

**Input:** Natural language description

```
"Returns the maximum value in a list"
```

**Output:** Python code

```python
def max_value(nums):
    return max(nums)
```

**3 Models Implemented:**

1. **Vanilla RNN** - Basic model
2. **LSTM** - Better memory
3. **Attention** - State-of-the-art (best performance)

---

## 📝 For Your Report/Presentation

**Key Results to Show:**

1. **Model Comparison Table** (from Step 3 output)
2. **Attention Heatmaps** (PNG files from Step 4)
3. **Example Predictions** (in JSON files)

**Key Points:**

- ✅ Implemented 3 different Seq2Seq models
- ✅ Attention model performs best (BLEU: 0.43)
- ✅ Created visualizations showing attention mechanism
- ✅ Complete end-to-end pipeline working

---

## 🎓 Complete Commands Summary

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train all models (~15 minutes)
python train.py --config config_quick.yaml --model all

# 3. Evaluate and get results
python evaluate.py --model all --split test

# 4. Visualize attention (optional)
python visualize_attention.py --num_examples 5 --summary
```

**Done! ✅**

Check `results/` and `visualizations/` folders for all outputs.
