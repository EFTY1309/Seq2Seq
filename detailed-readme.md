

---

## Complete Codebase Analysis Against Assignment Requirements

### Overall Verdict: Your code **fully satisfies all requirements**. Here is a detailed breakdown:

---

## 1. Architecture Requirements

### Model 1 — Vanilla RNN Seq2Seq ✅ (vanilla_rnn.py)

```
Docstring (text)  →  [Embedding]  →  [RNN Encoder]  →  Hidden State (context vector)
                                                              ↓
Generated Code    ←  [Linear]  ←  [RNN Decoder] ← Token-by-token decoding
```

- `EncoderRNN`: Single-layer `nn.RNN`, produces one final hidden state as the **fixed-length context vector**
- `DecoderRNN`: Single-layer `nn.RNN`, uses ONLY that one hidden state — no access to per-step encoder outputs
- **Why this is the baseline**: The entire docstring is compressed into a single vector. For long docstrings, information is lost (the model "forgets" early words). This is what your teacher wants you to demonstrate.

---

### Model 2 — LSTM Seq2Seq ✅ (lstm.py)

```
Docstring  →  [Embedding]  →  [LSTM Encoder]  →  (h_n, c_n) — hidden + cell state
                                                        ↓
Generated Code  ←  [Linear]  ←  [LSTM Decoder (h, c state passed step-by-step)]
```

- `EncoderLSTM`: Uses `nn.LSTM` which adds a **cell state** `c` alongside the hidden state `h`
- **The key improvement over vanilla RNN**: The cell state acts like a "memory tape." LSTM gates (forget, input, output) decide what to remember and what to forget, solving the **vanishing gradient problem** that makes vanilla RNNs forget early tokens
- Still uses a **fixed-length context vector** (no access to all encoder steps)

---

### Model 3 — LSTM + Bahdanau Attention ✅ (attention_lstm.py)

This is the most complex model. Here is the full flow:

```
Docstring
   ↓
[BiLSTM Encoder] → encoder_outputs (all steps), hidden state
   |                    (batch, src_len, hidden*2)
   ↓                          ↓
[fc_hidden / fc_cell]         |
   ↓                          |
Decoder LSTM State ──────→ [BahdanauAttention]
   (h_t)                        ↓
                          attn_weights (softmax)
                                ↓
                          context vector (weighted sum of encoder_outputs)
                                ↓
                   [LSTM input = embedding + context]
                                ↓
                        [Linear: hidden + context + embedding → vocab]
```

**Key classes:**
- `EncoderBiLSTM`: Reads docstring in **both forward and backward** directions. `fc_hidden` and `fc_cell` project the concatenated bidirectional state back to decoder size
- `BahdanauAttention`: At each decoding step, computes $e_{t,i} = v^T \tanh(W_h h_t + W_e \text{enc}_i)$, then $\alpha_{t,i} = \text{softmax}(e_{t,i})$. Context = $\sum_i \alpha_{t,i} \cdot \text{enc}_i$
- `AttentionDecoderLSTM`: Concatenates embedding + context as LSTM input; final prediction uses hidden + context + embedding together

---

## 2. Training Configuration ✅ (config.yaml + train.py)

| Requirement | Your Code |
|---|---|
| Embedding dim 128–256 | `embedding_dim: 256` ✅ |
| Hidden dim 256 | `hidden_dim: 256` ✅ |
| Loss: Cross-entropy | `nn.CrossEntropyLoss(ignore_index=0)` ✅ (ignores padding) |
| Optimizer: Adam | `optim.Adam(...)` ✅ |
| Teacher forcing | `teacher_forcing_ratio: 0.5` ✅ |
| Same train/val/test split | All 3 models share same data loader ✅ |
| Gradient clipping | `clip_grad_norm_(..., 5.0)` ✅ |

### What is Teacher Forcing?
During training, instead of always feeding the model's own predicted token as the next input, **50% of the time** you feed the actual correct token. This speeds up training and prevents error accumulation. At inference (`teacher_forcing_ratio=0`), only the model's own predictions are used.

---

## 3. Dataset ✅ (data_loader.py)

| Requirement | Your Code |
|---|---|
| Dataset: Nan-Do/code-search-net-python | ✅ |
| Train 5,000–10,000 | `train_size: 10000` ✅ |
| Max docstring length: 50 tokens | `max_docstring_length: 50` ✅ |
| Max code length: 80 tokens | `max_code_length: 80` ✅ |
| Vocab management | `Vocabulary` class with PAD/SOS/EOS/UNK ✅ |

### How `Vocabulary` Works
```
<PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3, then top 9,996 most frequent words
    encode("return max nums")  →  [342, 17, 891]
    decode([342, 17, 891, 2]) →  "return max nums"   (stops at EOS=2)
```

---

## 4. Evaluation Metrics ✅ (evaluate.py)

| Metric | Implementation |
|---|---|
| **Token Accuracy** | Counts matching tokens position-by-position, adds penalty for length mismatch |
| **BLEU Score** | Uses `sacrebleu` library — measures n-gram overlap (1,2,3,4-grams) between prediction and reference |
| **Exact Match** | Checks if normalized prediction equals normalized reference exactly |
| **By length analysis** | Groups test examples into buckets (0–10, 11–20, 21–30, 31+ tokens) and calculates all 3 metrics per bucket |

---

## 5. Attention Visualization ✅ (visualize_attention.py)

- Generates **heatmap images** with docstring tokens on X-axis and generated code tokens on Y-axis
- Analyzes **which source word each generated token attends to most strongly**
- Calculates **attention entropy** (low entropy = focused/sharp attention, high = diffuse)
- Saves PNGs to attention

---

## 6. Report Plots ✅ (generate_report_plots.py)

- `plot_training_curves()` → Training vs Validation loss across epochs
- `plot_model_comparison()` → Bar chart of BLEU / Token Accuracy / Exact Match for all 3 models
- `plot_performance_by_length()` → Line chart showing performance degradation for longer docstrings
- `generate_results_table()` → LaTeX table ready to paste into your report PDF

---

## 7. What Each File Does — Summary

| File | Purpose |
|---|---|
| config.yaml | Central configuration — change hyperparameters here |
| data_loader.py | Downloads dataset, builds vocabulary, returns DataLoaders |
| vanilla_rnn.py | Baseline RNN encoder-decoder |
| lstm.py | LSTM encoder-decoder (adds cell state memory) |
| attention_lstm.py | BiLSTM + Bahdanau Attention (full model) |
| train.py | Trains all 3 models, saves checkpoints + training history |
| evaluate.py | Loads trained models, computes BLEU/accuracy/exact match |
| visualize_attention.py | Generates attention heatmaps for 3+ test examples |
| generate_report_plots.py | Generates all graphs for the report PDF |

---

## One Gap to Address: Error Type Analysis

The assignment requires analyzing **types of errors** (syntax errors, missing indentation, incorrect operators). Your code does not currently classify errors this way. This is needed for the qualitative analysis section of your report.

Would you like me to add an error analysis function to evaluate.py that automatically categorizes prediction errors into: missing indentation, syntax errors, wrong operators/variables, and length mismatches?