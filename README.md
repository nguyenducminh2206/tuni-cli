# mi-race — Machine Learning for Science (CLI)

A lightweight command‑line tool to train and evaluate ML models directly from the terminal on tabular and sequence‑like data. It supports:
- Fast CSV/TSV/Parquet loading
- Balancing dataset by undersampling to minimum label count
- Config‑driven training for MLP and 1D CNN
- Sequence columns (e.g. `time_trace`) → statistical expansion or split into steps
- Clean terminal report: class counts, accuracy, macro‑F1, confusion matrix
- Noise analysis (when a noise column exists): train per‑noise and print reports
- Compare overall accuracy and accuracy‑vs‑noise across models (terminal charts)
- Artifacts saved to `outputs/`

---

## 1. Environment Setup (Recommended)

### Windows (PowerShell)
```powershell
python -m venv .venv              # create virtual environment
.\.venv\Scripts\activate          # activate it
python -m pip install -U pip      # upgrade pip (optional but good practice)
cd mi-race                        # move to the working directory
pip install -e .                  # editable install to use `mi-race` CLI
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
cd mi-race
pip install -e .
```

To leave the environment:
```bash
deactivate
```

Why venv?
- Keeps dependencies isolated
- Reproducible experiments
- Avoids polluting system Python

---

## 2. Quick CLI Usage

```bash
mi-race run --model mlp -c config.json
mi-race run --model cnn -c config.json

# Compare models from outputs/summary_models.csv (prints terminal bars)
mi-race compare

# or (module form)
python -m mi_race.cli.main run --model mlp -c config.json
```

### `compare`
Read `outputs/summary_models.csv` and print two terminal plots:
- Overall model accuracy (one horizontal bar per model)
- Accuracy vs noise (grouped horizontal bars per noise level)

```
mi-race compare
```

---

## 3. Configuration (config.json)

```json
{
  "data": {
    "path": "data_7x7",       // path/to_your/dataset
    "y_col": "dis_to_target", // labels for the model
    "x_cols": ["time_trace_0:time_trace_99", "cMax", "cVar"], // inputs for the model
    "sequence_mode": "split", // 3 modes: "split" | "stats" | "ignore"
    "balance": true // "false" if balancing data is not necessary 
  },
  "model": {
        "mlp": {
            "type": "mlp",
            "hidden_layers": [128, 128],
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "alpha": 0.0001,
            "batch_size": "auto"
        },
        "cnn": {
            "type": "cnn",
            "channels": [16, 32],
            "kernel_size": 5,
            "pool": 2,
            "fc": 128,
            "epochs": 15,
            "lr": 0.001,
            "batch_size": 64
        }
    },
    "train": {
        "test_size": 0.2,
        "random_state": 42,
        "max_iter": 500,
        "standardize": true
    },
    "output": {
        "dir": "outputs",
        "show_report": true
    }

}
```

### `data` section
- **`path`**: file path **or** dataset id. The loader resolves:
  - If `path` exists and is a **file**: read by extension (`.csv`, `.tsv`, `.parquet`).
  - If `path` exists and is a **directory** *or* has **no extension**: treated as a **dataset id** → `build_df(path.name)` is called.
  - If `path` does **not** exist: treated as **dataset id** → `build_df(path)`.
- **`id`**: alternative to `path` (explicit dataset id). One of `path` or `id` is required.
- **`y_col`**: label/target column (must exist).
- **`x_cols`**: feature columns. If omitted, all **numeric** columns except `y_col` are used.
  - Supports **column ranges** for split sequence columns: Use `"prefix_start:prefix_end"` notation to select a range of columns.
  - Example: `"time_trace_1:time_trace_50"` selects columns `time_trace_1` through `time_trace_50`.
  - Can mix ranges with regular column names: `["time_trace_10:time_trace_100", "cMax", "cVar"]`
- **`sequence_mode`**:
  - `"stats"` (default): sequence‑like columns (Python list/ndarray) are expanded into statistical features:
    - `len, mean, std, min, max, q10, q25, q50, q75, q90`
  - `"split"`: sequence‑like columns are split into individual columns (e.g., `time_trace_0`, `time_trace_1`, etc.). Each element of the sequence becomes a separate feature column, preserving all temporal information.
  - `"ignore"`: sequence‑like columns are skipped.

> **Note**: If you load **CSV** files where sequence columns are stored as **strings** (e.g., `"[1.0, 1.1, ...]"`), convert them to real lists first or use Parquet/PKL to preserve list/array types. The summarizer detects Python lists/arrays, not strings.


### Model Section 
- MLP
  - type: mlp
  - hidden_layers: list of layer sizes
  - activation: relu | tanh | logistic | identity
  - alpha: L2 regularization
  - Other: solver, learning_rate_init, batch_size, etc.

#### CNN options
- type: cnn
- channels: list of conv channel sizes (e.g., [16, 32])
- kernel_size, pool, fc, epochs, lr, batch_size
- sequence_prefix: name prefix of the split sequence group (required if multiple groups exist). Example: with `time_trace_0..time_trace_99`, use `"sequence_prefix": "time_trace"`.
- Logs: epoch summaries print exactly every 5 epochs with train/test loss and accuracy; a final test summary is printed at the end.
- Requires: `data.sequence_mode: "split"` and that your `x_cols` include the split sequence range.

### Train Section
- test_size: test split fraction
- random_state: reproducibility
- max_iter: MLP training iterations
- standardize: adds StandardScaler

### Output Section
- dir: artifact folder
- show_report: print precision/recall/F1 per class

---

## 4. What Happens on `run`
1. **Load data**.
2. Resolve **features** (`x_cols`) & **target** (`y_col`). Sequence columns are summarized if `sequence_mode="stats"`.
3. Split train/test (optionally stratified and/or balanced as configured).
4. Fit the selected model (MLP or CNN) with optional standardization.
5. Print a terminal report and **save artifacts**.

Example terminal output (abridged):

```
[mi-race] Loaded dataset with columns: ['cell_id', 'time_trace', 'dis_to_target', 'simulation_file', 'noise', 'cMax', 'cVar']
[mi-race] Balancing enabled: undersampled each class to min_count=4800
[mi-race] Class distribution (before): {0: 4800, 1: 28800, 2: 57600, 3: 86400, 4: 48000, 5: 9600}
[mi-race] Class distribution (after):  {0: 4800, 1: 4800, 2: 4800, 3: 4800, 4: 4800, 5: 4800}
[mi-race] Final feature columns (n=1003):
Split groups:
- time_trace_*: time_trace_0..time_trace_1000 (1001 cols)
- Other features (2): cMax, cVar
[mi-race] Saved processed features to: outputs\processed_features.csv
[mi-race] Class distribution (full): {0: 4800, 1: 4800, 2: 4800, 3: 4800, 4: 4800, 5: 4800}

[mi-race] ===== Running model: cnn =====
[mi-race][cnn] Using split sequence group 'time_trace' with 1001 steps
[mi-race][cnn] Total rows: 28800
[mi-race][cnn] Class distribution (train): {0: 3840, 1: 3840, 2: 3840, 3: 3840, 4: 3840, 5: 3840}
[mi-race][cnn] Class distribution (test):  {0: 960, 1: 960, 2: 960, 3: 960, 4: 960, 5: 960}
[mi-race][cnn] epoch 5/15  train_loss=1.0270 train_acc=0.5427  test_loss=1.1568 test_acc=0.4837
[cnn] epoch 10/15:  66%|███████████████████████████████████████▉                     | 236/360 [00:04<00:02, 54.20it/s]

=== Training for noise 0.01 (cnn) ===
[mi-race][cnn] Using split sequence group 'time_trace' with 1001 steps
[mi-race][cnn] Total rows: 6120
[mi-race][cnn] Class distribution (train): {0: 800, 1: 805, 2: 835, 3: 801, 4: 844, 5: 811}
[mi-race][cnn] Class distribution (test):  {0: 200, 1: 201, 2: 209, 3: 200, 4: 211, 5: 203}
[mi-race][cnn] epoch 5/15  train_loss=0.7874 train_acc=0.6591  test_loss=0.9462 test_acc=0.5711
[mi-race][cnn] epoch 10/15  train_loss=0.4550 train_acc=0.8346  test_loss=1.1939 test_acc=0.5564
[mi-race][cnn] epoch 15/15  train_loss=0.0470 train_acc=0.9953  test_loss=2.2383 test_acc=0.5547
[mi-race][cnn] final test: loss=2.2383 acc=0.5547
[mi-race][cnn] noise=0.01  accuracy=0.5547  epochs=15
Confusion Matrix (noise=0.01):
        pred_0  pred_1  pred_2  pred_3  pred_4  pred_5
true_0     198       2       0       0       0       0
true_1       0     181      10       6       2       2
true_2       0      11     103      21      43      31
true_3       0       1      43      49      59      48
true_4       0       4      29      49      83      46
true_5       0       2      27      48      61      65
MI: I(true;pred)=1.1216  NMI_sqrt=0.4347  NMI_min=0.4354  NMI_max=0.4340

...
```
---

## 5. What Happens on `compare`
- Read the outputs of training process in `outputs/processed_features.csv`
- Plot the graphs for comparing models' accuracy and accuracy vs. noise for each model

Example terminal output:
```
=== Model Accuracy (Overall) ===
   cnn |█████████████████████████████████████████                                                 | 0.4528
   mlp |███████████████████████████████████████                                                   | 0.4335

=== Accuracy vs Noise (grouped bars) ===
  Legend: ■ mlp  ■ cnn

noise 0.01:
     mlp |███████████████████████████████████████████████████                                             | 0.5327
     cnn |█████████████████████████████████████████████████████                                           | 0.5547

noise 0.02:
     mlp |█████████████████████████████████████████████                                                   | 0.4711
     cnn |██████████████████████████████████████████████                                                  | 0.4770

```

--- 
Artifacts saved:
- `outputs/processed_features.csv` (features actually used for training)
- `outputs/summary_models.csv` (row‑per‑model summary)
  - columns: `model`, `accuracy`, and any `noise_XX.XX` columns populated when per‑noise training runs
- Per‑model directory with detailed artifacts:
  - `outputs/mlp/` and/or `outputs/cnn/`
    - `confusion_matrix.csv`
    - `confusion_matrix_info.json` (mutual information and related stats computed from the confusion matrix)
    - `report.txt` (human‑readable summary)

Per‑noise training and summaries:
- If your dataset filenames contain a pattern like `noise_0.01`, a `noise` column is parsed automatically.
- When present, the CLI also trains per noise level and prints accuracy, confusion matrix, and MI per noise.
- Per‑noise overall accuracies are written to `outputs/summary_models.csv` under columns like `noise_0.01`, `noise_0.02`, ...

---

## License
MIT License