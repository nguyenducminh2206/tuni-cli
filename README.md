# mi-race — Machine Learning for Science (CLI)

A lightweight command‑line tool to train and evaluate ML models directly from the terminal on tabular and sequence‑like data. It supports:
- Fast CSV/TSV/Parquet loading
- Config‑driven training (currently MLP; CNN placeholder in config)
- Sequence columns (e.g. `time_trace`) → statistical feature expansion
- Clean terminal report: preview, class counts, accuracy, macro‑F1, confusion matrix
- Artifacts saved to `outputs/`

---

## 1. Environment Setup (Recommended)

### Windows (PowerShell)
```powershell
python -m venv .venv              # create virtual environment
.\.venv\Scripts\activate          # activate it
python -m pip install -U pip      # upgrade pip (optional but good practice)
pip install -r requirements.txt
pip install -e .                  # editable install to use `mi-race` CLI
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
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
# or (module form)
python -m mi_race.cli.main run --model mlp -c config.json
```

Preview a file:
```bash
mi-race load processed_data/iris_ds.csv --label species
```

---

## 3. Configuration (config.json)

```json
{
  "data": {
    "path": "data_7x7",       // path/to_your/dataset
    "y_col": "dis_to_target", // labels for the model
    "x_cols": ["time_trace_0:time_trace_99", "cMax", "cVar"], // inputs for the model
    "sequence_mode": "split" // 3 modes: "split" | "stats" | "ignore"
  },
  "model": {
    "type": "mlp",
    "hidden_layers": [128, 128],
    "activation": "relu",
    "alpha": 0.0001
  },
  "train": {
    "test_size": 0.2,
    "random_state": 42,
    "max_iter": 200,
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
- **`x_cols`**/**`x_col`**: feature columns. If omitted, all **numeric** columns except `y_col` are used.
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
- type: mlp
- hidden_layers: list of layer sizes
- activation: relu | tanh | logistic | identity
- alpha: L2 regularization
(You may add solver, learning_rate_init, batch_size, etc.)

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
3. Split train/test (optionally stratified).
4. Build a `Pipeline(StandardScaler? → MLPClassifier)` and **fit**.
5. Print a terminal report and **save artifacts**.

Example terminal output (abridged):

```
[mi-race] Loaded dataset with columns: ['cell_id', 'time_trace', 'dis_to_target', 'cMax', 'cVar']
[mi-race] Sequence column 'time_trace' converted to stats: ['time_trace_len', 'time_trace_mean', ...]
[mi-race] Final feature columns (n=...): ['cMax', 'cVar', 'time_trace_len', ...]
[mi-race] Class distribution (full): {0: 90, 1: 732, ...}

=== Preview (first rows) ===
 cell_id  time_trace                        dis_to_target   cMax     cVar
 0        [1.0000, 1.0001, …, 1.0366] ...  4               1.1296   0.0020
 ...

Total rows: 235,200  •  Classes: [0, 1, 2, 3, 4, 5]

=== Metrics (test) ===
Accuracy: 41.65%
Macro F1: 30.12%

=== Confusion Matrix (test) ===
          0     1     2     3     4     5
    ------------------------------------
  0 |    86     7     1     2     0     0
  1 |     0   388    81    69    35     3
  2 |     0    71   433   426   197    25
  3 |     2    82   380   750   444    70
  4 |     0    52   170   386   295    57
  5 |     0     6    39    86    51    10

=== Classification Report (test) ===
precision recall f1-score support
...
```

Artifacts saved:
- `outputs/confusion_matrix_global.csv` (counts matrix; unlabeled CSV for easy downstream use)

---

## Command reference

### `load`
Preview a CSV quickly (first 5 rows), and optionally show the number of unique classes in a label column.
```
mi-race load path/to/file.csv --label dis_to_target
```

### `run`
Run training and evaluation from a JSON config.
```
mi-race run --model mlp -c config.json
# or
python -m mi_race.cli.main run --model mlp -c config.json
```

---

## License
MIT License