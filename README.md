# mi-race — Machine Learning for Science (CLI)

A single CLI that runs the full **encoder → channel → decoder** pipeline:

- **Encoder** (`mi_race/encoder/`) — maps symbols to release schedules `[(t, amount), ...]` on compartment 0.
- **Channel** (`mi_race/channel/`) — runs an exact SSA simulation of 1D diffusion across `L` compartments.
- **Decoder** (`mi_race/train/`) — trains a classifier (MLP / 1D CNN / RNN / Random Forest) to recover the symbol from the observed trajectory.

Everything is config-driven. One JSON file describes the codebook, the simulator parameters, the dataset split, and the model hyperparameters. Artifacts (confusion matrix, MI, NMI, KSG MI, classification report) land in `outputs/`.

---

## Install

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
cd mi-race
pip install -e .
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
cd mi-race
pip install -e .
```

This makes the `mi-race` command available. To leave the venv: `deactivate`.

---

## Quickstart

The full pipeline is three commands. Run from the repo root with the venv active:

```bash
mi-race generate-data -c configs/encoder.json   # 1. simulate dataset
mi-race run --model cnn -c configs/encoder.json # 2. train decoder
mi-race compare                                        # 3. plot accuracy
```

What you get:

| Step | Produces |
|------|----------|
| `generate-data` | `data/encoder.csv` — 900 rows × 202 cols (3 symbols × 300 runs, 201 timesteps + 1 label).<br>Plus `data/encoder_signal.png` — one diagnostic figure showing all compartments under the merged release schedule. Set `channel.make_plots: false` to skip. |
| `run` | `outputs/cnn/{confusion_matrix.csv, confusion_matrix_info.json, report.txt}` and a row in `outputs/summary_models.csv` |
| `compare` | Terminal bar charts of overall accuracy and (if applicable) per-split accuracy |

To train every supported model in one go, swap step 2 for `mi-race run-all -c configs/encoder.json`.

Run `mi-race` with no arguments to drop into an interactive shell with the same subcommands plus an `edit` shortcut for the config.

---

## Commands

| Command | What it does |
|---------|--------------|
| `mi-race generate-data -c CONFIG [--out FILE]` | Run the SSA channel over the codebook in `CONFIG.channel.symbols` and write a labeled CSV. |
| `mi-race run --model {mlp\|cnn\|rnn\|rf} -c CONFIG` | Train one model. If `data.split_by` names a column in the dataset, additionally retrain per unique value of that column. |
| `mi-race run-all -c CONFIG` | Run every supported model sequentially. Failures in one model don't abort the rest. Prints a summary table. |
| `mi-race compare [--split PREFIX]` | Read `outputs/summary_models.csv` and print two terminal bar charts: overall accuracy and accuracy-vs-split. |
| `mi-race concat -c CONFIG [--recursive] [--pattern *.csv] [--out FILE]` | Concatenate CSVs from a folder into one file. Auto-detects nested subfolders. |
| `mi-race filter -c CONFIG [--file F] [--column C --value V] [--out FILE]` | Interactive row filter (single column equals value) over a concatenated CSV. |

Global flags: `-v`/`--version`. The startup banner and tqdm progress bars auto-suppress when stdout isn't a TTY (CI, pipes, captured output), so scripts stay clean without needing a flag.

---

## Configuration

A single JSON file drives everything. Sections are independent — `generate-data` only reads `channel` (and `data.path`), while `run` only reads `data` / `model` / `train` / `output`.

### `channel` — used by `generate-data` only

```json
"channel": {
  "L": 10,                          // # compartments in the 1D lattice
  "S": 1.0,                         // compartment size [micron]
  "D": 2.0,                         // diffusion coefficient [micron^2/s]
  "dt": 0.01,                       // snapshot timestep [s]
  "T": 2.0,                         // total simulation time [s]
  "observed_compartment": 9,        // which compartment is recorded (default: L-1)
  "save_all_compartments": false,   // if true, write comp0_*..comp{L-1}_*
  "runs_per_symbol": 300,           // SSA trajectories per symbol → rows = runs × len(symbols)
  "seed": 12345,                    // master RNG seed (per-run seeds are derived from this)
  "symbols": {                      // codebook: symbol_id → list of [t, amount] pulses on compartment 0
    "0": [[0.0, 100]],
    "1": [[0.5, 100]],
    "2": [[1.0, 100]]
  }
}
```

- Output column count is `1 + (int(T/dt) + 1)` — one label column plus one column per timestep.
- A symbol can have multiple pulses, e.g. `"0": [[0.0, 50], [1.0, 50]]`.
- Pulses outside `[0, T]` are dropped with a warning.

### `data` — used by `run` / `run-all`

```json
"data": {
  "path": "data/encoder.csv",                // file, folder (auto-concat), or dataset id
  "x_cols": "comp9_0:comp9_200",             // range notation, list, or omitted (all numeric)
  "y_col": "symbol",                         // required: target column
  "sequence_mode": "split",                  // "split" (timestep cols → CNN) or "ignore"
  "balance": false,                          // undersample to min class count
  "split_by": "sigma",                       // optional: also train per unique value
  "filter": "sigma==0.5 & mu==1.5"           // optional: pandas-style row filter
}
```

- **`path` resolution:** existing file → read by extension (`.csv`/`.tsv`/`.parquet`). Existing directory or extensionless string → treated as a dataset id (`build_df(name)`).
- **`x_cols` ranges:** `"prefix_start:prefix_end"` expands to all matching columns *inclusive on both ends*. If you change `T`, `dt`, or `observed_compartment` in the `channel` block, this range must change to match — a mismatch produces a "missing feature columns" error.
- **`sequence_mode`:** `"split"` is required for CNN. RNN ignores split columns and reads raw lists from the column named by `sequence_col`.
- CSV cannot round-trip Python lists. If your sequence column is `"[1.0, 1.1, ...]"` strings, use Parquet/PKL instead.

### `model` — per-model hyperparameters

```json
"model": {
  "mlp": { "hidden_layers": [128, 128], "activation": "relu", "lr": 0.001,
           "dropout": 0.0, "batch_size": 64, "epochs": 15 },
  "cnn": { "channels": [16, 32], "kernel_size": 5, "pool": 2, "fc": 128,
           "lr": 0.001, "batch_size": 32, "epochs": 15, "device": "auto" },
  "rnn": { "sequence_col": "time_trace", "max_len": 500, "pad_value": 0.0,
           "hidden_size": 128, "num_layers": 1, "bidirectional": true,
           "lr": 0.001, "batch_size": 8, "epochs": 5, "device": "auto" },
  "rf":  { "n_estimators": 200, "max_depth": null, "random_state": 42 }
}
```

- **CNN** needs `data.sequence_mode: "split"` and `x_cols` covering the timestep range. Use `sequence_prefix` when multiple sequence groups exist.
- **RNN** reads raw sequences from `sequence_col` in the *original* dataframe — not the split columns.
- **RF** labels are internally cast to floats for sklearn compatibility.
- `device: "auto"` selects CUDA when available, else CPU.

### `train` / `output`

```json
"train":  { "test_size": 0.2, "random_state": 42, "standardize": true },
"output": { "dir": "outputs", "show_report": true, "ksg_k": 5 }
```

- `standardize` adds a `StandardScaler` on features.
- `show_report` controls whether the per-class precision/recall/F1 table is printed.
- `ksg_k` is the neighborhood size for the KSG MI estimator; only computed when the model emits probabilities (MLP, RF).

---

## What gets saved

Each `mi-race run` writes:

```
outputs/
├── processed_features.csv          # features actually fed to the model (skipped for RNN)
├── processed_features_rnn.csv      # padded sequence matrix (RNN runs only)
├── summary_models.csv              # one row per model; columns: model, accuracy, <split>_<value>...
├── report_detailed_global.txt      # appended on every run (delete manually if it grows)
└── <model>/
    ├── confusion_matrix.csv
    ├── confusion_matrix_info.json  # I(true;pred), NMI_sqrt, NMI_min, NMI_max, KSG MI (if available)
    └── report.txt                  # human-readable summary
```

`compare` reads `outputs/summary_models.csv` and prints two terminal bar charts.

---

## Architecture at a glance

```
configs/*.json
      │
      ▼
mi-race generate-data        →   mi_race/encoder/codebook.py     (symbol → schedule)
                                  mi_race/encoder/dataset_gen.py  (orchestrator)
                                  mi_race/channel/simulation.py   (simulate_ssa_with_schedule)
                                  → data/<name>.csv

mi-race run / run-all        →   mi_race/cli/main.py
                                  mi_race/train/orchestrator.py   (central controller)
                                  mi_race/train/registry.py       (ModelSpec adapters)
                                  mi_race/train/models/{mlp,cnn,rnn,random_forest}.py
                                  mi_race/analysis/metrics.py     (confusion matrix → MI)
                                  mi_race/reporting/report.py
                                  → outputs/

mi-race compare              →   mi_race/reporting/compare_models.py
                                  (reads outputs/summary_models.csv)
```

The orchestrator never branches on model type. Each entry in `train/registry.py` is a `ModelSpec` with a uniform-signature adapter; the orchestrator just calls `spec.adapter(...)`. To add a new model, register a new `ModelSpec` — no orchestrator change needed.

---

## Tests

```bash
cd mi-race
pytest                                                  # full suite
pytest tests/test_phase2_registry.py::test_name         # single test
```

`tests/conftest.py` provides synthetic-data fixtures (`synthetic_df`, `synthetic_csv`, `synthetic_config`) that mirror the real dataset shape (`mu` target, `sigma` split, `time_point_1..11` features). Reuse them in new tests rather than fabricating fresh data.

---

## License

MIT — see `LICENSE`.
