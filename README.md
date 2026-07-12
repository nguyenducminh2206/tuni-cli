# mi-race — Machine Learning for Science (CLI)


- **Encoder** (`mi_race/encoder/`) — maps each symbol to a **release vector**: how many molecules to drop into compartment 0 at each time slot (e.g. `[50, 0, 0, 20, …]`). Every symbol shares a fixed molecule **budget**.
- **Channel** (`mi_race/channel/`) — runs a **pluggable** simulation of the molecules across `L` compartments. Built-in: exact SSA diffusion. Or plug in your own physics (see [Custom channels](#custom-channels)).
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

Run from the repo root with the venv active. The pipeline is three commands:

```bash
# 1. define the codebook — writes release vectors into the config
mi-race symbols       -c configs/encoder.json --type time --n 3 --slots 20 --budget 100 --yes

# 2. run the channel over every symbol → labeled dataset + diagnostic plot
mi-race generate-data -c configs/encoder.json

# 3. train the decoder, compute MI/accuracy, render an HTML report
mi-race report        -c configs/encoder.json --open
```

| Step              | Produces                                                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `symbols`       | writes`channel.symbols` (dense release vectors) into the config; previews each symbol as bars                                                                              |
| `generate-data` | `data/encoder.csv` — one row per SSA run, all compartments recorded — plus `data/encoder_signal.png` (diagnostic). Set `channel.make_plots: false` to skip the plot. |
| `report`        | `experiments/encoder/report.html` (clean messages, averaged-received trajectories, confusion matrix, MI + accuracy) and `result.json`                                    |

Prefer terminal-only output? `mi-race run --model cnn -c configs/encoder.json` trains one model into `outputs/`, and `mi-race compare` prints accuracy bars. Running `mi-race` with no arguments opens an interactive shell.

---

## Commands

| Command                                                                                                   | What it does                                                                                                                                                                          |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mi-race symbols -c CONFIG [--type time\|two-pulse\|uniform\|manual] [--n N --slots S --budget B] [--yes]` | Build a codebook of release vectors and write it into the config. Interactive when`--type` is omitted; `manual` lets you type `slot:amount` pairs (checked against the budget). |
| `mi-race channels [--new FILE]`                                                                         | List built-in channels and their parameters.`--new` scaffolds a ready-to-edit custom-channel template.                                                                              |
| `mi-race generate-data -c CONFIG [--out FILE]`                                                          | Run the configured channel over`CONFIG.channel.symbols` and write a labeled CSV (all compartments).                                                                                 |
| `mi-race report -c CONFIG [--model M] [--name N] [--out DIR] [--open]`                                  | Train the decoder and render an HTML experiment report (MI, accuracy, confusion matrix, message/received figures).`--open` opens it in the browser.                                 |
| `mi-race run --model {mlp\|cnn\|rnn\|rf} -c CONFIG`                                                        | Train one model into`outputs/`. If `data.split_by` names a column, additionally retrain per unique value of it.                                                                   |
| `mi-race run-all -c CONFIG`                                                                             | Run every supported model sequentially. Failures in one model don't abort the rest. Prints a summary table.                                                                           |
| `mi-race compare [--split PREFIX]`                                                                      | Read`outputs/summary_models.csv` and print two terminal bar charts: overall accuracy and accuracy-vs-split.                                                                         |
| `mi-race concat -c CONFIG [--recursive] [--pattern *.csv] [--out FILE]`                                 | Concatenate CSVs from a folder into one file. Auto-detects nested subfolders.                                                                                                         |
| `mi-race filter -c CONFIG [--file F] [--column C --value V] [--out FILE]`                               | Interactive row filter (single column equals value) over a concatenated CSV.                                                                                                          |

Global flags: `-v`/`--version`. The startup banner and tqdm progress bars auto-suppress when stdout isn't a TTY (CI, pipes, captured output), so scripts stay clean without needing a flag.

---

## Configuration

A single JSON file drives everything. Sections are independent — `generate-data` only reads `channel` (and `data.path`), while `run` only reads `data` / `model` / `train` / `output`.

### `channel` — used by `generate-data` only

```json
"channel": {
  "type": "ssa",                    // "ssa", "ssa_absorbing", or "custom" — see `mi-race channels`
  "L": 10,                          // # compartments in the 1D lattice
  "S": 1.0,                         // compartment size [micron]
  "D": 2.0,                         // diffusion coefficient [micron^2/s]
  "dt": 0.01,                       // snapshot timestep [s]
  "T": 2.0,                         // total simulation time [s]
  "runs_per_symbol": 300,           // trajectories per symbol → rows = runs × len(symbols)
  "seed": 12345,                    // master RNG seed (per-run seeds are derived from this)
  "n_slots": 20,                    // release-vector length
  "slot_dt": 0.1,                   // seconds per slot → slot i releases at t = i * slot_dt
  "budget": 100,                    // total molecules per symbol (kept equal across symbols)
  "symbols": {                      // codebook: symbol_id → length-n_slots release vector
    "0": [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "1": [0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0]
  }
}
```

- **All compartments are always recorded** — the CSV has `comp0_*..comp{L-1}_*`. The decoder picks which to observe via `data.x_cols` (so one dataset serves every observation experiment). Column count is `1 + L·(int(T/dt) + 1)`.
- **Symbols are dense release vectors** (built with `mi-race symbols`). Each entry is the molecule count released at that slot; the vector sums to `budget`. Legacy sparse `[[t, amount]]` symbols still parse.
- **`type`** selects the channel implementation — see [Custom channels](#custom-channels). Run `mi-race channels` to list built-ins.

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
- **`x_cols` selects the observed compartment(s):** `"comp3_0:comp3_200"` observes one compartment (the bottleneck), a list like `["comp4_0:comp4_200", "comp5_0:comp5_200"]` observes several, and omitting `x_cols` uses all. Ranges are *inclusive on both ends*; if you change `T` or `dt`, the range end must change to match (`int(T/dt)`) or you get a "missing feature columns" error. To feed several compartments to the CNN at once, also set `model.cnn.multi_channel: true`.
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

## Custom channels

The channel is pluggable — you can replace the built-in SSA with your own physics without touching mi-race's source. A channel is just one function:

```python
simulate(schedule, cfg, rng) -> (times, X)
```

- `schedule` — list of `(release_time, amount)` tuples, molecules injected into compartment 0.
- `cfg` — the `channel` config block, so you can read your own parameters (`cfg["my_param"]`).
- `rng` — a NumPy random generator.
- returns `(times, X)` — `times` is a 1-D array of snapshot times (length `n_steps`); `X` is a 2-D int array of shape `(n_steps, L)` where `X[t, k]` is the molecule count in compartment `k` at time `t`.

### Step by step

**1. Scaffold a template** (gives you a working file with the right signature):

```bash
mi-race channels --new my_channel.py
```

**2. Edit `simulate()`** in `my_channel.py` — fill in your physics. The template already handles injecting the schedule; you add how molecules move between compartments, then return `(times, X)`.

**3. Point the config at it** (`configs/mine.json`):

```json
"channel": {
  "type": "custom",
  "impl": "my_channel.py:simulate",
  "my_param": 0.25,
  "L": 8, "dt": 0.01, "T": 2.0,
  "runs_per_symbol": 300,
  "n_slots": 20, "slot_dt": 0.1, "budget": 100
}
```

**4. Run the pipeline exactly as before** — nothing else changes:

```bash
mi-race symbols       -c configs/mine.json --type time --n 8 --slots 20 --budget 100 --yes
mi-race generate-data -c configs/mine.json          # generate-data prints "channel=custom"
mi-race report        -c configs/mine.json --open
```

`impl` accepts either a file path (`my_channel.py:simulate`) or an installed module (`my_pkg.channels:simulate`). Custom channels are output-validated on every call, so a wrong return shape fails with a clear message instead of a deep crash. Run `mi-race channels` any time to list the built-ins.

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
mi-race symbols              →   mi_race/encoder/symbols.py       (build release vectors → config)

mi-race generate-data        →   mi_race/encoder/codebook.py      (symbol vector → schedule)
                                  mi_race/encoder/dataset_gen.py   (orchestrator)
                                  mi_race/channel/registry.py      (build_channel: ssa / custom)
                                  mi_race/channel/simulation.py    (SSA engine)
                                  → data/<name>.csv  +  <name>_signal.png

mi-race report               →   mi_race/reporting/experiment_report.py
                                  (train decoder → MI/accuracy/figures → experiments/<name>/report.html)

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
