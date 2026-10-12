
# mi-race rebuild — roadmap

Pipeline: **symbols → encoder → SSA channel (absorbing receiver) → decoder**.
The rules live in [SPEC.md](SPEC.md). This file is the order of work: tick a box when it's done.

## How each part works

1. **Code** — write the piece (one function or class).
2. **Try it** — run a small `python -c "..."` command and look at the output.
3. **Test** — add tests to `tests/`, then run `.venv/bin/python -m pytest` until everything passes.
4. **Commit** — one small commit per part.

Don't start the next part until all four boxes of the current one are ticked.

---

## Phase 0 — clean start ✅

- [X] Freeze the old version: branch `legacy`, tag `legacy-v1`, pushed to GitHub
- [X] Reference copy of the old code in `../tuni-cli-legacy`
- [X] Untracked results and notes zipped to `../tuni-cli-legacy-data.zip`
- [X] Remove the old code from `main`
- [X] Write `SPEC.md`
- [X] Skeleton: `pyproject.toml`, `src/mi_race/__init__.py`, `tests/test_smoke.py`
- [ ] Add the pending sections to `SPEC.md`: "CLI look", and the extra config rules (output.dir, decoders may be empty, declared decoder fields, separate random streams, path rule decided in step 4)

---

## Step 1 — config loading and validation (`src/mi_race/config.py`)

### Part 1 — `configs/example.json`

- [X] Code: create the file
- [X] Try it: `python3 -m json.tool configs/example.json`
- [X] Commit

### Part 2 — `ConfigError`

- [X] Code
- [X] Try it: raise one and catch one
- [X] Commit

### Part 3 — data classes (the shape of a valid config)

**3a — `ChannelConfig`** (`@dataclass`, `frozen`, `@property`)

- [X] Code
- [X] Try it
- [X] Test: jump rate, n_intervals, float rounding, can't be changed
- [ ] Commit

**3b — `EncoderConfig`** (tuples, `n_symbols`)

- [ ] Code
- [ ] Try it: tuples vs lists
- [ ] Test: n_symbols, symbols can't be changed
- [ ] Commit

**3c — `DataConfig` and `OutputConfig`** (default values)

- [ ] Edit `configs/example.json`: rename `hidden` → `hidden_layers`, add `"output": {"dir": "outputs"}`
- [ ] Code
- [ ] Try it
- [ ] Test: defaults are 300 runs, 0.2 test fraction, `outputs`
- [ ] Commit

**3d — `Config`** (puts it together, `field(default_factory=…)`)

- [ ] Code
- [ ] Try it
- [ ] Test
- [ ] Commit

### Part 4 — `load_config`: read the file

- [ ] Code: file not found → `ConfigError`; broken JSON → `ConfigError`; duplicate key → `ConfigError`
- [ ] Try it
- [ ] Test: missing file, broken JSON, duplicate key
- [ ] Commit

### Part 5 — `_check_keys`: missing and unknown keys

- [ ] Code: "did you mean …?" for typos
- [ ] Try it: `runs_per_symbo`
- [ ] Test: unknown key, missing key, section that isn't an object
- [ ] Commit

### Part 6 — `_whole` and `_positive`: number checks

- [ ] Code: reject `true`/`false`, decimals where whole numbers are needed, `NaN`, infinity, ≤ 0
- [ ] Try it
- [ ] Test
- [ ] Commit

### Part 7 — `_parse_channel`

- [ ] Code: L ≥ 2; h, D, dt, T > 0; T / dt whole (with float tolerance)
- [ ] Try it
- [ ] Test: each rule
- [ ] Commit

### Part 8 — `_parse_encoder`

- [ ] Code: n_slots ≥ 1, slot_dt > 0, last slot before T, ≥ 2 symbols, each vector has n_slots non-negative whole numbers summing to the budget; lists → tuples
- [ ] Try it
- [ ] Test: each rule
- [ ] Commit

### Part 9 — `_parse_data`, `_parse_output`, `_parse_decoders`

- [ ] Code: path non-empty; 0 < test_fraction < 1; every symbol gets ≥ 1 training and ≥ 1 test run; warn below 300 runs; output.dir default; decoder names and their declared fields; empty decoders allowed
- [ ] Try it
- [ ] Test: each rule, plus the warning
- [ ] Commit

### Part 10 — `parse_config`: put it all together

- [ ] Code: `schema_version` must be 1; `seed` default 0; builds the full `Config`
- [ ] Try it: load `configs/example.json`, then break it on purpose and read the messages
- [ ] Test: the example loads; optional keys get defaults; wrong schema version
- [ ] Commit — **step 1 done**

---

## Step 2 — SSA channel with absorbing receiver *(draft — refined when we get there)*

- [ ] 2a — release vector → release schedule `[(time, molecules), …]`
- [ ] 2b — jump rates (propensities) for the current molecule counts
- [ ] 2c — one SSA event: when the next jump happens and which one
- [ ] 2d — the main loop: releases as events (discard the sampled jump), empty tube jumps ahead
- [ ] 2e — snapshots and the receiver signal: `rx_m` = absorptions in (t_{m-1}, t_m]
- [ ] 2f — tests: conservation, release at t = 0, release timing, same seed → same result, L = 2
- [ ] 2g — look at a few signals by eye

## Step 3 — symbols: interactive prompt logic *(draft)*

- [ ] 3a — turn one typed line into a release vector
- [ ] 3b — the prompt loop with budget check (keyboard input replaceable in tests)
- [ ] 3c — write the symbols into a config
- [ ] 3d — tests with simulated keyboard input

## Step 4 — dataset generation *(draft)*

- [ ] 4a — random streams from the master seed (simulation / split / each decoder)
- [ ] 4b — simulate every symbol `runs_per_symbol` times
- [ ] 4c — write the CSV `symbol, rx_0 … rx_M` (decide the path rule here)
- [ ] 4d — inspect the signals by eye (mean signal per symbol)

## Step 5 — decoders *(draft)*

- [ ] 5a — load the dataset and make the stratified split once
- [ ] 5b — check each decoder's settings
- [ ] 5c — random forest (raw counts)
- [ ] 5d — MLP (per-feature scaler fitted on training rows)
- [ ] 5e — CNN (one global scaler fitted on training values)
- [ ] 5f — seeding and determinism tests

## Step 6 — metrics *(draft)*

- [ ] Accuracy
- [ ] Confusion matrix with all labels 0 … N-1
- [ ] Macro-F1
- [ ] Hard-decision MI I(U;Û), reported with n_test

## Step 7 — CLI and report *(draft)*

- [ ] `ui.py`: `[mi-race]` prefix, warnings and errors, boxes, banner
- [ ] Commands, including `mi-race symbols`
- [ ] Report

## Step 8 — (v2) encoder learning *(later)*

- [ ] To be planned after v1 works
