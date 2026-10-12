# mi-race — Specification (v1)

## Goal

Given N symbols and a fixed molecule budget, measure how reliably a receiver can
tell which symbol was sent through a diffusion channel with an absorbing receiver,
and how much information about the symbol survives.

v1 does **not** measure channel capacity. Results hold for one fixed codebook,
uniformly distributed symbols, one fixed channel and a trained decoder.

## Pipeline

```text
symbols (config or prompt) → encoder (release schedule) → SSA channel
  → receiver signal → decoder → predicted symbol
```

## Notation

| Name | Meaning |
| --- | --- |
| U | transmitted symbol, uniform on {0, …, N-1} |
| Û | decoder's prediction |
| Y | receiver signal (`rx_0 … rx_M`) |
| L | positions: compartments 0 … L-2, receiver at L-1 |
| h | compartment width |
| D | diffusion coefficient |
| λ = D / h² | jump rate of one molecule to each neighbour |
| dt, T | snapshot interval and duration; M = T / dt; t_m = m · dt |
| n_slots, slot_dt | number of release slots and their spacing |
| B | budget: molecules per symbol |
| pθ(u \| y) | decoder's probability of symbol u given signal y |

## Encoder (release vectors)

- Symbol u has a release vector r_u of n_slots non-negative integers, with Σ r_u = B for every u.
- Slot k releases r_u[k] molecules into compartment 0 at t = k · slot_dt, and every slot satisfies t < T.

## Channel

- Positions 0 … L-2 are compartments that hold molecules. Position L-1 is the
  receiver, an absorbing sink. All compartments start empty.
- Each molecule jumps to each neighbouring position at rate λ. Compartment 0 has
  a reflecting wall on its left side. Simulated exactly with SSA (Gillespie).
- A jump from L-2 into L-1 removes the molecule and increases the cumulative
  absorbed count A(t).
- Releases are deterministic external events, applied in time order:
  - If the next release comes before the next sampled reaction: advance to the
    release time, inject the molecules into compartment 0, discard the sampled
    reaction and recompute all propensities (exact, because waiting times are memoryless).
  - When no molecules are in the tube, time jumps straight to the next release (or to T).
  - A release at time t is applied before the snapshot at time t.
- Snapshots only record the state; they never alter or restart the SSA process.
- Conservation: molecules released up to t = molecules in compartments 0 … L-2 + A(t).

## Receiver signal

`rx_m` counts the absorptions in the interval (t_{m-1}, t_m]:

```text
rx_0 = 0
rx_m = A(t_m) - A(t_{m-1}),   m = 1 … M
```

`cumsum(rx)` recovers A(t).

## Config

- JSON with `"schema_version": 1`. The complete reference example is `configs/example.json`.
- Symbols are a list: the k-th release vector is symbol k.
- Unknown keys are an error, so typos are caught.

## Symbols input

- Written in the config file, or entered with an interactive prompt that checks
  the budget and writes the config.
- The prompt logic is built and tested in step 3 (simulated keyboard input) and
  becomes the `mi-race symbols` command in step 7.

## Data

- `runs_per_symbol` defaults to 300. Dataset columns: `symbol, rx_0 … rx_M`.
- One master seed, from which every simulation run gets its own seed.
- A stratified train/test split (`test_fraction`, default 0.2) is made once and
  shared by every decoder.

## Decoders

- Random forest (baseline): raw counts, no scaling.
- MLP: `StandardScaler` with each `rx_m` as one feature. Mean and variance are fitted
  on training rows only, then applied to training and test rows.
- CNN: one global mean and standard deviation, fitted on all training values
  (this keeps the shape of the signal over time), applied to training and test rows.
- CPU by default. NumPy, scikit-learn and PyTorch are seeded, with PyTorch
  deterministic mode on. "Same result" means same machine, same environment, CPU.
- Hyperparameters come from the config and are never tuned on the test set. If
  tuning is added, it uses a validation split carved from the training data.

## Metrics (on the test set)

- Primary: accuracy.
- Confusion matrix (all labels 0 … N-1, including never-predicted ones) and macro-F1.
- Hard-decision MI, the plug-in estimate from the test confusion matrix:

  ```text
  p(u, û) = confusion[u, û] / n_test
  I(U;Û) = Σ p(u,û) · log₂[ p(u,û) / (p(u) · p(û)) ]
  (0 · log 0 = 0)
  ```

  Never called channel MI or capacity. It is biased upward on small test sets,
  by roughly (N-1)² / (2 · n_test · ln 2) bits, so `n_test` is always reported with it.
- Later, for probabilistic decoders, the Barber–Agakov lower bound on I(U;Y):
  I_BA = log₂ N + mean(log₂ pθ(U | Y)).

## Config validation (step 1): reject before simulating

- `schema_version == 1`, and no unknown keys
- N ≥ 2
- L ≥ 2
- h, D, dt, T > 0, and T / dt is an integer (within float tolerance)
- n_slots ≥ 1, slot_dt > 0, (n_slots - 1) · slot_dt < T
- B > 0; every vector has exactly n_slots entries, all non-negative integers, summing to B
- 0 < test_fraction < 1, and runs_per_symbol · test_fraction ≥ 1 (each symbol has a
  test run); warn when runs_per_symbol < 300

## Out of scope (v1)

HDF5/OU data, RNN, KSG MI, reflecting receiver, observing other compartments,
custom channels, sparse `[[t, amount]]` symbols, channel capacity.

## Later (v2)

Learning the encoder (joint training), possibly with a pre-training stage.

## Build order (each step: explanation → tests → commit)

1. Example config, config loading and validation (no simulator or models yet)
2. SSA channel with absorbing receiver, including a conservation test
3. Symbols: interactive prompt logic
4. Dataset generation, then inspect the signals by eye
5. Decoders: random forest → MLP → CNN, sharing one split
6. Metrics
7. CLI (including `mi-race symbols`) and report
8. (v2) Encoder learning

## Lessons from legacy (do not repeat)

- Fit scalers after the split, on training data only.
- Seed torch for every model.
- Releases at exactly T were silently dropped, so slots must satisfy t < T.
- Validate configs before simulating.
