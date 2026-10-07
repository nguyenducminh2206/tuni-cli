# mi-race Improvement Recommendations

This document consolidates recommendations for improving the `mi-race` encoder-decoder optimization workflow after comparing it with the `Optimal_Encoder_for_Noisy_Channel` project and the model-free end-to-end communication approach.

## Executive summary

The current `mi-race` design makes the correct fundamental choice for its channel:

- The decoder is trained by ordinary backpropagation.
- The encoder is trained with REINFORCE because the stochastic simulation algorithm (SSA) is not differentiable.
- The encoder directly represents a discrete, molecule-budget-constrained codebook.

The highest-value improvement is not to replace the encoder with a conventional neural network. It is to make joint training more stable:

1. Initialize an exploratory encoder policy.
2. Pretrain the decoder while the encoder is frozen.
3. Alternate several decoder updates with one encoder update.
4. Anneal encoder exploration during training.
5. Select the best encoder using independent validation simulations.
6. Evaluate the final codebook with more trajectories and multiple random seeds.

## 1. What should remain unchanged

The following parts of the current architecture are well suited to molecular communication and should be preserved:

- The `EncoderPolicy` logit table for per-symbol release-slot probabilities.
- Multinomial sampling of molecule packets.
- Exact normalization to a fixed molecule budget.
- Masking of release slots outside the channel time window.
- SSA as the authoritative physical channel.
- REINFORCE for updating the encoder.
- The leave-one-out per-symbol reward baseline.
- `log1p` preprocessing of molecule counts.
- Fresh final datasets and fresh decoders for baseline-versus-optimized evaluation.
- Accuracy, macro F1, confusion-matrix MI, and NMI as evaluation metrics.
- The standard model registry for final decoder evaluation.

A general CNN, RNN, or MLP transmitter is not necessary for the present finite-symbol problem. With one-hot symbol input, the current trainable table is already the direct parameterization of a codebook. A larger neural encoder would become useful only if symbols had meaningful structure, the symbol space became very large, or the encoder needed to condition on additional context.

## 2. Highest-priority training improvement

### Current limitation

The decoder and encoder currently begin training together. Early encoder rewards therefore come from an almost random decoder and contain little useful information.

Within one iteration, the current process is approximately:

1. Simulate one batch.
2. Compute decoder logits and correct-class log-probabilities.
3. Update the decoder.
4. Update the encoder using rewards calculated before the decoder update.

This makes the encoder learn from a noisy and slightly stale reward signal.

### Recommended three-stage process

#### Stage A: initialize an exploratory encoder

Start from the configured baseline codebook while retaining broad exploration:

```json
{
  "init": "baseline",
  "init_mix": 0.5
}
```

An `init_mix` of `0.5` assigns meaningful probability to the baseline release pattern while still exploring other valid slots.

Do not initialize the policy as a completely deterministic baseline. A deterministic policy can prevent discovery of better schedules. It also trains the decoder only on the baseline distribution, leaving the decoder unreliable on new schedules.

#### Stage B: pretrain the decoder

Before updating the encoder, freeze the encoder and train the decoder for approximately 50-100 steps.

Each decoder warm-up step should:

1. Sample balanced schedules for every symbol from the fixed exploratory policy.
2. Simulate fresh SSA trajectories.
3. Compute decoder logits.
4. Minimize ordinary categorical cross-entropy.
5. Update only decoder parameters.

The warm-up data should come from the exploratory policy, not only from deterministic baseline schedules. This helps the decoder assign meaningful rewards to schedules outside the original codebook.

#### Stage C: alternate decoder and encoder updates

After decoder warm-up, use explicit alternating phases. A reasonable starting ratio is five decoder updates for every encoder update.

```text
repeat:
    decoder phase:
        perform 5 updates using fresh SSA batches
        encoder remains frozen

    encoder phase:
        perform 1 update using a new independent SSA batch
        decoder remains frozen
```

The decoder objective remains:

```text
L_decoder = -mean(log q(true_symbol | received_trace))
```

The encoder objective remains:

```text
L_encoder = -mean((reward - baseline) * log policy(action | symbol))
            - entropy_coefficient * policy_entropy
```

The encoder phase should simulate a fresh batch after the decoder phase. This ensures that encoder rewards are calculated with the current decoder and avoids reusing the decoder's training batch as policy feedback.

### Suggested optimizer configuration

Add settings similar to these to `OPTIMIZE_DEFAULTS`:

```python
"decoder_pretrain_steps": 100,
"decoder_steps": 5,
"encoder_steps": 1,
"decoder_weight_decay": 1e-4,
"grad_clip": 1.0,
"entropy_coef_start": 0.04,
"entropy_coef_end": 0.003,
"validation_every": 20,
"validation_runs_per_symbol": 32,
```

Separate the training code into helpers such as:

```python
pretrain_decoder(...)
update_decoder(...)
update_encoder(...)
evaluate_policy(...)
```

This separation will make it easier to test that parameters change only during their intended phase.

## 3. Decoder improvements

### Use AdamW

Use AdamW for the joint-training decoder with modest weight decay:

```python
dec_opt = torch.optim.AdamW(
    decoder.parameters(),
    lr=lr_decoder,
    weight_decay=decoder_weight_decay,
)
```

This can improve regularization without changing the decoder objective.

### Clip decoder gradients

After decoder backpropagation, clip gradients:

```python
torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
```

A default maximum norm of `1.0` is a reasonable starting point.

### Consider normalization inside the CNN

If decoder training is unstable, consider `GroupNorm` after convolutional layers. It is preferable to batch normalization when batch sizes may change or be small.

Do not normalize each received trace independently, because absolute molecule counts may contain useful symbol information. If input standardization is added, estimate global per-channel statistics from decoder warm-up data and reuse them consistently.

### Avoid label smoothing initially

The encoder reward is the correct-class log-probability and is interpreted through a Barber-Agakov mutual-information lower bound. Label smoothing changes probability calibration and makes this interpretation less direct.

Label smoothing can be investigated later as an ablation, but should not be the initial default.

### Avoid fixed temperature scaling initially

Arbitrary temperature scaling also changes the probabilities used for the encoder reward. If temperature scaling is tested, compare it explicitly against an unscaled baseline and keep the reward definition clearly documented.

## 4. Encoder improvements

### Keep the discrete policy

The existing policy matches the physical action space:

- Releases are non-negative.
- Releases occur in discrete slots.
- Molecule amounts are integers.
- Every symbol has the same total molecule budget.

This is more appropriate than producing an unconstrained continuous waveform and projecting it afterward.

### Anneal entropy

The current entropy coefficient remains constant. Replace it with a schedule, for example:

```text
start: 0.04
end:   0.003
```

The coefficient can decay linearly over the first 70-90% of training.

High early entropy encourages exploration. Low late entropy allows the policy to concentrate into a deterministic, usable codebook.

### Increase policy-gradient sample count

For the eight-symbol hard experiment, `per_symbol = 8` may produce a noisy gradient estimate. Test:

```text
per_symbol = 16
```

and compare against `32` if simulation cost is acceptable.

Larger per-symbol batches improve both the leave-one-out baseline and the policy-gradient estimate.

### Preserve the leave-one-out baseline

The current leave-one-out reward baseline is a useful variance-reduction technique and should remain. It is better matched to the balanced per-symbol batch structure than a single global reward average.

Possible later extensions include:

- An exponential-moving-average baseline per symbol.
- A learned value baseline.
- Advantage normalization per symbol rather than across the complete batch.

These should be treated as ablations rather than introduced simultaneously.

## 5. Independent validation and checkpoint selection

Do not automatically return the policy from the last training iteration.

Every `validation_every` outer iterations:

1. Convert policy probabilities into a deterministic, budgeted codebook.
2. Generate validation trajectories using an independent RNG stream.
3. Evaluate the codebook with the current decoder.
4. Record validation accuracy, cross-entropy, and MI.
5. Save copies of the policy logits and decoder parameters when validation MI improves.

At the end of training, restore the best policy and decoder states.

Training diagnostics should be clearly separated from validation results. Accuracy and MI calculated on batches used to train the decoder are useful progress indicators but are not unbiased estimates of generalization.

The existing final evaluation protocol should remain authoritative:

- Generate a fresh baseline dataset.
- Generate a fresh optimized-codebook dataset.
- Train a fresh decoder for each codebook.
- Compare them using the same model, split, and number of trajectories.

## 6. Evaluation quality and reproducibility

### Use more final trajectories

The current edited hard configuration uses 50 runs per symbol:

```text
8 symbols x 50 runs = 400 rows
20% test split = 80 test rows
approximately 10 test examples per class
```

This is adequate for a smoke test but too small for a stable scientific comparison. Per-class accuracy moves in approximately ten-percentage-point increments, and confusion-matrix MI is noisy.

For final experiments, use at least:

```json
"runs_per_symbol": 300
```

Alternatively, keep a small generation setting for development and override final optimization evaluation:

```json
"optimize": {
  "eval_runs_per_symbol": 300
}
```

### Run multiple seeds

Policy-gradient results should be evaluated over at least five independent seeds.

Report:

- Mean accuracy and standard deviation.
- Mean confusion-matrix MI and standard deviation.
- Mean NMI and standard deviation.
- Individual seed results.
- Best, median, and worst learned codebooks when useful.

Use identical evaluation budgets and decoder settings for baseline and optimized codebooks.

### Add scientific baselines

Compare at least:

1. Hand-designed baseline codebook.
2. Random valid codebook.
3. Learned codebook.
4. Learned codebook without decoder pretraining.
5. Learned codebook without entropy annealing.
6. Different `per_symbol` values.
7. Different `quanta` values.

This will establish which parts of the optimizer provide the improvement.

## 7. Configuration and documentation changes

### Clarify the observed compartment

The edited `encoder_hard.json` observes compartment 7, while the README describes it as observing compartment 3.

If compartment 7 is intentional, update the documentation. A clearer experimental organization would use two configs:

```text
configs/encoder_hard_midpoint.json
configs/encoder_hard_receiver.json
```

The midpoint config would observe compartment 3. The receiver config would observe compartment 7.

### Separate development and final-evaluation settings

Make the distinction explicit:

```text
fast development:
    fewer optimization steps
    fewer runs per symbol
    one seed

final experiment:
    full optimization steps
    at least 300 evaluation runs per symbol
    at least five seeds
```

This avoids accidentally publishing results produced by smoke-test settings.

## 8. Correctness and robustness fixes

### Enforce matching multi-compartment time ranges

`observed_compartments()` currently reduces all requested compartment ranges to one combined minimum-to-maximum slice.

For example:

```json
[
  "comp4_0:comp4_10",
  "comp6_100:comp6_200"
]
```

would effectively select timesteps 0-200 for both compartments.

Choose one of these behaviors:

1. Require all selected compartments to use the same start and end indices and reject inconsistent ranges.
2. Implement independent time ranges and define how unequal lengths are aligned.

The first option is simpler and matches the current CNN input representation.

### Validate optimizer settings

Validate the following before simulation begins:

- `steps >= 1`
- `per_symbol >= 2`
- `quanta >= 1`
- `lr_encoder > 0`
- `lr_decoder > 0`
- `0 <= init_mix <= 1`
- Entropy coefficients are non-negative.
- At least one release slot lies within `[0, T]`.
- All codebook vectors have the declared `n_slots` length.
- All molecule amounts are non-negative integers.
- Every symbol has the declared molecule budget.
- Symbol identifiers are unique and convertible to integers.
- Validation and evaluation run counts are positive.

Fail with actionable configuration errors instead of allowing NaNs or silently changing an experiment.

### Fix headless plotting

The test suite can abort on macOS when Matplotlib selects the GUI backend during dataset generation. It succeeds when run with:

```bash
MPLBACKEND=Agg ../.venv/bin/pytest
```

Because the CLI saves figures rather than requiring an interactive window, configure a noninteractive backend for saved plots or set `MPLBACKEND=Agg` in test configuration.

Possible fixes include:

- Set the backend before importing `matplotlib.pyplot` in plotting modules.
- Add an environment setting in test configuration.
- Use Matplotlib's object-oriented noninteractive canvas directly.

### Correct supported Python metadata

The package metadata currently declares Python 3.9 support, but the source uses `X | None` type-union syntax, which requires Python 3.10 or newer.

Change:

```toml
requires-python = ">=3.9"
```

to:

```toml
requires-python = ">=3.10"
```

Alternatively, replace the newer union syntax with `Optional[...]`, but raising the minimum version is simpler.

### Validate baseline codebooks before optimization

The optimizer currently derives `n_slots` and `budget` from configuration and the first codebook entry. Add validation that every baseline symbol:

- Has the same vector length.
- Has the same total molecule budget.
- Uses no negative entries.
- Does not place molecules after `T`.

This prevents the optimizer from silently training against a malformed baseline.

## 9. Testing recommendations

Add tests for each training phase and invariant.

### Decoder pretraining tests

- Encoder logits do not change during decoder warm-up.
- Decoder parameters do change.
- Decoder loss decreases on a deterministic toy channel.
- Warm-up batches contain equal samples per symbol.

### Alternating training tests

- Decoder parameters remain unchanged during an encoder-only update.
- Encoder logits remain unchanged during a decoder-only update.
- The encoder phase uses a fresh simulation batch.
- The configured decoder-to-encoder update ratio is respected.

### Entropy schedule tests

- The coefficient begins at the configured start value.
- It decreases monotonically.
- It reaches the configured final value.

### Validation tests

- Validation uses an RNG independent of training.
- Best policy parameters are restored at the end.
- Training metrics and validation metrics are stored separately.
- Validation performs no parameter updates.

### Configuration validation tests

- Reject `init_mix` outside `[0, 1]`.
- Reject unequal codebook lengths.
- Reject inconsistent molecule budgets.
- Reject negative molecule amounts.
- Reject configurations with no valid release slots.
- Reject inconsistent multi-compartment time ranges.

### Headless plotting test

- Generate a dataset and report using a noninteractive backend in a headless environment.

Avoid strict unit tests asserting that stochastic optimization always beats the baseline after only a few steps. Test invariants deterministically and reserve performance comparisons for longer integration or experiment tests.

## 10. Suggested implementation order

### Phase 1: engineering correctness

1. Fix the Matplotlib headless backend.
2. Change the declared Python requirement to 3.10 or newer.
3. Add optimizer and codebook validation.
4. Reject inconsistent multi-compartment time ranges.
5. Resolve the compartment 3 versus compartment 7 documentation mismatch.

### Phase 2: training stability

1. Add decoder pretraining.
2. Split the combined loop into decoder and encoder update helpers.
3. Use fresh batches for each phase.
4. Add the configurable decoder-to-encoder update ratio.
5. Use AdamW and decoder gradient clipping.
6. Add entropy annealing.

### Phase 3: model selection and evaluation

1. Add independent validation simulations.
2. Save and restore the best policy.
3. Separate training, validation, and final evaluation metrics.
4. Raise final evaluation to at least 300 runs per symbol.
5. Add multi-seed experiment aggregation.

### Phase 4: research extensions

Only after the preceding changes are validated, consider:

- A learned value-function baseline.
- A differentiable surrogate of the SSA channel.
- Surrogate-assisted encoder gradients combined with real-channel REINFORCE updates.
- More expressive temporal decoder architectures.
- Larger structured neural encoders for non-finite or contextual messages.
- Curriculum training across receiver distance or channel difficulty.

A differentiable surrogate must always be checked against real SSA rollouts. Otherwise, the encoder may exploit inaccuracies in the surrogate rather than improve performance on the actual channel.

## 11. Proposed first experiment

The first controlled experiment should compare:

### Existing optimizer

```text
random decoder
one decoder update per batch
one encoder update using the same batch's pre-update rewards
fixed entropy coefficient
```

### Proposed optimizer

```text
100 decoder warm-up steps
5 decoder updates per encoder update
fresh encoder-reward batch
entropy annealing from 0.04 to 0.003
best-policy checkpointing
```

Keep all other settings identical:

- Same channel configuration.
- Same observed compartment.
- Same number of symbols and slots.
- Same molecule budget.
- Same total SSA simulation budget when possible.
- Same final decoder model.
- Same final evaluation size.
- Same set of random seeds.

Report accuracy, MI, NMI, runtime, number of SSA simulations, and variability across seeds. This experiment will show whether the staged alternating method improves sample efficiency and final communication quality rather than merely using more simulations.

## 12. Definition of success

The proposed changes should be considered successful when:

- Training remains reproducible for a fixed seed.
- The encoder receives informative rewards after decoder warm-up.
- Validation MI is more stable across training.
- The best checkpoint is at least as good as the final unchecked policy.
- Optimized codebooks outperform hand-designed and random baselines across multiple seeds.
- Improvements remain present when evaluated using fresh datasets and fresh decoders.
- Reported gains include uncertainty rather than relying on a single run.
- All unit and integration tests pass in a headless environment.

