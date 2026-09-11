# Model optimization experiments

Sweep spaces for `pact-tools inference sweep`. Each file is a space TOML: scalar
keys are fixed, list keys are search axes, and the runner expands the cartesian
product. See [`docs/tools/inference/sweep.md`](../docs/tools/inference/sweep.md).

## Stages

A later stage consumes the result of an earlier one. Searching architecture and
training recipe at the same time would multiply into a grid too large to finish,
and would confound the two effects.

| Stage | Files | Purpose |
| --- | --- | --- |
| 0 | `00_baseline_*.toml` | Record the current architecture on a fair recipe. Every later result is compared against these numbers. |
| 1 | `10_recipe_*.toml` | Find a training recipe on one fixed architecture. Axes are optimizer, learning rate, and objective. |
| 2 | `20_arch_*.toml` | Hold the recipe from stage 1 and search the architecture catalog. This produces the size against quality frontier. |
| 2.5 | `25_confirm_*.toml` | Repeat the architectures at and beside the knee under further seeds, so the choice is not made on one draw. Fill `arch` after stage 2. |
| 2.7 | `27_refine_*.toml` | Tune the settings stage 1 held fixed -- weight decay, positive-class weight, augmentation, batch size -- on the selected architecture. |
| 3 | `30_final_*.toml` | Retrain the selected points at the flight input size for the numbers that are reported and exported. |

Stage 1 searches the settings that decide whether a run converges at all, and
stage 2.7 searches the settings that decide how well a converged run
generalises. They are split around architecture selection deliberately. The
second group regularises capacity, so its best values depend on the capacity
being regularised: a weight decay tuned on a 23.5 M parameter stand-in does not
transfer to a network three orders of magnitude smaller. Searching them once, on
the architecture that was actually chosen, is both more relevant and far cheaper
than searching them across the whole catalog.

The segmentor stage-2 file now carries the recipe from the completed stage 1
sweep: AdamW at 0.001 with `focal_dice`. The classifier stage-2 file carries
AdamW at 0.001 with BCE, the robust member of a two-way tie on the same
pattern. See [`experiments/results/README.md`](results/README.md).

## Selection rule

Stage 2 produces a frontier, not a winner. The point taken from it is the knee:
the largest size reduction that still holds the metric, measured against the
stage 0 baseline. A point further down the frontier is only preferred when the
metric it gives up is within the seed-to-seed spread measured at that
architecture, because a drop smaller than the noise is not a drop.

Both the frontier and the knee are read from validation results. The test split
is scored once, on the stage 3 finalists, so that it does not steer the search.

The catalog mixes stages, so the frontier is drawn from one sweep JSONL, with
seeds of the same architecture averaged:

```text
uv run pact-tools inference pareto --run-dir artifacts/runs --kind segmentor --metric mean_iou --split val --from-jsonl artifacts/runs/20_arch_segmentor.jsonl --by-arch --baseline 0.5509 --write-space experiments/25_confirm_segmentor.toml
uv run pact-tools inference pareto --run-dir artifacts/runs --kind classifier --metric f1 --split val --from-jsonl artifacts/runs/20_arch_classifier.jsonl --by-arch --baseline 0.9311 --write-space experiments/25_confirm_classifier.toml
```

Training boxes often keep the catalog outside the repo. Pass that path to
`--run-dir` and `--from-jsonl`. Repeat with `--cost flops` for the latency
axis. `--baseline` is the stage 0 validation score. `--spread` (default 0)
is the seed-to-seed range measured at stage 2.5; pass it when that stage has
run. The printed rows are the knee plus one neighbour on each side, which is
the `arch` list stage 2.5 trains.

After stage 2.5, join the architecture JSONL with the extra-seed JSONL, take
the single knee, and fill the refine and 256 px spaces:

```text
uv run pact-tools inference pareto --run-dir artifacts/runs --kind segmentor --metric mean_iou --split val --from-jsonl artifacts/runs/20_arch_segmentor.jsonl --from-jsonl artifacts/runs/25_confirm_segmentor.jsonl --by-arch --baseline 0.5509 --auto-spread --neighbors 0 --write-space experiments/27_refine_segmentor.toml --write-space experiments/30_final_segmentor.toml
uv run pact-tools inference pareto --run-dir artifacts/runs --kind classifier --metric f1 --split val --from-jsonl artifacts/runs/20_arch_classifier.jsonl --from-jsonl artifacts/runs/25_confirm_classifier.jsonl --by-arch --baseline 0.9311 --auto-spread --neighbors 0 --write-space experiments/27_refine_classifier.toml --write-space experiments/30_final_classifier.toml
```

ImageNet-pretrained backbones are permitted in the shipped model. Every
pretrained entry in stage 2 is paired with the same backbone from scratch, so
the frontier records what the pretrained weights are worth rather than assuming
it.

## Repeated seeds

A frontier drawn through single runs treats seed-to-seed spread as a result, and
that spread can exceed the gap between neighbouring architectures, which would
select an architecture on noise. Repeats are therefore required before a knee is
chosen, but they are not required everywhere.

The segmentor searches two seeds at once. Its corpus is 1,437 images and the
small corpus is where seed spread is widest, so at least one repeat is needed
before the frontier means anything.

The classifier searches one seed, then repeats. A classifier trial costs several
times more, and repeating all 21 architectures would spend most of the budget
resolving differences between points far from the frontier, where the ranking
does not change any decision. Stage 2.5 repeats only the architectures on or
beside the knee.

## Epoch budget

Stage 2 raises the budget for both models, for the same underlying reason: a
frontier measured at an epoch cap ranks architectures by how quickly they start
rather than by where they arrive.

The classifier goes from 15 epochs to 30. Its two families do not converge at
the same rate, because a pretrained backbone begins near its answer while a
compact network trained from scratch is still improving well past the point
where the pretrained one has stopped.

The segmentor goes from 50 epochs to 120. Stage 1 measured the need directly:
Dice-family trials peaked at epoch 43 on average out of 50, and 7 of those 18
trials peaked within two epochs of the cap, so a meaningful share of them were
still improving when the budget ended them.

Early stopping is what keeps the larger budgets affordable. A run that has
genuinely settled exits at its patience bound rather than running to the cap, so
the extra epochs are spent only on the runs that use them.

## Input size

Stages 0 to 2 run at 128 px. Source tiles are 120 px, so 128 px is close to
native and about four times cheaper than the flight size, which buys a much
wider search. Parameter count does not depend on input size, so the size
frontier that stage 2 finds holds at either resolution.

Stage 3 runs at the 256 px flight input size. Only the selected points pay that
cost, and their metrics are the ones the acceptance gate checks.

Stage 3 also retrains the baseline architecture beside the selected one, under
the same resolution, epoch budget, recipe and seed. Stage 0 recorded the
baseline at 128 px with BCE under SGD, so reporting a 256 px candidate against
that record would credit the candidate for the resolution and the objective as
well as for the architecture. Retraining both is what makes the reported
improvement attributable to the model.

After each stage 3 run, `pact-tools inference finalize --run <dir>` scores the
test split, exports FP32 and INT8 ONNX, and runs the golden-scene gate. That is
the command that produces the artifacts the definition of done requires. The
shipped graphs are FP32. INT8 stays beside them as a measured alternative.

## Data

The packs come from `scripts/fetch_smoke_plume_dataset.py --download
--preprocess`. That step writes two packs, because the corpus labels the two
models differently: all 21,350 images carry a plume-presence label, and 1,437
of them carry polygon masks. Set `data_dir` to the pack that matches `kind`.
