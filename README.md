# rsi-methods

Recursive self-improvement (RSI) built on top of **Tumble**, an autonomous ML
research agent that writes and iteratively improves `train.py` for a dataset.

**Problem area: clinical risk prediction.** The agent builds risk models from
patient records (heart disease, diabetes, liver disease, hospital readmission),
and the RSI layer tries to make the agent itself a better clinical-ML researcher.

## Datasets

Six public OpenML/UCI datasets, all binary classification scored by **ROC-AUC**
(target = 1 for the clinical outcome). `scripts/make_datasets.py` downloads them
by fixed OpenML id. Only the target is standardized and pure ID columns dropped;
missing values, zeros-meaning-missing and messy codes are left in on purpose.

| Dataset | OpenML id | Rows | Features | Positive % | Split | Baseline LR AUC |
|---|---|---|---|---|---|---|
| Heart disease, Cleveland | 49 | 303 | 13 | 46% | train | 0.930 |
| Heart disease, Hungarian (20% missing) | 51 | 294 | 13 | 36% | train | 0.812 |
| Pima diabetes onset | 37 | 768 | 8 | 35% | held-out | 0.884 |
| Blood donation (returns to donate) | 1464 | 748 | 4 | 24% | held-out | 0.727 |
| Indian liver patients (ILPD) | 1480 | 583 | 10 | 71% | test | 0.736 |
| Diabetes 130-US hospitals, readmitted <30 days (8k sample) | 4541 | 8000 | 47 | 11% | test | 0.581 |

Baseline = plain logistic regression trained on the dev part and scored on the
hidden part, exactly the way the harness scores the agent.
Datasets were chosen from 12 clinical candidates by probing: excluded were
ones already near-solved (thyroid "sick" AUC 0.999, SPECTF 0.976,
heart-statlog 0.947), ones with almost no signal (thoracic surgery ~0.52), and
ones too small/noisy to tell real gains from luck (Haberman, breast-cancer:
AUC varies by ±0.07 across splits). The two heart datasets share a schema, so
they sit in the same split to avoid leakage across splits.

## The three objects

| Object | What it is here | Who changes it |
|---|---|---|
| Task model | `train.py` | The Tumble loop (propose, run, keep/revert) |
| Improver | `research_prompt`: how the agent researches | The previous generation's improver |
| Meta-level | `meta_prompt`: how an improver rewrites itself | Also the previous generation (self-referential) |

Tumble alone is first-order self-improvement. The **generation loop** makes it
recursive: each improver writes its own successor, and the successor is kept
only if it beats its parent on held-out datasets under the same budget.

```
gen 0  seed (Tumble's prompt) ── run on TRAIN (evidence) + HELD-OUT (score)
gen g  child = champion.meta_prompt(champion, evidence)
       child runs on HELD-OUT, same budget → paired comparison
       ACCEPT → child is champion  |  REJECT → champion stays, sees why
stop   max generations or `patience` rejections in a row
```

## Layout

```
tumble_core/        Tumble's loop, copied with small changes (local executor,
                    budget as a parameter, no W&B logging)
rsi/improver.py     Improver = research_prompt + meta_prompt; fixed HARNESS_CONTRACT
rsi/run_task.py     one improver x one dataset = one Tumble session
rsi/meta.py         the self-rewrite step (parent writes child)
rsi/compare.py      paired, per-dataset comparison with a margin
rsi/generations.py  the generation loop
rsi/metrics.py      run-level metrics (gain over baseline, wasted rate, best-so-far curve...)
rsi/evaluate.py     final evaluation on untouched TEST datasets, with repeats and baselines
rsi/plots.py        generations.png and best_so_far.png
run_rsi.py          CLI entry point
scripts/make_datasets.py   downloads the clinical datasets, writes datasets/registry.json
```

## Run

```bash
uv sync
cp .env.example .env              # add your W&B Inference (or OpenRouter) key
uv run python scripts/make_datasets.py
uv run python run_rsi.py --budget 5 --generations 3 --repeats 3
```

Train / held-out / test datasets default to the splits in `datasets/registry.json`;
override with `--train`, `--heldout`, `--test`.

Each run writes to `runs/<timestamp>/`: every improver's JSON, every session's
workspace (with git history of kept edits), trajectory, `lineage.jsonl`, and
`eval/` with the final results. To (re-)run only the evaluation (finished runs are cached):

```bash
uv run python -m rsi.evaluate runs/<timestamp> --test liver_ilpd hospital_readmission --repeats 3
```

## How we know whether it improved

Three dataset splits, each with one job:

| Split | Used for | Seen by the loop? |
|---|---|---|
| train | evidence the improver learns from when rewriting itself | yes |
| held-out | accept/reject decisions for each child | yes (so it's biased by selection) |
| test | the final claim, used once at the end | **never** |

**Run-level metrics** (every Tumble session, `rsi/metrics.py`): final best score,
gain over the agent's own first attempt, experiments to reach the best, wasted
experiment rate, failure rate, best-so-far curve, tokens, wall time.

**Final evaluation** (`rsi/evaluate.py`): every contender runs on every test
dataset `--repeats` times. Contenders:

| Contender | Question it answers |
|---|---|
| gen0 | reference: Tumble's original prompt |
| gen1..genK | did each accepted generation actually improve on unseen data? |
| gen0-2x-budget | is recursion better than just running more experiments? |
| blind-rewrite | does learning from evidence matter, or does any rewrite help? |

Gains are relative to gen0's mean score on the same dataset. A gain is called
**clear** only if its 95% bootstrap confidence interval excludes zero; gen0's
spread against itself is reported as the noise floor. `generations.png` puts
held-out gain (what the loop selected on) next to test gain (what it's worth):
a large gap means the loop over-fit its held-out set.

## Fairness guards

- **The harness scores, not the agent.** Each dataset is split 80/20 into a dev
  part (`dataset.csv`, with labels) and a hidden part (`test_features.csv`, no
  labels). `train.py` writes `predictions.csv`; the harness scores it against
  hidden labels that exist only in its memory. This was added after a real run
  in which an agent target-encoded over all rows before its own split and
  reported a fake 0.80 AUC on readmission; under harness scoring that same
  script gets 0.58, below an honest version (0.60).
- The improver can't change the scoring: the predictions protocol, the hidden
  split, budget and keep/revert rule live in the harness (`HARNESS_CONTRACT`, `Workspace`).
- Improvers are judged only on held-out datasets that never appear in their evidence.
- A child must beat its parent by a mean relative margin (`--min-gain`, default 1%) **and**
  win on more held-out datasets than it loses; per-dataset differences under 0.5% are ties.
- The number of generations is bounded, with an early stop on repeated rejections.

## Known limitations (to address)

- The loop itself runs each (improver, dataset) once; only the final evaluation repeats.
  A noisy held-out result can still accept a lucky child, which the test split then exposes.
- Tool results are capped at 40k characters (an agent reading a whole 8k-row CSV
  otherwise overflows the model's context window).
- Within a session keep/revert uses the hidden-split score, so the agent still
  hill-climbs on one fixed split (it never sees the labels, but it does see which
  edits scored higher). On these small datasets AUC moves by ±0.02-0.05 between
  random splits, which is the scale of "improvement" to be skeptical of.
- Runs execute LLM-written code locally. Use the E2B sandbox from Tumble for anything untrusted.
