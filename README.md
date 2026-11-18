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
rsi/budget/         Budget RSI: edit features, scorer, candidate workspace, experiment
run_budget_rsi.py   Budget RSI entry point
rsi/sorsi/          SO-RSI: anomaly detector, inquiry memory, diagnosing workspace, experiment
rsi/flow/           RSIFlow: workflow schema, router, flow workspace, flow report
run_so_rsi.py       SO-RSI entry point
run_rsi.py          CLI entry point
scripts/make_datasets.py   downloads the clinical datasets, writes datasets/registry.json
scripts/export_run.py      copies a finished run into results/ and updates the results log
results/                   committed results of each run (see Results log)
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

## Method 1: Budget RSI

Based on *Recursive Self-Improvement on a Budget* (arXiv 2610.29121; course
labs 10.03-10.05). Running an experiment is the expensive step, so a cheap
scorer predicts which proposed edit is worth running.

- After a normal baseline experiment, each round the agent writes 3 candidate
  scripts (`candidate_1.py` ... `candidate_3.py`) and calls `submit_candidates`.
- The harness, not the agent, picks one to run; the round still costs one experiment.
- The scorer is a linear readout (ridge regression, solved with numpy) over
  the parent's score and 8 static features of the edit: size, model switch,
  ensemble added, preprocessing touched, new columns, hyperparameter changes,
  search/CV added, and whether it compiles (`rsi/budget/features.py`).
- It is trained on labeled edits and refit on the outcomes of its own picks.

The experiment (`run_budget_rsi.py`) has four phases:

1. **collect**: on the train + held-out datasets every candidate is executed,
   so each edit gets a true outcome. With the default budget of 4 experiments
   (1 baseline + 3 candidate rounds): 4 datasets x 2 repeats x 3 rounds x 3
   candidates = 72 labeled edits (the paper fits on 96; `--rounds 5` gives 96).
2. **analyze**: replay every round offline, testing the scorer on a dataset it
   wasn't trained on: scorer pick vs random vs the agent's first idea vs the
   best possible pick, plus a learning curve over training-set size.
3. **live**: on the test datasets, same budget: plain Tumble vs propose-3 and
   run {first, random, scorer pick}.
4. **refit**: scorer v2 = v1's data + the outcomes of its live picks.

```bash
uv run python run_budget_rsi.py                          # all phases
uv run python run_budget_rsi.py --phases collect analyze # offline part only
uv run python run_budget_rsi.py --out runs/budget-<ts>   # resume
```

## Method 2: SO-RSI

Based on *Second-Order Problem Solving for Recursive Self-Improvement*
(arXiv 2610.05701; course labs 10.06-10.08): investigate WHY an experiment
fails before patching it again.

- **Expectations:** `run_experiment` also takes `expected_direction`
  (improve / no_change / worse) and a one-sentence expectation.
- **Anomaly detection** (harness-side, `rsi/sorsi/detector.py`) after every experiment:
  - *recurrence*: the same error signature again, or the same kind of change
    reverted twice in a row
  - *opposing edits*: an edit brings back lines a recent edit removed (A -> B -> A);
    merely re-tuning a recently added line is not flagged
  - *expectation mismatch*: "improve" expected, but it crashed or the score
    fell by more than 1%
- **Diagnose before patching:** when an anomaly fires, `run_experiment` is
  blocked until the agent calls `record_diagnosis` (root cause, evidence,
  change that addresses the cause). It may first run up to 6 `run_probe`
  scripts on its workspace data (hidden labels never touch disk; probes
  don't use the experiment budget but are counted).
- **Inquiry memory:** the next experiment tests each diagnosis (an
  error-triggered one is confirmed by a clean run, a score-triggered one by a
  kept change). Confirmed diagnoses become lessons shown to later sessions;
  memory built over rounds is the recursive part.

The experiment (`run_so_rsi.py`): **build** memory over rounds on the train +
held-out datasets (tracking thrashing per round), then **test** on the test
datasets with the same budget: plain Tumble vs SO-RSI without memory vs SO-RSI
with the frozen memory. Default budget 6, since anomalies need some history.

On the real trace from the first valid run (gen0's hospital session, four
crashes), the detector flags a mismatch after the first crash and recurrence
on the second identical `unseen labels` error -- the point where SO-RSI would
have forced a diagnosis instead of three more blind patches.

```bash
uv run python run_so_rsi.py                       # build -> test
uv run python run_so_rsi.py --out runs/sorsi-<ts> # resume
```

## Method 3: RSIFlow

Based on *RSIFlow: Adaptive Self-Improvement Workflows with an Evolving
Meta-Agent* (arXiv 2610.21791; course labs 10.12-10.14). Instead of rewriting
one prompt, the improver is a **workflow**: stages with short instructions,
an order, and routing rules (`rsi/flow/workflow.py`). Seed stages: baseline,
fix_errors, preprocess, model, features, tune, ensemble.

- **Adaptive orchestration** (`rsi/flow/router.py`, harness-side and
  deterministic): every experiment is assigned a stage from the results so
  far -- a crash goes to fix_errors (then the interrupted stage resumes), a
  stage that just improved continues, a stage with no gain for
  `plateau_patience` experiments has stalled and the next one in `order`
  takes over, and when everything has stalled the most productive stage
  returns. Each `run_experiment` result ends with `NEXT STAGE -> ...`.
- **Execution feedback** (`rsi/flow/workspace.py`): each experiment is logged
  with its stage, relative gain, and whether the edit actually matched the
  stage (same static edit features as Budget RSI).
- **Evolving meta-agent:** the generation loop's children are rewritten
  workflows (instructions, stages, order, thresholds), written from the
  per-stage statistics, validated against a schema, and accepted only if they
  beat their parent on held-out datasets.

The final evaluation adds two contenders: **plain Tumble** (no workflow) and
**flow-static**, the seed workflow with fixed-order routing -- the ablation
that isolates what adaptive routing adds. `flow_report.md` shows per-stage
statistics, routing decisions, and how the workflow evolved.

```bash
uv run python run_rsi.py --method flow --budget 6 --generations 2 --repeats 2
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

## Results log

Each finished run is exported from `runs/` (gitignored) into a committed
`results/<run>/` folder with its config, harness commit, lineage, every
improver's prompts, a per-session table and the final evaluation:

```bash
uv run python scripts/export_run.py runs/<timestamp> --title "Short name" --notes "Context"
```

The table below is regenerated by that script; don't edit it by hand.

<!-- results-log:start -->
| Run | Title | Harness commit | Budget / gens / repeats | Outcome |
|---|---|---|---|---|
| [20261117-164852](results/20261117-164852/README.md) | Prompt-only RSI loop, first valid run | `6cf9de2` | 4 / 2 / 2 | 0/2 children accepted; champion gen0; no champion beyond gen0 to test |
<!-- results-log:end -->

## Findings from earlier runs

The first four real runs (2026-11-17, budget 4, 2 generations, 2 repeats,
DeepSeek-V3.1 via W&B Inference) were stopped early because each exposed a
harness problem. Their results are not valid and were discarded; what they
showed is kept here because each one shaped the harness.

| Run | What happened | Evidence | Fix |
|---|---|---|---|
| 1 | Session crashed after 2 experiments with HTTP 400 from W&B | "Failed to apply prompt template: ... + operator on unsupported types undefined and string". A tool-call-only turn (content=None) was resent without a `content` key; reproduced with a 2-message request | Always send `content` (`679847a`) |
| 2 | Experiments crashed on `import xgboost` | 2 of the first 8 experiments (heart_cleveland exp 3, heart_hungarian exp 2): `ModuleNotFoundError`. Risk: the meta-step "learns" a missing package and it looks like RSI progress | List available libraries in the fixed contract (`6cf9de2`) |
| 3 | Session died: context overflow | Agent called `read_file("dataset.csv")` on the 8k-row readmission CSV: 1.5M characters, 981,740 tokens vs a 161,000 limit. Also: gen 1 was rejected for a 0.0004 AUC "loss" on blood donation despite +3.0% on Pima | Cap tool results at 40k chars (`679847a`); count <0.5% differences as ties (`ed8d6d1`) |
| 4 | Fake 0.80 AUC on hospital readmission (honest ~0.60) | Agent target-encoded categoricals with `df.groupby(col)['target'].mean()` over all rows *before* its own split, leaking test labels through ~700 rare diagnosis codes; +32.8% in one experiment. The harness trusted the self-printed METRIC | Harness-side scoring on hidden labels (`6cf9de2`). The same leaky script now scores 0.580 vs 0.596 for an honest one |

Runs 3 and 4 also completed their generation loops (both children rejected
each time), but under self-reported scoring, so those decisions are not
evidence either way.

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
