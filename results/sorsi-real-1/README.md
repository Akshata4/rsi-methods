# SO-RSI, first real run

SO-RSI · run `sorsi-real-1`

The uncommitted changes recorded in provenance were exactly the opposing-edits fix and runner provenance, committed right after as 98369c7 and b0039d5 (one earlier session with the old detector was discarded). Build: 2 rounds x 4 datasets, budget 6; round 2, with lessons from round 1, had fewer failed experiments (4% vs 17%) and fewer anomalies. Test (2 repeats): no AUC difference beyond noise; memory cut failed experiments (8% vs 12%) and gave the most stable hospital result; 19 of 24 test mismatches were score drops within noise, so few diagnoses could be confirmed. DeepSeek-V3.1 via W&B Inference.

- Harness commit: `cdcc869` (with uncommitted changes)
- Model: wandb / (provider default)
- Settings: budget 6 · 2 build rounds · 2 test repeats

Budget 6 experiments per session · build: heart_cleveland, heart_hungarian, pima_diabetes, blood_donation × 2 rounds · test: liver_ilpd, hospital_readmission × 2 repeats

## 1. Building inquiry memory, round by round

Each session starts with the lessons confirmed in earlier finished sessions.

| Round | Sessions | Lessons available (mean) | Failure rate | Repeated errors / session | Anomalies / session | Diagnoses confirmed | Mean best AUC |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 2.2 | 17% | 0.00 | 3.50 | 5/8 | 0.828 |
| 2 | 4 | 6.8 | 4% | 0.00 | 2.50 | 3/7 | 0.839 |

## 2. Comparison on untouched test datasets

Same budget for every contender. Gain is relative to plain Tumble's mean score.

| Contender | liver_ilpd AUC | hospital_readmission AUC | Gain vs plain [95% CI] | Verdict |
|---|---|---|---|---|
| Plain Tumble (first-order) | 0.756 ± 0.019 | 0.634 ± 0.006 | -0.0% [-1.2%, +1.2%] | reference |
| SO-RSI, no memory | 0.766 ± 0.023 | 0.610 ± 0.036 | -1.2% [-5.8%, +2.5%] | within noise |
| **SO-RSI with inquiry memory** | 0.740 ± 0.006 | 0.641 ± 0.003 | -0.5% [-2.0%, +1.0%] | within noise |

Noise floor: plain Tumble varies by ±1.5% against itself across repeats.

### Thrashing and inquiry (per session, mean)

| Contender | Failed experiments | Wasted experiments | Repeated errors | Anomalies (rec / opp / mis) | Diagnoses confirmed | Probes | Tokens |
|---|---|---|---|---|---|---|---|
| Plain Tumble (first-order) | 12% | 50% | 0.00 | — | — | — | 531,490 |
| SO-RSI, no memory | 12% | 50% | 0.25 | 0.2 / 0.8 / 2.5 | 2/8 | 5.8 | 677,210 |
| **SO-RSI with inquiry memory** | 8% | 75% | 0.25 | 0.8 / 0.2 / 3.5 | 1/13 | 6.0 | 279,017 |

Repeated errors are counted the same way for every contender (failed experiments whose error matches an earlier one in the session); anomalies are only detected for SO-RSI sessions.

## 3. The inquiry memory used in testing

```text
- Lessons from earlier inquiries (diagnoses confirmed by the next experiment in earlier
  sessions on other clinical datasets; general patterns, not facts about this dataset):
  * [mismatch] When expected an improvement but the score fell -3.9% vs the best so far: root cause was HistGradientBoostingClassifier underperforms compared to LogisticRegression on this specific dataset, likely due to the small dataset size (235 samples) where simpler models like LogisticRegression with proper regularization can generalize better; what worked: Return to LogisticRegression but optimize hyperparameters and feature engineering based on the correlation analysis
  * [mismatch] When expected 'improve' but the script failed (NameError: name <v> is not defined): root cause was Missing import statement for OneHotEncoder in the Logistic Regression implementation; what worked: Add the missing OneHotEncoder import to the Logistic Regression implementation
  * [mismatch] When expected an improvement but the score fell -6.6% vs the best so far: root cause was HistGradientBoostingClassifier underperforms compared to RandomForest on this specific dataset, likely due to the small dataset size (242 samples) where tree-based ensembles like RandomForest are more robust and better at handling the mixed data types and missing values; what worked: Return to RandomForest but focus on feature engineering and potentially try logistic regression as an alternative strong baseline
  * [mismatch] When expected 'improve' but the script failed (sklearn.exceptions.NotFittedError: This Pipeline instance is not fitted yet. Call <v> with appropriate arguments before using this estimator.): root cause was The custom transformer approach failed because the pipeline structure was complex and the transformer wasn't properly integrated into the sklearn pipeline framework, causing a NotFittedError; what worked: Use a simpler approach by manually creating the missing indicator features before the pipeline to avoid complex transformer integration issues
  * [mismatch] When expected an improvement but the score fell -2.1% vs the best so far: root cause was RandomForest performs better than HistGradientBoosting on this specific dataset, likely due to the small dataset size (235 samples) and high percentage of missing values where tree-based ensembles like RandomForest are more robust; what worked: Return to RandomForest but optimize hyperparameters and improve feature engineering to boost performance further
  * [mismatch] When expected 'improve' but the script failed (TypeError: LabelEncoder.fit_transform() takes <n> positional arguments but <n> were given): root cause was LabelEncoder does not accept a y parameter when used in a ColumnTransformer pipeline, causing a TypeError because the pipeline passes both X and y to all transformers; what worked: Replace LabelEncoder with OneHotEncoder which properly handles the pipeline interface and is more appropriate for categorical variables
  * [recurrence] When the same kind of change (log_hyperparams_changed, model_changed, preprocessing_touched) was tried twice in a row and reverted both times: root cause was HistGradientBoostingClassifier may not be well-suited for this particular dataset, possibly due to the small dataset size (242 samples) or specific feature characteristics where tree-based ensembles like RandomForest perform better; what worked: Return to RandomForest but optimize hyperparameters and improve feature engineering based on correlation analysis
  * [mismatch] When expected an improvement but the score fell -5.2% vs the best so far: root cause was The stacking ensemble performed worse because the base models (HGB and RF) make highly correlated predictions (correlation > 0.89), reducing the benefit of ensemble diversity. Additionally, the meta-learner (LogisticRegression) was already the best individual performer, so stacking it with weaker, correlated models didn't help.; what worked: Focus on optimizing LogisticRegression with the best feature engineering and regularization, as it showed the strongest individual performance.
```

## Files

- `frozen_lessons.txt`: the inquiry memory given to the test sessions
- `memory.jsonl`: every tested diagnosis (root cause, evidence, change, confirmed or not)
- `inquiry_events.jsonl`: every anomaly, diagnosis and test of a diagnosis, per session
- `results.json`, `sessions.csv`, `config.json`, `provenance.json`
- 20 sessions
