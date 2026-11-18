# RSI evaluation on untouched test datasets

Each contender ran on each test dataset 2 time(s). Scores are mean ± std over repeats.
Gain is relative to gen0's mean score on the same dataset (+ = better), averaged over all runs.
A gain only counts as real if its 95% bootstrap CI excludes 0.

| Contender | Budget | liver_ilpd (roc_auc ↑) | hospital_readmission (roc_auc ↑) | Gain vs gen0 [95% CI] | Verdict | Wasted | Exp. to best | Tokens/run |
|---|---|---|---|---|---|---|---|---|
| gen0 | 4 | 0.7688 ± 0.049 | 0.6336 ± 0 (1 failed) | -25.0% [-75.0%, +2.3%] | reference | 62% | 2.3 | 388,365 |
| gen0-2x-budget | 8 | 0.7861 ± 0.011 | 0.6342 ± 0.013 | +1.2% [-0.6%, +2.8%] | within noise | 62% | 5.8 | 853,195 |
| blind-rewrite | 4 | 0.7516 ± 0.013 | 0.5905 ± 0.049 | -4.5% [-9.5%, -1.2%] | clear loss | 44% | 3.8 | 411,202 |

**Noise floor:** gen0 varies by ±4.5% against itself across repeats. Gains smaller than this are not evidence.

## Generation loop

- Children proposed: 2, accepted: 0 (acceptance rate 0%)
- Meta-step tokens (writing children): 5,072

## How to read this

- **genK vs gen0:** did recursion produce a better improver on data it never saw?
- **genK vs gen0-2x-budget:** is recursion better than simply running more experiments?
- **genK vs blind-rewrite:** does learning from run evidence matter, or does any rewrite help?
- **Wasted / Exp. to best:** a better improver should waste fewer experiments and peak sooner.
- **generations.png:** if test gain is far below held-out gain, the loop over-fit its held-out set.
