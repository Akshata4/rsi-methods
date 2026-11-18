# RSI evaluation on untouched test datasets

Each contender ran on each test dataset 2 time(s). Scores are mean ± std over repeats.
Gain is relative to gen0's mean score on the same dataset (+ = better), averaged over all runs.
A gain only counts as real if its 95% bootstrap CI excludes 0.

| Contender | Budget | liver_ilpd (roc_auc ↑) | hospital_readmission (roc_auc ↑) | Gain vs gen0 [95% CI] | Verdict | Wasted | Exp. to best | Tokens/run |
|---|---|---|---|---|---|---|---|---|
| gen0 | 6 | 0.7411 ± 0.00025 | 0.6158 ± 0.0099 | +0.0% [-0.8%, +0.8%] | reference | 54% | 4.0 | 404,732 |
| gen0-2x-budget | 12 | 0.7656 ± 0.026 | 0.6379 ± 0.017 | +3.4% [+1.2%, +5.7%] | clear gain | 58% | 10.5 | 829,390 |
| blind-rewrite | 6 | 0.7406 ± 0.006 | 0.6096 ± 0.041 | -0.5% [-4.2%, +2.6%] | within noise | 71% | 2.8 | 559,640 |
| plain-tumble | 6 | 0.7567 ± 0.0093 | 0.6271 ± 0.012 | +2.0% [+0.8%, +3.1%] | clear gain | 33% | 6.0 | 600,960 |
| flow-static | 6 | 0.7695 ± 0.032 | 0.618 ± 0.0075 | +2.1% [-0.1%, +5.3%] | within noise | 42% | 5.2 | 521,341 |

**Noise floor:** gen0 varies by ±0.9% against itself across repeats. Gains smaller than this are not evidence.

## Generation loop

- Children proposed: 2, accepted: 0 (acceptance rate 0%)
- Meta-step tokens (writing children): 6,641

## How to read this

- **genK vs gen0:** did recursion produce a better improver on data it never saw?
- **genK vs gen0-2x-budget:** is recursion better than simply running more experiments?
- **genK vs blind-rewrite:** does learning from run evidence matter, or does any rewrite help?
- **Wasted / Exp. to best:** a better improver should waste fewer experiments and peak sooner.
- **generations.png:** if test gain is far below held-out gain, the loop over-fit its held-out set.
