"""Budget RSI (arXiv 2610.29121): score candidate code edits before running them.

    features.py   what the scorer sees about an edit (8 diff features + the parent's score)
    scorer.py     a linear readout: features -> predicted gain, refit from its own outcomes
    workspace.py  the candidate protocol: the agent proposes k edits, the harness runs one
    experiment.py collect labels -> offline comparison -> live runs -> refit (run_budget_rsi.py)
"""
