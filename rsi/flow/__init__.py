"""RSIFlow (arXiv 2610.21791): adaptive workflow orchestration with an evolving meta-agent.

    workflow.py   the evolvable object: stages, their instructions, order and routing rules
    router.py     the orchestrator: picks each experiment's stage from the results so far
    workspace.py  FlowWorkspace: tags experiments with stages, logs per-stage statistics
    report.py     per-stage statistics and the workflow's evolution, for a finished run

Evolution reuses the generation loop (rsi/generations.py): a child improver is
a rewritten workflow, judged on held-out datasets like any other child.
"""
