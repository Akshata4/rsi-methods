"""Charts for the RSI evaluation (static PNGs for the README and the video).

Colors follow the entity, never the rank: gen0 is always blue, the final
champion always orange, and so on, in every chart.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from rsi.metrics import relative_gain  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4, fixed order


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.axhline(0, color=TEXT_2, linewidth=0.8, zorder=1)


def plot_generations(path: str, lineage: list[dict], summary: dict, minimize: dict[str, bool]) -> None:
    """Held-out gain vs gen0 for every child the loop evaluated (what it selected on),
    next to test gain for each accepted champion (what it is actually worth)."""
    gen0_heldout = lineage[0]["heldout"]

    def heldout_gain(scores: dict) -> float:
        gains = [relative_gain(scores[ds], gen0_heldout[ds], minimize[ds]) for ds in gen0_heldout]
        gains = [-1.0 if g is None else g for g in gains]
        return sum(gains) / len(gains)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _style(ax, "Gain over gen0 by generation", "Generation", "Mean relative gain vs gen0")

    acc = [(e["generation"], heldout_gain(e["heldout"])) for e in lineage if e["accepted"] and e.get("heldout")]
    rej = [(e["generation"], heldout_gain(e["heldout"])) for e in lineage if not e["accepted"] and e.get("heldout")]
    ax.plot([g for g, _ in acc], [v for _, v in acc], color=SERIES[0], linewidth=2, zorder=3)
    ax.scatter([g for g, _ in acc], [v for _, v in acc], s=48, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
               zorder=4, label="Held-out, accepted (used for selection)")
    if rej:
        ax.scatter([g for g, _ in rej], [v for _, v in rej], s=48, facecolor=SURFACE, edgecolor=SERIES[0],
                   linewidth=2, zorder=4, label="Held-out, rejected child")

    test = [(int(name[3:]), s) for name, s in summary.items() if name[3:].isdigit()]
    if test:
        xs = [g + 0.08 for g, _ in test]  # small offset so test points don't sit on held-out points
        ys = [s["mean_gain_vs_gen0"] for _, s in test]
        lo = [max(0.0, y - s["ci95"][0]) if s["ci95"][0] == s["ci95"][0] else 0 for y, (_, s) in zip(ys, test)]
        hi = [max(0.0, s["ci95"][1] - y) if s["ci95"][1] == s["ci95"][1] else 0 for y, (_, s) in zip(ys, test)]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES[1], linewidth=2, capsize=4, marker="s", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4, label="Test, with 95% CI (final claim)")

    gens = sorted({e["generation"] for e in lineage})
    ax.set_xticks(gens)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_best_so_far(path: str, curves: dict[str, list], budget: int) -> None:
    """Average best-so-far curve: how fast each contender climbs, experiment by experiment."""
    champions = sorted((n for n in curves if n[3:].isdigit()), key=lambda n: int(n[3:]))
    final = champions[-1] if len(champions) > 1 else None
    shown = [("gen0", "gen0 (seed)", SERIES[0])]
    if final:
        shown.append((final, f"{final} (final champion)", SERIES[1]))
    shown += [(n, label, c) for n, label, c in [("gen0-2x-budget", "gen0, 2x budget", SERIES[2]),
                                                 ("blind-rewrite", "Blind rewrite", SERIES[3])] if n in curves]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _style(ax, "Best score so far, per experiment", "Experiment", "Relative gain vs gen0's final score")
    ends = []
    for name, label, color in shown:
        ys = curves[name]
        pts = [(i + 1, y) for i, y in enumerate(ys) if y is not None]
        if not pts:
            continue
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, linewidth=2, label=label,
                linestyle="--" if name == "gen0-2x-budget" else "-")
        ends.append((pts[-1], label.split(" (")[0]))
    ax.axvline(budget, color=GRID, linewidth=1, linestyle=":")
    ax.set_xlim(0.8, max(len(c) for c in curves.values()) + 1.6)

    # Direct-label line ends, skipping any label that would collide with one already
    # placed (overlapping lines are still identified by the legend).
    lo, hi = ax.get_ylim()
    min_gap = 0.06 * (hi - lo)
    placed = []
    for (x, y), text in ends:
        if any(abs(x - px) < 1.5 and abs(y - py) < min_gap for px, py in placed):
            continue
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", color=TEXT_2, fontsize=8, va="center")
        placed.append((x, y))
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
