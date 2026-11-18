"""Budget RSI charts, in the same style and palette as rsi/plots.py."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rsi.plots import GRID, SERIES, SURFACE, TEXT, TEXT_2, _style  # noqa: E402


def plot_offline(path: str, offline: dict) -> None:
    """Mean gain of the edit each strategy would have picked (offline replay)."""
    g = offline["mean_realized_gain"]
    order = [("Best possible (oracle)", g["oracle"]), ("Scorer (Budget RSI)", g["scorer"]),
             ("Random candidate", g["random"]), ("Agent's first idea", g["first"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    _style(ax, "Gain of the picked edit, by pick strategy", "Mean relative gain over the parent script", "")
    for line in list(ax.lines):  # _style draws a y=0 baseline; this chart's baseline is x=0
        line.remove()
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    labels = [name for name, _ in order][::-1]
    values = [v for _, v in order][::-1]
    colors = [GRID if "oracle" in n else SERIES[0] if "Scorer" in n else "#a9a7a0" for n in labels]
    bars = ax.barh(range(len(labels)), values, color=colors, height=0.55, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(labels)), labels)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:+.2%}", (v, bar.get_y() + bar.get_height() / 2), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", ha="left" if v >= 0 else "right", va="center", color=TEXT, fontsize=8)
    ax.tick_params(axis="y", labelsize=9, labelcolor=TEXT)
    lo = min(0, min(values)); hi = max(0, max(values))
    pad = (hi - lo) * 0.25 or 0.01
    ax.set_xlim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_curve(path: str, offline: dict) -> None:
    """Scorer pick quality vs how many labeled edits it was trained on."""
    curve = offline["learning_curve"]
    g = offline["mean_realized_gain"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    _style(ax, "Scorer pick quality vs. training-set size", "Labeled edits the scorer was trained on",
           "Mean gain of the scorer's pick")
    xs = [p["n_labels"] for p in curve]
    ys = [p["mean_realized_gain"] for p in curve]
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Scorer pick", zorder=3)
    if len(xs) > 1:  # the largest size is one fit on the whole pool, not an average over random subsets
        ax.scatter([xs[-1]], [ys[-1]], s=60, facecolor=SURFACE, edgecolor=SERIES[0], linewidth=2, zorder=4)
        ax.annotate("single fit (noisier)", (xs[-1], ys[-1]), xytext=(-6, -2), textcoords="offset points",
                    ha="right", va="top", color=TEXT_2, fontsize=8)
    for value, label, style in [(g["oracle"], "Best possible", ":"), (g["random"], "Random pick", "--")]:
        ax.axhline(value, color=TEXT_2, linewidth=1, linestyle=style, zorder=2)
        ax.annotate(label, (xs[-1] if xs else 0, value), xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", color=TEXT_2, fontsize=8)
    if xs:
        ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
