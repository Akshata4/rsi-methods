"""The scorer: a linear readout from edit features to predicted gain.

Deliberately tiny and transparent (this is the "assistant" in the explanation):
standardize each feature, then ridge regression solved in closed form with
numpy, so the learned weights can be read off and explained. It learns from
labeled edits -- (features, what actually happened) -- and is refit whenever
new outcomes arrive, which is the self-improving part.

Label = relative gain of the candidate over its parent, clipped to
[-CLIP, +CLIP]; a candidate that crashed or timed out gets -CLIP.
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from rsi.budget.features import FEATURE_NAMES

CLIP = 0.10  # +-10% relative gain; a crash counts as the worst outcome


def label_for(metric: float | None, parent_score: float, minimize: bool) -> float:
    if metric is None:
        return -CLIP
    gain = (metric - parent_score) / max(abs(parent_score), 1e-9)
    if minimize:
        gain = -gain
    return float(np.clip(gain, -CLIP, CLIP))


@dataclass
class EditScorer:
    alpha: float = 1.0
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    n_train: int = 0

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        return np.array([[r[name] for name in self.feature_names] for r in rows], dtype=float)

    def fit(self, features: list[dict], labels: list[float]) -> "EditScorer":
        X, y = self._matrix(features), np.asarray(labels, dtype=float)
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0  # a constant feature carries no information; keep its weight at ~0
        Z = (X - mean) / std
        intercept = y.mean()
        coef = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - intercept))
        self.mean, self.std, self.coef = mean.tolist(), std.tolist(), coef.tolist()
        self.intercept, self.n_train = float(intercept), len(y)
        return self

    @property
    def fitted(self) -> bool:
        return self.n_train > 0

    def predict(self, features: list[dict]) -> list[float]:
        if not self.fitted:
            raise RuntimeError("scorer has not been fit")
        Z = (self._matrix(features) - np.array(self.mean)) / np.array(self.std)
        return (Z @ np.array(self.coef) + self.intercept).tolist()

    def weights(self) -> dict[str, float]:
        """Standardized weights: + means 'edits with more of this tend to help'."""
        return dict(zip(self.feature_names, self.coef))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EditScorer":
        with open(path) as f:
            return cls(**json.load(f))
