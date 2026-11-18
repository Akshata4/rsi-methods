"""Write placeholder datasets (no download needed) plus datasets/registry.json.

These are stand-ins so the loop can be tested end to end. Replace them with
datasets from your chosen problem area. Keep each one small (a run must
finish in seconds) and not already solved (avoid ~1.0 accuracy datasets
like iris/wine -- there's nothing left to improve).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_classification, make_friedman1, make_friedman3

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def save(name: str, df: pd.DataFrame, target: str, metric: str, minimize: bool, registry: list) -> None:
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    registry.append({"name": name, "path": f"{name}.csv", "target": target, "metric": metric, "minimize": minimize})
    print(f"wrote {name}.csv {df.shape}")


def with_messy_columns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add a categorical column and some missing values so preprocessing matters."""
    df = df.copy()
    df["region"] = rng.choice(["north", "south", "east", "west"], size=len(df))
    cols = [c for c in df.columns if c not in ("target", "region")]
    for c in rng.choice(cols, size=min(3, len(cols)), replace=False):
        df.loc[rng.random(len(df)) < 0.08, c] = np.nan
    return df


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    registry: list = []

    d = load_diabetes(as_frame=True).frame
    save("diabetes", with_messy_columns(d, rng), "target", "rmse", True, registry)

    X, y = make_friedman1(n_samples=1200, n_features=10, noise=1.0, random_state=1)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]).assign(target=y)
    save("friedman", with_messy_columns(df, rng), "target", "rmse", True, registry)

    for name, seed, sep in [("noisy_clf", 2, 0.6), ("hard_clf", 3, 0.4)]:
        X, y = make_classification(n_samples=1500, n_features=20, n_informative=6, n_redundant=4,
                                   flip_y=0.08, class_sep=sep, random_state=seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).assign(target=y)
        save(name, with_messy_columns(df, rng), "target", "accuracy", False, registry)

    X, y = make_friedman3(n_samples=1000, noise=0.15, random_state=4)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]).assign(target=y)
    save("friedman3", with_messy_columns(df, rng), "target", "rmse", True, registry)

    X, y = make_classification(n_samples=2000, n_features=15, n_informative=5, weights=[0.8, 0.2],
                               flip_y=0.05, class_sep=0.7, random_state=5)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).assign(target=y)
    save("imbalanced_clf", with_messy_columns(df, rng), "target", "accuracy", False, registry)

    with open(os.path.join(OUT, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    print(f"wrote registry.json ({len(registry)} datasets)")


if __name__ == "__main__":
    main()
