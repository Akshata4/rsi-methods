"""Download the clinical risk prediction datasets and write datasets/registry.json.

Problem area: clinical risk prediction. Every dataset is binary classification
scored by ROC-AUC, with target = 1 for the clinically relevant outcome
(disease present, readmission, donation, ...).

Sources are public OpenML/UCI datasets, fetched by fixed OpenML id so the data
is identical on every machine. Cleaning is deliberately minimal: only the
target is standardized and pure ID columns are dropped. Messiness that a real
analyst would have to handle (missing values, zeros that mean "missing",
high-cardinality codes, imbalance) is left in on purpose -- that is the
research the agent has to do.

    uv run python scripts/make_datasets.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datasets")
CACHE = os.path.join(OUT, ".openml_cache")

# name -> (OpenML id, label that becomes target=1, columns to drop, row sample size or None, description)
DATASETS = {
    "heart_cleveland": (49, ">50_1", [], None, "Heart disease (Cleveland): >50% vessel narrowing"),
    "heart_hungarian": (51, ">50_1", [], None, "Heart disease (Hungarian): >50% vessel narrowing; ~20% missing values"),
    "pima_diabetes": (37, "tested_positive", [], None, "Diabetes onset in Pima women; zeros in some columns mean 'missing'"),
    "blood_donation": (1464, "2", [], None, "Whether a past blood donor donated again (2 = donated)"),
    "liver_ilpd": (1480, "1", [], None, "Indian liver patient records (1 = liver patient)"),
    "hospital_readmission": (4541, "<30", ["encounter_id", "patient_nbr"], 8000,
                             "Diabetes 130-US hospitals: readmitted within 30 days (8,000-row sample)"),
}

# Which datasets play which role in the RSI experiment (see README).
SPLITS = {
    "train": ["heart_cleveland", "heart_hungarian"],
    "heldout": ["pima_diabetes", "blood_donation"],
    "test": ["liver_ilpd", "hospital_readmission"],
}


def reference_auc(df: pd.DataFrame) -> float:
    """A plain logistic-regression baseline on the harness's fixed split,
    so we know each dataset's starting point and that it isn't already solved."""
    X, y = df.drop(columns=["target"]), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), make_column_selector(dtype_include="number")),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
         make_column_selector(dtype_exclude="number")),
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    return float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))


def load(name: str) -> pd.DataFrame:
    did, positive, drop, sample, _ = DATASETS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # liac-arff parser: the default pandas ARFF parser breaks on some
        # categorical columns with recent pandas versions.
        bunch = fetch_openml(data_id=did, as_frame=True, data_home=CACHE, parser="liac-arff")
    X = bunch.data.drop(columns=drop)
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype(object)
    labels = bunch.target.astype(str)
    if positive not in set(labels):
        raise ValueError(f"{name}: positive label {positive!r} not in {sorted(set(labels))}")
    df = X.assign(target=(labels == positive).astype(int).values)
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=0).reset_index(drop=True)
    return df


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    registry = []
    print(f"{'dataset':22s} {'split':8s} {'rows':>6s} {'cols':>5s} {'pos%':>5s} {'missing%':>9s} {'LR AUC':>7s}")
    for split, names in SPLITS.items():
        for name in names:
            df = load(name)
            df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
            auc = reference_auc(df)
            missing = df.drop(columns=["target"]).isna().mean().mean()
            print(f"{name:22s} {split:8s} {len(df):6d} {df.shape[1] - 1:5d} {df['target'].mean():5.0%} {missing:9.1%} {auc:7.3f}")
            registry.append({
                "name": name, "path": f"{name}.csv", "target": "target", "metric": "roc_auc", "minimize": False,
                "split": split, "openml_id": DATASETS[name][0], "description": DATASETS[name][4],
                "reference_lr_auc": round(auc, 4),
            })
    with open(os.path.join(OUT, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nwrote {len(registry)} datasets + registry.json to {OUT}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
