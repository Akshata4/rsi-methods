"""What the scorer "sees" about a proposed edit.

The paper's readout uses "the parent's steering-split solve rate and eight
coordinates of the proposed diff". Here: the parent's current score plus eight
cheap, static features of the change from the parent train.py to a candidate.
Nothing here runs the code or looks at data -- scoring an edit costs microseconds.
"""

import ast
import difflib
import math
import re
from collections import Counter

FEATURE_NAMES = [
    "parent_score",          # how good the script being edited already is
    "log_lines_changed",     # size of the edit: log(1 + lines added + lines removed)
    "model_changed",         # 1 if the set of estimator classes changed
    "ensemble_added",        # 1 if a Voting/Stacking/Bagging ensemble was introduced
    "preprocessing_touched", # 1 if imputation/scaling/encoding lines changed
    "log_features_added",    # log(1 + new engineered columns, e.g. X['ratio'] = ...)
    "log_hyperparams_changed",  # log(1 + numeric literals that changed)
    "search_added",          # 1 if a grid/random search or cross-validation was introduced
    "compiles",              # 1 if the candidate is valid Python
]

# Matched only where a class is instantiated or a function is called ("Name("),
# not in import lines -- scripts often import more than they use.
_MODEL = re.compile(r"\b([A-Z][A-Za-z]*(?:Classifier|Regressor|Regression|SVC|SVR|NB))\s*\(")
_ENSEMBLE = re.compile(r"\b(?:Voting|Stacking|Bagging)(?:Classifier|Regressor)\s*\(")
_PREPROC = re.compile(r"Imputer|Scaler|Encoder|fillna|get_dummies|ColumnTransformer|KBinsDiscretizer"
                      r"|PolynomialFeatures|QuantileTransformer|PowerTransformer|dropna")
_NEW_COLUMN = re.compile(r"""^\s*\w+\[\s*['"][^'"]+['"]\s*\]\s*=""")
_SEARCH = re.compile(r"(?:GridSearchCV|RandomizedSearchCV|Halving\w*SearchCV|cross_val_score|cross_validate|KFold)\s*\(")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?")


def _changed_lines(parent: str, candidate: str) -> tuple[list[str], list[str]]:
    added, removed = [], []
    for line in difflib.unified_diff(parent.splitlines(), candidate.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    return added, removed


def _compiles(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def edit_features(parent: str, candidate: str, parent_score: float) -> dict[str, float]:
    added, removed = _changed_lines(parent, candidate)
    changed = added + removed
    numbers_before, numbers_after = Counter(_NUMBER.findall(parent)), Counter(_NUMBER.findall(candidate))
    numeric_changes = sum(((numbers_before - numbers_after) + (numbers_after - numbers_before)).values())
    return {
        "parent_score": float(parent_score),
        "log_lines_changed": math.log1p(len(changed)),
        "model_changed": float(set(_MODEL.findall(parent)) != set(_MODEL.findall(candidate))),
        "ensemble_added": float(bool(_ENSEMBLE.search(candidate)) and not _ENSEMBLE.search(parent)),
        "preprocessing_touched": float(any(_PREPROC.search(line) for line in changed)),
        "log_features_added": math.log1p(sum(bool(_NEW_COLUMN.search(line)) for line in added)),
        "log_hyperparams_changed": math.log1p(numeric_changes),
        "search_added": float(bool(_SEARCH.search(candidate)) and not _SEARCH.search(parent)),
        "compiles": float(_compiles(candidate)),
    }
