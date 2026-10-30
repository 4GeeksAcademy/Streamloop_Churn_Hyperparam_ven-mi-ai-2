"""StreamLoop churn classifier: baseline, random search, then a narrowed grid.

The test set is scored twice: once for the default pipeline, once for the
selected tuned pipeline. Search objects are fit on the training split only.

See PLAN.md. This module does not write tuning_report.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ITER = 24
NEAR_TIE_BAND = 0.01
MAX_GRID_COMBOS = 36
TOP_REGION = 5

# Ordered values used both as the random-search lists and as the neighbor axis
# when the grid is cut down. Do not reorder these without updating PLAN.md.
ORDERED_SPACES = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [4, 8, 12, 20, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8],
}
UNORDERED_KEYS = ["model__max_features", "model__class_weight"]

PARAM_DISTRIBUTIONS = {
    **ORDERED_SPACES,
    "model__max_features": ["sqrt", "log2", 0.5],
    "model__class_weight": [None, "balanced", "balanced_subsample"],
}

# Module-level scorer so joblib workers on Windows can import it.
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


F2_SCORER = make_scorer(f2_score)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "run_results.json"


def load_customers(url: str = DATA_URL) -> tuple[pd.DataFrame, dict]:
    """Load the public churn table and apply row-local cleaning only."""
    df = pd.read_csv(url)
    expected = {
        ID_COL,
        TARGET,
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    raw_rows = int(len(df))
    df = df.drop(columns=[ID_COL]).copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_account = df["TotalCharges"].isna() & (df["tenure"] == 0)
    filled_new_accounts = int(new_account.sum())
    df.loc[new_account, "TotalCharges"] = 0.0
    profile = {
        "source": url,
        "rows": raw_rows,
        "columns_after_id_drop": int(df.shape[1]),
        "churn_rate": float((df[TARGET] == "Yes").mean()),
        "totalcharges_filled_as_zero_for_tenure_0": filled_new_accounts,
        "totalcharges_still_missing": int(df["TotalCharges"].isna().sum()),
    }
    return df, profile


def split_customers(df: pd.DataFrame):
    """Split before any estimator is fit. Target map is a fixed label rule."""
    y = (df[TARGET] == "Yes").astype(int)
    x = df.drop(columns=[TARGET])
    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_pipeline() -> Pipeline:
    """Preprocessing estimated from data lives entirely inside this pipeline."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    # Column lists are completed at fit time from the frame's columns via
    # a wrapper so the transformer always sees the post-split schema.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, _categorical_columns),
        ]
    )
    forest = RandomForestClassifier(random_state=RANDOM_STATE)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", forest),
        ]
    )


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in NUMERIC_FEATURES]


def score_predictions(estimator, x, y) -> dict:
    """One metric bundle. Call this on the test set only at the two allowed times."""
    predicted = estimator.predict(x)
    probability = estimator.predict_proba(x)[:, 1]
    return {
        "f2": float(fbeta_score(y, predicted, beta=2)),
        "recall": float(recall_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
    }


def _plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def candidate_table(search, limit: int = 8) -> list[dict]:
    results = search.cv_results_
    n_splits = search.n_splits_
    order = np.argsort(results["mean_test_score"])[::-1][:limit]
    rows = []
    for rank, index in enumerate(order, start=1):
        folds = [
            float(results[f"split{split}_test_score"][index])
            for split in range(n_splits)
        ]
        rows.append(
            {
                "rank_by_mean": rank,
                "cv_index": int(index),
                "mean_f2": float(results["mean_test_score"][index]),
                "std_f2": float(results["std_test_score"][index]),
                "fold_f2": folds,
                "fold_range": float(max(folds) - min(folds)),
                "params": _plain(results["params"][index]),
            }
        )
    return rows


def _neighbor_values(space: list, value) -> list:
    try:
        index = space.index(value)
    except ValueError:
        return [value]
    start = max(0, index - 1)
    stop = min(len(space), index + 2)
    return list(space[start:stop])


def _unique(values: list) -> list:
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def narrow_search_space(random_search: RandomizedSearchCV) -> dict:
    """Cut the random-search lists down using the rule in PLAN.md."""
    results = random_search.cv_results_
    order = np.argsort(results["mean_test_score"])[::-1]
    top_indexes = order[:TOP_REGION]
    best = _plain(random_search.best_params_)
    grid = {}

    for key in UNORDERED_KEYS:
        top_values = [_plain(results["params"][index][key]) for index in top_indexes]
        best_value = best[key]
        if top_values.count(best_value) >= 3:
            grid[key] = [best_value]
        else:
            values = _unique(top_values)
            if best_value not in values:
                values.insert(0, best_value)
            grid[key] = values

    for key, space in ORDERED_SPACES.items():
        grid[key] = _neighbor_values(space, best[key])

    def product_size(candidate: dict) -> int:
        size = 1
        for values in candidate.values():
            size *= len(values)
        return size

    while product_size(grid) > MAX_GRID_COMBOS:
        ordered_widths = [
            (len(grid[key]), key) for key in ORDERED_SPACES if len(grid[key]) > 1
        ]
        if not ordered_widths:
            break
        _, widest = max(ordered_widths)
        grid[widest] = [best[widest]]

    return grid


def select_from_grid(grid_search: GridSearchCV) -> dict:
    """Apply the near-tie stability rule. Does not touch the test set."""
    results = grid_search.cv_results_
    means = np.asarray(results["mean_test_score"], dtype=float)
    stds = np.asarray(results["std_test_score"], dtype=float)
    best_index = int(np.argmax(means))
    best_mean = float(means[best_index])
    band = np.flatnonzero(means >= best_mean - NEAR_TIE_BAND)

    # Lowest std, then highest mean, then the refit index if still tied.
    def sort_key(index: int):
        is_refit = 0 if index == best_index else 1
        return (stds[index], -means[index], is_refit)

    chosen_index = int(sorted(band.tolist(), key=sort_key)[0])
    outside_tighter = None
    if chosen_index == best_index:
        # Note a tighter model that lost only because it fell outside the band.
        tighter_order = np.argsort(stds)
        for index in tighter_order:
            index = int(index)
            if index == best_index:
                continue
            if means[index] < best_mean - NEAR_TIE_BAND and stds[index] < stds[best_index]:
                outside_tighter = index
                break

    chosen_params = _plain(results["params"][chosen_index])
    use_refit_estimator = chosen_index == best_index
    if use_refit_estimator:
        reason = (
            "The refit winner is inside the 0.01 F2 near-tie band and has the "
            "lowest fold standard deviation in that band (ties on spread go to "
            "the higher mean). best_estimator_ is used as trained by refit=True "
            "and is not fit again."
        )
    else:
        reason = (
            "A grid candidate within 0.01 mean F2 of the leader has a lower "
            "fold standard deviation. That candidate is the shipped model. "
            "It is fit once on the training split. best_estimator_ is not fit again."
        )

    return {
        "near_tie_band_f2": NEAR_TIE_BAND,
        "refit_index": best_index,
        "refit_mean_f2": best_mean,
        "refit_std_f2": float(stds[best_index]),
        "refit_params": _plain(results["params"][best_index]),
        "chosen_index": chosen_index,
        "chosen_mean_f2": float(means[chosen_index]),
        "chosen_std_f2": float(stds[chosen_index]),
        "chosen_params": chosen_params,
        "use_refit_estimator": use_refit_estimator,
        "reason": reason,
        "tighter_model_outside_band": (
            None
            if outside_tighter is None
            else {
                "cv_index": outside_tighter,
                "mean_f2": float(means[outside_tighter]),
                "std_f2": float(stds[outside_tighter]),
                "params": _plain(results["params"][outside_tighter]),
            }
        ),
    }


def _fit_selected(grid_search, selection, x_train, y_train):
    if selection["use_refit_estimator"]:
        return grid_search.best_estimator_, False
    pipeline = build_pipeline()
    pipeline.set_params(**selection["chosen_params"])
    pipeline.fit(x_train, y_train)
    return pipeline, True


def _search_summary(search, kind: str) -> dict:
    return {
        "kind": kind,
        "scoring": "f2",
        "n_splits": int(search.n_splits_),
        "n_candidates": int(len(search.cv_results_["params"])),
        "best_params": _plain(search.best_params_),
        "best_mean_f2": float(search.best_score_),
        "best_std_f2": float(
            search.cv_results_["std_test_score"][search.best_index_]
        ),
        "top_candidates": candidate_table(search),
    }


def run() -> dict:
    df, profile = load_customers()
    x_train, x_test, y_train, y_test = split_customers(df)

    # --- Baseline. Test set touch 1. Happens before either search exists. ---
    baseline = build_pipeline()
    baseline_params = _plain(baseline.named_steps["model"].get_params())
    baseline.fit(x_train, y_train)
    baseline_test = score_predictions(baseline, x_test, y_test)

    # --- Broad search, training split only. ---
    random_search = RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring=F2_SCORER,
        cv=CV,
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        error_score="raise",
    )
    random_search.fit(x_train, y_train)
    grid = narrow_search_space(random_search)

    # --- Focused search, training split only. ---
    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=grid,
        scoring=F2_SCORER,
        cv=CV,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    grid_search.fit(x_train, y_train)

    selection = select_from_grid(grid_search)
    tuned, extra_train_fit = _fit_selected(
        grid_search, selection, x_train, y_train
    )
    selection["extra_train_fit"] = extra_train_fit

    # --- Test set touch 2. No further search after this. ---
    tuned_test = score_predictions(tuned, x_test, y_test)
    comparison = {
        metric: {
            "baseline": baseline_test[metric],
            "tuned": tuned_test[metric],
            "tuned_minus_baseline": tuned_test[metric] - baseline_test[metric],
        }
        for metric in baseline_test
    }

    payload = {
        "data_profile": profile,
        "split": {
            "test_size": TEST_SIZE,
            "stratify": True,
            "random_state": RANDOM_STATE,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "scoring": {
            "name": "f2",
            "beta": 2,
            "reason": (
                "A missed churn costs more than an unneeded retention offer. "
                "F2 weights recall more than precision and still penalizes a "
                "model that flags every customer."
            ),
        },
        "baseline": {
            "hyperparameters": {
                key: baseline_params[key]
                for key in (
                    "n_estimators",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "max_features",
                    "class_weight",
                    "random_state",
                )
            },
            "test_metrics": baseline_test,
        },
        "random_search": _search_summary(random_search, "RandomizedSearchCV"),
        "narrowed_grid": _plain(grid),
        "grid_search": _search_summary(grid_search, "GridSearchCV"),
        "selection": selection,
        "tuned_test_metrics": tuned_test,
        "comparison": comparison,
        "test_set_scored_times": 2,
        "versions": {
            "pandas": pd.__version__,
            "scikit-learn": __import__("sklearn").__version__,
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_report(payload)
    return payload


def _print_report(payload: dict) -> None:
    print("Train / test:", payload["split"]["n_train"], "/", payload["split"]["n_test"])
    print("Churn rate (all rows):", round(payload["data_profile"]["churn_rate"], 4))
    print("Baseline test:", json.dumps(payload["baseline"]["test_metrics"], indent=2))
    print("Random search best:", payload["random_search"]["best_params"])
    print(
        "Random search mean F2 / std:",
        round(payload["random_search"]["best_mean_f2"], 4),
        "/",
        round(payload["random_search"]["best_std_f2"], 4),
    )
    print("Narrowed grid:", payload["narrowed_grid"])
    print("Grid best:", payload["grid_search"]["best_params"])
    print(
        "Grid mean F2 / std:",
        round(payload["grid_search"]["best_mean_f2"], 4),
        "/",
        round(payload["grid_search"]["best_std_f2"], 4),
    )
    print("Selection:", payload["selection"]["reason"])
    print("Chosen params:", payload["selection"]["chosen_params"])
    print("Tuned test:", json.dumps(payload["tuned_test_metrics"], indent=2))
    print("Wrote", ARTIFACT_PATH)


if __name__ == "__main__":
    run()
