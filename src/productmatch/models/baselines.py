"""Deterministic supervised baselines for pairwise entity matching."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SEED = 42
MODEL_CHOICES = ("logistic_regression", "lightgbm")


def make_estimator(name: str, seed: int = DEFAULT_SEED) -> object:
    """Return a classifier with predict_proba; fits occur on train data only."""
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),

            ]
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM not installed. Run pip install -e '.[dev,models]'"
            ) from exc
        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=15,
            colsample_bytree=1.0,
            subsample=1.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"unknown model: {name!r}; choose from {MODEL_CHOICES}")
