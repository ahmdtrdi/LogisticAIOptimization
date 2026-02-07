from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from .utils import Paths, ensure_dir, get_logger, save_joblib, save_json


log = get_logger("training_pipeline")


@dataclass
class ModelBundle:
    pipeline: Any                 # sklearn Pipeline (preprocess + model)
    decision_threshold: float
    feature_cols: List[str]
    target_col: str


def find_best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray, steps: int = 200) -> Tuple[float, float]:
    """
    Find threshold maximizing F1 on validation set.
    Returns (best_threshold, best_f1).
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, steps):
        pred = (proba >= t).astype(int)
        # F1 for positive class
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = Paths.from_cfg(cfg)
    ensure_dir(paths.model_dir)

    schema = cfg["schema"]
    target_col = schema["target_col"]

    train_path = paths.preprocessed_dir / "train_ready.parquet"
    log.info(f"Loading train-ready data: {train_path}")
    df = pd.read_parquet(train_path)

    # Split X/y
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col], errors="ignore")

    # Identify columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    # Model
    rf_params = cfg.get("model", {}).get("rf_params", {})
    clf = RandomForestClassifier(**rf_params)

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    # Split train/val/test
    seed = int(cfg["train"].get("random_seed", 42))
    test_size = float(cfg["train"].get("test_size", 0.2))
    val_size = float(cfg["train"].get("val_size", 0.2))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=seed, stratify=y
    )
    # val relative to temp
    val_rel = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_rel), random_state=seed, stratify=y_temp
    )

    log.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Fit
    model.fit(X_train, y_train)

    # Validation threshold tuning
    val_proba = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold_by_f1(y_val.values, val_proba)
    log.info(f"Best threshold on val: {best_t:.4f} | Best F1: {best_f1:.4f}")

    # Test evaluation
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_t).astype(int)

    acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_proba)

    report = classification_report(y_test, test_pred, output_dict=True)
    cm = confusion_matrix(y_test, test_pred)

    metrics = {
        "test_accuracy": float(acc),
        "test_auc": float(auc),
        "val_best_threshold": float(best_t),
        "val_best_f1": float(best_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }

    # Save bundle
    bundle = ModelBundle(
        pipeline=model,
        decision_threshold=best_t,
        feature_cols=list(X.columns),
        target_col=target_col,
    )

    bundle_path = paths.model_dir / "champion_bundle.joblib"
    metrics_path = paths.model_dir / "metrics.json"

    save_joblib(bundle, bundle_path)
    save_json(metrics, metrics_path)

    log.info(f"Saved model bundle: {bundle_path}")
    log.info(f"Saved metrics: {metrics_path}")

    return metrics
