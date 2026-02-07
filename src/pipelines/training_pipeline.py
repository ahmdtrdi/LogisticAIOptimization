from __future__ import annotations

import time
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import optuna
from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from optuna.samplers import NSGAIISampler
from optuna.samplers import CmaEsSampler

from .utils import Paths, ensure_dir, get_logger, save_joblib, save_json


log = get_logger("training_pipeline")

_HAS_XGB = True
_HAS_CAT = True
try:
    from xgboost import XGBClassifier
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
except Exception:
    _HAS_CAT = False

# Utilities
def _fmt_sec(sec: float) -> str:
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def optimize_with_tqdm(
    study: optuna.Study,
    objective,
    n_trials: int,
    timeout_sec=None,
    desc: str = "Optuna"
):
    pbar = tqdm(total=n_trials, desc=desc)

    def _cb(study, trial):
        pbar.update(1)
        try:
            if hasattr(study, "best_value") and study.best_value is not None:
                pbar.set_postfix(best=f"{study.best_value:.4f}")
            elif hasattr(study, "best_trials") and len(study.best_trials) > 0:
                # show best F1 among pareto trials
                best_f1 = max(t.values[0] for t in study.best_trials if t.values is not None)
                pbar.set_postfix(best_f1=f"{best_f1:.4f}")
        except Exception:
            pass

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_sec,
            callbacks=[_cb],
            n_jobs=1,  # penting untuk M1 8GB: aman RAM
        )
    finally:
        pbar.close()


class StepTimer:
    def __init__(self, log, total_steps: int):
        self.log = log
        self.total_steps = total_steps
        self.start = time.time()
        self.step = 0

    def mark(self, msg: str) -> None:
        self.step += 1
        elapsed = time.time() - self.start
        avg = elapsed / max(self.step, 1)
        remaining = max(self.total_steps - self.step, 0)
        eta = avg * remaining
        self.log.info(f"{msg} | elapsed={_fmt_sec(elapsed)} | eta~{_fmt_sec(eta)}")



@dataclass
class ModelBundle:
    pipeline: Any
    decision_threshold: float
    feature_cols: List[str]
    target_col: str
    # for ensemble bundles:
    meta: Optional[Dict[str, Any]] = None


def find_best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray, steps: int = 200) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, steps):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def _build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

# Candidate model factory
def build_candidate_model(name: str, preprocess: ColumnTransformer, seed: int, cfg: Dict[str, Any], tuned_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    name = name.lower()
    tuned_params = tuned_params or {}

    if name == "rf":
        base = cfg.get("model", {}).get("rf_params", {})
        params = {**base, **tuned_params, "random_state": seed, "n_jobs": 2}
        clf = RandomForestClassifier(**params)
        return Pipeline([("preprocess", preprocess), ("clf", clf)])

    if name == "lr":
        # keep it simple
        params = {"max_iter": 3000, "random_state": seed, **tuned_params}
        clf = LogisticRegression(**params)
        return Pipeline([("preprocess", preprocess), ("clf", clf)])

    if name == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        # reasonable defaults
        params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": 2,
            "eval_metric": "logloss",
            "tree_method": "hist",
            **tuned_params,
        }
        clf = XGBClassifier(**params)
        return Pipeline([("preprocess", preprocess), ("clf", clf)])

    if name == "cat":
        if not _HAS_CAT:
            raise ImportError("catboost not installed. Run: pip install catboost")
        params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.08,
            "loss_function": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "thread_count": 2,
            **tuned_params,
        }
        clf = CatBoostClassifier(**params)
        return Pipeline([("preprocess", preprocess), ("clf", clf)])

    raise ValueError(f"Unknown candidate model: {name}")


def eval_on_val_default_threshold(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_val, proba)),
        "f1": float(f1_score(y_val, pred)),
    }

# Optuna tuning (used in both single & ensemble)
def tune_model_optuna(
    cfg: Dict[str, Any],
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocess: ColumnTransformer,
    seed: int,
    n_trials: int,
    timeout_sec: Optional[int],
    metric: str,
) -> Dict[str, Any]:
    metric = metric.lower()

    def objective(trial: optuna.Trial) -> float:
        m = model_name.lower()

        if m == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 800),
                "max_depth": trial.suggest_int("max_depth", 6, 28),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

        elif m == "lr":
            params = {
                "C": trial.suggest_float("C", 1e-3, 30.0, log=True),
                "solver": "lbfgs",
            }

        elif m == "xgb":
            if not _HAS_XGB:
                raise optuna.TrialPruned()
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 700),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0, log=True),
            }

        elif m == "cat":
            if not _HAS_CAT:
                raise optuna.TrialPruned()
            params = {
                "iterations": trial.suggest_int("iterations", 250, 700),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            }

        else:
            raise ValueError(f"Optuna tuning not implemented for: {model_name}")

        model = build_candidate_model(m, preprocess, seed, cfg=cfg, tuned_params=params)
        model.fit(X_train, y_train)

        scores = eval_on_val_default_threshold(model, X_val, y_val)
        return scores[metric]

    study = optuna.create_study(direction="maximize")
    optimize_with_tqdm(
        study,
        objective,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        desc=f"Tuning {model_name.upper()} ({metric})",
    )
    return dict(study.best_params)


# -------------------------
# Mode 1: Single model flow (notebook-like)
# -------------------------
def run_single_model_flow(
    cfg: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocess: ColumnTransformer,
    seed: int,
    timer: StepTimer,
) -> Dict[str, Any]:
    select_metric = str(cfg.get("train", {}).get("select_metric", "auc")).lower()
    tune_cfg = cfg.get("train", {}).get("tune", {})
    use_tune = bool(tune_cfg.get("enabled", False))
    n_trials = int(tune_cfg.get("n_trials", 30))
    timeout_sec = tune_cfg.get("timeout_sec", None)

    candidates = cfg.get("train", {}).get("candidates", ["rf", "lr", "xgb", "cat"])
    log.info(f"[SINGLE] Candidates: {candidates} | select_metric={select_metric}")

    # 1) baseline train all candidates (untuned) and pick best
    scores_map: Dict[str, Dict[str, float]] = {}
    for m in candidates:
        log.info(f"[SINGLE] Fit baseline candidate: {m}")
        model = build_candidate_model(m, preprocess, seed, cfg)
        model.fit(X_train, y_train)
        scores = eval_on_val_default_threshold(model, X_val, y_val)
        scores_map[m] = scores
        log.info(f"[SINGLE] {m} | val_auc={scores['auc']:.4f} | val_f1@0.5={scores['f1']:.4f}")

    timer.mark("[SINGLE 1/4] Baseline candidates evaluated")

    best_model = sorted(scores_map.keys(), key=lambda k: scores_map[k][select_metric], reverse=True)[0]
    log.info(f"[SINGLE] Best model selected: {best_model} | scores={scores_map[best_model]}")

    timer.mark("[SINGLE 2/4] Best model selected")

    # 2) tune only best
    best_params: Dict[str, Any] = {}
    if use_tune:
        log.info(f"[SINGLE] Optuna tuning best='{best_model}' | n_trials={n_trials} | timeout={timeout_sec} | metric={select_metric}")
        best_params = tune_model_optuna(
            cfg=cfg,
            model_name=best_model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            preprocess=preprocess,
            seed=seed,
            n_trials=n_trials,
            timeout_sec=timeout_sec,
            metric=select_metric,
        )
        log.info(f"[SINGLE] Best params for {best_model}: {best_params}")
    else:
        log.info("[SINGLE] Tuning disabled (using defaults/config).")

    timer.mark("[SINGLE 3/4] Hyperparameter tuning completed")

    # 3) fit final tuned model + threshold tuning
    final_model = build_candidate_model(best_model, preprocess, seed, cfg, tuned_params=best_params)
    final_model.fit(X_train, y_train)

    val_proba = final_model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold_by_f1(y_val.values, val_proba)
    log.info(f"[SINGLE] Threshold tuning: best_t={best_t:.4f} | val_best_f1={best_f1:.4f}")

    # 4) test eval
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_t).astype(int)
    acc = float(accuracy_score(y_test, test_pred))
    auc = float(roc_auc_score(y_test, test_proba))

    cm = confusion_matrix(y_test, test_pred).tolist()
    report = classification_report(y_test, test_pred, output_dict=True)

    timer.mark("[SINGLE 4/4] Test evaluation completed")

    return {
        "name": f"single::{best_model}",
        "model_type": best_model,
        "best_params": best_params,
        "threshold": float(best_t),
        "val_best_f1": float(best_f1),
        "test_accuracy": acc,
        "test_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "candidate_scores": scores_map,
        "pipeline": final_model,
    }


# -------------------------
# Mode 2: Ensemble flow
# -------------------------
def _stack_proba(models: Dict[str, Pipeline], X: pd.DataFrame) -> np.ndarray:
    # shape: (n_samples, n_models)
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models.values()])

def soft_voting(proba_mat: np.ndarray) -> np.ndarray:
    return proba_mat.mean(axis=1)

def hill_climb_weights(
    proba_mat: np.ndarray,
    y_val: np.ndarray,
    iters: int = 400,
    step: float = 0.10,
    seed: int = 42
) -> Tuple[np.ndarray, float, float]:
    """
    Optimize weights + threshold by greedy hill-climbing on validation F1.
    Returns: (best_weights, best_threshold, best_f1)
    """
    rng = np.random.default_rng(seed)
    n_models = proba_mat.shape[1]
    w = rng.random(n_models)
    w = w / (w.sum() + 1e-12)

    def score(weights: np.ndarray) -> Tuple[float, float]:
        p = (proba_mat @ weights)
        t, f1 = find_best_threshold_by_f1(y_val, p, steps=120)
        return f1, t

    best_f1, best_t = score(w)

    for _ in range(iters):
        j = rng.integers(0, n_models)
        delta = step * (1 if rng.random() < 0.5 else -1)
        w2 = w.copy()
        w2[j] = np.clip(w2[j] + delta, 0.0, 1.0)
        w2 = w2 / (w2.sum() + 1e-12)

        f1, t = score(w2)
        if f1 > best_f1:
            w, best_f1, best_t = w2, f1, t

    return w, float(best_t), float(best_f1)


def optimize_ensemble_cmaes(
    P_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    seed: int,
    timeout_sec: Optional[int] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Optimize softmax(logits) weights + threshold using Optuna CMA-ES.
    Returns: (best_weights, best_threshold, best_f1)
    """
    yv = np.asarray(y_val).astype(int)
    n_models = P_val.shape[1]

    sampler = CmaEsSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        logits = np.array([trial.suggest_float(f"w{i}", -4.0, 4.0) for i in range(n_models)], dtype=float)
        w = _softmax(logits)

        t = trial.suggest_float("threshold", 0.05, 0.95)
        p = P_val @ w

        return _f1_at_threshold(yv, p, t)

    optimize_with_tqdm(
        study,
        objective,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        desc="CMA-ES weights+threshold (val F1)",
    )

    best = study.best_trial
    logits = np.array([best.params[f"w{i}"] for i in range(n_models)], dtype=float)
    w = _softmax(logits)
    t = float(best.params["threshold"])
    f1 = float(best.value)
    return w, t, f1

def optimize_ensemble_nsga2(
    P_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    seed: int,
    timeout_sec: Optional[int] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Optimize weights+threshold using Optuna NSGA-II multi-objective.
    Objectives: maximize (F1, AUC) on validation.
    Returns: (best_weights, best_threshold, best_f1, best_auc)
    """
    yv = np.asarray(y_val).astype(int)
    n_models = P_val.shape[1]

    sampler = NSGAIISampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)

    def objective(trial: optuna.Trial):
        logits = np.array([trial.suggest_float(f"w{i}", -4.0, 4.0) for i in range(n_models)], dtype=float)
        w = _softmax(logits)

        t = trial.suggest_float("threshold", 0.05, 0.95)
        p = P_val @ w

        f1 = _f1_at_threshold(yv, p, t)
        auc = _auc(yv, p)
        return f1, auc

    optimize_with_tqdm(
        study,
        objective,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        desc="NSGA-II weights+threshold (val F1,AUC)",
    )

    best_trial = sorted(
        study.best_trials,
        key=lambda tr: (tr.values[0], tr.values[1]),
        reverse=True
    )[0]

    logits = np.array([best_trial.params[f"w{i}"] for i in range(n_models)], dtype=float)
    w = _softmax(logits)
    t = float(best_trial.params["threshold"])
    best_f1, best_auc = float(best_trial.values[0]), float(best_trial.values[1])
    return w, t, best_f1, best_auc

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def _f1_at_threshold(y_true: np.ndarray, proba: np.ndarray, t: float) -> float:
    y_true = y_true.astype(int)
    pred = (proba >= t).astype(int)
    return float(f1_score(y_true, pred))

def _auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, proba))

def run_ensemble_flow(
    cfg: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocess: ColumnTransformer,
    seed: int,
    timer: StepTimer,
) -> Dict[str, Any]:
    tune_cfg = cfg.get("train", {}).get("tune", {})
    use_tune = bool(tune_cfg.get("enabled", False))
    n_trials = int(tune_cfg.get("n_trials", 20))
    timeout_sec = tune_cfg.get("timeout_sec", None)

    base_models = cfg.get("train", {}).get("ensemble_models", ["rf", "xgb", "cat", "lr"])
    metric = str(cfg.get("train", {}).get("select_metric", "auc")).lower()

    log.info(f"[ENSEMBLE] Base models: {base_models} | tune={use_tune} | metric={metric}")

    # 1) tune all base models (or use defaults)
    tuned_params_map: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Pipeline] = {}

    for m in base_models:
        if use_tune:
            log.info(f"[ENSEMBLE] Tuning model: {m}")
            params = tune_model_optuna(
                cfg=cfg,
                model_name=m,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                preprocess=preprocess,
                seed=seed,
                n_trials=n_trials,
                timeout_sec=timeout_sec,
                metric=metric,
            )
            tuned_params_map[m] = params
        else:
            tuned_params_map[m] = {}

        model = build_candidate_model(m, preprocess, seed, cfg, tuned_params=tuned_params_map[m])
        log.info(f"[ENSEMBLE] Fit base model: {m}")
        model.fit(X_train, y_train)
        models[m] = model

    timer.mark("[ENSEMBLE 1/4] Base models trained (tuned if enabled)")

    # 2) proba matrices
    P_val = _stack_proba(models, X_val)
    P_test = _stack_proba(models, X_test)
    yv = y_val.values.astype(int)

    p_val_soft = soft_voting(P_val)
    t_soft, f1_soft = find_best_threshold_by_f1(yv, p_val_soft, steps=160)

    w_hill, t_hill, f1_hill = hill_climb_weights(P_val, yv, iters=500, step=0.08, seed=seed)

    w_cma, t_cma, f1_cma = optimize_ensemble_cmaes(
        P_val=P_val,
        y_val=yv,
        n_trials=int(cfg.get("train", {}).get("ensemble_opt", {}).get("cmaes_trials", 30)),
        seed=seed,
        timeout_sec=cfg.get("train", {}).get("ensemble_opt", {}).get("timeout_sec", None),
    )

    w_nsga, t_nsga, f1_nsga, auc_nsga = optimize_ensemble_nsga2(
        P_val=P_val,
        y_val=yv,
        n_trials=int(cfg.get("train", {}).get("ensemble_opt", {}).get("nsga2_trials", 30)),
        seed=seed,
        timeout_sec=cfg.get("train", {}).get("ensemble_opt", {}).get("timeout_sec", None),
    )

    timer.mark("[ENSEMBLE 2/4] Ensemble weight/threshold search completed")

    cand = [
        ("soft_voting", None, t_soft, f1_soft),
        ("hill_climbing", w_hill, t_hill, f1_hill),
        ("cmaes", w_cma, t_cma, f1_cma),
        ("nsga2", w_nsga, t_nsga, f1_nsga),
    ]
    cand_sorted = sorted(cand, key=lambda x: x[3], reverse=True)
    champ_name, champ_w, champ_t, champ_val_f1 = cand_sorted[0]
    log.info(f"[ENSEMBLE] Champion on VAL: {champ_name} | val_f1={champ_val_f1:.4f} | t={champ_t:.4f}")

    timer.mark("[ENSEMBLE 3/4] Ensemble champion selected")

    if champ_name == "soft_voting":
        p_test = soft_voting(P_test)
    else:
        p_test = P_test @ champ_w

    test_pred = (p_test >= champ_t).astype(int)
    acc = float(accuracy_score(y_test, test_pred))
    auc = float(roc_auc_score(y_test, p_test))
    cm = confusion_matrix(y_test, test_pred).tolist()
    report = classification_report(y_test, test_pred, output_dict=True)

    timer.mark("[ENSEMBLE 4/4] Test evaluation completed")
 
    ensemble_meta = {
        "strategy": champ_name,
        "threshold": champ_t,
        "val_f1": champ_val_f1,
        "weights": None if champ_w is None else champ_w.tolist(),
        "base_models": list(models.keys()),
        "tuned_params": tuned_params_map,
        "candidates_val": [
            {"name": n, "threshold": float(t), "val_f1": float(f1), "weights": (None if w is None else np.asarray(w).tolist())}
            for (n, w, t, f1) in cand
        ],
    }
    return {
        "name": f"ensemble::{champ_name}",
        "model_type": "ensemble",
        "threshold": float(champ_t),
        "val_best_f1": float(champ_val_f1),
        "test_accuracy": acc,
        "test_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "pipeline": models,  # dict of pipelines
        "meta": ensemble_meta,
    }

def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = Paths.from_cfg(cfg)
    ensure_dir(paths.model_dir)

    schema = cfg["schema"]
    target_col = schema["target_col"]

    train_path = paths.preprocessed_dir / "train_ready.parquet"
    log.info(f"Loading train-ready data: {train_path}")
    df = pd.read_parquet(train_path)

    # safety guard
    feat_cfg = cfg.get("features", {})
    leak_cols = feat_cfg.get("leakage_cols", [])
    final_drops = feat_cfg.get("final_drops", [])
    df = df.drop(columns=[c for c in (leak_cols + final_drops) if c in df.columns], errors="ignore")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col], errors="ignore")

    seed = int(cfg["train"].get("random_seed", 42))
    test_size = float(cfg["train"].get("test_size", 0.2))
    val_size = float(cfg["train"].get("val_size", 0.2))

    run_mode = int(cfg.get("train", {}).get("run_mode", 1))
    log.info(f"[RUN] mode={run_mode} (1=single, 2=ensemble, 3=both)")
    log.info(f"[RUN] Feature count: {X.shape[1]} | Positive rate: {(y.mean() * 100):.2f}%")

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=seed, stratify=y
    )
    val_rel = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_rel), random_state=seed, stratify=y_temp
    )

    preprocess = _build_preprocess(X_train)

    # timers (rough ETA)
    timer = StepTimer(log, total_steps=4)

    timer.mark("[STEP 1/4] Data loaded & split ready")

    results: Dict[str, Any] = {"runs": []}
    champion: Dict[str, Any] = {}

    # mode 1
    if run_mode in (1, 3):
        single_timer = StepTimer(log, total_steps=4)
        single = run_single_model_flow(
            cfg, X_train, y_train, X_val, y_val, X_test, y_test, preprocess, seed, single_timer
        )
        results["runs"].append({k: v for k, v in single.items() if k != "pipeline"})  # keep metrics only
      
        single_bundle = ModelBundle(
            pipeline=single["pipeline"],
            decision_threshold=single["threshold"],
            feature_cols=list(X.columns),
            target_col=target_col,
            meta={"type": "single", "model_type": single["model_type"], "best_params": single.get("best_params", {})},
        )
        save_joblib(single_bundle, paths.model_dir / "single_bundle.joblib")
        save_json({k: v for k, v in single.items() if k != "pipeline"}, paths.model_dir / "single_metrics.json")
        
        del single["pipeline"]
        gc.collect()
        
        champion = single  # provisional champion
        timer.mark("[STEP 2/4] Single flow finished")

    # mode 2
    if run_mode in (2, 3):
        ens_timer = StepTimer(log, total_steps=4)
        ens = run_ensemble_flow(
            cfg, X_train, y_train, X_val, y_val, X_test, y_test, preprocess, seed, ens_timer
        )
        results["runs"].append({k: v for k, v in ens.items() if k not in {"pipeline"}})

        ens_bundle = ModelBundle(
            pipeline=ens["pipeline"],  # dict of pipelines
            decision_threshold=ens["threshold"],
            feature_cols=list(X.columns),
            target_col=target_col,
            meta=ens.get("meta", {}),
        )
        save_joblib(ens_bundle, paths.model_dir / "ensemble_bundle.joblib")
        save_json({k: v for k, v in ens.items() if k != "pipeline"}, paths.model_dir / "ensemble_metrics.json")

        del ens["pipeline"]
        gc.collect()    
       
        if not champion:
            champion = ens
        else:
            if float(ens["test_auc"]) >= float(champion["test_auc"]):
                champion = ens

        timer.mark("[STEP 3/4] Ensemble flow finished")

    # save overall champion pointer
    results["champion"] = {
        "name": champion["name"],
        "test_accuracy": champion["test_accuracy"],
        "test_auc": champion["test_auc"],
        "threshold": champion["threshold"],
        "type": champion.get("model_type", ""),
    }

    save_json(results, paths.model_dir / "run_summary.json")
    timer.mark("[STEP 4/4] Run summary saved")

    log.info(f"[CHAMPION] {results['champion']}")
    
    try:
        del df, X, y, X_train, X_val, X_test, y_train, y_val, y_test
    except Exception:
        pass
    gc.collect()
    
    return results