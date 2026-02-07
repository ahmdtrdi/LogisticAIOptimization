from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from .utils import Paths, ensure_dir, get_logger, load_joblib


log = get_logger("inference_pipeline")


@dataclass(frozen=True)
class Action:
    name: str
    cost: float
    risk_reduction: float
    capacity_weight: int = 1


def expected_net_benefit(p_late: float, penalty_late: float, action: Action) -> float:
    cost_without = p_late * penalty_late
    p_after = p_late * (1.0 - action.risk_reduction)
    cost_with = p_after * penalty_late + action.cost
    return cost_without - cost_with


def best_action(p_late: float, penalty_late: float, actions: Dict[str, Action], min_positive: float = 0.0):
    best_name, best_benefit = None, -np.inf
    for name, act in actions.items():
        b = expected_net_benefit(p_late, penalty_late, act)
        if b > best_benefit:
            best_name, best_benefit = name, b
    if best_benefit <= min_positive:
        return None, best_benefit
    return best_name, best_benefit


def optimize_interventions(
    proba: np.ndarray,
    penalty_late: float,
    actions: Dict[str, Action],
    max_capacity: Optional[int] = None,
    budget: Optional[float] = None,
    min_positive_benefit: float = 0.0
) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(proba):
        act_name, benefit = best_action(float(p), penalty_late, actions, min_positive=min_positive_benefit)
        if act_name is None:
            continue
        act = actions[act_name]
        roi = (benefit / act.cost * 100.0) if act.cost > 0 else np.inf
        rows.append({
            "idx": i,
            "p_late": float(p),
            "action": act_name,
            "action_cost": act.cost,
            "capacity_weight": act.capacity_weight,
            "expected_net_benefit": float(benefit),
            "expected_roi_percent": float(roi),
        })

    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan

    plan = plan.sort_values(["expected_net_benefit", "expected_roi_percent"], ascending=[False, False]).reset_index(drop=True)

    selected = []
    used_capacity, used_budget = 0, 0.0

    for _, r in plan.iterrows():
        w = int(r["capacity_weight"])
        c = float(r["action_cost"])
        if max_capacity is not None and (used_capacity + w) > max_capacity:
            continue
        if budget is not None and (used_budget + c) > budget:
            continue
        selected.append(r)
        used_capacity += w
        used_budget += c

    selected_df = pd.DataFrame(selected)
    if not selected_df.empty:
        selected_df["cum_capacity"] = selected_df["capacity_weight"].cumsum()
        selected_df["cum_budget"] = selected_df["action_cost"].cumsum()

    return selected_df


def run(cfg: Dict[str, Any], input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = Paths.from_cfg(cfg)
    ensure_dir(paths.predictions_dir)

    bundle_path = paths.model_dir / "champion_bundle.joblib"
    bundle = load_joblib(bundle_path)

    # Read input batch
    df = pd.read_csv(input_path)

    # Ensure same feature columns (drop extra, add missing)
    X = df.copy()
    for c in bundle.feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[bundle.feature_cols]

    proba = bundle.pipeline.predict_proba(X)[:, 1]
    risk = (proba >= bundle.decision_threshold).astype(int)

    pred_df = df.copy()
    pred_df["late_risk_proba"] = proba
    pred_df["risk_label"] = risk
    pred_df["decision_threshold"] = float(bundle.decision_threshold)

    # Optimization settings
    biz = cfg.get("business", {})
    opt = cfg.get("optimization", {})

    penalty_late = float(biz.get("cost_penalty_late", 50.0))
    max_capacity = opt.get("max_capacity", None)
    budget = opt.get("budget", None)

    actions = {
        "EXPEDITE": Action("EXPEDITE", cost=float(biz.get("cost_intervention", 15.0)), risk_reduction=0.70, capacity_weight=2),
        "PRIORITY_HANDLING": Action("PRIORITY_HANDLING", cost=8.0, risk_reduction=0.45, capacity_weight=1),
        "PROACTIVE_CALL": Action("PROACTIVE_CALL", cost=3.0, risk_reduction=0.18, capacity_weight=1),
    }

    plan = optimize_interventions(
        proba=np.asarray(proba),
        penalty_late=penalty_late,
        actions=actions,
        max_capacity=max_capacity,
        budget=budget,
        min_positive_benefit=0.0
    )

    # Attach readable metadata if present in input
    if not plan.empty:
        plan_out = plan.copy()
        plan_out["source_row"] = plan_out["idx"]
        # join by row position
        joined = pred_df.reset_index(drop=True).iloc[plan_out["idx"].astype(int).values].reset_index(drop=True)
        plan_out = pd.concat([joined, plan_out.drop(columns=["idx"]).reset_index(drop=True)], axis=1)
    else:
        plan_out = plan

    # Save outputs
    pred_path = paths.predictions_dir / "predictions.csv"
    plan_path = paths.predictions_dir / "action_plan.csv"
    pred_df.to_csv(pred_path, index=False)
    plan_out.to_csv(plan_path, index=False)

    log.info(f"[EXPLAIN] Saved predictions: {pred_path}")
    log.info(f"[EXPLAIN] Saved action plan: {plan_path}")

    return pred_df, plan_out
