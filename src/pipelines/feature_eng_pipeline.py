from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import re

from .utils import Paths, ensure_dir, get_logger


log = get_logger("feature_eng_pipeline")


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    """
    Robust CSV loader for datasets that aren't UTF-8.
    Tries common encodings and falls back safely.
    """
    path = str(path)
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e

    return pd.read_csv(path, encoding="utf-8", errors="replace")

def resolve_date_column(df: pd.DataFrame, preferred: str) -> str:
    """
    Resolve date column robustly:
    1) If preferred exists, use it.
    2) Else find columns containing keywords like 'date' and 'order' (case-insensitive).
    3) Else find any column containing 'date'.
    """
    if preferred in df.columns:
        return preferred

    cols = list(df.columns)
    cols_lower = {c: c.lower() for c in cols}

    # strongest match: both 'order' and 'date'
    strong = [c for c in cols if ("order" in cols_lower[c] and "date" in cols_lower[c])]
    if len(strong) == 1:
        log.warning(f"Configured date_col '{preferred}' not found. Auto-selected '{strong[0]}'.")
        return strong[0]
    if len(strong) > 1:
        # prefer ones with parentheses like DateOrders
        strong_sorted = sorted(strong, key=lambda x: ("dateorders" not in cols_lower[x], len(x)))
        pick = strong_sorted[0]
        log.warning(f"Configured date_col '{preferred}' not found. Multiple candidates {strong}. Picked '{pick}'.")
        return pick

    # weaker match: any 'date'
    weak = [c for c in cols if "date" in cols_lower[c]]
    if len(weak) == 1:
        log.warning(f"Configured date_col '{preferred}' not found. Auto-selected '{weak[0]}'.")
        return weak[0]
    if len(weak) > 1:
        weak_sorted = sorted(weak, key=lambda x: ("order" not in cols_lower[x], len(x)))
        pick = weak_sorted[0]
        log.warning(f"Configured date_col '{preferred}' not found. Multiple candidates {weak}. Picked '{pick}'.")
        return pick

    sample_cols = ", ".join(cols[:25])
    raise KeyError(
        f"Could not resolve a date column. preferred='{preferred}'. "
        f"No columns matched pattern. Sample columns: {sample_cols}"
    )


def resolve_column_by_substring(df: pd.DataFrame, preferred: str, required_substrings: list[str]) -> str:
    """
    Generic resolver: match preferred, else column containing all required substrings (case-insensitive).
    """
    if preferred in df.columns:
        return preferred

    cols = list(df.columns)
    cols_lower = {c: c.lower() for c in cols}

    matches = []
    for c in cols:
        if all(s.lower() in cols_lower[c] for s in required_substrings):
            matches.append(c)

    if len(matches) == 1:
        log.warning(f"Configured column '{preferred}' not found. Auto-selected '{matches[0]}'.")
        return matches[0]
    if len(matches) > 1:
        pick = sorted(matches, key=len)[0]
        log.warning(f"Configured column '{preferred}' not found. Multiple matches {matches}. Picked '{pick}'.")
        return pick

    sample_cols = ", ".join(cols[:25])
    raise KeyError(
        f"Could not resolve column. preferred='{preferred}', required_substrings={required_substrings}. "
        f"Sample columns: {sample_cols}"
    )

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized haversine distance in km.
    """
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def _build_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dt = _safe_to_datetime(df[date_col])
    df["order_year"] = dt.dt.year
    df["order_month"] = dt.dt.month
    df["order_dow"] = dt.dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5, 6]).astype(int)
    return df


def _build_peak_season(df: pd.DataFrame, peak_months: List[int]) -> pd.DataFrame:
    df["is_peak_season"] = df["order_month"].isin(peak_months).astype(int)
    return df


def _build_sla_features(df: pd.DataFrame, sla_col: str, fast_threshold: int) -> pd.DataFrame:
    # Example: aggressive SLA if scheduled_days <= fast_threshold
    df["sla_days"] = pd.to_numeric(df[sla_col], errors="coerce")
    df["sla_aggressive"] = (df["sla_days"] <= fast_threshold).astype(int)
    return df


def _build_distance(df: pd.DataFrame, lat_col: str, lon_col: str, wh_lat: float, wh_lon: float) -> pd.DataFrame:
    if lat_col in df.columns and lon_col in df.columns:
        df["distance_km"] = haversine_km(df[lat_col], df[lon_col], wh_lat, wh_lon)
    else:
        df["distance_km"] = np.nan
    return df

def derive_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Ensure target_col exists.
    Priority:
      1) If target_col already exists -> use it
      2) Else if 'Delivery Status' exists -> derive from Late delivery
      3) Else if 'Late_delivery_risk' exists -> use it as fallback binary target
    """
    if target_col in df.columns:
        return df

    if "Delivery Status" in df.columns:
        df[target_col] = np.where(df["Delivery Status"].astype(str).str.lower().eq("late delivery"), 1, 0)
        log.info(f"Derived target '{target_col}' from 'Delivery Status' == 'Late delivery'.")
        return df

    if "Late_delivery_risk" in df.columns:
        df[target_col] = pd.to_numeric(df["Late_delivery_risk"], errors="coerce").fillna(0).astype(int)
        log.warning(f"Derived target '{target_col}' from fallback column 'Late_delivery_risk'.")
        return df

    sample_cols = ", ".join(list(df.columns)[:25])
    raise KeyError(
        f"Could not derive target '{target_col}'. "
        f"Expected either '{target_col}' or 'Delivery Status' or 'Late_delivery_risk'. "
        f"Sample columns: {sample_cols}"
    )


def run(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - train_ready_df: features + target (for training)
      - meta_df: human-readable metadata for dashboard/action plan
    Saves:
      - data/02-preprocessed/train_ready.parquet
      - data/03-features/order_metadata.parquet
    """
    paths = Paths.from_cfg(cfg)
    ensure_dir(paths.preprocessed_dir)
    ensure_dir(paths.features_dir)

    # Load raw data (robust IO)
    raw_cfg = cfg["data"]
    main_path = paths.raw_dir / raw_cfg["main_csv"]
    log.info(f"Reading main raw data: {main_path}")
    df = read_csv_safely(main_path)

 
    schema = cfg["schema"]
    date_col_cfg = schema["date_col"]
    target_col_cfg = schema["target_col"]
    id_col_cfg = schema["id_col"]

    date_col = resolve_date_column(df, preferred=date_col_cfg)

    if id_col_cfg in df.columns:
        id_col = id_col_cfg
    else:
        id_col = resolve_column_by_substring(df, preferred=id_col_cfg, required_substrings=["order", "id"])

    target_col = target_col_cfg
    df = derive_target(df, target_col=target_col)

    # Helpful logging
    log.info(f"Resolved columns -> date_col='{date_col}', id_col='{id_col}', target_col='{target_col}'")

    # Basic cleaning
    df = df.dropna(subset=[date_col, target_col]).copy()

    # Feature engineering
    df = _build_time_features(df, date_col=date_col)

    peak_months = cfg["features"].get("peak_months", [11, 12])
    df = _build_peak_season(df, peak_months=peak_months)

    sla_col = cfg["features"].get("sla_col", "Days for shipment (scheduled)")
    fast_thr = int(cfg["features"].get("fast_sla_threshold", 2))
    if sla_col in df.columns:
        df = _build_sla_features(df, sla_col=sla_col, fast_threshold=fast_thr)
    else:
        log.warning(f"SLA column '{sla_col}' not found. Skipping SLA features.")

    lat_col = cfg["features"].get("lat_col", "Latitude")
    lon_col = cfg["features"].get("lon_col", "Longitude")
    wh_lat = float(cfg["features"].get("wh_lat", 0.0))
    wh_lon = float(cfg["features"].get("wh_lon", 0.0))
    df = _build_distance(df, lat_col=lat_col, lon_col=lon_col, wh_lat=wh_lat, wh_lon=wh_lon)

    # Ensure target int
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    # Build metadata for dashboard/action plan
    meta_cols = [
        id_col,
        "Order Country",
        "Order Region",
        "Customer Segment",
        "Shipping Mode",
        "Category Name",
        "Department Name",
        lat_col,
        lon_col,
        "distance_km",
        "sla_days",
        date_col,
        target_col,
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta_df = df[meta_cols].copy()

    # Training-ready dataset
    drop_cols = [id_col] if id_col in df.columns else []
    train_ready_df = df.drop(columns=drop_cols, errors="ignore").copy()

    # Save outputs
    train_path = paths.preprocessed_dir / "train_ready.parquet"
    meta_path = paths.features_dir / "order_metadata.parquet"

    log.info(f"Saving train-ready data: {train_path}")
    train_ready_df.to_parquet(train_path, index=False)

    log.info(f"Saving order metadata: {meta_path}")
    meta_df.to_parquet(meta_path, index=False)

    return train_ready_df, meta_df
