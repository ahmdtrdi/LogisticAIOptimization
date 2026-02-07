from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import copy
from typing import Any, Dict, Optional, List, Tuple

import joblib
import yaml


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def cleanup_run_artifacts(
    paths: "Paths",
    *,
    remove_models: bool = True,
    remove_predictions: bool = True,
    remove_features: bool = False,
    remove_preprocessed: bool = False,
    keep: Optional[Iterable[str]] = None,
) -> None:
    """
    Delete old run artifacts to prevent disk from growing.
    - remove_models: delete models/*.joblib, *.json
    - remove_predictions: delete data/04-predictions/*
    - remove_features: delete data/03-features/* (meta/action plan)
    - remove_preprocessed: delete data/02-preprocessed/* (train_ready)
    keep: list of filename substrings to keep (safety).
    """
    keep = set(keep or [])

    def _safe_rm(p: Path):
        if not p.exists():
            return
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p, ignore_errors=True)

    def _rm_dir_contents(d: Path):
        if not d.exists():
            return
        for item in d.iterdir():
            name = item.name
            if any(k in name for k in keep):
                continue
            _safe_rm(item)

    if remove_models:
        _rm_dir_contents(paths.model_dir)

    if remove_predictions:
        _rm_dir_contents(paths.predictions_dir)

    if remove_features:
        _rm_dir_contents(paths.features_dir)

    if remove_preprocessed:
        _rm_dir_contents(paths.preprocessed_dir)


def save_json(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_joblib(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: str | os. PathLike) -> Any: 
    return joblib.load(path)


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    raw_dir: Path
    preprocessed_dir: Path
    features_dir: Path
    predictions_dir: Path
    model_dir: Path

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "Paths":
        p = cfg["paths"]
        return Paths(
            data_dir=Path(p["data_dir"]),
            raw_dir=Path(p["raw_dir"]),
            preprocessed_dir=Path(p["preprocessed_dir"]),
            features_dir=Path(p["features_dir"]),
            predictions_dir=Path(p["predictions_dir"]),
            model_dir=Path(p["model_dir"]),
        )
        
def resolve_cfg_paths(cfg: Dict[str, Any], config_path: str | os.PathLike) -> Dict[str, Any]:
    """
    Make cfg['paths'][*] absolute, relative to project root inferred from config location.
    Assumption: config is in <root>/config/*.yaml
    """
    cfg2 = copy.deepcopy(cfg)
    config_path = Path(config_path).resolve()

    # <root>/config/local.yaml -> root=<root>
    root = config_path.parents[1]

    def _resolve(p: str) -> str:
        pth = Path(p)
        if pth.is_absolute():
            return str(pth)
        return str((root / pth).resolve())

    if "paths" not in cfg2:
        raise ValueError("Config missing required section: 'paths'")

    for k in ["data_dir", "raw_dir", "preprocessed_dir", "features_dir", "predictions_dir", "model_dir"]:
        if k not in cfg2["paths"]:
            raise ValueError(f"Config missing required key: paths.{k}")
        cfg2["paths"][k] = _resolve(cfg2["paths"][k])

    return cfg2


def _require_keys(cfg: Dict[str, Any], required: List[Tuple[str, ...]]) -> None:
    for path in required:
        cur = cfg
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                raise ValueError(f"Config missing required key: {'.'.join(path)}")
            cur = cur[k]


def validate_config(cfg: Dict[str, Any]) -> None:
    required = [
        ("paths", "raw_dir"),
        ("paths", "preprocessed_dir"),
        ("paths", "features_dir"),
        ("paths", "predictions_dir"),
        ("paths", "model_dir"),
        ("data", "main_csv"),
        ("schema", "target_col"),
        ("schema", "date_col"),
        ("schema", "id_col"),
        ("train", "random_seed"),
        ("train", "test_size"),
        ("train", "val_size"),
    ]
    _require_keys(cfg, required)

    ts = float(cfg["train"]["test_size"])
    vs = float(cfg["train"]["val_size"])
    if ts <= 0 or vs <= 0 or (ts + vs) >= 0.9:
        raise ValueError("train.test_size and train.val_size must be >0 and sum to <0.9")
