import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.utils import load_config, get_logger, resolve_cfg_paths, validate_config
from src.pipelines import feature_eng_pipeline, training_pipeline 

log = get_logger("train")


def main():
    parser = argparse.ArgumentParser(description="Train pipeline entrypoint")
    parser.add_argument("--config", required=True, help="Path to config yaml (e.g. config/local.yaml)")
    parser.add_argument("--skip-feature-eng", action="store_true", help="Skip feature engineering step")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_cfg_paths(cfg, args.config)
    validate_config(cfg)

    if not args.skip_feature_eng:
        log.info("Running feature engineering pipeline...")
        feature_eng_pipeline.run(cfg)
    else:
        log.info("Skipping feature engineering pipeline (using existing preprocessed data).")

    log.info("Running training pipeline...")
    metrics = training_pipeline.run(cfg)

    log.info(
        f"Training done. "
        f"Test AUC={metrics['test_auc']:.4f} | Test Acc={metrics['test_accuracy']:.4f} | "
        f"Best Threshold={metrics['val_best_threshold']:.4f}"
    )


if __name__ == "__main__":
    main()
