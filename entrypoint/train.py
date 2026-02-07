import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.utils import load_config, get_logger, resolve_cfg_paths, validate_config, Paths, cleanup_run_artifacts  # noqa: E402
from src.pipelines import feature_eng_pipeline, training_pipeline  # noqa: E402

log = get_logger("train")


def _ask_mode_interactive() -> int:
    print("\nChoose training mode:")
    print("  1) Single Model (multi->select best->tune best->threshold)")
    print("  2) Ensemble (tune all->voting/hill/cmaes/nsga2)")
    print("  3) Both (run 1 then 2)")
    while True:
        v = input("Input [1/2/3]: ").strip()
        if v in {"1", "2", "3"}:
            return int(v)
        print("Invalid input. Please type 1, 2, or 3.")
    


def main():
    parser = argparse.ArgumentParser(description="Train pipeline entrypoint")
    parser.add_argument("--config", required=True, help="Path to config yaml (e.g. config/local.yaml)")
    parser.add_argument("--skip-feature-eng", action="store_true", help="Skip feature engineering step")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=None, help="1=single, 2=ensemble, 3=both")

    # CLEAN FLAGS
    parser.add_argument("--clean", action="store_true", help="Delete old models + predictions before run")
    parser.add_argument("--clean-all", action="store_true", help="Delete models + predictions + features + preprocessed")

    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_cfg_paths(cfg, args.config)
    validate_config(cfg)

    # CLEANUP BEFORE RUN
    if args.clean or args.clean_all:
        p = Paths.from_cfg(cfg)
        log.info("[CLEAN] Removing old artifacts...")
        cleanup_run_artifacts(
            p,
            remove_models=True,
            remove_predictions=True,
            remove_features=bool(args.clean_all),
            remove_preprocessed=bool(args.clean_all),
            keep=[".gitkeep"],
        )
        log.info("[CLEAN] Done.")

    mode = args.mode if args.mode is not None else _ask_mode_interactive()
    cfg.setdefault("train", {})
    cfg["train"]["run_mode"] = mode

    if not args.skip_feature_eng:
        log.info("Running feature engineering pipeline...")
        feature_eng_pipeline.run(cfg)
    else:
        log.info("Skipping feature engineering pipeline (using existing preprocessed data).")

    log.info("Running training pipeline...")
    metrics = training_pipeline.run(cfg)

    champ = metrics.get("champion", {})
    log.info(
        f"Done. Champion={champ.get('name')} | "
        f"TestAUC={champ.get('test_auc', float('nan')):.4f} | "
        f"TestAcc={champ.get('test_accuracy', float('nan')):.4f} | "
        f"Threshold={champ.get('threshold', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()