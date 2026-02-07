import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.utils import load_config, get_logger, resolve_cfg_paths, validate_config  # noqa: E402
from src.pipelines import inference_pipeline  # noqa: E402

log = get_logger("inference")


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline entrypoint")
    parser.add_argument("--config", required=True, help="Path to config yaml (e.g. config/local.yaml)")
    parser.add_argument("--input", required=True, help="CSV/Parquet input path for batch prediction")
    parser.add_argument("--output-dir", default=None, help="Override predictions output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_cfg_paths(cfg, args.config)
    validate_config(cfg)

    log.info("Running inference pipeline...")
    inference_pipeline.run(cfg, input_path=args.input, output_dir=args.output_dir)
    log.info("Inference completed.")


if __name__ == "__main__":
    main()