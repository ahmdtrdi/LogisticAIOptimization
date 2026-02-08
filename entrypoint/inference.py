import argparse
import sys
import os

# Add root path
sys.path.append(os.getcwd())

from src.pipelines.utils import load_config, get_logger
from src.pipelines import inference_pipeline

log = get_logger("entrypoint_inference")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml", help="Path to config file")
    parser.add_argument("--input", required=True, help="Path to input CSV file (Raw Data)")
    parser.add_argument("--model", required=True, help="Path to model .pkl file")
    parser.add_argument("--output", default=None, help="Path to save predictions")
    
    args = parser.parse_args()
    
    # 1. Load Config
    cfg = load_config(args.config)
    
    # 2. Validate Inputs
    if not os.path.exists(args.input):
        log.error(f"Input file not found: {args.input}")
        return
        
    if not os.path.exists(args.model):
        log.error(f"Model file not found: {args.model}")
        return

    # 3. Run Inference
    inference_pipeline.run(
        cfg=cfg,
        input_path=args.input,
        model_path=args.model,
        output_path=args.output
    )

if __name__ == "__main__":
    main()