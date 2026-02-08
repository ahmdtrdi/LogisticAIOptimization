import argparse
import sys
import os
import time
import pandas as pd
from datetime import timedelta

# Add root path
sys.path.append(os.getcwd())

from src.pipelines.utils import load_config, get_logger
from src.pipelines import feature_eng_pipeline, training_pipeline

log = get_logger("entrypoint")

def main():
    # Start Timer
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml", help="Path to config file")
    parser.add_argument("--mode", type=int, choices=[1, 2], help="1=Single Model, 2=Ensemble")
    parser.add_argument("--skip-fe", action="store_true", help="Skip Feature Engineering")
    args = parser.parse_args()
    
    # Load Config
    cfg = load_config(args.config)
    
    # Run Feature Engineering
    if not args.skip_fe:
        feature_eng_pipeline.run(cfg)
    else:
        log.info("Skipping Feature Engineering (Assumed data exists)...")
        
    # Load Data
    train_path = os.path.join(cfg['paths']['features_dir'], cfg['data']['train_ready_parquet'])
    if not os.path.exists(train_path):
        log.error("Train data not found! Run without --skip-fe first.")
        return

    df = pd.read_parquet(train_path)
    X = df.drop(columns=[cfg['schema']['target']])
    y = df[cfg['schema']['target']]
    
    # Mode Selection
    mode = args.mode
    if not mode:
        print("\n=== SELECT TRAINING MODE ===")
        print("1) Single Model (Compare -> Select Best -> Tune -> Threshold)")
        print("2) Ensemble (Train All -> Optimize Weights + Threshold)")
        try:
            mode = int(input("Input [1/2]: "))
        except:
            mode = 1
            
    # Run Pipeline
    if mode == 1:
        training_pipeline.run_single_model_flow(cfg, X, y)
    elif mode == 2:
        training_pipeline.run_ensemble_flow(cfg, X, y)

    # End Timer
    end_time = time.time()
    elapsed = end_time - start_time
    duration = str(timedelta(seconds=int(elapsed)))
    
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins} minutes and {secs} seconds"

    print("\n" + "="*50)
    log.info("PIPELINE EXECUTION COMPLETE")
    log.info(f"Total duration: {time_str}.")
    log.info("="*50)

if __name__ == "__main__":
    main()