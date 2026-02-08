import pandas as pd
import numpy as np
import os
import joblib
from .utils import get_logger

log = get_logger("inference")

def apply_fe_for_inference(df, cfg):
    log.info("Applying Feature Engineering on new data...")

    wh_lat = cfg['feature_engineering']['warehouse_lat']
    wh_lon = cfg['feature_engineering']['warehouse_lon']
    lat_col = cfg['feature_engineering']['lat_col']
    lon_col = cfg['feature_engineering']['lon_col']
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
        
    df['distance_km'] = haversine(wh_lat, wh_lon, df[lat_col], df[lon_col])
    
    drop_cols = cfg['schema']['drop_cols']
    existing_drops = [c for c in drop_cols if c in df.columns]
    df_clean = df.drop(columns=existing_drops)
    
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_clean[col] = df_clean[col].astype('category').cat.codes
        
    df_clean = df_clean.fillna(0)
    
    tgt = cfg['schema']['target']
    if tgt in df_clean.columns:
        df_clean = df_clean.drop(columns=[tgt])
        
    return df_clean

def align_features(df_input, model):
    expected_cols = []
    
    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_
    elif hasattr(model, "feature_names_"): 
        expected_cols = model.feature_names_
    
    else:
        return df_input

    missing_cols = set(expected_cols) - set(df_input.columns)
    if missing_cols:
        for c in missing_cols:
            df_input[c] = 0 
            
    return df_input[expected_cols].copy()

def generate_business_report(df_result):
    """
    Menghasilkan laporan ROI jika Ground Truth tersedia.
    """
    log.info("\n" + "="*50)
    log.info(" BUSINESS IMPACT SIMULATION (ROI ANALYSIS)")
    log.info("="*50)
    
    total_orders = len(df_result)
    log.info(f"Total Orders Evaluated   : {total_orders:,}")
    log.info("-" * 50)
    
    has_ground_truth = False
    actual_col = None
    
    if 'Delivery Status' in df_result.columns:
        y_true = np.where(df_result['Delivery Status'] == 'Late delivery', 1, 0)
        has_ground_truth = True
    elif 'is_late' in df_result.columns:
        y_true = df_result['is_late'].values
        has_ground_truth = True
        
    # Cost Parameters
    COST_INTERVENTION = 15.0
    COST_PENALTY = 50.0
    
    if has_ground_truth:
        total_late = np.sum(y_true)
        cost_without_ai = total_late * COST_PENALTY
        
        log.info("1. WITHOUT AI (Reactive):")
        log.info(f"   Total Late Orders     : {total_late:,}")
        log.info(f"   Total Penalty Cost    : ${cost_without_ai:,.2f}")
        log.info("-" * 50)
        
        
        recs = df_result['recommendation'].values
        intervened_mask = (recs == 'INTERVENE')
        ignored_mask = (recs == 'IGNORE')
        
     
        total_intervention_cost = np.sum(intervened_mask) * COST_INTERVENTION
    
        missed_late_count = np.sum((y_true == 1) & ignored_mask)
        missed_late_cost = missed_late_count * COST_PENALTY
        
        total_cost_with_ai = total_intervention_cost + missed_late_cost
        
        net_savings = cost_without_ai - total_cost_with_ai
        roi_pct = (net_savings / cost_without_ai) * 100 if cost_without_ai > 0 else 0
        
        log.info("2. WITH AI (Proactive):")
        log.info(f"   Interventions Triggered: {np.sum(intervened_mask):,} (Cost: ${total_intervention_cost:,.2f})")
        log.info(f"   Missed Late Orders     : {missed_late_count:,} (Cost: ${missed_late_cost:,.2f})")
        log.info(f"   Total Cost with AI     : ${total_cost_with_ai:,.2f}")
        log.info("=" * 50)
        log.info(f" NET SAVINGS           : ${net_savings:,.2f}")
        log.info(f" ROI / EFFICIENCY UP   : {roi_pct:.2f}%")
        log.info("=" * 50)
        
    else:
        risk_scores = df_result['risk_score'].values
        proj_loss_no_ai = np.sum(risk_scores * COST_PENALTY)
        
        recs = df_result['recommendation'].values
        intervened_mask = (recs == 'INTERVENE')
        ignored_mask = (recs == 'IGNORE')
        
        cost_intervention = np.sum(intervened_mask) * COST_INTERVENTION
        residual_risk_cost = np.sum(risk_scores[ignored_mask] * COST_PENALTY)
        
        proj_cost_ai = cost_intervention + residual_risk_cost
        proj_savings = proj_loss_no_ai - proj_cost_ai
        
        log.info("REPORT (PROJECTED / FUTURE DATA):")
        log.info(f"   Est. Loss (No AI)     : ${proj_loss_no_ai:,.2f}")
        log.info(f"   Est. Cost (With AI)   : ${proj_cost_ai:,.2f}")
        log.info(f"   Projected Savings     : ${proj_savings:,.2f}")
        log.info("=" * 50)


def run(cfg, input_path, model_path, output_path=None):
    log.info(f"Starting Inference on: {input_path}")
    
    if input_path.endswith('.parquet'):
        df_raw = pd.read_parquet(input_path)
    else:
        df_raw = pd.read_csv(input_path, encoding='latin-1')
        
    X_new = apply_fe_for_inference(df_raw.copy(), cfg)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    log.info(f"Loading model from {model_path}...")
    artifact = joblib.load(model_path)
    
    y_proba = None
    threshold = 0.5
    model_type = "Unknown"
    
    if 'weights' in artifact: # Ensemble
        model_type = f"Ensemble ({artifact.get('algorithm', 'Custom')})"
        threshold = artifact['threshold']
        weights = artifact['weights']
        models = artifact['models']
        
        log.info(f"Predicting with Ensemble (Thresh={threshold:.4f})...")
        weighted_sum = 0
        total_weight = 0
    
        name_map = {
            'RandomForest': 'w_rf',
            'XGBoost': 'w_xgb',
            'CatBoost': 'w_cat',
            'LogisticReg': 'w_lr'
        }
        
        for name, model in models.items():
            # Cari bobot dengan key yang benar
            key_weight = name_map.get(name, f"w_{name}")
            w = weights.get(key_weight, 0)
            
            if w > 0:
                X_aligned = align_features(X_new.copy(), model)
                try:
                    p = model.predict_proba(X_aligned)[:, 1]
                    weighted_sum += w * p
                    total_weight += w
                except Exception as e:
                    log.error(f"Error predicting with {name}: {e}")

        if total_weight == 0:
            y_proba = np.zeros(len(X_new))
        else:
            y_proba = weighted_sum / total_weight

    elif 'model' in artifact: # Single Model
        model_type = artifact.get('type', 'SingleModel')
        threshold = artifact.get('threshold', 0.5)
        model = artifact['model']
        
        log.info(f"Predicting with {model_type} (Thresh={threshold:.4f})...")
        X_aligned = align_features(X_new.copy(), model)
        y_proba = model.predict_proba(X_aligned)[:, 1]
            
    if np.isscalar(y_proba):
        y_proba = np.full(len(X_new), y_proba)
        
    y_pred = (y_proba >= threshold).astype(int)
    cost_intervention = 15.0
    cost_penalty = 50.0
    expected_loss = y_proba * cost_penalty
    should_intervene = expected_loss > cost_intervention
    
    # Result DataFrame
    result_df = df_raw.copy()
    
    result_df['risk_score'] = np.round(y_proba, 4)
    result_df['prediction'] = np.where(y_pred == 1, 'LATE', 'ON-TIME')
    result_df['recommendation'] = np.where(should_intervene, 'INTERVENE', 'IGNORE')
    result_df['potential_saving'] = np.where(should_intervene, expected_loss - cost_intervention, 0)
    result_df = result_df.sort_values(by='risk_score', ascending=False)
    
    # 6. Save
    if output_path is None:
        output_path = os.path.join(cfg['paths']['raw_dir'], "predictions.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result_df.to_csv(output_path, index=False)
    
    generate_business_report(result_df)
    
    log.info(f"Detailed row-level predictions saved to: {output_path}")
    
    return result_df