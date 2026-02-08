import pandas as pd
import numpy as np
import os
from .utils import get_logger

log = get_logger("feature_eng")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Menghitung jarak (km) menggunakan rumus Haversine."""
    R = 6371  # Radius bumi (km)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def run(cfg):
    log.info("Starting Feature Engineering Pipeline...")
    
    # Load Data
    raw_path = os.path.join(cfg['paths']['raw_dir'], cfg['data']['main_csv'])
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")
        
    log.info(f"Loading raw data: {raw_path}")
    df = pd.read_csv(raw_path, encoding='latin-1')

    # Create Target (is_late)
    target_col = cfg['schema']['target']
    if target_col not in df.columns:
        log.info("Creating target variable 'is_late' from 'Delivery Status'...")
        df[target_col] = np.where(df['Delivery Status'] == 'Late delivery', 1, 0)

    # Feature Construction
    log.info("Calculating Haversine Distance...")
    wh_lat = cfg['feature_engineering']['warehouse_lat']
    wh_lon = cfg['feature_engineering']['warehouse_lon']
    lat_col = cfg['feature_engineering']['lat_col']
    lon_col = cfg['feature_engineering']['lon_col']

    df['distance_km'] = haversine_distance(
        wh_lat, wh_lon, df[lat_col], df[lon_col]
    )

    meta_cols = ['Order Id', 'Order City', 'Order Country', 'Category Name', 
                 'Customer Segment', 'Product Name', target_col, 'distance_km']
    available_meta = [c for c in meta_cols if c in df.columns]
    
    meta_path = os.path.join(cfg['paths']['features_dir'], cfg['data']['metadata_parquet'])
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    df[available_meta].to_parquet(meta_path, index=False)
    log.info(f"Metadata saved to: {meta_path}")

    # Drop Leakage & Noise
    drop_cols = cfg['schema']['drop_cols']
    log.info(f"Dropping {len(drop_cols)} leakage/noise columns...")
    df_clean = df.drop(columns=drop_cols, errors='ignore')

    # Simple Encoding (Label Encode Objects)
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_clean[col] = df_clean[col].astype('category').cat.codes

    # Handle Missing Values (Fill 0)
    df_clean = df_clean.fillna(0)

    # Save Train Ready Data
    train_path = os.path.join(cfg['paths']['features_dir'], cfg['data']['train_ready_parquet'])
    df_clean.to_parquet(train_path, index=False)
    log.info(f"Train-ready data saved to: {train_path}. Shape: {df_clean.shape}")