import pandas as pd
import numpy as np
import os
import joblib
import optuna
import warnings
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from optuna.samplers import CmaEsSampler, NSGAIISampler
from .utils import get_logger

# Suppress Warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = get_logger("training")

def optimize_with_progress(study, objective, n_trials, desc="Tuning"):
    """Wrapper untuk progress bar Optuna"""
    with tqdm(total=n_trials, desc=desc, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        def callback(study, trial):
            if study.best_trial:
                best_val = study.best_value
                pbar.set_description(f"{desc} (Best: {best_val:.4f})")
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

def find_optimal_threshold(model, X_val, y_val):
    try:
        y_proba = model.predict_proba(X_val)[:, 1]
    except:
        return 0.5
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def run_single_model_flow(cfg, X, y):
    print("\n" + "="*50)
    print(" MODE 1: SINGLE MODEL OPTIMIZATION")
    print("="*50)
    
    train_cfg = cfg.get('training', {})
    seed = train_cfg.get('random_seed', 42)
    test_size = train_cfg.get('test_size', 0.2)
    n_jobs = train_cfg.get('n_jobs', 1)
    metric = train_cfg.get('metric', 'f1_macro')
    n_trials = train_cfg.get('n_trials', 10)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    # Candidates
    candidates = {
        'LogisticReg': LogisticRegression(solver='liblinear', class_weight='balanced', random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, class_weight='balanced', random_state=seed),
        'XGBoost': XGBClassifier(n_estimators=100, n_jobs=n_jobs, eval_metric='logloss', random_state=seed),
        'CatBoost': CatBoostClassifier(iterations=100, verbose=0, allow_writing_files=False, random_state=seed)
    }
    
    # Compare
    print(f"\n[1/4] Comparing Base Models (CV metric={metric})...")
    best_score = -1
    best_name = ""
    
    for name, model in tqdm(candidates.items(), desc="Comparing"):
        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring=metric)
            mean_score = scores.mean()
            # Log bersih sesuai request
            log.info(f"{name:<15} : {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
        except Exception as e:
            log.warning(f"Failed to eval {name}: {e}")
            
    if best_name == "": return
    print(f"\n>>> WINNER: {best_name.upper()} (Score: {best_score:.4f})")
    
    # Tune
    print(f"\n[2/4] Tuning {best_name} ({n_trials} trials)...")
    
    def objective(trial):
        if best_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'n_jobs': n_jobs, 'class_weight': 'balanced', 'random_state': seed
            }
            clf = RandomForestClassifier(**params)
        elif best_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_jobs': n_jobs, 'eval_metric': 'logloss', 'random_state': seed
            }
            clf = XGBClassifier(**params)
        elif best_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 4, 10),
                'verbose': 0, 'allow_writing_files': False, 'random_state': seed
            }
            clf = CatBoostClassifier(**params)
        else:
            params = {'C': trial.suggest_float('C', 0.1, 10.0, log=True)}
            clf = LogisticRegression(**params, solver='liblinear', class_weight='balanced', random_state=seed)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        if metric == 'f1_macro': return f1_score(y_test, preds, average='macro')
        elif metric == 'accuracy': return accuracy_score(y_test, preds)
        else: return accuracy_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    optimize_with_progress(study, objective, n_trials, desc=f"Tuning {best_name}")
    
    print(f"   Best Params: {study.best_params}")
    
    # Final Fit
    print(f"\n[3/4] Refitting {best_name} & Optimizing Threshold...")
    best_params = study.best_params
    
    if best_name == 'RandomForest':
        final_model = RandomForestClassifier(**best_params, n_jobs=-1, class_weight='balanced', random_state=seed)
    elif best_name == 'XGBoost':
        final_model = XGBClassifier(**best_params, n_jobs=-1, eval_metric='logloss', random_state=seed)
    elif best_name == 'CatBoost':
        final_model = CatBoostClassifier(**best_params, verbose=0, allow_writing_files=False, random_state=seed)
    else:
        final_model = LogisticRegression(**best_params, solver='liblinear', class_weight='balanced', random_state=seed)
        
    final_model.fit(X_train, y_train)
    
    optimal_thresh = find_optimal_threshold(final_model, X_test, y_test)
    print(f"   Optimal Threshold Found: {optimal_thresh:.4f}")
    
    # Eval
    y_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred_opt = (y_proba >= optimal_thresh).astype(int)
    
    print("\n" + "="*50)
    print(" FINAL REPORT (SINGLE MODEL)")
    print("="*50)
    print(classification_report(y_test, y_pred_opt))
    
    # Save
    save_dir = cfg['paths']['model_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "champion_single_model.pkl")
    joblib.dump({'model': final_model, 'threshold': optimal_thresh, 'type': best_name}, save_path)
    print(f"\nModel saved to {save_path}")

def run_ensemble_flow(cfg, X, y):
    print("\n" + "="*50)
    print(" MODE 2: ENSEMBLE OPTIMIZATION")
    print("="*50)
    
    train_cfg = cfg.get('training', {})
    seed = train_cfg.get('random_seed', 42)
    n_jobs = train_cfg.get('n_jobs', 1)
    n_trials = train_cfg.get('n_trials', 10)

    # split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=seed)
    
    # Train Squad
    squad = {
        'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=seed),
        'XGBoost': XGBClassifier(n_estimators=100, n_jobs=n_jobs, eval_metric='logloss', random_state=seed),
        'CatBoost': CatBoostClassifier(iterations=100, verbose=0, allow_writing_files=False, random_state=seed),
        'LogisticReg': LogisticRegression(solver='liblinear', random_state=seed)
    }
    
    print("\n[1/3] Training Base Models (SQUAD)...")
    p_val, p_test = {}, {}
    
    for name, model in tqdm(squad.items(), desc="Fitting Models"):
        model.fit(X_train, y_train)
        
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        p_val[name] = val_proba
        p_test[name] = model.predict_proba(X_test)[:, 1]
        
        log.info(f"{name:<15} : {val_f1:.4f} (Val F1)")
        
    # Optimize Weights
    print(f"\n[2/3] Optimizing Ensemble Weights ({n_trials} trials)...")
    
    def objective_ensemble(trial):
        w_rf = trial.suggest_float('w_rf', 0, 1)
        w_xgb = trial.suggest_float('w_xgb', 0, 1)
        w_cat = trial.suggest_float('w_cat', 0, 1)
        w_lr = trial.suggest_float('w_lr', 0, 1)
        thresh = trial.suggest_float('threshold', 0.3, 0.7)
        
        weighted_prob = (
            w_rf * p_val['RandomForest'] + w_xgb * p_val['XGBoost'] + 
            w_cat * p_val['CatBoost'] + w_lr * p_val['LogisticReg']
        ) / (w_rf + w_xgb + w_cat + w_lr + 1e-10)
        
        pred = (weighted_prob >= thresh).astype(int)
        return f1_score(y_val, pred, average='macro')

    strategies = {
        'HillClimbing': optuna.create_study(direction='maximize'),
        'GeneticAlgo': optuna.create_study(direction='maximize', sampler=CmaEsSampler(seed=seed))
    }
    
    best_algo_name = ""
    best_algo_score = 0
    best_algo_params = {}
    
    for name, study in strategies.items():
        print(f"\n>>> Running Strategy: {name}")
        optimize_with_progress(study, objective_ensemble, n_trials, desc=name)
        
        if study.best_value > best_algo_score:
            best_algo_score = study.best_value
            best_algo_name = name
            best_algo_params = study.best_params
            
            # Save Artifact
            save_dir = cfg['paths']['model_dir']
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"ensemble_{name}.pkl")
            
            joblib.dump({
                'models': squad,
                'weights': best_algo_params, 
                'threshold': best_algo_params['threshold'],
                'algorithm': name
            }, save_path)
            
    print(f"\n>>> Ensemble Finished. Champion: {best_algo_name}")
    
    print("\n" + "="*50)
    print(f" FINAL REPORT (ENSEMBLE: {best_algo_name})")
    print("="*50)
    
    w_rf = best_algo_params.get('w_rf', 0)
    w_xgb = best_algo_params.get('w_xgb', 0)
    w_cat = best_algo_params.get('w_cat', 0)
    w_lr = best_algo_params.get('w_lr', 0)
    final_thresh = best_algo_params.get('threshold', 0.5)
    
    final_weighted_prob = (
        w_rf * p_test['RandomForest'] + 
        w_xgb * p_test['XGBoost'] + 
        w_cat * p_test['CatBoost'] + 
        w_lr * p_test['LogisticReg']
    ) / (w_rf + w_xgb + w_cat + w_lr + 1e-10)
    
    final_preds = (final_weighted_prob >= final_thresh).astype(int)
    
    print(classification_report(y_test, final_preds))
    print(f"Optimal Threshold: {final_thresh:.4f}")
    print("="*50)