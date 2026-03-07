#!/usr/bin/env python3
"""
Final Optimized 220kV Prediction
==========================================
- Separate models for Lateral vs Longitudinal profiles
- Optuna HPO for XGBoost + LightGBM calibration
- Rich feature engineering with physics transfer features
- Profile-aware stacking ensemble
"""

import json, pickle, warnings, os, time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

MU0 = 4e-7 * np.pi
SQRT3 = np.sqrt(3)
XGB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ──────────────────────────────────────────────────────
# Load synthetic 220kV models for transfer features
# ──────────────────────────────────────────────────────

class EMFNet(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.15):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


LOCATION_MAP = {
    'Ibri-Ibri City': dict(feeder='Ibri-Ibri City', substation='Ibri 220kV', config='horizontal'),
    'Mahda-Oha':      dict(feeder='Mahda-Oha',      substation='Mahda 220kV', config='horizontal'),
    'Barka-Rustaq':   dict(feeder='Barka-Rustaq',    substation='Barka 220kV', config='delta'),
}


def safe_le(enc, val):
    return enc.transform([val])[0] if val in enc.classes_ else 0


def load_220kv_models():
    feat_cols = json.load(open('outputs/feature_columns_220.json'))
    scaler = pickle.load(open('outputs/scaler_220.pkl', 'rb'))
    le = pickle.load(open('outputs/label_encoders_220.pkl', 'rb'))
    models = {}
    for t_label, t_key in [('e', 'E'), ('h', 'H')]:
        xm = pickle.load(open(f'outputs/xgb_220_{t_label}.pkl', 'rb'))
        lm = pickle.load(open(f'outputs/lgb_220_{t_label}.pkl', 'rb'))
        mm = pickle.load(open(f'outputs/meta_220_{t_label}.pkl', 'rb'))
        bp = json.load(open(f'outputs/best_params_220_{t_label}.json'))
        nn_bp = bp['nn']
        nm = EMFNet(len(feat_cols), [nn_bp['hd']]*nn_bp['nl'], nn_bp['dp']).to(device)
        nm.load_state_dict(torch.load(f'outputs/nn_220_{t_label}.pt', map_location=device, weights_only=True))
        nm.eval()
        models[t_key] = (xm, lm, nm, mm)
    return models, feat_cols, scaler, le


def get_synth_preds(df_sub, models, ft, feat_cols, scaler, le):
    n = len(df_sub)
    feat = pd.DataFrame(index=range(n))
    lf = 0.85; I = 800 * lf
    feat['voltage_kV'] = 220; feat['current_A'] = I
    feat['distance_m'] = df_sub['Distance'].values
    feat['measurement_height_m'] = 1.0; feat['height_m'] = 30
    feat['phase_spacing_m'] = 7.0; feat['span_length_m'] = 350
    feat['conductor_radius_cm'] = 7.75; feat['conductor_diameter_m'] = 0.030
    feat['bundle_count'] = 2; feat['sag_m'] = 5.5; feat['ground_clearance_m'] = 24.5
    feat['elevation_m'] = 300; feat['wind_speed_ms'] = 2.0
    feat['soil_resistivity'] = 100; feat['solar_irradiance'] = 800
    feat['load_factor'] = lf; feat['power_factor'] = 0.90
    feat['active_power_MW'] = SQRT3 * 220 * I * 0.90 / 1000
    feat['frequency_Hz'] = 50; feat['phase_angle_deg'] = 0.0
    feat['conductor_temp_C'] = 55.0
    feat['temperature_C'] = df_sub['Temperature'].values
    feat['humidity_pct'] = df_sub['Humidity'].values
    feat['circuit_id'] = df_sub['Circuit_ID'].values
    
    cfgs, fdrs, sbs, cts, pfs = [], [], [], [], []
    for _, row in df_sub.iterrows():
        m = LOCATION_MAP.get(row['Location'], LOCATION_MAP['Ibri-Ibri City'])
        cfgs.append(m['config']); fdrs.append(m['feeder']); sbs.append(m['substation'])
        cts.append('Double-Circuit 3ph AC' if m['config']=='double_circuit' else '3-Phase AC')
        pfs.append(row['Profile_Type'])
    
    feat['configuration'] = [safe_le(le['configuration'], c) for c in cfgs]
    feat['feeder'] = [safe_le(le['feeder'], f) for f in fdrs]
    feat['substation'] = [safe_le(le['substation'], s) for s in sbs]
    feat['profile_type'] = [safe_le(le['profile_type'], p) for p in pfs]
    feat['weather'] = [safe_le(le['weather'], 'Clear')] * n
    feat['time_of_day'] = [safe_le(le['time_of_day'], 'Morning')] * n
    feat['season'] = [safe_le(le['season'], 'Summer' if df_sub['Temperature'].values[i] > 30 else 'Spring/Autumn') for i in range(n)]
    feat['circuit_type'] = [safe_le(le['circuit_type'], c) for c in cts]
    
    feat['voltage_current_product'] = feat['voltage_kV'] * feat['current_A']
    feat['distance_to_height_ratio'] = feat['distance_m'] / feat['height_m']
    feat['sag_to_span_ratio'] = feat['sag_m'] / feat['span_length_m']
    feat['log_distance'] = np.log1p(feat['distance_m'])
    feat['sqrt_distance'] = np.sqrt(feat['distance_m'])
    feat['inv_distance'] = 1.0 / (feat['distance_m'] + 1)
    feat['inv_distance_sq'] = 1.0 / (feat['distance_m']**2 + 1)
    feat['temp_humidity'] = feat['temperature_C'] * feat['humidity_pct']
    feat['power_density'] = feat['active_power_MW'] / (feat['distance_m'] * feat['height_m'] + 1)
    feat['height_spacing_ratio'] = feat['height_m'] / feat['phase_spacing_m']
    
    X = feat[feat_cols].values.astype(np.float32)
    X_s = scaler.transform(X)
    xm, lm, nm, mm = models[ft]
    p1 = xm.predict(X_s); p2 = lm.predict(X_s)
    with torch.no_grad(): p3 = nm(torch.from_numpy(X_s).to(device)).cpu().numpy()
    p4 = mm.predict(np.column_stack([p1, p2, p3]))
    return p1, p2, p3, p4


# ──────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────

def build_features(df_sub, synth_preds=None, for_profile=None):
    """
    Rich features, optionally specialized for a given profile type.
    """
    d = df_sub['Distance'].values.astype(float)
    T = df_sub['Temperature'].values.astype(float)
    H = df_sub['Humidity'].values.astype(float)
    cid = df_sub['Circuit_ID'].values.astype(float)
    
    f = {}
    
    # Distance transforms  
    f['d'] = d
    f['d_log'] = np.log1p(d)
    f['d_sqrt'] = np.sqrt(d)
    f['d_inv'] = 1.0 / (d + 1.0)
    f['d_inv2'] = 1.0 / (d**2 + 1.0)
    f['d_inv3'] = 1.0 / (d**3 + 1.0)
    f['d_pow15'] = d**1.5
    f['d_pow2'] = d**2
    f['d_cbrt'] = np.cbrt(d)
    
    # Profile-specific decay models
    if for_profile == 'Lateral':
        # Lateral: steep decay ~ 1/r^2 to 1/r^3
        f['d_exp5'] = np.exp(-d / 5.0)
        f['d_exp10'] = np.exp(-d / 10.0)
        f['d_exp20'] = np.exp(-d / 20.0)
        f['d_exp30'] = np.exp(-d / 30.0)
        f['d_gauss10'] = np.exp(-d**2 / (2 * 10**2))
        f['d_gauss20'] = np.exp(-d**2 / (2 * 20**2))
    elif for_profile == 'Longitudinal':
        # Longitudinal: slower decay, possible span modulation
        f['d_exp50'] = np.exp(-d / 50.0)
        f['d_exp100'] = np.exp(-d / 100.0)
        f['d_exp200'] = np.exp(-d / 200.0)
        f['d_cos350'] = np.cos(2 * np.pi * d / 350.0)   # span modulation
        f['d_cos700'] = np.cos(2 * np.pi * d / 700.0)
        f['d_span_mod'] = 0.5 + 0.5 * np.cos(2 * np.pi * d / 350.0)
    else:
        # General: include both
        f['d_exp10'] = np.exp(-d / 10.0)
        f['d_exp30'] = np.exp(-d / 30.0)
        f['d_exp100'] = np.exp(-d / 100.0)
    
    # Environment
    f['temp'] = T
    f['humid'] = H
    f['temp_humid'] = T * H
    f['temp_norm'] = (T - 30.0) / 5.0
    f['humid_norm'] = (H - 30.0) / 10.0
    
    # Circuit
    f['cid'] = cid
    f['cid_is2'] = (cid == 2).astype(float)
    
    # Location one-hot  
    for loc in ['Ibri-Ibri City', 'Mahda-Oha', 'Barka-Rustaq']:
        f[f'loc_{loc[:4]}'] = (df_sub['Location'].values == loc).astype(float)
    
    # Profile one-hot (only in combined models)
    if for_profile is None:
        f['is_lat'] = (df_sub['Profile_Type'].values == 'Lateral').astype(float)
        f['is_lon'] = (df_sub['Profile_Type'].values == 'Longitudinal').astype(float)
    
    # Location × Distance
    for loc in ['Ibri-Ibri City', 'Mahda-Oha', 'Barka-Rustaq']:
        lm = (df_sub['Location'].values == loc).astype(float)
        f[f'loc_{loc[:4]}_d'] = lm * d
        f[f'loc_{loc[:4]}_dlog'] = lm * np.log1p(d)
        f[f'loc_{loc[:4]}_dinv'] = lm / (d + 1.0)
    
    # Profile × Distance (only combined)
    if for_profile is None:
        is_lat = (df_sub['Profile_Type'].values == 'Lateral').astype(float)
        is_lon = 1.0 - is_lat
        f['lat_d'] = is_lat * d
        f['lon_d'] = is_lon * d
        f['lat_dinv'] = is_lat / (d + 1.0)
        f['lon_dinv'] = is_lon / (d + 1.0)
    
    # Circuit × Distance
    f['cid_d'] = cid * d
    f['cid_dinv'] = cid / (d + 1.0)
    
    # Location × Circuit × Distance (3-way)
    for loc in ['Ibri-Ibri City', 'Mahda-Oha', 'Barka-Rustaq']:
        lm = (df_sub['Location'].values == loc).astype(float)
        f[f'loc_{loc[:4]}_cid_d'] = lm * cid * d
        f[f'loc_{loc[:4]}_cid_dinv'] = lm * cid / (d + 1.0)
    
    # Dist × Environment
    f['d_temp'] = d * T
    f['d_humid'] = d * H
    
    # Transfer features from synthetic model
    if synth_preds is not None:
        p1, p2, p3, p4 = synth_preds
        f['syn_xgb'] = p1
        f['syn_lgb'] = p2
        f['syn_nn'] = p3
        f['syn_ens'] = p4
        f['syn_mean'] = (p1 + p2 + p3) / 3.0
        f['syn_log'] = np.log1p(np.abs(p4))
        f['syn_sqrt'] = np.sqrt(np.abs(p4))
        f['syn_spread'] = np.abs(p1 - p2)
        f['syn_ens_d'] = p4 * d
        f['syn_ens_dinv'] = p4 / (d + 1.0)
    
    return pd.DataFrame(f)


# ──────────────────────────────────────────────────────
# Optuna HPO for XGBoost calibration
# ──────────────────────────────────────────────────────

def optuna_xgb(X, y, cv, n_trials=50):
    """Tune XGBoost on real data using CV R2."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 800),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 100, log=True),
            'tree_method': 'hist',
            'device': XGB_DEVICE,
            'random_state': 42,
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def optuna_lgb(X, y, cv, n_trials=50):
    """Tune LightGBM on real data using CV R2."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 800),
            'num_leaves': trial.suggest_int('num_leaves', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 100, log=True),
            'device_type': 'cpu',
            'verbose': -1,
            'random_state': 42,
        }
        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def optuna_et(X, y, cv, n_trials=30):
    """Tune ExtraTrees on real data."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42,
            'n_jobs': -1,
        }
        model = ExtraTreesRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def optuna_ridge(X, y, cv, n_trials=30):
    """Tune Ridge alpha."""
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 1000, log=True)
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


# ──────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "=" * 72)
    print("  FINAL OPTIMIZED 220kV PREDICTION")
    print("  Profile-Specific Models + Optuna HPO + Rich Features")
    print("=" * 72)
    
    df = pd.read_csv('standardized_data.csv')
    print(f"\n  Real data: {len(df)} rows")
    
    # Load synthetic models
    print("  Loading 220kV synthetic models ...")
    models_220, feat_cols, scaler_220, le_220 = load_220kv_models()
    print("  -> Loaded")
    
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_outputs = []
    final_summary = {}
    
    for ft_name in ['E', 'H']:
        print(f"\n{'='*72}")
        print(f"  {ft_name}-FIELD")
        print(f"{'='*72}")
        
        df_ft = df[df['Field_Type'] == ft_name].copy().reset_index(drop=True)
        if len(df_ft) == 0: continue
        
        actual = df_ft['Field_Value'].values
        
        # Synthetic transfer predictions (full dataset)
        sp_all = get_synth_preds(df_ft, models_220, ft_name, feat_cols, scaler_220, le_220)
        
        # ══════════════════════════════════════════════════════
        # Strategy A: Combined model (all profiles together)
        # ══════════════════════════════════════════════════════
        print(f"\n  ── Strategy A: Combined model ──")
        
        X_comb_df = build_features(df_ft, sp_all, for_profile=None)
        X_comb = X_comb_df.values.astype(np.float64)
        
        # Quick scan of default models
        quick_models = {
            'xgb': xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, 
                                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, 
                                      reg_lambda=2.0, tree_method='hist', device=XGB_DEVICE, random_state=42),
            'et': ExtraTreesRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, 
                                       max_features=0.7, random_state=42, n_jobs=-1),
            'ridge': Ridge(alpha=1.0),
            'lgb': lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                                       reg_lambda=2.0, device_type='cpu', verbose=-1, random_state=42),
        }
        
        print(f"    Quick scan ({len(X_comb_df.columns)} features):")
        for name, model in quick_models.items():
            pred = cross_val_predict(model, X_comb, actual, cv=cv5)
            pred = np.maximum(pred, 0)
            r2 = r2_score(actual, pred)
            print(f"      {name:10s}  R2={r2:.4f}")
        
        # Optuna HPO for combined
        print(f"\n    Optuna XGB (40 trials) ...")
        bp_xgb, r2_xgb = optuna_xgb(X_comb, actual, cv5, n_trials=40)
        print(f"      best R2={r2_xgb:.4f}")
        
        print(f"    Optuna LGB (40 trials) ...")
        bp_lgb, r2_lgb = optuna_lgb(X_comb, actual, cv5, n_trials=40)
        print(f"      best R2={r2_lgb:.4f}")
        
        print(f"    Optuna ET  (30 trials) ...")
        bp_et, r2_et = optuna_et(X_comb, actual, cv5, n_trials=30)
        print(f"      best R2={r2_et:.4f}")
        
        print(f"    Optuna Ridge (20 trials) ...")
        bp_ridge, r2_ridge = optuna_ridge(X_comb, actual, cv5, n_trials=20)
        print(f"      best R2={r2_ridge:.4f}")
        
        # Get best tuned predictions
        tuned_models_comb = {
            'xgb_tuned': xgb.XGBRegressor(**{**bp_xgb, 'tree_method': 'hist', 'device': XGB_DEVICE, 'random_state': 42}),
            'lgb_tuned': lgb.LGBMRegressor(**{**bp_lgb, 'device_type': 'cpu', 'verbose': -1, 'random_state': 42}),
            'et_tuned': ExtraTreesRegressor(**{**bp_et, 'random_state': 42, 'n_jobs': -1}),
            'ridge_tuned': Ridge(alpha=bp_ridge['alpha']),
        }
        
        comb_preds = {}
        print(f"\n    Tuned model results:")
        for name, model in tuned_models_comb.items():
            pred = cross_val_predict(model, X_comb, actual, cv=cv5)
            pred = np.maximum(pred, 0)
            comb_preds[name] = pred
            r2 = r2_score(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            print(f"      {name:15s}  R2={r2:.4f}  RMSE={rmse:.4f}")
        
        # Stack combined
        stack_comb = np.column_stack(list(comb_preds.values()))
        stack_pred_comb = cross_val_predict(Ridge(alpha=0.5), stack_comb, actual, cv=cv5)
        stack_pred_comb = np.maximum(stack_pred_comb, 0)
        r2_stack = r2_score(actual, stack_pred_comb)
        print(f"      {'stacked':15s}  R2={r2_stack:.4f}  RMSE={np.sqrt(mean_squared_error(actual, stack_pred_comb)):.4f}")
        comb_preds['stacked'] = stack_pred_comb
        
        best_comb_name = max(comb_preds, key=lambda k: r2_score(actual, comb_preds[k]))
        best_comb_pred = comb_preds[best_comb_name]
        best_comb_r2 = r2_score(actual, best_comb_pred)
        print(f"    >> Best combined: {best_comb_name}  R2={best_comb_r2:.4f}")
        
        # ══════════════════════════════════════════════════════
        # Strategy B: Profile-specific models
        # ══════════════════════════════════════════════════════
        print(f"\n  ── Strategy B: Profile-specific models ──")
        
        profile_preds = np.zeros(len(df_ft))
        
        for profile in ['Lateral', 'Longitudinal']:
            mask = df_ft['Profile_Type'] == profile
            if mask.sum() < 5: continue
            
            df_prof = df_ft[mask].reset_index(drop=True)
            actual_prof = df_prof['Field_Value'].values
            
            # Get synthetic predictions for this subset
            sp_prof = get_synth_preds(df_prof, models_220, ft_name, feat_cols, scaler_220, le_220)
            
            X_prof_df = build_features(df_prof, sp_prof, for_profile=profile)
            X_prof = X_prof_df.values.astype(np.float64)
            
            # Smaller CV for smaller subsets
            n_splits = min(5, max(3, mask.sum() // 10))
            cv_prof = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            print(f"\n    {profile} (n={mask.sum()}, {X_prof.shape[1]} features, {n_splits}-fold):")
            
            # Optuna HPO per profile
            print(f"      Optuna XGB ({30} trials) ...")
            bp_x, r2_x = optuna_xgb(X_prof, actual_prof, cv_prof, n_trials=30)
            print(f"        best R2={r2_x:.4f}")
            
            print(f"      Optuna LGB ({30} trials) ...")
            bp_l, r2_l = optuna_lgb(X_prof, actual_prof, cv_prof, n_trials=30)
            print(f"        best R2={r2_l:.4f}")
            
            print(f"      Optuna ET  ({20} trials) ...")
            bp_e, r2_e = optuna_et(X_prof, actual_prof, cv_prof, n_trials=20)
            print(f"        best R2={r2_e:.4f}")
            
            print(f"      Optuna Ridge ({15} trials) ...")
            bp_r, r2_r = optuna_ridge(X_prof, actual_prof, cv_prof, n_trials=15)
            print(f"        best R2={r2_r:.4f}")
            
            # Build tuned models
            prof_models = {
                'xgb': xgb.XGBRegressor(**{**bp_x, 'tree_method': 'hist', 'device': XGB_DEVICE, 'random_state': 42}),
                'lgb': lgb.LGBMRegressor(**{**bp_l, 'device_type': 'cpu', 'verbose': -1, 'random_state': 42}),
                'et': ExtraTreesRegressor(**{**bp_e, 'random_state': 42, 'n_jobs': -1}),
                'ridge': Ridge(alpha=bp_r['alpha']),
            }
            
            prof_preds_dict = {}
            for name, model in prof_models.items():
                pred = cross_val_predict(model, X_prof, actual_prof, cv=cv_prof)
                pred = np.maximum(pred, 0)
                prof_preds_dict[name] = pred
                r2 = r2_score(actual_prof, pred)
                print(f"      {name:10s}  R2={r2:.4f}")
            
            # Stack profile-specific
            stack_prof = np.column_stack(list(prof_preds_dict.values()))
            stack_pred_prof = cross_val_predict(Ridge(alpha=0.5), stack_prof, actual_prof, cv=cv_prof)
            stack_pred_prof = np.maximum(stack_pred_prof, 0)
            r2_sp = r2_score(actual_prof, stack_pred_prof)
            print(f"      {'stacked':10s}  R2={r2_sp:.4f}")
            prof_preds_dict['stacked'] = stack_pred_prof
            
            best_prof_name = max(prof_preds_dict, key=lambda k: r2_score(actual_prof, prof_preds_dict[k]))
            best_prof_pred = prof_preds_dict[best_prof_name]
            best_prof_r2 = r2_score(actual_prof, best_prof_pred)
            print(f"      >> Best {profile}: {best_prof_name}  R2={best_prof_r2:.4f}")
            
            profile_preds[mask.values] = best_prof_pred
        
        # Overall profile-specific R2
        r2_profile = r2_score(actual, profile_preds)
        rmse_profile = np.sqrt(mean_squared_error(actual, profile_preds))
        print(f"\n    >> Overall profile-specific: R2={r2_profile:.4f}  RMSE={rmse_profile:.4f}")
        
        # ══════════════════════════════════════════════════════
        # Strategy C: Blend combined + profile-specific
        # ══════════════════════════════════════════════════════
        print(f"\n  ── Strategy C: Blending ──")
        
        blend_X = np.column_stack([best_comb_pred, profile_preds])
        blend_pred = cross_val_predict(Ridge(alpha=0.5), blend_X, actual, cv=cv5)
        blend_pred = np.maximum(blend_pred, 0)
        r2_blend = r2_score(actual, blend_pred)
        print(f"    Blend (combined + profile-specific): R2={r2_blend:.4f}")
        
        # Weighted average exploration
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            wavg = w * best_comb_pred + (1 - w) * profile_preds
            r2_w = r2_score(actual, wavg)
            print(f"    w_comb={w:.1f}: R2={r2_w:.4f}")
        
        # ══════════════════════════════════════════════════════
        # FINAL SELECTION
        # ══════════════════════════════════════════════════════
        candidates = {
            'combined_best': best_comb_pred,
            'profile_specific': profile_preds,
            'blend_ridge': blend_pred,
        }
        # Add weighted averages
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            candidates[f'wavg_{w:.1f}'] = w * best_comb_pred + (1 - w) * profile_preds
        
        print(f"\n  >>> FINAL SELECTION <<<")
        print(f"  {'Method':25s}  {'R2':>8s}  {'RMSE':>8s}  {'MAE':>8s}")
        best_final_r2 = -999
        best_final_pred = None
        best_final_name = ''
        
        for name, pred in candidates.items():
            r2 = r2_score(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            print(f"  {name:25s}  {r2:8.4f}  {rmse:8.4f}  {mae:8.4f}")
            if r2 > best_final_r2:
                best_final_r2 = r2; best_final_pred = pred; best_final_name = name
        
        print(f"\n  >>> {ft_name}-FIELD BEST: {best_final_name}  R²={best_final_r2:.4f}")
        final_summary[ft_name] = dict(method=best_final_name, r2=best_final_r2,
                                       rmse=float(np.sqrt(mean_squared_error(actual, best_final_pred))),
                                       mae=float(mean_absolute_error(actual, best_final_pred)))
        
        # Per-location + per-profile breakdown
        print(f"\n  Per-location:")
        for loc in df_ft['Location'].unique():
            m = df_ft['Location'] == loc
            if m.sum() < 2: continue
            r2_loc = r2_score(actual[m], best_final_pred[m])
            mae_loc = mean_absolute_error(actual[m], best_final_pred[m])
            print(f"    {loc:20s}  n={m.sum():3d}  R2={r2_loc:.4f}  MAE={mae_loc:.4f}")
        
        print(f"  Per-profile:")
        for pt in df_ft['Profile_Type'].unique():
            m = df_ft['Profile_Type'] == pt
            if m.sum() < 2: continue
            r2_pt = r2_score(actual[m], best_final_pred[m])
            print(f"    {pt:15s}  n={m.sum():3d}  R2={r2_pt:.4f}")
        
        # Save
        df_ft[f'Pred_{ft_name}'] = best_final_pred
        df_ft[f'Method_{ft_name}'] = best_final_name
        df_ft[f'Residual_{ft_name}'] = actual - best_final_pred
        all_outputs.append(df_ft)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        
        # 1: Actual vs Predicted (overall)
        ax = axes[0, 0]
        ax.scatter(actual, best_final_pred, alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
        lo, hi = min(actual.min(), best_final_pred.min()), max(actual.max(), best_final_pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.set_title(f'Overall R²={best_final_r2:.4f}')
        ax.grid(True, alpha=0.3)
        
        # 2: By Lateral
        ax = axes[0, 1]
        m_lat = df_ft['Profile_Type'] == 'Lateral'
        if m_lat.any():
            ax.scatter(actual[m_lat], best_final_pred[m_lat], alpha=0.6, s=30, c='orange')
            r2_lat = r2_score(actual[m_lat], best_final_pred[m_lat])
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
            ax.set_title(f'Lateral R²={r2_lat:.4f}')
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.grid(True, alpha=0.3)
        
        # 3: By Longitudinal
        ax = axes[0, 2]
        m_lon = df_ft['Profile_Type'] == 'Longitudinal'
        if m_lon.any():
            ax.scatter(actual[m_lon], best_final_pred[m_lon], alpha=0.6, s=30, c='blue')
            r2_lon = r2_score(actual[m_lon], best_final_pred[m_lon])
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
            ax.set_title(f'Longitudinal R²={r2_lon:.4f}')
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.grid(True, alpha=0.3)
        
        # 4: Distance profile by location
        ax = axes[1, 0]
        colors = {'Ibri-Ibri City': 'blue', 'Mahda-Oha': 'red', 'Barka-Rustaq': 'green'}
        for loc in df_ft['Location'].unique():
            m = df_ft['Location'] == loc
            c = colors.get(loc, 'gray')
            ax.scatter(df_ft.loc[m, 'Distance'], actual[m], marker='o', s=15, alpha=0.5, color=c, label=f'{loc} act')
            ax.scatter(df_ft.loc[m, 'Distance'], best_final_pred[m], marker='x', s=15, alpha=0.5, color=c, label=f'{loc} pred')
        ax.set_xlabel('Distance (m)'); ax.set_ylabel(ft_name)
        ax.set_title('Distance Profile')
        ax.legend(fontsize=5); ax.grid(True, alpha=0.3)
        
        # 5: Residuals
        ax = axes[1, 1]
        res = actual - best_final_pred
        ax.scatter(df_ft['Distance'], res, alpha=0.6, s=20, c='steelblue')
        ax.axhline(0, color='red', ls='--', lw=1.5)
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('Residual')
        ax.set_title(f'Residuals (std={res.std():.3f})')
        ax.grid(True, alpha=0.3)
        
        # 6: Residual histogram
        ax = axes[1, 2]
        ax.hist(res, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='red', ls='--', lw=1.5)
        ax.set_xlabel('Residual'); ax.set_ylabel('Count')
        ax.set_title('Residual Distribution')
        
        fig.suptitle(f'{ft_name}-Field: Final Optimized 220kV (R²={best_final_r2:.4f})', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'outputs/220kv_final_{ft_name.lower()}.png', dpi=150)
        plt.close(fig)
        print(f"\n  -> outputs/220kv_final_{ft_name.lower()}.png")
    
    # Save combined
    if all_outputs:
        out = pd.concat(all_outputs, ignore_index=True)
        out.to_csv('outputs/real_predictions_220kv.csv', index=False)
        print(f"\n  -> outputs/real_predictions_220kv.csv ({len(out)} rows)")
    
    # Final summary
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    for ft, info in final_summary.items():
        print(f"  {ft}-field: R²={info['r2']:.4f}  RMSE={info['rmse']:.4f}  MAE={info['mae']:.4f}  Method={info['method']}")
    
    json.dump(final_summary, open('outputs/final_220kv_summary.json', 'w'), indent=2)
    print("\n  Done!")


if __name__ == '__main__':
    run_pipeline()
