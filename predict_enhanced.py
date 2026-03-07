#!/usr/bin/env python3
"""
Enhanced 220kV Prediction – Maximize calibrated R² on real data.
Uses the 220kV synthetic models as physics-informed transfer features,
combined with rich feature engineering and stacking ensemble.

Target: E R² > 0.80, H R² > 0.75 (up from 0.72 / 0.66)
"""

import json, pickle, warnings, os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import (KFold, RepeatedKFold,
                                     cross_val_predict, GroupKFold, 
                                     LeaveOneGroupOut)
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, StackingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

MU0 = 4e-7 * np.pi
SQRT3 = np.sqrt(3)

# ──────────────────────────────────────────────────────────────
# Load synthetic 220kV models for transfer features
# ──────────────────────────────────────────────────────────────

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


def load_220kv_models():
    """Load pre-trained 220kV synthetic models."""
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
        nm.load_state_dict(torch.load(f'outputs/nn_220_{t_label}.pt', 
                                       map_location=device, weights_only=True))
        nm.eval()
        models[t_key] = (xm, lm, nm, mm)
    
    return models, feat_cols, scaler, le


def safe_le(enc, val):
    if val in enc.classes_: return enc.transform([val])[0]
    return 0


# ──────────────────────────────────────────────────────────────
# Build 220kV features for synthetic model  
# ──────────────────────────────────────────────────────────────

LOCATION_MAP = {
    'Ibri-Ibri City': dict(feeder='Ibri-Ibri City', substation='Ibri 220kV', config='horizontal'),
    'Mahda-Oha':      dict(feeder='Mahda-Oha',      substation='Mahda 220kV', config='horizontal'),
    'Barka-Rustaq':   dict(feeder='Barka-Rustaq',    substation='Barka 220kV', config='delta'),
}


def build_synth_features(df_sub, feat_cols, le):
    """Build feature vector for synthetic model inference."""
    n = len(df_sub)
    feat = pd.DataFrame(index=range(n))
    
    lf = 0.85
    I = 800 * lf
    
    feat['voltage_kV']           = 220
    feat['current_A']            = I
    feat['distance_m']           = df_sub['Distance'].values
    feat['measurement_height_m'] = 1.0
    feat['height_m']             = 30
    feat['phase_spacing_m']      = 7.0
    feat['span_length_m']        = 350
    feat['conductor_radius_cm']  = 7.75
    feat['conductor_diameter_m'] = 0.030
    feat['bundle_count']         = 2
    feat['sag_m']                = 5.5
    feat['ground_clearance_m']   = 24.5
    feat['elevation_m']          = 300
    feat['wind_speed_ms']        = 2.0
    feat['soil_resistivity']     = 100
    feat['solar_irradiance']     = 800
    feat['load_factor']          = lf
    feat['power_factor']         = 0.90
    feat['active_power_MW']      = SQRT3 * 220 * I * 0.90 / 1000
    feat['frequency_Hz']         = 50
    feat['phase_angle_deg']      = 0.0
    feat['conductor_temp_C']     = 55.0
    feat['temperature_C']        = df_sub['Temperature'].values
    feat['humidity_pct']         = df_sub['Humidity'].values
    feat['circuit_id']           = df_sub['Circuit_ID'].values
    
    configs, feeders, subs, circuits, profiles = [], [], [], [], []
    for _, row in df_sub.iterrows():
        m = LOCATION_MAP.get(row['Location'], LOCATION_MAP['Ibri-Ibri City'])
        configs.append(m['config'])
        feeders.append(m['feeder'])
        subs.append(m['substation'])
        circuits.append('Double-Circuit 3ph AC' if m['config']=='double_circuit' else '3-Phase AC')
        profiles.append(row['Profile_Type'])
    
    feat['configuration'] = [safe_le(le['configuration'], c) for c in configs]
    feat['feeder']        = [safe_le(le['feeder'], f) for f in feeders]
    feat['substation']    = [safe_le(le['substation'], s) for s in subs]
    feat['profile_type']  = [safe_le(le['profile_type'], p) for p in profiles]
    feat['weather']       = [safe_le(le['weather'], 'Clear')] * n
    feat['time_of_day']   = [safe_le(le['time_of_day'], 'Morning')] * n
    feat['season']        = [safe_le(le['season'], 'Summer' if df_sub['Temperature'].values[i] > 30 else 'Spring/Autumn') for i in range(n)]
    feat['circuit_type']  = [safe_le(le['circuit_type'], c) for c in circuits]
    
    feat['voltage_current_product']  = feat['voltage_kV'] * feat['current_A']
    feat['distance_to_height_ratio'] = feat['distance_m'] / feat['height_m']
    feat['sag_to_span_ratio']        = feat['sag_m'] / feat['span_length_m']
    feat['log_distance']             = np.log1p(feat['distance_m'])
    feat['sqrt_distance']            = np.sqrt(feat['distance_m'])
    feat['inv_distance']             = 1.0 / (feat['distance_m'] + 1)
    feat['inv_distance_sq']          = 1.0 / (feat['distance_m']**2 + 1)
    feat['temp_humidity']            = feat['temperature_C'] * feat['humidity_pct']
    feat['power_density']            = feat['active_power_MW'] / (feat['distance_m'] * feat['height_m'] + 1)
    feat['height_spacing_ratio']     = feat['height_m'] / feat['phase_spacing_m']
    
    return feat[feat_cols].values.astype(np.float32)


def get_synth_predictions(df_sub, models, field_type, feat_cols, scaler, le):
    """Get physics-informed predictions from synthetic 220kV models."""
    X = build_synth_features(df_sub, feat_cols, le)
    X_s = scaler.transform(X)
    
    xm, lm, nm, mm = models[field_type]
    p_xgb = xm.predict(X_s)
    p_lgb = lm.predict(X_s)
    with torch.no_grad():
        p_nn = nm(torch.from_numpy(X_s).to(device)).cpu().numpy()
    p_ens = mm.predict(np.column_stack([p_xgb, p_lgb, p_nn]))
    
    return p_xgb, p_lgb, p_nn, p_ens


# ──────────────────────────────────────────────────────────────
# Rich Feature Engineering for Calibration
# ──────────────────────────────────────────────────────────────

def build_rich_features(df_sub, synth_preds=None):
    """
    Build maximally rich feature set for calibration on real data.
    
    Features:
    - Distance transforms (log, sqrt, inv, inv_sq, power)
    - Location one-hot
    - Profile_Type one-hot  
    - Circuit_ID
    - Temperature & Humidity + interactions
    - Location × Distance interactions
    - Profile × Distance interactions
    - Physics-informed synthetic model predictions
    """
    n = len(df_sub)
    d = df_sub['Distance'].values.astype(float)
    T = df_sub['Temperature'].values.astype(float)
    H = df_sub['Humidity'].values.astype(float)
    cid = df_sub['Circuit_ID'].values.astype(float)
    
    features = {}
    
    # ── Distance transforms ──
    features['d'] = d
    features['d_log'] = np.log1p(d)
    features['d_sqrt'] = np.sqrt(d)
    features['d_inv'] = 1.0 / (d + 1.0)
    features['d_inv2'] = 1.0 / (d**2 + 1.0)
    features['d_inv3'] = 1.0 / (d**3 + 1.0)
    features['d_pow15'] = d**1.5
    features['d_pow2'] = d**2
    features['d_cbrt'] = np.cbrt(d)
    features['d_exp_neg'] = np.exp(-d / 30.0)   # fast decay
    features['d_exp_neg2'] = np.exp(-d / 100.0)  # slow decay
    features['d_exp_neg3'] = np.exp(-d / 10.0)   # very fast decay
    
    # ── Environment ──
    features['temp'] = T
    features['humid'] = H
    features['temp_humid'] = T * H
    features['temp2'] = T ** 2
    features['humid2'] = H ** 2
    features['temp_norm'] = (T - 30.0) / 5.0
    features['humid_norm'] = (H - 30.0) / 10.0
    
    # ── Circuit ──
    features['circuit_id'] = cid
    features['circuit_is_2'] = (cid == 2).astype(float)
    
    # ── Location one-hot ──
    for loc in ['Ibri-Ibri City', 'Mahda-Oha', 'Barka-Rustaq']:
        features[f'loc_{loc[:4]}'] = (df_sub['Location'].values == loc).astype(float)
    
    # ── Profile one-hot ──
    features['is_lateral'] = (df_sub['Profile_Type'].values == 'Lateral').astype(float)
    features['is_longitudinal'] = (df_sub['Profile_Type'].values == 'Longitudinal').astype(float)
    
    # ── Location × Distance interactions ──
    for loc in ['Ibri-Ibri City', 'Mahda-Oha', 'Barka-Rustaq']:
        loc_mask = (df_sub['Location'].values == loc).astype(float)
        features[f'loc_{loc[:4]}_d'] = loc_mask * d
        features[f'loc_{loc[:4]}_d_log'] = loc_mask * np.log1p(d)
        features[f'loc_{loc[:4]}_d_inv'] = loc_mask / (d + 1.0)
        features[f'loc_{loc[:4]}_d_exp'] = loc_mask * np.exp(-d / 30.0)
    
    # ── Profile × Distance interactions ──
    is_lat = (df_sub['Profile_Type'].values == 'Lateral').astype(float)
    is_lon = 1.0 - is_lat
    features['lat_d'] = is_lat * d
    features['lon_d'] = is_lon * d
    features['lat_d_log'] = is_lat * np.log1p(d)
    features['lon_d_log'] = is_lon * np.log1p(d)
    features['lat_d_inv'] = is_lat / (d + 1.0)
    features['lon_d_inv'] = is_lon / (d + 1.0)
    features['lat_d_exp'] = is_lat * np.exp(-d / 20.0)  # lateral decays faster
    features['lon_d_exp'] = is_lon * np.exp(-d / 100.0)  # longitudinal slower
    
    # ── Circuit × Distance ──
    features['cid_d'] = cid * d
    features['cid_d_inv'] = cid / (d + 1.0)
    
    # ── Profile × Circuit ──
    features['lat_cid'] = is_lat * cid
    features['lon_cid'] = is_lon * cid
    
    # ── Distance × Environment ──
    features['d_temp'] = d * T
    features['d_humid'] = d * H
    features['d_inv_temp'] = features['d_inv'] * T
    features['d_inv_humid'] = features['d_inv'] * H
    
    # ── Location × Profile ──
    for loc in ['Ibri-Ibri City', 'Mahda-Oha', 'Barka-Rustaq']:
        loc_mask = (df_sub['Location'].values == loc).astype(float)
        features[f'loc_{loc[:4]}_lat'] = loc_mask * is_lat
        features[f'loc_{loc[:4]}_lon'] = loc_mask * is_lon
        features[f'loc_{loc[:4]}_lat_d'] = loc_mask * is_lat * d
        features[f'loc_{loc[:4]}_lon_d'] = loc_mask * is_lon * d
        features[f'loc_{loc[:4]}_lat_dinv'] = loc_mask * is_lat / (d + 1.0)
        features[f'loc_{loc[:4]}_lon_dinv'] = loc_mask * is_lon / (d + 1.0)
    
    # ── Physics-informed transfer features (from synthetic model) ──
    if synth_preds is not None:
        p_xgb, p_lgb, p_nn, p_ens = synth_preds
        features['synth_xgb'] = p_xgb
        features['synth_lgb'] = p_lgb
        features['synth_nn'] = p_nn
        features['synth_ens'] = p_ens
        features['synth_mean'] = (p_xgb + p_lgb + p_nn) / 3.0
        features['synth_log'] = np.log1p(np.abs(p_ens))
        features['synth_inv'] = 1.0 / (np.abs(p_ens) + 0.01)
        features['synth_sqrt'] = np.sqrt(np.abs(p_ens))
        features['synth_ens_d'] = p_ens * d
        features['synth_ens_dinv'] = p_ens / (d + 1.0)
        features['synth_residual_proxy'] = p_ens / (np.abs(p_ens) + 1.0)  # saturation
        # Ratio features
        features['synth_xgb_lgb_ratio'] = p_xgb / (p_lgb + 0.001)
        features['synth_spread'] = np.abs(p_xgb - p_lgb)
    
    return pd.DataFrame(features)


# ──────────────────────────────────────────────────────────────
# Stacking Ensemble Calibration
# ──────────────────────────────────────────────────────────────

def train_stacking_ensemble(X, y, cv, feature_names=None):
    """
    Train a stacking ensemble with diverse base learners.
    Returns cross-validated predictions and per-fold metrics.
    """
    XGB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_models = {
        'xgb_deep': xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=2.0, min_child_weight=3, gamma=0.1,
            tree_method='hist', device=XGB_DEVICE, random_state=42),
        'xgb_shallow': xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_alpha=1.0,
            reg_lambda=5.0, min_child_weight=5, gamma=0.5,
            tree_method='hist', device=XGB_DEVICE, random_state=43),
        'lgb_deep': lgb.LGBMRegressor(
            n_estimators=500, num_leaves=63, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=2.0, min_child_samples=5,
            device_type='cpu', verbose=-1, random_state=42),
        'lgb_shallow': lgb.LGBMRegressor(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_alpha=1.0,
            reg_lambda=5.0, min_child_samples=10,
            device_type='cpu', verbose=-1, random_state=43),
        'rf': RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            max_features=0.6, random_state=42, n_jobs=-1),
        'et': ExtraTreesRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            max_features=0.7, random_state=42, n_jobs=-1),
        'gbr': GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42),
        'knn5': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'knn10': KNeighborsRegressor(n_neighbors=10, weights='distance'),
        'ridge': Ridge(alpha=1.0),
        'bayesian': BayesianRidge(),
        'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    }
    
    # Level-1: cross-val predictions from each base model
    oof_preds = {}
    
    for name, model in base_models.items():
        try:
            pred = cross_val_predict(model, X, y, cv=cv)
            pred = np.maximum(pred, 0)
            oof_preds[name] = pred
            r2 = r2_score(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            print(f"    {name:15s}  R2={r2:.4f}  RMSE={rmse:.4f}")
        except Exception as e:
            print(f"    {name:15s}  FAILED: {e}")
    
    if not oof_preds:
        return None, {}
    
    # Level-2: Stack predictions with Ridge meta-learner
    stack_X = np.column_stack(list(oof_preds.values()))
    
    # Also add original top features
    top_feats = ['d', 'd_log', 'd_inv', 'd_exp_neg', 'is_lateral', 'circuit_id']
    for f in top_feats:
        if feature_names is not None and f in feature_names:
            idx = list(feature_names).index(f)
            stack_X = np.column_stack([stack_X, X[:, idx]])
    
    # Meta-learner: Ridge on stacked predictions
    meta_pred = cross_val_predict(Ridge(alpha=0.5), stack_X, y, cv=cv)
    meta_pred = np.maximum(meta_pred, 0)
    r2_meta = r2_score(y, meta_pred)
    rmse_meta = np.sqrt(mean_squared_error(y, meta_pred))
    print(f"    {'STACKED':15s}  R2={r2_meta:.4f}  RMSE={rmse_meta:.4f}")
    
    # Also try XGBoost as meta-learner
    meta_xgb_pred = cross_val_predict(
        xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                          reg_alpha=1.0, reg_lambda=3.0, random_state=42),
        stack_X, y, cv=cv)
    meta_xgb_pred = np.maximum(meta_xgb_pred, 0)
    r2_meta_xgb = r2_score(y, meta_xgb_pred)
    rmse_meta_xgb = np.sqrt(mean_squared_error(y, meta_xgb_pred))
    print(f"    {'STACKED-XGB':15s}  R2={r2_meta_xgb:.4f}  RMSE={rmse_meta_xgb:.4f}")
    
    # Select best approach
    all_results = {}
    for name, pred in oof_preds.items():
        all_results[name] = pred
    all_results['stacked_ridge'] = meta_pred
    all_results['stacked_xgb'] = meta_xgb_pred
    
    best_name = max(all_results, key=lambda k: r2_score(y, all_results[k]))
    best_pred = all_results[best_name]
    best_r2 = r2_score(y, best_pred)
    
    print(f"    >> BEST: {best_name}  R2={best_r2:.4f}")
    
    return best_pred, all_results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  ENHANCED 220kV PREDICTION – STACKING ENSEMBLE + RICH FEATURES")
    print("=" * 72)
    
    # Load real data
    df = pd.read_csv('standardized_data.csv')
    print(f"\n  Real data: {len(df)} rows")
    
    # Load synthetic models for transfer features
    print("  Loading 220kV synthetic models ...")
    try:
        models_220, feat_cols, scaler_220, le_220 = load_220kv_models()
        has_synth = True
        print("  -> Loaded successfully")
    except Exception as e:
        print(f"  -> Failed: {e}, proceeding without transfer features")
        has_synth = False
    
    # Cross-validation strategies
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv10 = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    all_outputs = []
    
    for ft_name in ['E', 'H']:
        print(f"\n{'='*72}")
        print(f"  {ft_name}-FIELD PREDICTION")
        print(f"{'='*72}")
        
        df_sub = df[df['Field_Type'] == ft_name].copy().reset_index(drop=True)
        if len(df_sub) == 0: continue
        
        actual = df_sub['Field_Value'].values
        
        # Get synthetic transfer predictions
        synth_preds = None
        if has_synth:
            synth_preds = get_synth_predictions(df_sub, models_220, ft_name, 
                                                 feat_cols, scaler_220, le_220)
            p_xgb_s, p_lgb_s, p_nn_s, p_ens_s = synth_preds
            r2_raw = r2_score(actual, p_ens_s)
            print(f"\n  Raw synthetic ensemble: R2={r2_raw:.4f}")
        
        # Build rich features
        X_df = build_rich_features(df_sub, synth_preds)
        feature_names = X_df.columns.tolist()
        X = X_df.values.astype(np.float64)
        
        print(f"  Features: {X.shape[1]} ({len(feature_names)} named)")
        
        # Also build version without synth features
        X_no_synth_df = build_rich_features(df_sub, None)
        X_no_synth = X_no_synth_df.values.astype(np.float64)
        
        # ── Approach 1: Rich features WITHOUT synthetic (5-fold) ──
        print(f"\n  --- WITHOUT synthetic transfer (5-fold CV) ---")
        best_pred_nosyn, all_nosyn = train_stacking_ensemble(
            X_no_synth, actual, cv5, X_no_synth_df.columns.tolist())
        
        # ── Approach 2: Rich features WITH synthetic (5-fold) ──
        if has_synth:
            print(f"\n  --- WITH synthetic transfer (5-fold CV) ---")
            best_pred_syn, all_syn = train_stacking_ensemble(
                X, actual, cv5, feature_names)
        
        # ── Approach 3: Rich features WITH synthetic (repeated 5-fold) ──
        if has_synth:
            print(f"\n  --- WITH synthetic transfer (repeated 5x3 CV) ---")
            best_pred_rep, all_rep = train_stacking_ensemble(
                X, actual, cv10, feature_names)
        
        # ── Select best overall ──
        print(f"\n  >>> FINAL COMPARISON <<<")
        candidates = {}
        if best_pred_nosyn is not None:
            candidates['NoSynth'] = best_pred_nosyn
        if has_synth and best_pred_syn is not None:
            candidates['WithSynth'] = best_pred_syn
        if has_synth and best_pred_rep is not None:
            candidates['Repeated'] = best_pred_rep
        
        # Add individual best models
        for name, pred in (all_syn if has_synth else all_nosyn).items():
            candidates[f'single_{name}'] = pred
        
        print(f"  {'Method':25s}  {'R2':>8s}  {'RMSE':>8s}  {'MAE':>8s}  {'MAPE%':>8s}")
        best_r2_overall = -999
        best_pred_overall = None
        best_name_overall = ''
        
        for name, pred in candidates.items():
            r2 = r2_score(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            mape = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-8))) * 100
            print(f"  {name:25s}  {r2:8.4f}  {rmse:8.4f}  {mae:8.4f}  {mape:8.1f}")
            if r2 > best_r2_overall:
                best_r2_overall = r2
                best_pred_overall = pred
                best_name_overall = name
        
        print(f"\n  >>> BEST: {best_name_overall}  R2={best_r2_overall:.4f}")
        
        # Per-location breakdown
        print(f"\n  Per-location breakdown ({best_name_overall}):")
        for loc in df_sub['Location'].unique():
            mask = df_sub['Location'] == loc
            if mask.sum() < 2: continue
            a = actual[mask]
            p = best_pred_overall[mask]
            r2_loc = r2_score(a, p)
            mae_loc = mean_absolute_error(a, p)
            print(f"    {loc:20s}  n={mask.sum():3d}  R2={r2_loc:.4f}  MAE={mae_loc:.4f}  "
                  f"actual=[{a.min():.2f}-{a.max():.2f}]  pred=[{p.min():.2f}-{p.max():.2f}]")
        
        # Per-profile breakdown
        print(f"\n  Per-profile breakdown:")
        for pt in df_sub['Profile_Type'].unique():
            mask = df_sub['Profile_Type'] == pt
            if mask.sum() < 2: continue
            a = actual[mask]
            p = best_pred_overall[mask]
            r2_pt = r2_score(a, p)
            print(f"    {pt:15s}  n={mask.sum():3d}  R2={r2_pt:.4f}")
        
        # Save predictions
        df_sub[f'Pred_{ft_name}_Best'] = best_pred_overall
        df_sub[f'Best_Method_{ft_name}'] = best_name_overall
        df_sub[f'Residual_{ft_name}'] = actual - best_pred_overall
        all_outputs.append(df_sub)
        
        # ── Plot ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(actual, best_pred_overall, alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
        lo, hi = min(actual.min(), best_pred_overall.min()), max(actual.max(), best_pred_overall.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
        ax.set_xlabel(f'Actual {ft_name}'); ax.set_ylabel('Predicted')
        ax.set_title(f'{best_name_overall}: R²={best_r2_overall:.4f}')
        ax.grid(True, alpha=0.3)
        
        # Distance profile
        ax = axes[0, 1]
        colors = {'Ibri-Ibri City': 'blue', 'Mahda-Oha': 'red', 'Barka-Rustaq': 'green'}
        for loc in df_sub['Location'].unique():
            mask = df_sub['Location'] == loc
            c = colors.get(loc, 'gray')
            ax.scatter(df_sub.loc[mask, 'Distance'], actual[mask],
                      label=f'{loc} actual', marker='o', s=20, alpha=0.6, color=c)
            ax.scatter(df_sub.loc[mask, 'Distance'], best_pred_overall[mask],
                      label=f'{loc} pred', marker='x', s=20, alpha=0.6, color=c)
        ax.set_xlabel('Distance (m)'); ax.set_ylabel(f'{ft_name} field')
        ax.set_title('Distance Profile'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        
        # Residuals vs distance
        ax = axes[1, 0]
        residuals = actual - best_pred_overall
        ax.scatter(df_sub['Distance'], residuals, alpha=0.6, s=20, c='steelblue')
        ax.axhline(0, color='red', linestyle='--', lw=1.5)
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('Residual (actual - predicted)')
        ax.set_title('Residuals vs Distance'); ax.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax = axes[1, 1]
        ax.hist(residuals, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', lw=1.5)
        ax.set_xlabel('Residual'); ax.set_ylabel('Count')
        ax.set_title(f'Residual Distribution (std={residuals.std():.3f})')
        
        fig.suptitle(f'{ft_name}-Field: Enhanced 220kV Prediction (R²={best_r2_overall:.4f})', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'outputs/220kv_enhanced_{ft_name.lower()}.png', dpi=150)
        plt.close(fig)
        print(f"\n  Saved -> outputs/220kv_enhanced_{ft_name.lower()}.png")
    
    # Save combined predictions
    if all_outputs:
        out = pd.concat(all_outputs, ignore_index=True)
        out.to_csv('outputs/real_predictions_220kv.csv', index=False)
        print(f"\n  Combined predictions -> outputs/real_predictions_220kv.csv ({len(out)} rows)")
    
    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == '__main__':
    main()
