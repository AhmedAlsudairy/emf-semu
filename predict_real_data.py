#!/usr/bin/env python3
"""
Predict E and H/B field values from real Narda EHP-50F measurements
===================================================================
Reads  : standardized_data.csv  (272 rows, 8 columns – Oman 220 kV)
Uses   : Trained XGBoost + LightGBM + NN ensemble from outputs/
Outputs: predictions CSV + comparison plots + console metrics

The real CSV has only:
  Location, Profile_Type, Field_Type, Distance, Circuit_ID,
  Field_Value, Temperature, Humidity

We reconstruct the full 44-feature vector the models expect using
known 220 kV line parameters (from the Oman grid) and sensible
environmental defaults.
"""

import json, pickle, warnings, os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Device ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── NN architecture (must match train_emf_models.py) ──────────
class EMFNet(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(),
                       nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Load trained artefacts ────────────────────────────────────
print("Loading trained models ...")
feat_cols   = json.load(open('outputs/feature_columns.json'))
nn_arch     = json.load(open('outputs/nn_architecture.json'))
scaler      = pickle.load(open('outputs/scaler.pkl', 'rb'))
label_encs  = pickle.load(open('outputs/label_encoders.pkl', 'rb'))


def load_models(target: str):
    """Load XGB, LGB, NN, Meta for a target ('b' or 'e')."""
    xgb_m  = pickle.load(open(f'outputs/xgb_{target}.pkl', 'rb'))
    lgb_m  = pickle.load(open(f'outputs/lgb_{target}.pkl', 'rb'))
    meta_m = pickle.load(open(f'outputs/meta_{target}.pkl', 'rb'))

    # Read per-target NN best params (B and E may differ)
    bp = json.load(open(f'outputs/best_params_{target}.json'))
    nn_bp = bp['nn']
    h_dims = [nn_bp['hidden_dim']] * nn_bp['n_layers']
    dropout = nn_bp['dropout']

    nn_m = EMFNet(nn_arch['input_dim'], h_dims, dropout).to(device)
    nn_m.load_state_dict(torch.load(f'outputs/nn_{target}.pt',
                                     map_location=device,
                                     weights_only=True))
    nn_m.eval()
    return xgb_m, lgb_m, nn_m, meta_m

xgb_b, lgb_b, nn_b, meta_b = load_models('b')
xgb_e, lgb_e, nn_e, meta_e = load_models('e')
print("  All models loaded.")


# ── 220 kV line defaults (Oman) ───────────────────────────────
# These match the training generator's DEFAULT_LINES[220]
LINE_220 = dict(
    voltage_kV          = 220,
    I_base_A            = 800,
    height_m            = 30,
    phase_spacing_m     = 7.0,
    span_length_m       = 350,
    conductor_radius_cm = 7.75,   # bundle-equivalent
    conductor_diameter_m= 0.030,
    bundle_count        = 2,
    conductor_weight_kg_m = 1.6,
    R_per_km            = 0.04,
    rated_tensile_N     = 120_000,
    elevation_m         = 300,     # Oman inland ~300 m
    wind_speed_ms       = 2.0,
    soil_resistivity_ohm_m = 100,
    solar_irradiance_W_m2 = 800,
    ice_thickness_mm    = 0,
    frequency_Hz        = 50,
)

LOCATION_MAP = {
    'Ibri-Ibri City': dict(feeder='Ibri-Ibri City', substation='Ibri 220kV',
                           config='horizontal'),
    'Mahda-Oha':      dict(feeder='Mahda-Oha', substation='Mahda 220kV',
                           config='horizontal'),
    'Barka-Rustaq':   dict(feeder='Barka-Rustaq', substation='Barka 220kV',
                           config='delta'),
}


def safe_le_transform(le, value):
    """encode value using a fitted LabelEncoder; unseen → nearest class idx 0."""
    if value in le.classes_:
        return le.transform([value])[0]
    return 0


# ── Build feature matrix ─────────────────────────────────────
def build_features(df_real: pd.DataFrame) -> np.ndarray:
    """
    Convert the 8-column real CSV into the 44-feature matrix
    the trained models expect.
    """
    n = len(df_real)
    feat = pd.DataFrame(index=range(n))

    L = LINE_220

    # --- direct from real CSV ---
    feat['voltage_kV']           = L['voltage_kV']
    feat['distance_m']           = df_real['Distance'].values
    feat['temperature_C']        = df_real['Temperature'].values
    feat['humidity_pct']         = df_real['Humidity'].values

    # --- current: base × load_factor (use 0.85 as typical) ---
    load_factor = 0.85
    current_A   = L['I_base_A'] * load_factor
    feat['current_A']            = current_A

    # --- measurement height: Narda EHP-50F on tripod ≈ 1.0 m ---
    feat['measurement_height_m'] = 1.0

    # --- line geometry ---
    feat['height_m']             = L['height_m']
    feat['phase_spacing_m']      = L['phase_spacing_m']
    feat['span_length_m']        = L['span_length_m']
    feat['conductor_radius_cm']  = L['conductor_radius_cm']
    feat['conductor_diameter_m'] = L['conductor_diameter_m']
    feat['bundle_count']         = L['bundle_count']
    feat['conductor_weight_kg_m']= L['conductor_weight_kg_m']

    # --- sag (approx for Oman summer) ---
    sag = 5.5
    feat['sag_m']                = sag
    feat['ground_clearance_m']   = L['height_m'] - sag

    # --- environmental ---
    feat['elevation_m']            = L['elevation_m']
    feat['wind_speed_ms']          = L['wind_speed_ms']
    feat['soil_resistivity_ohm_m'] = L['soil_resistivity_ohm_m']
    feat['solar_irradiance_W_m2']  = L['solar_irradiance_W_m2']
    feat['ice_thickness_mm']       = L['ice_thickness_mm']
    feat['frequency_Hz']           = L['frequency_Hz']

    # --- conductor temp (approx IEEE 738 for 680 A, 31°C ambient) ---
    feat['conductor_temp_C']     = 55.0

    # --- load / power ---
    feat['load_factor']          = load_factor
    pf = 0.90
    feat['power_factor']         = pf
    feat['active_power_MW']      = np.sqrt(3) * L['voltage_kV'] * current_A * pf / 1000
    feat['phase_angle_deg']      = 0.0      # averaged out in RMS

    # --- per-row mapping from location ---
    configs  = []
    feeders  = []
    subs     = []
    circuits = []
    for _, row in df_real.iterrows():
        loc = row['Location']
        m   = LOCATION_MAP.get(loc, LOCATION_MAP['Ibri-Ibri City'])
        configs.append(m['config'])
        feeders.append(m['feeder'])
        subs.append(m['substation'])
        is_dc = m['config'] == 'double_circuit'
        circuits.append('Double-Circuit 3ph AC' if is_dc else '3-Phase AC')

    # --- weather heuristic ---
    temps = df_real['Temperature'].values
    hums  = df_real['Humidity'].values
    weathers = []
    for t, h in zip(temps, hums):
        if h > 70 and t > 30:
            weathers.append('Hot/Humid')
        elif t > 35:
            weathers.append('Clear')
        else:
            weathers.append('Clear')

    seasons = ['Summer' if t > 30 else 'Spring/Autumn' for t in temps]

    # --- encode categoricals ---
    feat['configuration'] = [safe_le_transform(label_encs['configuration'], c) for c in configs]
    feat['feeder']        = [safe_le_transform(label_encs['feeder'], f) for f in feeders]
    feat['substation']    = [safe_le_transform(label_encs['substation'], s) for s in subs]
    feat['weather']       = [safe_le_transform(label_encs['weather'], w) for w in weathers]
    feat['time_of_day']   = [safe_le_transform(label_encs['time_of_day'], 'Morning')] * n
    feat['season']        = [safe_le_transform(label_encs['season'], s) for s in seasons]
    feat['profile_type']  = [safe_le_transform(label_encs['profile_type'], 'Overhead Transmission')] * n
    feat['circuit_type']  = [safe_le_transform(label_encs['circuit_type'], c) for c in circuits]

    # --- engineered features ---
    feat['voltage_current_product'] = feat['voltage_kV'] * feat['current_A']
    feat['height_to_spacing_ratio'] = feat['height_m'] / feat['phase_spacing_m']
    feat['distance_to_height_ratio']= feat['distance_m'] / feat['height_m']
    feat['sag_to_span_ratio']       = feat['sag_m'] / feat['span_length_m']
    feat['log_distance']            = np.log1p(feat['distance_m'])
    feat['sqrt_distance']           = np.sqrt(feat['distance_m'])
    feat['inv_distance']            = 1.0 / (feat['distance_m'] + 1)
    feat['inv_distance_sq']         = 1.0 / (feat['distance_m']**2 + 1)
    feat['temp_humidity_interaction']= feat['temperature_C'] * feat['humidity_pct']
    feat['power_density']           = feat['active_power_MW'] / (feat['distance_m'] * feat['height_m'] + 1)

    # --- reorder to match training feature order ---
    feat = feat[feat_cols]
    return feat.values.astype(np.float32)


# ── Predict ───────────────────────────────────────────────────
def ensemble_predict(X_scaled, xgb_m, lgb_m, nn_m, meta_m):
    p_xgb = xgb_m.predict(X_scaled)
    p_lgb = lgb_m.predict(X_scaled)
    with torch.no_grad():
        p_nn = nn_m(torch.from_numpy(X_scaled).to(device)).cpu().numpy()
    Xm = np.column_stack([p_xgb, p_lgb, p_nn])
    return meta_m.predict(Xm), p_xgb, p_lgb, p_nn


# ── Main ──────────────────────────────────────────────────────
def main():
    csv_path = 'standardized_data.csv'
    print(f"\nLoading real data: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # Split by field type
    df_E = df[df['Field_Type'] == 'E'].copy().reset_index(drop=True)
    df_H = df[df['Field_Type'] == 'H'].copy().reset_index(drop=True)
    print(f"  E-field rows: {len(df_E)},  H-field rows: {len(df_H)}")

    os.makedirs('outputs', exist_ok=True)
    all_results = []

    # ─── E-field predictions ──────────────────────────────────
    if len(df_E) > 0:
        print("\n" + "="*70)
        print("  E-FIELD PREDICTION  (real Narda EHP-50F readings)")
        print("="*70)

        X_E = build_features(df_E)
        X_E_s = scaler.transform(X_E)
        ens_e, p_xgb_e, p_lgb_e, p_nn_e = ensemble_predict(
            X_E_s, xgb_e, lgb_e, nn_e, meta_e)
        actual_e = df_E['Field_Value'].values

        print(f"\n  {'Model':12s}  {'RMSE':>8s}  {'MAE':>8s}  {'R²':>8s}  {'MAPE%':>8s}")
        print("  " + "-"*52)
        for name, pred in [('XGBoost', p_xgb_e), ('LightGBM', p_lgb_e),
                           ('NeuralNet', p_nn_e), ('Ensemble', ens_e)]:
            rmse = np.sqrt(mean_squared_error(actual_e, pred))
            mae  = mean_absolute_error(actual_e, pred)
            r2   = r2_score(actual_e, pred)
            mape = np.mean(np.abs((actual_e - pred) / (np.abs(actual_e) + 1e-8))) * 100
            print(f"  {name:12s}  {rmse:8.4f}  {mae:8.4f}  {r2:8.4f}  {mape:8.2f}")

        df_E['Predicted_E_XGB']  = p_xgb_e
        df_E['Predicted_E_LGB']  = p_lgb_e
        df_E['Predicted_E_NN']   = p_nn_e
        df_E['Predicted_E_Ens']  = ens_e
        df_E['Residual_E']       = actual_e - ens_e
        all_results.append(df_E)

        # ── scatter plot ──
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (nm, p) in zip(axes, [('XGBoost', p_xgb_e), ('LightGBM', p_lgb_e),
                                       ('NeuralNet', p_nn_e), ('Ensemble', ens_e)]):
            r2 = r2_score(actual_e, p)
            ax.scatter(actual_e, p, alpha=0.6, s=20, edgecolors='k', linewidths=0.3)
            lo, hi = min(actual_e.min(), p.min()), max(actual_e.max(), p.max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideal')
            ax.set_xlabel('Actual E (V/m)'); ax.set_ylabel('Predicted E (V/m)')
            ax.set_title(f'{nm}  R²={r2:.4f}')
            ax.legend()
        fig.suptitle('E-Field: Predicted vs Actual (Narda EHP-50F)', fontsize=14)
        fig.tight_layout()
        fig.savefig('outputs/real_E_predictions.png', dpi=150)
        plt.close(fig)
        print("  Saved -> outputs/real_E_predictions.png")

        # ── distance profile ──
        fig, ax = plt.subplots(figsize=(10, 5))
        for loc in df_E['Location'].unique():
            mask = df_E['Location'] == loc
            ax.scatter(df_E.loc[mask, 'Distance'], actual_e[mask],
                       label=f'{loc} (actual)', marker='o', s=30)
            ax.scatter(df_E.loc[mask, 'Distance'], ens_e[mask],
                       label=f'{loc} (pred)', marker='x', s=30)
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('E-field (V/m)')
        ax.set_title('E-Field vs Distance: Actual vs Ensemble Prediction')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig('outputs/real_E_distance_profile.png', dpi=150)
        plt.close(fig)
        print("  Saved -> outputs/real_E_distance_profile.png")

    # ─── H-field predictions (→ B-field model, convert) ───────
    if len(df_H) > 0:
        print("\n" + "="*70)
        print("  H-FIELD PREDICTION  (via B-field model, H = B / mu_0)")
        print("="*70)

        X_H = build_features(df_H)
        X_H_s = scaler.transform(X_H)
        # B model predicts B in µT; convert to H in A/m
        MU0 = 4e-7 * np.pi
        ens_b, p_xgb_b, p_lgb_b, p_nn_b = ensemble_predict(
            X_H_s, xgb_b, lgb_b, nn_b, meta_b)

        # B (µT) → H (A/m):  H = B * 1e-6 / µ₀
        conv = 1e-6 / MU0
        ens_h   = ens_b   * conv
        p_xgb_h = p_xgb_b * conv
        p_lgb_h = p_lgb_b * conv
        p_nn_h  = p_nn_b  * conv

        actual_h = df_H['Field_Value'].values

        print(f"\n  {'Model':12s}  {'RMSE':>8s}  {'MAE':>8s}  {'R²':>8s}  {'MAPE%':>8s}")
        print("  " + "-"*52)
        for name, pred in [('XGBoost', p_xgb_h), ('LightGBM', p_lgb_h),
                           ('NeuralNet', p_nn_h), ('Ensemble', ens_h)]:
            rmse = np.sqrt(mean_squared_error(actual_h, pred))
            mae  = mean_absolute_error(actual_h, pred)
            r2   = r2_score(actual_h, pred)
            mape = np.mean(np.abs((actual_h - pred) / (np.abs(actual_h) + 1e-8))) * 100
            print(f"  {name:12s}  {rmse:8.4f}  {mae:8.4f}  {r2:8.4f}  {mape:8.2f}")

        df_H['Predicted_H_XGB']  = p_xgb_h
        df_H['Predicted_H_LGB']  = p_lgb_h
        df_H['Predicted_H_NN']   = p_nn_h
        df_H['Predicted_H_Ens']  = ens_h
        df_H['Predicted_B_uT']   = ens_b
        df_H['Residual_H']       = actual_h - ens_h
        all_results.append(df_H)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (nm, p) in zip(axes, [('XGBoost', p_xgb_h), ('LightGBM', p_lgb_h),
                                       ('NeuralNet', p_nn_h), ('Ensemble', ens_h)]):
            r2 = r2_score(actual_h, p)
            ax.scatter(actual_h, p, alpha=0.6, s=20, edgecolors='k', linewidths=0.3)
            lo, hi = min(actual_h.min(), p.min()), max(actual_h.max(), p.max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideal')
            ax.set_xlabel('Actual H (A/m)'); ax.set_ylabel('Predicted H (A/m)')
            ax.set_title(f'{nm}  R²={r2:.4f}')
            ax.legend()
        fig.suptitle('H-Field: Predicted vs Actual (Narda EHP-50F)', fontsize=14)
        fig.tight_layout()
        fig.savefig('outputs/real_H_predictions.png', dpi=150)
        plt.close(fig)
        print("  Saved -> outputs/real_H_predictions.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        for loc in df_H['Location'].unique():
            mask = df_H['Location'] == loc
            ax.scatter(df_H.loc[mask, 'Distance'], actual_h[mask],
                       label=f'{loc} (actual)', marker='o', s=30)
            ax.scatter(df_H.loc[mask, 'Distance'], ens_h[mask],
                       label=f'{loc} (pred)', marker='x', s=30)
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('H-field (A/m)')
        ax.set_title('H-Field vs Distance: Actual vs Ensemble Prediction')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig('outputs/real_H_distance_profile.png', dpi=150)
        plt.close(fig)
        print("  Saved -> outputs/real_H_distance_profile.png")

    # ─── Combined output ──────────────────────────────────────
    if all_results:
        out = pd.concat(all_results, ignore_index=True)
        out.to_csv('outputs/real_predictions.csv', index=False)
        print(f"\n  All predictions -> outputs/real_predictions.csv  ({len(out)} rows)")

    # ─── Calibrated predictions (domain adaptation) ─────────────
    print("\n" + "="*70)
    print("  CALIBRATED MODEL  (post-hoc linear correction on ensemble)")
    print("="*70)
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.linear_model import Ridge as CalibRidge
    from sklearn.ensemble import GradientBoostingRegressor
    import xgboost as xgb_cal

    for ft, df_sub, ens_pred in [('E', df_E, ens_e), ('H', df_H, ens_h)]:
        if len(df_sub) == 0:
            continue
        actual = df_sub['Field_Value'].values

        # --- Method 1: Linear calibration on ensemble output ---
        Xc = np.column_stack([ens_pred, ens_pred**2, np.log1p(np.abs(ens_pred)),
                              df_sub['Distance'].values,
                              np.log1p(df_sub['Distance'].values)])
        cal_model = CalibRidge(alpha=1.0)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cal_pred = cross_val_predict(cal_model, Xc, actual, cv=kf)
        cal_pred = np.maximum(cal_pred, 0)

        rmse_c = np.sqrt(mean_squared_error(actual, cal_pred))
        mae_c = mean_absolute_error(actual, cal_pred)
        r2_c = r2_score(actual, cal_pred)
        print(f"\n  {ft}-field (Calibrated Ensemble, 5-fold CV):")
        print(f"    RMSE={rmse_c:.4f}  MAE={mae_c:.4f}  R²={r2_c:.4f}")

        # --- Method 2: Direct XGBoost on real data features ---
        Xd = np.column_stack([
            df_sub['Distance'].values,
            np.log1p(df_sub['Distance'].values),
            np.sqrt(df_sub['Distance'].values),
            1.0 / (df_sub['Distance'].values + 1),
            1.0 / (df_sub['Distance'].values**2 + 1),
            df_sub['Temperature'].values,
            df_sub['Humidity'].values,
            pd.Categorical(df_sub['Location']).codes,
            pd.Categorical(df_sub['Profile_Type']).codes,
            df_sub['Circuit_ID'].values,
            df_sub['Temperature'].values * df_sub['Humidity'].values,
        ])
        direct_model = xgb_cal.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42)
        dir_pred = cross_val_predict(direct_model, Xd, actual, cv=kf)
        dir_pred = np.maximum(dir_pred, 0)

        rmse_d = np.sqrt(mean_squared_error(actual, dir_pred))
        mae_d = mean_absolute_error(actual, dir_pred)
        r2_d = r2_score(actual, dir_pred)
        print(f"  {ft}-field (Direct XGBoost on real data, 5-fold CV):")
        print(f"    RMSE={rmse_d:.4f}  MAE={mae_d:.4f}  R²={r2_d:.4f}")

        # --- Method 3: Hybrid = ensemble predictions + real features ---
        Xh = np.column_stack([Xd, ens_pred, np.log1p(np.abs(ens_pred))])
        hybrid_model = xgb_cal.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42)
        hyb_pred = cross_val_predict(hybrid_model, Xh, actual, cv=kf)
        hyb_pred = np.maximum(hyb_pred, 0)

        rmse_h = np.sqrt(mean_squared_error(actual, hyb_pred))
        mae_h = mean_absolute_error(actual, hyb_pred)
        r2_h = r2_score(actual, hyb_pred)
        print(f"  {ft}-field (Hybrid: Ensemble+Real features, 5-fold CV):")
        print(f"    RMSE={rmse_h:.4f}  MAE={mae_h:.4f}  R²={r2_h:.4f}")

        # save best calibration predictions
        best_name = min(
            [('Calibrated', cal_pred, r2_c), ('Direct', dir_pred, r2_d),
             ('Hybrid', hyb_pred, r2_h)],
            key=lambda x: -x[2])
        df_sub[f'Predicted_{ft}_Best'] = best_name[1]
        df_sub[f'Best_Method_{ft}'] = best_name[0]
        print(f"  >> Best: {best_name[0]}  R²={best_name[2]:.4f}")

        # ── scatter for calibrated methods ──
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (nm, p, r2v) in zip(axes, [
             ('Calibrated', cal_pred, r2_c),
             ('Direct XGB', dir_pred, r2_d),
             ('Hybrid', hyb_pred, r2_h)]):
            ax.scatter(actual, p, alpha=0.6, s=25, edgecolors='k', linewidths=0.3)
            lo, hi = min(actual.min(), p.min()), max(actual.max(), p.max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideal')
            ax.set_xlabel(f'Actual {ft}'); ax.set_ylabel(f'Predicted {ft}')
            ax.set_title(f'{nm}  R²={r2v:.4f}')
            ax.legend()
        fig.suptitle(f'{ft}-Field: Calibrated Predictions (5-fold CV)', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'outputs/real_{ft}_calibrated.png', dpi=150)
        plt.close(fig)
        print(f"  Saved -> outputs/real_{ft}_calibrated.png")

    # Re-save with calibrated predictions
    if all_results:
        out = pd.concat(all_results, ignore_index=True)
        out.to_csv('outputs/real_predictions.csv', index=False)

    # ─── Per-location summary ─────────────────────────────────
    print("\n" + "="*70)
    print("  PER-LOCATION SUMMARY")
    print("="*70)
    for loc in df['Location'].unique():
        print(f"\n  {loc}")
        for ft in ['E', 'H']:
            sub = df[(df['Location']==loc) & (df['Field_Type']==ft)]
            if len(sub) == 0:
                continue
            actual = sub['Field_Value'].values
            # get matching predictions
            if ft == 'E' and len(df_E) > 0:
                mask = df_E['Location'] == loc
                pred = ens_e[mask.values] if mask.any() else None
            elif ft == 'H' and len(df_H) > 0:
                mask = df_H['Location'] == loc
                pred = ens_h[mask.values] if mask.any() else None
            else:
                pred = None
            if pred is not None and len(pred) > 0:
                r2 = r2_score(actual, pred) if len(actual) > 1 else float('nan')
                mae = mean_absolute_error(actual, pred)
                print(f"    {ft}: n={len(actual):3d}  actual=[{actual.min():.3f} - {actual.max():.3f}]  "
                      f"pred=[{pred.min():.3f} - {pred.max():.3f}]  MAE={mae:.4f}  R2={r2:.4f}")

    print("\n  Done.")


if __name__ == '__main__':
    main()
