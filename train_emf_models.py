#!/usr/bin/env python3
"""
EMF Models Training Pipeline v3
================================
Matches generate_emf_dataset.py v4 column schema.
Features:
  - XGBoost + LightGBM gradient boosting (GPU / CUDA)
  - PyTorch 4-layer MLP (CUDA)
  - Stacked ensemble with Ridge meta-learner
  - Optuna hyperparameter optimization (subsampled for speed)
  - Trains for both B-field (µT) and E-field (V/m)
  - All artifacts saved to outputs/
"""

import numpy as np
import pandas as pd
import pickle, json, argparse, os, warnings, time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import optuna

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Device selection – CUDA preferred, CPU fallback with flag
# ---------------------------------------------------------------------------
USE_GPU = True                        # set False to force CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if USE_GPU and device.type != 'cuda':
    print("WARNING: CUDA requested but not available – falling back to CPU.")
    print("  Install torch+cu126 in a Python 3.12 venv for GPU acceleration.")

print(f"Device : {device}")
if device.type == 'cuda':
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")

os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------------------------
# Column schema  (must match generate_emf_dataset.py v4)
# ---------------------------------------------------------------------------
TARGET_COLS = ['B_field_uT', 'E_field_V_m']

# Columns to EXCLUDE from features (targets + physics reference + derived)
EXCLUDE_COLS = {
    'B_field_uT', 'E_field_V_m', 'H_field_A_m',
    'B_field_clean_uT', 'E_field_clean_V_m',
    'corona_onset_kV_cm', 'surface_gradient_kV_cm', 'corona_ratio',
    'ICNIRP_E_exceeded', 'ICNIRP_B_exceeded',
}

# Categorical columns to label-encode
CAT_COLS = [
    'configuration', 'feeder', 'substation',
    'weather', 'time_of_day', 'season',
    'profile_type', 'circuit_type',
]

# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------
def load_and_prepare(csv_path: str):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    # ---- additional engineered features (on top of those in the CSV) ----
    if 'voltage_current_product' not in df.columns:
        df['voltage_current_product'] = df['voltage_kV'] * df['current_A']
    if 'height_to_spacing_ratio' not in df.columns:
        df['height_to_spacing_ratio'] = df['height_m'] / df['phase_spacing_m']
    if 'distance_to_height_ratio' not in df.columns:
        df['distance_to_height_ratio'] = df['distance_m'] / df['height_m']
    if 'sag_to_span_ratio' not in df.columns:
        df['sag_to_span_ratio'] = df['sag_m'] / df['span_length_m']
    if 'log_distance' not in df.columns:
        df['log_distance'] = np.log1p(df['distance_m'])
    if 'sqrt_distance' not in df.columns:
        df['sqrt_distance'] = np.sqrt(df['distance_m'])
    if 'inv_distance' not in df.columns:
        df['inv_distance'] = 1 / (df['distance_m'] + 1)
    if 'inv_distance_sq' not in df.columns:
        df['inv_distance_sq'] = 1 / (df['distance_m']**2 + 1)
    if 'temp_humidity_interaction' not in df.columns:
        df['temp_humidity_interaction'] = df['temperature_C'] * df['humidity_pct']
    if 'power_density' not in df.columns:
        df['power_density'] = df['active_power_MW'] / (df['distance_m'] * df['height_m'] + 1)

    # ---- targets ----
    targets = {
        'B': df['B_field_uT'].values.astype(np.float32),
        'E': df['E_field_V_m'].values.astype(np.float32),
    }

    # ---- encode categoricals ----
    label_encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # ---- feature matrix ----
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # drop any remaining object columns that slipped through
    for c in list(feature_cols):
        if df[c].dtype == object:
            feature_cols.remove(c)

    X = df[feature_cols].values.astype(np.float32)

    print(f"  Features: {X.shape[1]}   Targets: B [{targets['B'].min():.4f}–{targets['B'].max():.2f}]  "
          f"E [{targets['E'].min():.4f}–{targets['E'].max():.2f}]")

    return X, targets, feature_cols, label_encoders


# ---------------------------------------------------------------------------
# HPO objectives
# ---------------------------------------------------------------------------
XGB_DEVICE = 'cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu'
LGB_DEVICE = 'gpu'  if (USE_GPU and torch.cuda.is_available()) else 'cpu'

def objective_xgb(trial, Xtr, ytr, Xv, yv):
    p = {
        'max_depth':        trial.suggest_int('max_depth', 4, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':            trial.suggest_float('gamma', 0, 5),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'tree_method': 'hist', 'device': XGB_DEVICE, 'random_state': 42,
    }
    m = xgb.XGBRegressor(**p)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return np.sqrt(mean_squared_error(yv, m.predict(Xv)))


def objective_lgb(trial, Xtr, ytr, Xv, yv):
    p = {
        'num_leaves':        trial.suggest_int('num_leaves', 20, 200),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'device_type': LGB_DEVICE, 'random_state': 42, 'verbose': -1,
    }
    m = lgb.LGBMRegressor(**p)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)])
    return np.sqrt(mean_squared_error(yv, m.predict(Xv)))


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class EMFNet(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.2):
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


def train_nn(model, Xtr, ytr, Xv, yv, epochs, bs, lr):
    crit = nn.HuberLoss(delta=1.0)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xv_t  = torch.from_numpy(Xv).to(device)
    yv_t  = torch.from_numpy(yv).to(device)

    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                     pin_memory=False, drop_last=True)

    best_val = float('inf')
    patience = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv_t), yv_t).item()
        if vl < best_val:
            best_val = vl
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    return best_val


def objective_nn(trial, Xtr, ytr, Xv, yv, in_dim):
    nl = trial.suggest_int('n_layers', 2, 5)
    hd = trial.suggest_categorical('hidden_dim', [128, 256, 512, 768])
    dp = trial.suggest_float('dropout', 0.0, 0.35)
    lr = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
    bs = trial.suggest_categorical('batch_size', [4096, 8192, 16384])
    model = EMFNet(in_dim, [hd]*nl, dp).to(device)
    return train_nn(model, Xtr, ytr, Xv, yv, epochs=60, bs=bs, lr=lr)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train_all(csv_path: str = 'grid_emf_dataset.csv',
              n_xgb_trials: int = 20,
              n_lgb_trials: int = 20,
              n_nn_trials:  int = 12,
              hpo_sample:   int = 150_000):

    wall0 = time.time()
    print("=" * 80)
    print("EMF MODEL TRAINING PIPELINE v3  (CUDA-accelerated)")
    print("=" * 80)

    X, targets, feat_cols, label_encoders = load_and_prepare(csv_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    results = {}

    for tgt in ['B', 'E']:
        print(f"\n{'='*80}\n  Training for {tgt}-field\n{'='*80}")
        y = targets[tgt]
        Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

        # HPO subsample
        ss = min(hpo_sample, len(Xtr))
        idx = np.random.choice(len(Xtr), ss, replace=False)
        Xh = Xtr[idx]; yh = ytr[idx]
        Xht, Xhv, yht, yhv = train_test_split(Xh, yh, test_size=0.2, random_state=42)

        print(f"  HPO: {len(Xht):,} train / {len(Xhv):,} val   |   Full: {len(Xtr):,} / {len(Xte):,}")

        # 1. XGBoost ---------------------------------------------------------
        print(f"\n  [1/5] XGBoost HPO ({n_xgb_trials} trials) ...")
        st = optuna.create_study(direction='minimize')
        st.optimize(lambda t: objective_xgb(t, Xht, yht, Xhv, yhv),
                    n_trials=n_xgb_trials, show_progress_bar=True)
        bp_xgb = st.best_params
        print(f"        best RMSE={st.best_value:.5f}")
        xgb_m = xgb.XGBRegressor(**bp_xgb, tree_method='hist',
                                  device=XGB_DEVICE, random_state=42)
        xgb_m.fit(Xtr, ytr)
        xgb_p = xgb_m.predict(Xte)

        # 2. LightGBM --------------------------------------------------------
        print(f"\n  [2/5] LightGBM HPO ({n_lgb_trials} trials) ...")
        st2 = optuna.create_study(direction='minimize')
        st2.optimize(lambda t: objective_lgb(t, Xht, yht, Xhv, yhv),
                     n_trials=n_lgb_trials, show_progress_bar=True)
        bp_lgb = st2.best_params
        print(f"        best RMSE={st2.best_value:.5f}")
        lgb_m = lgb.LGBMRegressor(**bp_lgb, device_type=LGB_DEVICE,
                                  verbose=-1, random_state=42)
        lgb_m.fit(Xtr, ytr)
        lgb_p = lgb_m.predict(Xte)

        # 3. Neural network ---------------------------------------------------
        print(f"\n  [3/5] NN HPO ({n_nn_trials} trials) ...")
        st3 = optuna.create_study(direction='minimize')
        st3.optimize(lambda t: objective_nn(t, Xht, yht, Xhv, yhv, Xtr.shape[1]),
                     n_trials=n_nn_trials, show_progress_bar=True)
        bp_nn = st3.best_params
        print(f"        best val_loss={st3.best_value:.6f}")
        h_dims = [bp_nn['hidden_dim']] * bp_nn['n_layers']
        nn_m = EMFNet(Xtr.shape[1], h_dims, bp_nn['dropout']).to(device)
        train_nn(nn_m, Xtr, ytr, Xte, yte, epochs=120,
                 bs=bp_nn['batch_size'], lr=bp_nn['lr'])
        nn_m.eval()
        with torch.no_grad():
            nn_p = nn_m(torch.from_numpy(Xte).to(device)).cpu().numpy()

        # 4. Ensemble ---------------------------------------------------------
        print(f"\n  [4/5] Ensemble meta-learner ...")
        with torch.no_grad():
            nn_tr_p = nn_m(torch.from_numpy(Xtr).to(device)).cpu().numpy()
        Xm_tr = np.column_stack([xgb_m.predict(Xtr), lgb_m.predict(Xtr), nn_tr_p])
        Xm_te = np.column_stack([xgb_p, lgb_p, nn_p])
        meta = Ridge(alpha=1.0)
        meta.fit(Xm_tr, ytr)
        ens_p = meta.predict(Xm_te)

        # 5. Evaluate ---------------------------------------------------------
        print(f"\n  [5/5] Evaluation:")
        res = {}
        for name, pred in [('XGBoost', xgb_p), ('LightGBM', lgb_p),
                           ('NeuralNet', nn_p), ('Ensemble', ens_p)]:
            rmse = np.sqrt(mean_squared_error(yte, pred))
            mae  = mean_absolute_error(yte, pred)
            r2   = r2_score(yte, pred)
            mape = np.mean(np.abs((yte - pred) / (np.abs(yte) + 1e-8))) * 100
            print(f"    {name:12s}  RMSE={rmse:9.5f}  MAE={mae:9.5f}  R²={r2:.6f}  MAPE={mape:.2f}%")
            res[name] = dict(rmse=float(rmse), mae=float(mae), r2=float(r2), mape=float(mape))
        results[tgt] = res

        # ---- save models ----
        pickle.dump(xgb_m, open(f'outputs/xgb_{tgt.lower()}.pkl', 'wb'))
        pickle.dump(lgb_m, open(f'outputs/lgb_{tgt.lower()}.pkl', 'wb'))
        torch.save(nn_m.state_dict(), f'outputs/nn_{tgt.lower()}.pt')
        pickle.dump(meta,  open(f'outputs/meta_{tgt.lower()}.pkl', 'wb'))
        json.dump({'xgb': bp_xgb, 'lgb': bp_lgb, 'nn': bp_nn},
                  open(f'outputs/best_params_{tgt.lower()}.json', 'w'), indent=2)

        # ---- plot ----
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, (name, pred) in zip(axes, [('XGBoost', xgb_p), ('LightGBM', lgb_p),
                                            ('NeuralNet', nn_p), ('Ensemble', ens_p)]):
            ax.scatter(yte, pred, alpha=0.15, s=1)
            lo, hi = yte.min(), yte.max()
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
            ax.set_xlabel(f'Actual {tgt}-field'); ax.set_ylabel('Predicted')
            ax.set_title(f'{name}  R²={res[name]["r2"]:.4f}')
        fig.tight_layout()
        fig.savefig(f'outputs/{tgt.lower()}_predictions.png', dpi=150)
        plt.close(fig)

    # ---- global artifacts ----
    pickle.dump(scaler, open('outputs/scaler.pkl', 'wb'))
    pickle.dump(label_encoders, open('outputs/label_encoders.pkl', 'wb'))
    json.dump(feat_cols, open('outputs/feature_columns.json', 'w'))
    json.dump({'input_dim': Xtr.shape[1], 'hidden_dims': h_dims,
               'dropout': bp_nn['dropout']},
              open('outputs/nn_architecture.json', 'w'), indent=2)
    json.dump(results, open('outputs/training_results.json', 'w'), indent=2)

    wall = time.time() - wall0
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE  ({wall/60:.1f} min)")
    print(f"  All artifacts → outputs/")
    print(f"{'='*80}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='EMF model training v3')
    ap.add_argument('--csv', default='grid_emf_dataset.csv')
    ap.add_argument('--xgb-trials', type=int, default=20)
    ap.add_argument('--lgb-trials', type=int, default=20)
    ap.add_argument('--nn-trials',  type=int, default=12)
    ap.add_argument('--hpo-sample', type=int, default=150_000)
    ap.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA available')
    args = ap.parse_args()

    if args.cpu:
        USE_GPU = False
        device = torch.device('cpu')
        print("Forced CPU mode.")

    train_all(csv_path=args.csv,
              n_xgb_trials=args.xgb_trials,
              n_lgb_trials=args.lgb_trials,
              n_nn_trials=args.nn_trials,
              hpo_sample=args.hpo_sample)
