"""
proto_multidisease_ews_v2.py

Robust multi-disease Early Warning System pipeline with:
 - adaptive per-disease outbreak percentile
 - cross-disease lag features
 - per-disease RF (with optional SMOTE resampling) + per-disease LR
 - per-disease PR->F1 threshold tuning and artifact saving

Usage:
    python proto_multidisease_ews_v2.py

Adjust CONFIG section to match your file paths and preferences.
"""

import os, json, warnings, math
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------- CONFIG ----------------
DATA_PATH = r"C:\Users\haris\OneDrive\Desktop\Project 1\DATASET_FINAL.csv"
OUT_DIR  = r"C:\Users\haris\OneDrive\Desktop\Project 1\outputs_multidisease_v2"
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL = "district"
TIME_COL = "week"   # either a date string or numeric period index
GEO_COLS = ["Latitude", "Longitude"]   # adjust if your file uses different names
CORE_WEATHER = ["week_avg_temp_C","week_avg_relative_humidity_percent","week_total_precip_mm","week_avg_wind_speed_ms"]

# Auto-percentile tuning
DEFAULT_OUTBREAK_PCT = 0.85
MIN_POS_RATE = 0.03    # target minimum positive rate after percentile tuning (e.g., 3% of rows)
MIN_PERCENTILE = 0.60  # lowest percentile allowed (don't go below this)

# SMOTE / resampling
USE_SMOTE = True        # requires imblearn
SMOTE_MIN_POS = 40      # if pos < this, use SMOTE for that disease for RF/LR training

# RF search / training
RF_SEARCH_ITERS = 20
RF_SEARCH_CV = 3
FINAL_RF_TREES = 600

# threshold tuning
MIN_THR = 0.20          # don't pick thresholds smaller than this for extremely rare diseases

# alert bins
ALERT_BINS = [-0.01, 0.3, 0.6, 1.01]
ALERT_LABELS = ["Safe","Watch","Alert"]
# ----------------------------------------

def clean_filename(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z_\-]', '_', str(s))

def safe_numeric(s):
    s = s.astype(str).str.strip()
    s = s.str.replace('%','', regex=False).str.replace(',','', regex=False)
    s = s.str.replace(r'[^0-9.\-eE]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

def safe_auc(y, p):
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y,p))
    except Exception:
        return np.nan

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Auto-detect disease columns (numeric excluding known)
exclude = set([ID_COL, TIME_COL] + CORE_WEATHER + GEO_COLS + ["year","month","key"])
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
disease_cols = [c for c in numeric_cols if c not in exclude]
if not disease_cols:
    if 'Dengue' in df.columns:
        disease_cols = ['Dengue']
    else:
        raise SystemExit("No disease columns detected. Update exclude or provide disease list.")
print("Detected diseases:", disease_cols)

# Coerce numeric for weather & geo & disease
for c in CORE_WEATHER + GEO_COLS:
    if c in df.columns:
        df[c] = safe_numeric(df[c])
for d in disease_cols:
    df[d] = safe_numeric(df[d]).fillna(0.0)

# TIME handling: try datetime then fallback to numeric
time_is_dt = False
try:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    if df[TIME_COL].notna().any():
        time_is_dt = True
        print("Parsed time column as datetime.")
    else:
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors='coerce')
        df[TIME_COL] = df[TIME_COL].astype('Int64')
except Exception:
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors='coerce')

df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

# fill weather/geo forward/back within district
for c in CORE_WEATHER + GEO_COLS:
    if c in df.columns:
        df[c] = df.groupby(ID_COL, group_keys=False)[c].transform(lambda x: x.ffill().bfill())

# week-of-year seasonality or period idx
if time_is_dt:
    df['woy'] = df[TIME_COL].dt.isocalendar().week.astype(int)
else:
    df['period_idx'] = df.groupby(ID_COL).cumcount()+1
    df['woy'] = df['period_idx'].mod(52)+1
df['woy_sin'] = np.sin(2*np.pi * df['woy']/52.0)
df['woy_cos'] = np.cos(2*np.pi * df['woy']/52.0)

# roll disease and weather
ROLL_WINDOWS = [3,4,6]
for w in ROLL_WINDOWS:
    for d in disease_cols:
        df[f"{d}_roll{w}"] = df.groupby(ID_COL)[d].transform(lambda s: s.rolling(w, min_periods=1).mean())
    for wc in CORE_WEATHER:
        if wc in df.columns:
            df[f"{wc}_roll{w}"] = df.groupby(ID_COL)[wc].transform(lambda s: s.rolling(w, min_periods=1).mean())

# interactions
df['temp_x_humidity'] = df.get('week_avg_temp_C', 0.0) * df.get('week_avg_relative_humidity_percent', 0.0)
df['rain_x_temp'] = df.get('week_total_precip_mm', 0.0) * df.get('week_avg_temp_C', 0.0)
if 'week_total_precip_mm' in df.columns:
    df['precip_log1p'] = np.log1p(df['week_total_precip_mm'].astype(float))
else:
    df['precip_log1p'] = 0.0
if 'precip_log1p_roll3' in df.columns:
    df['humidity_x_rain_roll3'] = df.get('week_avg_relative_humidity_percent', 0.0) * df['precip_log1p_roll3']
else:
    df['humidity_x_rain_roll3'] = df.get('week_avg_relative_humidity_percent', 0.0) * df['precip_log1p']

# ---------------- Adaptive outbreak percentile per disease ----------------
print("Adaptive outbreak percentile selection (ensures minimum positive rate)...")
percentile_map = {}
global_row_count = len(df)
for d in disease_cols:
    pct = DEFAULT_OUTBREAK_PCT
    # progressively lower percentile until positive rate >= MIN_POS_RATE or min percentile reached
    while pct >= MIN_PERCENTILE:
        thr = df.groupby(ID_COL)[d].transform(lambda s: s.quantile(pct))
        outbreak_flag = (df[d] >= thr).astype(int)
        pos_rate = outbreak_flag.mean()
        if pos_rate >= MIN_POS_RATE or pct == MIN_PERCENTILE:
            percentile_map[d] = pct
            print(f"  {d} -> percentile {pct:.2f}, positive rate {pos_rate:.4f}")
            break
        pct -= 0.05
    else:
        percentile_map[d] = MIN_PERCENTILE
print("Percentile map:", percentile_map)

# compute outbreak columns using final percentiles
for d in disease_cols:
    q = percentile_map.get(d, DEFAULT_OUTBREAK_PCT)
    thr = df.groupby(ID_COL)[d].transform(lambda s: s.quantile(q))
    df[f"{d}_outbreak"] = (df[d] >= thr).astype(int)

# create next-period target
for d in disease_cols:
    df[f"{d}_outbreak_next"] = df.groupby(ID_COL)[f"{d}_outbreak"].shift(-1)

# make modeling frame: drop rows missing any target
target_cols = [f"{d}_outbreak_next" for d in disease_cols]
df_model = df.dropna(subset=target_cols).reset_index(drop=True)

# ---------------- Cross-disease lag features (adds predictive signal) ----------------
print("Adding cross-disease lag features (lag-1 of other diseases)...")
for d1 in disease_cols:
    for d2 in disease_cols:
        if d1 == d2: 
            continue
        colname = f"{d1}_lag1_{clean_filename(d2)}"
        df_model[colname] = df_model.groupby(ID_COL)[d2].shift(1)

# ---------------- Lags for own disease and weather ----------------
LAG_PERIODS = [1,2,3]
feature_lags = []
for lag in LAG_PERIODS:
    for d in disease_cols:
        name = f"{d}_lag_{lag}"
        df_model[name] = df_model.groupby(ID_COL)[d].shift(lag)
        feature_lags.append(name)
    for wc in CORE_WEATHER + ['precip_log1p']:
        if wc in df_model.columns:
            name = f"{wc}_lag_{lag}"
            df_model[name] = df_model.groupby(ID_COL)[wc].shift(lag)
            feature_lags.append(name)

# drop any rows with missing features (safe)
df_model = df_model.dropna().reset_index(drop=True)

# build features list
roll_cols = [c for c in df_model.columns if '_roll' in c]
cyc_cols = ['woy_sin','woy_cos']
interaction_cols = [c for c in ['temp_x_humidity','rain_x_temp','humidity_x_rain_roll3'] if c in df_model.columns]
# collect names of cross-disease lag features we added earlier
cross_disease_cols = []
for d1 in disease_cols:
    for d2 in disease_cols:
        if d1 == d2:
            continue
        cname = f"{d1}_lag1_{clean_filename(d2)}"
        if cname in df_model.columns:
            cross_disease_cols.append(cname)
# now cross_disease_cols is a proper list of the created columns (possibly empty)

geo_cols = [c for c in GEO_COLS if c in df_model.columns]
feature_list = feature_lags + CORE_WEATHER + roll_cols + cyc_cols + interaction_cols + cross_disease_cols + geo_cols
feature_list = [c for c in feature_list if c in df_model.columns]

# X, Y
X = df_model[feature_list].copy()
Y = df_model[[f"{d}_outbreak_next" for d in disease_cols]].astype(int).copy()

# time split by index (80/20)
split_idx = int(len(df_model)*0.8)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
Y_train, Y_test = Y.iloc[:split_idx].copy(), Y.iloc[split_idx:].copy()

print("Shapes -> X_train:", X_train.shape, "Y_train:", Y_train.shape, "X_test:", X_test.shape, "Y_test:", Y_test.shape)

# scale for LR
scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler_v2.joblib"))

# ---------------- RF base param selection (heuristic on representative disease) ----------------
print("Heuristic RF base param tuning on the disease with most positives...")
pos_counts = Y_train.sum().sort_values(ascending=False)
rep_d = pos_counts.index[0] if len(pos_counts)>0 else disease_cols[0]
print("Representative disease:", rep_d, "positives:", int(pos_counts.loc[rep_d]))
base_rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
param_dist = {
    'n_estimators': [100,200,300],
    'max_depth': [8,12,16,None],
    'min_samples_split': [2,4,8],
    'min_samples_leaf': [1,2,4],
    'max_features': ['sqrt','log2',0.5]
}
tscv = TimeSeriesSplit(n_splits=RF_SEARCH_CV)
try:
    rs = RandomizedSearchCV(base_rf, param_dist, n_iter=RF_SEARCH_ITERS, cv=tscv, scoring='f1', n_jobs=-1, verbose=1, random_state=42)
    # target y for rep disease
    y_rep = Y_train[rep_d].values
    if len(np.unique(y_rep)) > 1:
        rs.fit(X_train, y_rep)
        best_base_params = rs.best_params_
        pd.DataFrame(rs.cv_results_).to_csv(os.path.join(OUT_DIR, "rf_base_rs_cv_results.csv"), index=False)
        print("Found base params:", best_base_params)
    else:
        best_base_params = {'n_estimators':200, 'max_depth':12, 'min_samples_split':2, 'min_samples_leaf':2, 'max_features':'sqrt'}
        print("Representative disease single-class; using default base params:", best_base_params)
except Exception as e:
    print("RandomizedSearchCV failed; using defaults. Error:", e)
    best_base_params = {'n_estimators':200, 'max_depth':12, 'min_samples_split':2, 'min_samples_leaf':2, 'max_features':'sqrt'}

# prepare per-disease RF training with optional SMOTE
print("Training per-disease RandomForest models (SMOTE where needed)...")
rf_models = {}
smote_avail = False
if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE
        smote_avail = True
        print("SMOTE available.")
    except Exception:
        print("SMOTE requested but imblearn not installed; continuing without SMOTE.")

rf_params = {k:v for k,v in best_base_params.items() if not k.startswith('estimator__')}
rf_params['n_estimators'] = FINAL_RF_TREES
rf_params['random_state'] = 42
rf_params['class_weight'] = 'balanced'
rf_params['n_jobs'] = -1

for i,d in enumerate(disease_cols):
    y_train = Y_train.iloc[:, i].values
    pos = int(y_train.sum())
    print(f"RF train for {d}: positives={pos}, total={len(y_train)}")
    if smote_avail and pos < SMOTE_MIN_POS and pos > 1:
        sm = SMOTE(random_state=42)
        Xr, yr = sm.fit_resample(X_train, y_train)
        rf = RandomForestClassifier(**rf_params)
        rf.fit(Xr, yr)
        print(f"  SMOTE applied for {d} -> resampled to {Xr.shape}")
    else:
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
    rf_models[d] = rf

joblib.dump(rf_models, os.path.join(OUT_DIR, "rf_models_per_disease_v2.joblib"))
print("Saved per-disease RF models.")

# ---------------- Per-disease Logistic Regression models ----------------
print("Training per-disease LogisticRegression models...")
lr_models = {}
for i,d in enumerate(disease_cols):
    y_train = Y_train.iloc[:, i].values
    pos = int(y_train.sum())
    print(f"LR train for {d}: positives={pos}")
    if smote_avail and pos < SMOTE_MIN_POS and pos > 1:
        sm = SMOTE(random_state=42)
        Xr, yr = sm.fit_resample(X_train_s, y_train)
        lr = LogisticRegression(C=1.0, penalty='l2', solver='saga', max_iter=2000, class_weight=None, n_jobs=-1)
        lr.fit(Xr, yr)
        print(f"  SMOTE used for LR {d}")
    else:
        lr = LogisticRegression(C=1.0, penalty='l2', solver='saga', max_iter=2000, class_weight='balanced', n_jobs=-1)
        lr.fit(X_train_s, y_train)
    lr_models[d] = lr

joblib.dump(lr_models, os.path.join(OUT_DIR, "lr_models_per_disease_v2.joblib"))
print("Saved per-disease LR models.")

# ---------------- Predictions and threshold tuning ----------------
print("Predicting test set and tuning thresholds (PR -> F1 max)...")
probs_rf = np.zeros((len(X_test), len(disease_cols)), dtype=float)
probs_lr = np.zeros_like(probs_rf)
for i,d in enumerate(disease_cols):
    rf = rf_models[d]
    try:
        probs_rf[:, i] = rf.predict_proba(X_test)[:, 1]
    except Exception:
        probs_rf[:, i] = rf.predict(X_test).astype(float)
    lr = lr_models[d]
    try:
        probs_lr[:, i] = lr.predict_proba(X_test_s)[:, 1]
    except Exception:
        probs_lr[:, i] = lr.predict(X_test_s).astype(float)

best_thresholds = {}
metrics_rows = []

for i,d in enumerate(disease_cols):
    y_true = Y_test.iloc[:, i].values
    p_rf = probs_rf[:, i]
    p_lr = probs_lr[:, i]

    # PR -> F1 max for RF
    try:
        prec, rec, thr = precision_recall_curve(y_true, p_rf)
        if len(thr) == 0:
            t_star = 0.5
        else:
            f1s = (2*prec*rec)/(prec+rec+1e-12)
            # f1s length = len(prec); thr length = len(prec)-1
            if len(f1s) > 1:
                idx = np.nanargmax(f1s[:-1])
                t_star = float(thr[idx]) if idx < len(thr) else float(thr[-1])
            else:
                t_star = float(thr[0]) if len(thr)>0 else 0.5
    except Exception:
        t_star = 0.5

    # enforce minimum threshold for very rare diseases
    pos_rate = Y_train.iloc[:, i].mean()
    if pos_rate < 0.01:
        t_star = max(t_star, MIN_THR)

    best_thresholds[d] = float(t_star)

    # compute metrics for RF and LR using thresholds
    yhat_rf = (p_rf >= t_star).astype(int)
    yhat_lr = (p_lr >= 0.5).astype(int)

    def safe(val):
        try: return float(val)
        except: return float('nan')

    def safe_auc_func(y,p): 
        try:
            return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float('nan')
        except:
            return float('nan')

    row_rf = dict(
        disease=d, model='RF',
        roc_auc=safe_auc_func(y_true, p_rf),
        ap= safe(average_precision_score(y_true, p_rf)) if len(np.unique(y_true))>1 else float('nan'),
        accuracy=safe(accuracy_score(y_true, yhat_rf)),
        precision=safe(precision_score(y_true, yhat_rf, zero_division=0)),
        recall=safe(recall_score(y_true, yhat_rf, zero_division=0)),
        f1=safe(f1_score(y_true, yhat_rf, zero_division=0)),
        thr=best_thresholds[d]
    )
    row_lr = dict(
        disease=d, model='LR',
        roc_auc=safe_auc_func(y_true, p_lr),
        ap= safe(average_precision_score(y_true, p_lr)) if len(np.unique(y_true))>1 else float('nan'),
        accuracy=safe(accuracy_score(y_true, yhat_lr)),
        precision=safe(precision_score(y_true, yhat_lr, zero_division=0)),
        recall=safe(recall_score(y_true, yhat_lr, zero_division=0)),
        f1=safe(f1_score(y_true, yhat_lr, zero_division=0)),
        thr=0.5
    )
    metrics_rows.extend([row_rf, row_lr])

    # safe plots: only if there is positive and negative class
    if len(np.unique(y_true)) > 1:
        try:
            fpr, tpr, _ = roc_curve(y_true, p_rf)
            fig = plt.figure(figsize=(6,5)); plt.plot(fpr,tpr,label=f"AUC={row_rf['roc_auc']:.3f}"); plt.plot([0,1],[0,1],'k--'); plt.title(f"ROC - {d} (RF)"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
            fig.savefig(os.path.join(OUT_DIR, f"roc_rf_{clean_filename(d)}.png"), bbox_inches='tight', dpi=150); plt.close(fig)
            prec, rec, _ = precision_recall_curve(y_true, p_rf)
            fig = plt.figure(figsize=(6,5)); plt.plot(rec,prec); plt.title(f"PR - {d} (RF)"); plt.xlabel("Recall"); plt.ylabel("Precision")
            fig.savefig(os.path.join(OUT_DIR, f"pr_rf_{clean_filename(d)}.png"), bbox_inches='tight', dpi=150); plt.close(fig)
        except Exception as e:
            print("Plotting skipped for",d, "error:",e)
    else:
        print(f"Skipping ROC/PR for {d} (single-class in test).")

# save metrics and thresholds
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_per_disease_v2.csv"), index=False)
with open(os.path.join(OUT_DIR, "best_thresholds_v2.json"), "w") as f:
    json.dump(best_thresholds, f, indent=2)
print("Saved metrics and thresholds.")

# save predictions for mapping & analysis (test period)
pred_df = df_model.iloc[split_idx:].reset_index(drop=True)[[ID_COL, TIME_COL] + GEO_COLS].copy()
for i,d in enumerate(disease_cols):
    pred_df[f"{d}_proba_rf"] = probs_rf[:, i]
    pred_df[f"{d}_pred_rf"] = (probs_rf[:, i] >= best_thresholds[d]).astype(int)
    pred_df[f"{d}_proba_lr"] = probs_lr[:, i]
    pred_df[f"{d}_pred_lr"] = (probs_lr[:, i] >= 0.5).astype(int)

pred_df.to_csv(os.path.join(OUT_DIR, "predictions_multidisease_v2.csv"), index=False)
print("Saved predictions CSV.")

# last-period alerts per district using tuned thresholds
last = pred_df.groupby(ID_COL).last().reset_index()
for d in disease_cols:
    last[f"{d}_alert"] = pd.cut(last[f"{d}_proba_rf"], bins=ALERT_BINS, labels=ALERT_LABELS)
last.to_csv(os.path.join(OUT_DIR, "last_period_alerts_multidisease_v2.csv"), index=False)
print("Saved last period alerts.")

# save models & metadata
joblib.dump(rf_models, os.path.join(OUT_DIR, "rf_models_per_disease_v2.joblib"))
joblib.dump(lr_models, os.path.join(OUT_DIR, "lr_models_per_disease_v2.joblib"))
with open(os.path.join(OUT_DIR, "proto_multidisease_v2_metadata.json"), "w") as f:
    json.dump({
        "feature_list": feature_list,
        "disease_cols": disease_cols,
        "percentile_map": percentile_map,
        "best_thresholds": best_thresholds,
        "rf_params": rf_params,
    }, f, indent=2)

print("All done. Artifacts saved to:", OUT_DIR)

# Helpful diagnostics summary to print
print("\n=== Diagnostics summary ===")
target_cols = Y_train.columns.tolist()   # e.g. ['Acute Diarrhoeal Disease_outbreak_next', ...]
for i, d in enumerate(disease_cols):
    target_col = target_cols[i]
    pos_rate_train = Y_train[target_col].mean()
    print(f"{d}: train positive rate={pos_rate_train:.4f}, chosen_percentile={percentile_map[d]:.2f}, tuned_threshold={best_thresholds.get(d, None)}")
print("===========================")
