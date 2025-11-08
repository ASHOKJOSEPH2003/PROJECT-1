import os, sys, warnings, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------- CONFIG ----------
DATA_PATH = r"C:\Users\ashok\Downloads\PROJECT 1\DATASET_FINAL.csv"
OUT_DIR   = r"C:\Users\ashok\Downloads\PROJECT 1\OUTPUT_MULTIDISEASE_EWS_XGB"
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL = "district"
TIME_COL = "week"
GEO_COLS = ["Latitude","Longitude"]
CORE_WEATHER = ["week_avg_temp_C","week_avg_relative_humidity_percent",
                "week_total_precip_mm","week_avg_wind_speed_ms"]

OUTBREAK_PERCENTILE = 0.85
LAG_PERIODS = [1,2,3]
ROLL_WINDOWS = [3,4,6]
XGB_N_ITER = 15
XGB_TSPLITS = 3
# -------------------------------

def safe_numeric(s):
    s = s.astype(str).str.strip()
    s = s.str.replace('%','', regex=False).str.replace(',','', regex=False)
    s = s.str.replace(r'[^0-9.\-eE]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except:
        return np.nan

# --- 1. Data Loading ---
print("Loading:", DATA_PATH)
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    sys.exit(1)

exclude = set([ID_COL, TIME_COL] + CORE_WEATHER + GEO_COLS + ["year","month"])
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
disease_cols = [c for c in numeric_cols if c not in exclude]

if not disease_cols:
    print("No disease-like numeric columns auto-detected. Exiting.")
    sys.exit(1)

print("Detected disease columns:", disease_cols)

for c in disease_cols + CORE_WEATHER + GEO_COLS:
    if c in df.columns:
        df[c] = safe_numeric(df[c])

try:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df['time_is_datetime'] = True
except Exception:
    df[TIME_COL] = df[TIME_COL].astype(int)
    df['time_is_datetime'] = False

df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
df[CORE_WEATHER] = df[CORE_WEATHER].fillna(method='ffill').fillna(method='bfill')
df[GEO_COLS] = df[GEO_COLS].fillna(method='ffill').fillna(method='bfill')
for d in disease_cols:
    df[d] = df[d].fillna(0)

# --- 2. Feature Engineering ---
if df['time_is_datetime'].any():
    df['month'] = df[TIME_COL].dt.month
else:
    df['period_idx'] = df.groupby(ID_COL).cumcount()+1
    df['month'] = ((df['period_idx']-1)//4+1).clip(1,12)

df['month_sin'] = np.sin(2*np.pi*df['month']/12.0)
df['month_cos'] = np.cos(2*np.pi*df['month']/12.0)
df['precip_log1p'] = np.log1p(df.get('week_total_precip_mm', 0.0).astype(float))

for w in ROLL_WINDOWS:
    for d in disease_cols:
        df[f"{d}_roll{w}"] = df.groupby(ID_COL)[d].transform(lambda x: x.rolling(w, min_periods=1).mean())
    for wc in CORE_WEATHER + ['precip_log1p']:
        if wc in df.columns:
            df[f"{wc}_roll{w}"] = df.groupby(ID_COL)[wc].transform(lambda s: s.rolling(w, min_periods=1).mean())

temp_c = df.get('week_avg_temp_C', 0.0)
humidity_p = df.get('week_avg_relative_humidity_percent', 0.0)
precip_mm = df.get('week_total_precip_mm', 0.0)
df['temp_x_humidity'] = temp_c * humidity_p
df['rain_x_temp'] = precip_mm * temp_c
df['humidity_x_rain_roll3'] = df.get('precip_log1p_roll3', df.get('precip_log1p', 0.0)) * humidity_p

# --- 3. Target Creation ---
for d in disease_cols:
    thr = df.groupby(ID_COL)[d].transform(lambda x: x.quantile(OUTBREAK_PERCENTILE))
    df[f"{d}_outbreak"] = (df[d] >= thr).astype(int)
    df[f"{d}_outbreak_next"] = df.groupby(ID_COL)[f"{d}_outbreak"].shift(-1)

target_cols = [f"{d}_outbreak_next" for d in disease_cols]
df_model = df.dropna(subset=target_cols).copy()

feature_lags = []
for lag in LAG_PERIODS:
    for d in disease_cols:
        name = f"{d}_lag_{lag}"
        df_model[name] = df_model.groupby(ID_COL)[d].shift(lag)
        feature_lags.append(name)
    for wc in CORE_WEATHER + ['precip_log1p']:
        if wc in df_model.columns:
            c = f"{wc}_lag_{lag}"
            df_model[c] = df_model.groupby(ID_COL)[wc].shift(lag)
            feature_lags.append(c)

df_model = df_model.dropna().reset_index(drop=True)

roll_cols = [c for c in df_model.columns if '_roll' in c]
cyc_cols = ['month_sin','month_cos']
interaction_cols = ['temp_x_humidity','rain_x_temp','humidity_x_rain_roll3']
feature_list = feature_lags + CORE_WEATHER + roll_cols + cyc_cols + interaction_cols + GEO_COLS
feature_list = [c for c in feature_list if c in df_model.columns]

X = df_model[feature_list].copy()
Y = df_model[[f"{d}_outbreak_next" for d in disease_cols]].astype(int).copy()

split_idx = int(len(df_model)*0.8)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
Y_train, Y_test = Y.iloc[:split_idx].copy(), Y.iloc[split_idx:].copy()

valid_targets = [col for col in Y_train.columns if Y_train[col].nunique() > 1]
Y_train = Y_train[valid_targets]
Y_test = Y_test[valid_targets]
disease_cols_final = [d.replace('_outbreak_next','') for d in valid_targets]

X_train_np, Y_train_np = X_train.values, Y_train.values
X_test_np, Y_test_np = X_test.values, Y_test.values

# --- 4. CHECKPOINTS ---
xgb_model_path = os.path.join(OUT_DIR, "xgb_multi_best.joblib")
quantum_weights_path = os.path.join(OUT_DIR, "quantum_weights.npy")
quantum_scaler_path = os.path.join(OUT_DIR, "quantum_scaler.joblib")

xgb_trained = os.path.exists(xgb_model_path)
quantum_trained = os.path.exists(quantum_weights_path) and os.path.exists(quantum_scaler_path)

# --- 5. XGBoost Training or Load ---
if xgb_trained:
    print("✅ Found saved XGBoost model. Loading...")
    best_multi_xgb = joblib.load(xgb_model_path)
else:
    print("⚙️ Training XGBoost...")
    xgb_base = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='gpu_hist',       # 🚀 GPU-accelerated tree building
        predictor='gpu_predictor',    # 🚀 GPU-based prediction
        gpu_id=0     
    )
    multi_xgb = MultiOutputClassifier(xgb_base, n_jobs=-1)
    param_dist_xgb = {
        'estimator__n_estimators': [200, 300, 500],
        'estimator__max_depth': [4, 6, 8],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.7, 0.8, 1.0],
        'estimator__colsample_bytree': [0.7, 0.8, 1.0]
    }
    rand_search_xgb = RandomizedSearchCV(
        estimator=multi_xgb,
        param_distributions=param_dist_xgb,
        n_iter=XGB_N_ITER,
        cv=TimeSeriesSplit(n_splits=XGB_TSPLITS),
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    rand_search_xgb.fit(X_train_np, Y_train_np)
    best_multi_xgb = rand_search_xgb.best_estimator_
    joblib.dump(best_multi_xgb, xgb_model_path)
    print("✅ Saved trained XGBoost model.")

# --- 6. Quantum Model (with GPU Option) ---
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import MinMaxScaler
from pennylane.optimize import AdamOptimizer

target_disease = disease_cols_final[6]
target_col = f"{target_disease}_outbreak_next"

quantum_features = ["week_avg_temp_C", "week_avg_relative_humidity_percent",
                    "week_total_precip_mm", "week_avg_wind_speed_ms"]

X_train_quantum = df_model.loc[X_train.index, quantum_features].copy()
X_test_quantum = df_model.loc[X_test.index, quantum_features].copy()
Y_train_quantum = Y_train[target_col].copy()
Y_test_quantum = Y_test[target_col].copy()

X_train_quantum.fillna(0, inplace=True)
X_test_quantum.fillna(0, inplace=True)

if quantum_trained:
    print("✅ Found saved Quantum model. Loading...")
    weights = pnp.array(np.load(quantum_weights_path), requires_grad=False)
    scaler_quantum = joblib.load(quantum_scaler_path)
else:
    print("⚙️ Training Quantum model...")
    scaler_quantum = MinMaxScaler((0, np.pi))
    X_train_scaled = scaler_quantum.fit_transform(X_train_quantum)
    X_test_scaled = scaler_quantum.transform(X_test_quantum)
    n_qubits = X_train_scaled.shape[1]

    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        print("🟢 Using GPU backend for quantum simulation.")
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        print("⚠️ GPU not available, using CPU (default.qubit).")

    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for i in range(n_qubits):
            qml.Rot(weights[i][0], weights[i][1], weights[i][2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = (n_qubits, 3)
    labels_train = pnp.array(Y_train_quantum.values * 2 - 1)
    features_train = pnp.array(X_train_scaled)
    np.random.seed(42)
    weights = pnp.array(np.random.normal(0, np.pi, weight_shapes), requires_grad=True)

    def loss(weights, features, labels):
        preds = []
        for f in features:
            circuit_out = quantum_circuit(f, weights)
            pred = pnp.mean(pnp.array(circuit_out))
            preds.append(pred)
        preds = pnp.array(preds)
        return pnp.mean((preds - labels) ** 2)

    opt = AdamOptimizer(stepsize=0.1)
    for epoch in range(50):
        weights, curr_loss = opt.step_and_cost(lambda w: loss(w, features_train, labels_train), weights)
        if epoch % 10 == 0:
            print(f"Quantum training epoch {epoch} - Loss: {curr_loss}")

    np.save(quantum_weights_path, weights)
    joblib.dump(scaler_quantum, quantum_scaler_path)
    print("✅ Saved quantum model weights and scaler.")

# --- 7. Quantum Evaluation ---
X_test_scaled = scaler_quantum.transform(X_test_quantum)
features_test = pnp.array(X_test_scaled)

@qml.qnode(qml.device("default.qubit", wires=X_test_scaled.shape[1]))
def quantum_circuit(inputs, weights):
    for i in range(len(inputs)):
        qml.RY(inputs[i], wires=i)
    for i in range(len(inputs)):
        qml.Rot(weights[i][0], weights[i][1], weights[i][2], wires=i)
    for i in range(len(inputs)-1):
        qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

def predict(weights, features):
    preds = []
    for f in features:
        measurement = quantum_circuit(f, weights)
        pred = pnp.mean(pnp.array(measurement))
        preds.append(pred)
    preds = np.array(preds)
    return (preds > 0).astype(int)

y_pred_quantum = predict(weights, features_test)
acc_quantum = accuracy_score(Y_test_quantum, y_pred_quantum)
f1_quantum = f1_score(Y_test_quantum, y_pred_quantum)

print(f"\nQuantum classifier accuracy for {target_disease}: {acc_quantum:.3f}")
print(f"Quantum classifier F1 score for {target_disease}: {f1_quantum:.3f}")

print("\n✅ Script completed. All models and results saved in:")
print(OUT_DIR)
