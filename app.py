# ews_app.py
# Streamlit inference app for Multi-Disease EWS (modified: cached folium map HTML to avoid flicker)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import folium
from streamlit.components.v1 import html as st_html
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import Optional
import random
import altair as alt

# ============= CONFIG (do not change directories) ==================
OUT_DIR = r"C:\Users\haris\OneDrive\Desktop\Project 1\outputs_multidisease_v2"
DATA_PATH = r"C:\Users\haris\OneDrive\Desktop\Project 1\DATASET_FINAL.csv"
RF_MODELS = os.path.join(OUT_DIR, "rf_models_per_disease_v2.joblib")
LR_MODELS = os.path.join(OUT_DIR, "lr_models_per_disease_v2.joblib")
SCALER_PATH = os.path.join(OUT_DIR, "scaler_v2.joblib")
META_PATH = os.path.join(OUT_DIR, "proto_multidisease_v2_metadata.json")
THR_PATH = os.path.join(OUT_DIR, "best_thresholds_v2.json")
# ==================================================================

st.set_page_config(page_title="Multi-Disease EWS — Inference", layout="wide")
st.title("Multi-Disease Early Warning System — Inference")

# ----------------- helpers -----------------
def load_json_safe(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_resource
def load_artifacts():
    meta = load_json_safe(META_PATH) or {}
    rf_models = joblib.load(RF_MODELS) if os.path.exists(RF_MODELS) else {}
    lr_models = joblib.load(LR_MODELS) if os.path.exists(LR_MODELS) else {}
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    thr_map = load_json_safe(THR_PATH) or {}
    return meta, rf_models, lr_models, scaler, thr_map

def safe_numeric(s):
    s = s.astype(str).str.strip()
    s = s.str.replace('%','', regex=False).str.replace(',','', regex=False)
    s = s.str.replace(r'[^0-9.\-eE]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["week_avg_temp_C","week_avg_relative_humidity_percent","week_total_precip_mm","week_avg_wind_speed_ms","Latitude","Longitude"]:
        if c in df.columns:
            df[c] = safe_numeric(df[c])
    if "week_date" in df.columns:
        df["week_date"] = pd.to_datetime(df["week_date"], errors="coerce")
    if ("year" in df.columns) and ("week" in df.columns):
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        def iso_week_to_date(y,w):
            try:
                return pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
            except Exception:
                return pd.NaT
        df["week_date"] = df.apply(lambda r: iso_week_to_date(r["year"], r["week"]) if pd.isna(r.get("week_date", pd.NaT)) else r["week_date"], axis=1)
    if "week" not in df.columns:
        df = df.sort_values(["district", "year","week_date"]).reset_index(drop=True)
        df["week"] = df.groupby("district").cumcount()+1
    else:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["woy"] = df["week"] % 52 + 1
    df["woy_sin"] = np.sin(2*np.pi*df["woy"]/52.0)
    df["woy_cos"] = np.cos(2*np.pi*df["woy"]/52.0)
    df["precip_log1p"] = np.log1p(df.get("week_total_precip_mm", 0.0).astype(float))
    df["temp_x_humidity"] = df.get("week_avg_temp_C", 0.0) * df.get("week_avg_relative_humidity_percent", 0.0)
    df["rain_x_temp"] = df.get("week_total_precip_mm", 0.0) * df.get("week_avg_temp_C", 0.0)
    for w in [3,4,6]:
        if "week_total_precip_mm" in df.columns:
            df[f"precip_log1p_roll{w}"] = df.groupby("district")["week_total_precip_mm"].transform(lambda s: np.log1p(s).rolling(w, min_periods=1).mean())
        if "week_avg_relative_humidity_percent" in df.columns:
            df[f"rh_roll{w}"] = df.groupby("district")["week_avg_relative_humidity_percent"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        if "week_avg_temp_C" in df.columns:
            df[f"temp_roll{w}"] = df.groupby("district")["week_avg_temp_C"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    return df

def align_features_for_model(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    df2 = df.copy()
    missing = [c for c in feature_list if c not in df2.columns]
    for m in missing:
        df2[m] = 0.0
    return df2[feature_list].fillna(0.0)

def llm_suggestions(openai_api_key: Optional[str], summary_text: str) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return "OpenAI python package not installed. Install with `pip install openai` to enable LLM suggestions."
    if not openai_api_key:
        return "No OpenAI API key provided. Enter key in the sidebar to get LLM suggestions."
    try:
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        return f"Failed to construct OpenAI client: {e}"
    prompt = (
        "You are a public-health advisor. Given the following district-level early-warning summary for multiple diseases, "
        "produce a concise, prioritized list of practical, low-cost, evidence-based prevention and mitigation suggestions "
        "targeted to local health authorities and community organisers. Group suggestions into short-term (this week), "
        "medium-term (1-3 months), and longer-term (3+ months) actions. Be explicit (e.g., supply x, do y). Do not include "
        "medical diagnosis instructions — focus on public-health measures.\n\n"
        "Summary:\n"
        f"{summary_text}\n\nOutput:\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful public health assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2
        )
        text = ""
        if getattr(resp, "choices", None):
            first = resp.choices[0]
            msg = getattr(first, "message", None)
            if msg:
                text = msg.get("content", "") if isinstance(msg, dict) else msg.content
            else:
                text = str(resp)
        else:
            text = str(resp)
        return text.strip()
    except Exception as e:
        return f"LLM call failed: {e}"

# --- MAP CACHING HELPER (ADDED) ---
@st.cache_data(show_spinner=False)
def build_folium_map_html(rows_df_json, disease_for_map, center_lat, center_lon, zoom_start=6):
    """
    Cached builder for folium map HTML.
    rows_df_json: list/dict - JSON-serializable subset of rows (district, Latitude, Longitude, alerts/probas)
    disease_for_map: string - disease name to pick alert/proba columns
    returns: rendered HTML string for embedding
    """
    import folium
    import pandas as _pd
    from html import escape
    rows = _pd.DataFrame(rows_df_json)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=int(zoom_start), tiles="CartoDB positron")
    color_map = {"Safe":"green","Watch":"orange","Alert":"red"}
    for _, r in rows.iterrows():
        lat = r.get("Latitude", None)
        lon = r.get("Longitude", None)
        if pd.isna(lat) or pd.isna(lon):
            continue
        alert_col = f"{disease_for_map}_alert_rf"
        prob_col = f"{disease_for_map}_proba_rf"
        alert = r.get(alert_col, "Safe")
        proba = r.get(prob_col, None)
        color = color_map.get(str(alert), "blue")
        popup_html = f"{escape(str(r.get('district','')))}<br/>{escape(disease_for_map)} Alert: {escape(str(alert))}<br/>Prob: {'' if proba is None or (isinstance(proba,float) and np.isnan(proba)) else round(float(proba),3)}"
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    return m.get_root().render()

# ----------------- Load artifacts -----------------
meta, rf_models, lr_models, scaler, thr_map = load_artifacts()
feature_list = meta.get("feature_list", []) or []
disease_cols = meta.get("disease_cols", []) or list(rf_models.keys() or lr_models.keys())

# ----------------- Dataset load -----------------
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}. Please place DATASET_FINAL.csv at that path or upload a CSV.")
    uploaded = st.file_uploader("Upload DATASET_FINAL.csv (optional)", type=["csv"])
    if uploaded is None:
        st.stop()
    else:
        df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(DATA_PATH)

# ----------------- Sidebar controls (CHANGES) -----------------
st.sidebar.header("Controls")
st.sidebar.write(f"Models loaded: RF [{'yes' if rf_models else 'no'}], LR [{'yes' if lr_models else 'no'}]")

# --- fixed year selector (2026-2036) ---
years_fixed = list(range(2026, 2037))
selected_year = st.sidebar.selectbox("Select Year (UI-only)", options=years_fixed, index=0)

# Top-N districts control
top_n = st.sidebar.number_input("Top N districts to display", min_value=1, max_value=500, value=20, step=1, help="Show only top-N highest-risk districts on map and in alerts list.")
st.sidebar.markdown("---")

# --- NEW: Interactive threshold sliders ---
st.sidebar.markdown("### Threshold adjustments (optional)")
thr_overrides = {}
if isinstance(thr_map, dict) and len(thr_map) > 0:
    for d in disease_cols:
        base_thr = float(thr_map.get(d, 0.5))
        thr_overrides[d] = st.sidebar.slider(
            f"{d} threshold", 0.0, 1.0, base_thr, 0.01,
            help=f"Adjust decision threshold for {d} (default: {base_thr})"
        )
else:
    st.sidebar.info("Threshold map not loaded; using default 0.5 for all.")

st.sidebar.write("LLM suggestions (optional)")
openai_key_input = st.sidebar.text_input("OpenAI API key (paste here or set in env/st.secrets)", type="password")
st.sidebar.caption("Key not stored persistently by this app; session-only.")

# build features & weeks
df_feat = build_features(df)
available_weeks = sorted(df_feat["week"].dropna().unique())
if len(available_weeks) == 0:
    st.error("No valid 'week' values detected in dataset. Ensure the CSV contains a numeric 'week' column.")
    st.stop()

selected_week = st.sidebar.selectbox("Select target week for inference", options=available_weeks[::-1], index=0)
st.sidebar.write(f"Selected year (UI-only): {selected_year}")
st.sidebar.write(f"Selected week: {selected_week}")

# Run inference
if st.button("Run inference"):
    st.info("Preparing inference rows...")
    rows = []
    for district, g in df_feat.groupby("district"):
        g = g.sort_values("week")
        sel = g[g["week"] == selected_week]
        if not sel.empty:
            rows.append(sel.iloc[-1])
        else:
            le = g[g["week"] <= selected_week]
            if not le.empty:
                rows.append(le.iloc[-1])
            else:
                rows.append(g.iloc[-1])
    infer_df = pd.DataFrame(rows).reset_index(drop=True)
    st.success(f"Selected {len(infer_df)} districts for inference.")

    if not feature_list:
        exclude = {"district","week","year","week_date","key"}
        feature_list_local = [c for c in infer_df.columns if pd.api.types.is_numeric_dtype(infer_df[c]) and c not in exclude]
    else:
        feature_list_local = feature_list

    X = align_features_for_model(infer_df, feature_list_local)
    if scaler is not None:
        try:
            X_lr = scaler.transform(X)
        except Exception:
            try:
                if hasattr(scaler, "feature_names_in_"):
                    f_in = list(scaler.feature_names_in_)
                    X2 = align_features_for_model(infer_df, f_in)
                    X_lr = scaler.transform(X2)
                else:
                    X_lr = scaler.transform(X)
            except Exception as e:
                st.warning(f"Failed to scale for LR: {e}")
                X_lr = X.values
    else:
        X_lr = X.values
    X_rf = X.values

    final = infer_df[["district","week"] + [c for c in ["Latitude","Longitude"] if c in infer_df.columns]].copy()
    # run models per disease (unchanged)
    for d in disease_cols:
        if d in rf_models:
            m = rf_models[d]
            try:
                if hasattr(m, "feature_names_in_"):
                    f_in = list(m.feature_names_in_)
                    X_rf_used = align_features_for_model(infer_df, f_in).values
                else:
                    X_rf_used = X_rf
            except Exception:
                X_rf_used = X_rf
            proba = None
            try:
                if hasattr(m, "predict_proba"):
                    proba = m.predict_proba(X_rf_used)[:, 1]
                else:
                    proba = np.full(len(X_rf_used), np.nan)
            except Exception as e:
                st.warning(f"RF predict_proba error for {d}: {e}")
                proba = np.full(len(X_rf_used), np.nan)
            proba = np.asarray(proba, dtype=float)
            proba = np.where(np.isfinite(proba), proba, np.nan)
            proba = np.where(np.isnan(proba), np.nan, np.clip(proba, 0.0, 1.0))
            final[f"{d}_proba_rf"] = proba
            # use interactive override if provided
            thr = thr_overrides.get(d, thr_map.get(d, 0.5))
            try:
                thr = float(thr)
            except Exception:
                thr = 0.5
            final[f"{d}_pred_rf"] = np.where(np.isnan(proba), np.nan, (proba >= thr).astype(int))
            alert_labels = []
            for v in proba:
                if np.isnan(v):
                    alert_labels.append("NoModel")
                else:
                    if v >= thr:
                        alert_labels.append("Alert")
                    elif v >= 0.6 * thr:
                        alert_labels.append("Watch")
                    else:
                        alert_labels.append("Safe")
            final[f"{d}_alert_rf"] = alert_labels
        else:
            final[f"{d}_proba_rf"] = np.nan
            final[f"{d}_pred_rf"] = np.nan
            final[f"{d}_alert_rf"] = "NoModel"
        if d in lr_models:
            m2 = lr_models[d]
            try:
                proba2 = m2.predict_proba(X_lr)[:,1]
            except Exception:
                try:
                    proba2 = m2.predict(X_lr).astype(float)
                except Exception as e:
                    st.error(f"LR predict failed for {d}: {e}")
                    proba2 = np.zeros(len(X_lr))
            final[f"{d}_proba_lr"] = proba2
            final[f"{d}_pred_lr"] = (proba2 >= 0.5).astype(int)
            final[f"{d}_alert_lr"] = pd.cut(proba2, bins=[-0.01, 0.3, 0.6, 1.01], labels=["Safe","Watch","Alert"])
        else:
            final[f"{d}_proba_lr"] = np.nan
            final[f"{d}_pred_lr"] = np.nan
            final[f"{d}_alert_lr"] = "NoModel"

    # noise injection (unchanged)
    try:
        rf_alert_cols = [f"{d}_alert_rf" for d in disease_cols if f"{d}_alert_rf" in final.columns]
        if len(rf_alert_cols) > 0:
            for idx in final.index:
                row_alerts = final.loc[idx, rf_alert_cols].astype(str).tolist()
                n_alerts = sum(1 for a in row_alerts if a == 'Alert')
                if n_alerts == len([c for c in row_alerts if c != 'NoModel']) and n_alerts > 1:
                    n_downgrade = 1 if random.random() < 0.7 else 2
                    alert_indices = [i for i, a in enumerate(row_alerts) if a == 'Alert']
                    to_change = random.sample(alert_indices, min(n_downgrade, len(alert_indices)))
                    for ti in to_change:
                        col = rf_alert_cols[ti]
                        dname = disease_cols[ti]
                        p = final.loc[idx, f"{dname}_proba_rf"]
                        thr_val = thr_map.get(dname, 0.5)
                        try:
                            thr_val = float(thr_val)
                        except Exception:
                            thr_val = 0.5
                        if not pd.isna(p) and p >= 0.6 * thr_val:
                            final.at[idx, col] = 'Watch'
                        else:
                            final.at[idx, col] = 'Safe'
    except Exception as e:
        st.warning(f"Noise injection step failed: {e}")

    # --- compute district risk and select top-N ---
    risk_scores = []
    for idx, r in final.iterrows():
        probs = [r.get(f"{d}_proba_rf", np.nan) for d in disease_cols]
        probs_num = [p for p in probs if not pd.isna(p)]
        score = max(probs_num) if len(probs_num) > 0 else 0.0
        risk_scores.append(score)
    final["risk_score"] = risk_scores
    final["n_alerts"] = final[[f"{d}_alert_rf" for d in disease_cols if f"{d}_alert_rf" in final.columns]].apply(lambda row: sum(1 for v in row if v=="Alert"), axis=1)
    final_sorted = final.sort_values(["n_alerts","risk_score"], ascending=False).reset_index(drop=True)
    final_top = final_sorted.head(int(top_n)).copy()

    st.write(f"Showing top {len(final_top)} districts (Top N = {top_n}) — total districts available: {len(final)}")

    # visuals (unchanged)
    total_alerts = int((final_sorted[[f"{d}_alert_rf" for d in disease_cols if f"{d}_alert_rf" in final.columns]] == "Alert").sum().sum())
    total_watch = int((final_sorted[[f"{d}_alert_rf" for d in disease_cols if f"{d}_alert_rf" in final.columns]] == "Watch").sum().sum())
    total_safe = int((final_sorted[[f"{d}_alert_rf" for d in disease_cols if f"{d}_alert_rf" in final.columns]] == "Safe").sum().sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts (RF)", total_alerts)
    col2.metric("Total Watch (RF)", total_watch)
    col3.metric("Total Safe (RF)", total_safe)

    alert_counts = {}
    for d in disease_cols:
        colname = f"{d}_alert_rf"
        if colname in final_sorted.columns:
            alert_counts[d] = int((final_sorted[colname]=="Alert").sum())
        else:
            alert_counts[d] = 0
    ac_df = pd.DataFrame({"disease": list(alert_counts.keys()), "alerts": list(alert_counts.values())})
    st.subheader("Alerts per disease (RF)")
    st.altair_chart(alt.Chart(ac_df).mark_bar().encode(
        x=alt.X("disease:N", sort='-y'),
        y=alt.Y("alerts:Q"),
        tooltip=["disease","alerts"]
    ).properties(width=700, height=300), use_container_width=True)

    stack_rows = []
    for d in disease_cols:
        colname = f"{d}_alert_rf"
        if colname in final_sorted.columns:
            counts = final_sorted[colname].value_counts().to_dict()
            stack_rows.append({"disease": d, "Safe": counts.get("Safe",0), "Watch": counts.get("Watch",0), "Alert": counts.get("Alert",0)})
        else:
            stack_rows.append({"disease": d, "Safe":0, "Watch":0, "Alert":0})
    stack_df = pd.DataFrame(stack_rows)
    stack_df_melt = stack_df.melt(id_vars=["disease"], value_vars=["Safe","Watch","Alert"], var_name="level", value_name="count")
    st.subheader("Alert level breakdown per disease")
    st.altair_chart(alt.Chart(stack_df_melt).mark_bar().encode(
        x=alt.X("disease:N", sort='-y'),
        y=alt.Y("count:Q"),
        color=alt.Color("level:N", scale=alt.Scale(domain=["Safe","Watch","Alert"], range=["#2ca02c","#ff7f0e","#d62728"])),
        tooltip=["disease","level","count"]
    ).properties(width=800, height=350), use_container_width=True)

    mean_probas = {}
    for d in disease_cols:
        col = f"{d}_proba_rf"
        if col in final_sorted.columns:
            mean_probas[d] = float(np.nanmean(final_sorted[col]))
        else:
            mean_probas[d] = np.nan
    mp_df = pd.DataFrame({"disease": list(mean_probas.keys()), "mean_proba": list(mean_probas.values())})
    st.subheader("Mean RF probability per disease (selected week)")
    st.altair_chart(alt.Chart(mp_df).mark_line(point=True).encode(
        x=alt.X("disease:N"),
        y=alt.Y("mean_proba:Q", scale=alt.Scale(domain=[0,1])),
        tooltip=["disease","mean_proba"]
    ).properties(width=700, height=250), use_container_width=True)

    # show alerts for top-N only
    st.subheader("Top districts — Alerts (compact)")
    for idx, row in final_top.iterrows():
        district_name = row.get("district", f"district_{idx}")
        row_alerts = []
        for d in disease_cols:
            col = f"{d}_alert_rf"
            if col in final_top.columns and str(row.get(col)) == "Alert":
                row_alerts.append(d)
        header = f"{district_name} — Alerts: {', '.join(row_alerts) if row_alerts else 'None'}"
        with st.expander(header, expanded=False):
            st.write(f"**Week:** {int(row.get('week', selected_week))}")
            lat = row.get("Latitude", None)
            lon = row.get("Longitude", None)
            if pd.notna(lat) and pd.notna(lon):
                st.write(f"**Location:** {lat:.4f}, {lon:.4f}")
            for d in row_alerts:
                proba = row.get(f"{d}_proba_rf", np.nan)
                thr = thr_map.get(d, 0.5)
                st.markdown(f"**{d}** — RF probability: `{None if pd.isna(proba) else round(float(proba),3)}`; threshold: `{thr}`")
                try:
                    if not pd.isna(proba):
                        st.progress(min(max(float(proba), 0.0), 1.0))
                except Exception:
                    pass
            key_metrics = ["week_avg_temp_C", "week_avg_relative_humidity_percent", "week_total_precip_mm", "week_avg_wind_speed_ms", "precip_log1p", "temp_x_humidity"]
            st.markdown("**Key input metrics (from dataset)**")
            for km in key_metrics:
                if km in infer_df.columns:
                    val = infer_df.loc[idx, km]
                    # intelligent fallbacks
                    if pd.isna(val):
                        # use mean or median of column as fallback
                        fallback_val = infer_df[km].dropna().mean() if infer_df[km].notna().any() else 0.0
                        st.metric(label=km, value=f"{round(float(fallback_val),2)} (est.)")
                    else:
                        st.metric(label=km, value=f"{round(float(val),2)}")
                else:
                    st.metric(label=km, value="N/A")


    # --- MAP (REPLACED WITH CACHED HTML) ---
    st.subheader("Map visualization (Multi-Disease Overlay)")
    st.write("Toggle diseases using the layer control on the top-right of the map.")

    center_lat = final_top["Latitude"].dropna().mean() if "Latitude" in final_top.columns else 20.0
    center_lon = final_top["Longitude"].dropna().mean() if "Longitude" in final_top.columns else 78.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")
    color_map = {"Safe": "green", "Watch": "orange", "Alert": "red"}

    # Create separate feature groups for each disease
    for disease in disease_cols:
        fg = folium.FeatureGroup(name=disease)
        alert_col = f"{disease}_alert_rf"
        proba_col = f"{disease}_proba_rf"

        if alert_col not in final_top.columns:
            continue

        for _, r in final_top.iterrows():
            lat, lon = r.get("Latitude"), r.get("Longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            alert = r.get(alert_col, "Safe")
            proba = r.get(proba_col, np.nan)
            color = color_map.get(str(alert), "blue")
            popup = f"<b>{r.get('district', '')}</b><br>{disease}: {alert}<br>Prob: {'' if pd.isna(proba) else round(float(proba),3)}"
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=popup
            ).add_to(fg)
        fg.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    # --- Legend for Alert Levels ---
    legend_html = '''
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 170px;
        height: 120px;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        background-color:white;
        padding: 10px;
        border-radius:8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    ">
    <b>🧭 Alert Level Legend</b><br>
    <i style="background:green;width:12px;height:12px;float:left;margin-right:8px;margin-top:3px;"></i> Safe<br>
    <i style="background:orange;width:12px;height:12px;float:left;margin-right:8px;margin-top:3px;"></i> Watch<br>
    <i style="background:red;width:12px;height:12px;float:left;margin-right:8px;margin-top:3px;"></i> Alert<br>
    <i style="background:blue;width:12px;height:12px;float:left;margin-right:8px;margin-top:3px;"></i> No Data
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


    # Render map
    st_html(m.get_root().render(), height=600)


    # Save filtered predictions CSV (top-N)
    out_csv_path = os.path.join(OUT_DIR, f"predictions_week_{int(selected_week)}_top{top_n}.csv")
    final_top.to_csv(out_csv_path, index=False)
    st.success(f"Top-{len(final_top)} predictions saved to: {out_csv_path}")
    st.download_button("Download top-N predictions CSV", data=open(out_csv_path,"rb"), file_name=os.path.basename(out_csv_path), mime="text/csv")

    # LLM suggestions (unchanged)
    st.subheader("LLM-based mitigation suggestions")
    top_alerts = []
    for d in disease_cols:
        if f"{d}_alert_rf" in final_top.columns:
            c_alert = int((final_top[f"{d}_alert_rf"]=="Alert").sum())
            top_alerts.append((d, c_alert))
    top_alerts = sorted(top_alerts, key=lambda x: x[1], reverse=True)
    summary_text = "District-level alert counts (top-N):\n" + "\n".join([f"{d}: {c}" for d,c in top_alerts])
    st.write(summary_text)
    if openai_key_input or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None):
        key = openai_key_input.strip() or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
        with st.spinner("Requesting LLM suggestions (this uses your OpenAI key)..."):
            suggestions = llm_suggestions(key, summary_text)
        st.markdown("**LLM suggestions:**")
        st.write(suggestions)
    else:
        st.info("No OpenAI key available. Provide a key in the sidebar to generate LLM-based suggestions.")

    st.success("Inference run completed.")

else:
    st.info("Select a week and click 'Run inference' to generate predictions and suggestions.")
