# Predictive Modeling of Climate Change Impacts on Infectious Disease Outbreaks Using Machine Learning

# Early Disease Outbreak Prediction System  
*A Data-Driven Early Warning Platform for Multi-Disease Outbreak Forecasting*

## Overview  
This project develops a **machine learning–powered early warning system** capable of predicting outbreaks of multiple communicable diseases using climatic, environmental, and epidemiological indicators.  
The system integrates **classical ML models (Logistic Regression, Random Forest, XGBoost)** with an experimental **Quantum Machine Learning (QML)** pipeline implemented using **PennyLane**, and provides an interactive **Streamlit Dashboard** for real-time exploration and visualization.

---

## Problem Statement  
Seasonal diseases such as **Cholera**, **Dengue**, **Malaria**, **Chikungunya**, and **Acute Diarrhoeal Diseases** continue to pose recurring public-health challenges in tropical regions.  
Outbreaks are influenced by climatic variables (temperature, humidity, precipitation, and wind speed) as well as socio-environmental conditions.  
Accurate early detection enables data-driven **health resource allocation**, **risk mitigation**, and **epidemiological surveillance**.

---

## Dataset  
**Source:** Aggregated weekly epidemiological and weather data (India, multi-district).  
**File:** `DATASET_FINAL.csv`

| Category | Variables |
|-----------|------------|
| Temporal | `week`, `month`, `year`, `date` |
| Geospatial | `district`, `Latitude`, `Longitude` |
| Weather | `week_avg_temp_C`, `week_avg_relative_humidity_percent`, `week_total_precip_mm`, `week_avg_wind_speed_ms` |
| Target Diseases | `Cholera`, `Malaria`, `Dengue`, `Chikungunya`, `Acute Diarrhoeal Disease`, `Acute Gastroenteritis`, `Acute Encephalitis Syndrome` |

---


---

## Modeling Pipeline  

### 1. Logistic Regression (Baseline)  
- Serves as an interpretable benchmark for binary outbreak prediction.  
- Handles class imbalance using weighted loss (`class_weight='balanced'`).  
- Evaluated using F1, Precision-Recall AUC, and ROC AUC.

### 2. Random Forest (Classical Ensemble)  
- Tuned using **TimeSeriesSplit GridSearchCV** for temporal validation.  
- Utilizes **balanced subsampling**, **cross-disease lag features**, and **rolling averages**.  
- SHAP interpretability is employed for feature importance analysis.

### 3. XGBoost (Gradient Boosting)  
- Fine-tuned with Bayesian optimization for rapid convergence.  
- Achieves superior recall and AUPRC on imbalanced datasets.  
- Integrated into the dashboard backend for live prediction.

### 4. Quantum Machine Learning (QML) with PennyLane  
- Implements a **Variational Quantum Classifier (VQC)** using PennyLane with the Qiskit simulator backend.  
- Hybrid classical–quantum approach: classical preprocessing with quantum feature mapping.  
- Explores performance gains from **quantum kernel embeddings** for low-sample outbreak prediction.

---

## Evaluation Metrics  

| Metric | Description |
|---------|--------------|
| **Precision, Recall, F1-Score** | Core classification metrics per disease |
| **PR-AUC (Average Precision)** | Emphasizes performance on imbalanced data |
| **ROC-AUC** | Overall separability of classes |
| **SHAP Analysis** | Feature contribution and interpretability |
| **Temporal Stability** | Rolling window validation over time |

All results are stored in `outputs/metrics_per_disease.csv` and visualized through **Precision-Recall curves** and **SHAP summary plots**.

---

## Streamlit Dashboard  

### Features
- Upload new regional data (CSV) for real-time outbreak prediction.  
- Visualize correlations between climatic variables and disease occurrence.  
- Compare predictions from **Logistic Regression**, **Random Forest**, **XGBoost**, and **Quantum ML** models.  
- View interactive heatmaps and outbreak risk by district.  
- Access SHAP-based interpretability visualizations.

Run locally:
```bash
streamlit run app.py

