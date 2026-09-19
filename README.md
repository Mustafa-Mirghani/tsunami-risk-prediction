# 🌊 Tsunami Risk Prediction from Earthquake Observations

![Python](https://img.shields.io/badge/Python-3.10-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-4.0-9cf) ![SHAP](https://img.shields.io/badge/SHAP-Interpretability-purple) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📋 Project Summary

Built and compared four machine learning models to predict whether an earthquake is likely to be flagged as tsunami-relevant, using standard USGS earthquake catalog data — magnitude, depth, location, and station-quality metrics. The project goes beyond model comparison alone, using SHAP to explain individual predictions and cross-checking feature importance across models to catch a misleading result before trusting it.

---

## 🌍 Why This Project

A few months ago I experienced an earthquake firsthand — thankfully light, and no one was hurt — but in the moment, it was genuinely frightening. That experience made the value of early-warning systems concrete to me in a way statistics alone never could: even a few minutes of advance warning can be the difference between people reaching safety and people being caught off guard.

This project is an early step toward understanding how machine learning could support that kind of system — not an attempt to build one. See **Limitations** below.

---

## 📊 Dataset Overview

| Field | Detail |
|---|---|
| **Source** | [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/) — FDSN Event Web Service API |
| **Scope** | All M4.5+ earthquakes, 2015–2024 |
| **Target** | Whether USGS flagged the event as tsunami-relevant (retrospective classification — see Limitations) |
| **Class Balance** | ~3% tsunami-relevant — severe class imbalance |

---

## 🔬 Methodology

### 1. Target Definition
Established explicitly, before any modelling, what the `tsunami` label actually represents — an institutional flag assigned after the fact, not confirmed physical tsunami generation — to avoid building a technically clean model on a scientifically ambiguous target.

### 2. Feature Engineering
- **Cyclic encoding** (sin/cos) applied to latitude and longitude, correcting the wrap-around problem at ±180° longitude, where raw values would make geographically adjacent locations appear artificially distant
- **Shallow-depth flag** engineered as an explicit binary feature, alongside continuous depth

### 3. Imbalance Handling
No default technique applied blindly. Class weighting (`class_weight='balanced'`) and `scale_pos_weight` were used instead of SMOTE, since synthetic oversampling risked producing unrealistic combinations of spatial/seismic variables.

### 4. Model Tuning
Every hyperparameter justified through explicit train/validation overfitting analysis — not defaults:

| Model | Tuning Approach | Final Configuration |
|---|---|---|
| Logistic Regression | Class-weighted baseline | `class_weight='balanced'` |
| Random Forest | Depth swept 1→None, smallest generalization gap selected | `max_depth=7` |
| XGBoost | Joint grid search: depth × boosting rounds | `max_depth=3, n_estimators=40` |
| LightGBM | Joint grid search: depth × boosting rounds | `max_depth=3, n_estimators=10` |

### 5. Interpretability
SHAP applied to XGBoost — a global feature-impact summary across the validation set, plus a detailed individual case study explaining a specific real prediction the model got wrong.

---

## 🎯 Results

*Final test-set performance — evaluated once, after all tuning was complete.*

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Rule baseline (M≥7.0, depth<70km) | 0.867 | 0.310 | 0.456 |
| Logistic Regression | 0.265 | **0.952** | 0.415 |
| Random Forest | 0.372 | 0.690 | 0.483 |
| XGBoost | 0.377 | 0.690 | 0.487 |
| **LightGBM** | 0.356 | 0.881 | **0.507** |

**LightGBM is the recommended model** — the strongest overall balance of any model tested, catching 88% of real tsunami-relevant events while achieving the highest F1 score, using a deliberately shallow, lightly-boosted configuration rather than the setting with the single highest validation score. Logistic Regression remains the right choice if recall alone is the priority, with zero tolerance for trading any of it away.

---

## 🔍 Key Findings

- **A cross-model check caught a misleading result.** Random Forest ranked station count (`nst`) as its second most important feature — a variable with no physical connection to tsunami risk. Checked against the other three models, this did not replicate, suggesting it was specific to how Random Forest split its trees rather than a genuine signal in the data.
- **Model complexity traded off against recall.** Simpler, less-boosted configurations (Logistic Regression, shallow LightGBM) consistently caught more real events; more complex ensembles favored precision instead — a pattern observed consistently across all four models.
- **SHAP explained a specific false negative in detail.** A magnitude-6.34 earthquake near Kamchatka was missed with only 5.8% predicted probability — despite magnitude correctly pushing the prediction upward, it was outweighed by depth, location-uncertainty metrics, and station count, exposing exactly which secondary factors overrode a genuine physical signal.

---

## ⚠️ Limitations

This is an early step, not a deployable system. Real tsunami early-warning depends on infrastructure this project doesn't attempt to replicate — deep-ocean pressure sensors (DART buoys), tide gauges, hydrodynamic modelling, and expert seismologist review. The dataset also lacks bathymetry and focal-mechanism data, both significant physical factors in tsunami generation, and the target label reflects retrospective classification rather than real-time ground truth. Full discussion in the notebook.

---

## 🛠️ Tech Stack

```
Python          3.10
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
matplotlib
seaborn
folium
```

---

## ▶️ How To Run

```bash
# 1. Clone the repository
git clone https://github.com/Mustafa-Mirghani/tsunami-risk-prediction.git

# 2. Navigate to project directory
cd tsunami-risk-prediction

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Predicting_Tsunami_Alerts_from_Earthquake_Observations.ipynb

# 5. Run all cells
```

> **Note:** Data is fetched live from the USGS API on run. If unavailable, a backup CSV fallback is used — see the notebook's Data Description section.

---

## 📈 Future Improvements

- [ ] Incorporate bathymetry and focal-mechanism data if a suitable public source is identified
- [ ] Extend SHAP analysis to LightGBM's individual predictions
- [ ] Test regional-only training (e.g. Pacific Ring of Fire) against the current global model
- [ ] Explore probability calibration for more reliable threshold-based alerting

---

## 👤 Author

**Mustafa Ahmed**
🐙 [GitHub](https://github.com/Mustafa-Mirghani)

---
