# Personalized Digital Twin for Type 2 Diabetes (T2D)

A data-driven **Digital Twin platform** for Type 2 Diabetes management and glycemic forecasting, built using the **CGMacros** dataset. This system integrates Continuous Glucose Monitoring (CGM) time-series data, physical activity logs, dietary macronutrient tracking, clinical metadata, and gut microbiome profiles to deliver real-time personalized glucose forecasting and causal "What-If" scenario simulations.

---

## Key Modules

1. **Postprandial Glycemic Response (PPGR) Predictor**: Forecasts peak glucose levels, time-to-peak, and post-meal Area Under the Curve (AUC) for 2–4 hour windows using dynamic meal inputs and clinical baselines.
2. **Gut Microbiome-Guided Personalization Engine**: Integrates gut bacterial profiles (1,979 microbial features) and health scores to personalize glycemic predictions based on unique microbiome signatures.
3. **Causal "What-If" Scenario Simulator**: Models counterfactual scenarios for dietary adjustments (e.g., adding fiber) and physical activity (e.g., post-meal walking) using causal inference models.
4. **Automated Dietary Logging (Computer Vision)**: Leverages pre- and post-meal images for automated food classification, macronutrient estimation, and portion waste analysis.
5. **Reinforcement Learning Lifestyle Coach**: Provides real-time activity and nutritional recommendations to maximize Time in Range (TIR: 70–180 mg/dL).

---

# Personalized Digital Twin for Type 2 Diabetes (T2D)

A data-driven **Digital Twin platform** for Type 2 Diabetes management and glycemic forecasting, built using the **CGMacros** dataset. This system integrates Continuous Glucose Monitoring (CGM) time-series data, physical activity logs, dietary macronutrient tracking, clinical metadata, and gut microbiome profiles to deliver real-time personalized glucose forecasting and causal "What-If" scenario simulations.

---

## 📌 Repository Structure

```
Digital-Twin-for-Diabetes/
├── notebooks/
│   ├── 01_EDA_and_Microbiome_PCA.ipynb              # Exploratory data analysis, wearable alignment & gut PCA
│   ├── 02_Postprandial_Glucose_Prediction.ipynb     # Tabular ML predictors (XGBoost, LightGBM, CatBoost)
│   └── 03_Deep_Learning_Continuous_Glucose_Forecasting.ipynb # PyTorch multi-horizon trajectory forecasting (LSTM, GRU, MLP)
├── src/
│   ├── preprocess.py                                 # Continuous time-series cleaning & meal event extraction
│   ├── dataset_dl.py                                 # PyTorch sequence extraction & DataLoader pipeline
│   └── models_dl.py                                  # PyTorch MLP, LSTM, and GRU forecasters with context fusion
├── Project_Modules_and_Phases.md                     # Detailed module design and project roadmap
├── requirements.txt                                  # Environment dependencies
└── README.md                                         # Project documentation
```

---

## 🚀 Accomplishments & Benchmarks

### 1. Data Pipeline & Alignment (`src/preprocess.py`)
- Cleaned 1-minute resolution dual continuous glucose sensor readings (`Libre GL` and `Dexcom GL` fused into `Unified GL`).
- Windowed **1,637 valid meal events** across **45 participants**, linking 60-minute pre-meal history to 120-minute post-meal trajectories.

### 2. Microbiome & Multimodal Feature Engineering (`notebooks/01_EDA_and_Microbiome_PCA.ipynb`)
- Reduced **1,979 sparse binary gut bacteria species** to 8 dense Principal Components using PCA.
- Combined meal macronutrients, pre-meal wearable statistics, clinical baselines (HbA1c, BMI, fasting insulin), 22 gut health test scores, and microbiome PCA components into a master dataset.

### 3. Tabular Machine Learning Predictors (`notebooks/02_Postprandial_Glucose_Prediction.ipynb`)
- Evaluated models across 5-Fold Group K-Fold Cross-Validation (unseen subject evaluation).
- `LightGBM` / `CatBoost` achieved $R^2 \approx 0.50$ and $MAE \approx 23.9$ mg/dL for postprandial peak glucose prediction.

### 4. Deep Learning Multi-Horizon Trajectory Forecasting (`notebooks/03_Deep_Learning_Continuous_Glucose_Forecasting.ipynb`)
- Built PyTorch **MLP**, **LSTM**, and **GRU** neural networks to forecast the **full 120-minute continuous postprandial glucose curve** (sampled every 5 minutes = 24 forecast steps).
- Achieved **22.16 mg/dL out-of-subject trajectory MAE** across all 24 forecasting horizons.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Setup Virtual Environment & Dependencies

```bash
# Clone the repository
git clone https://github.com/AoD-X-abhi/Digital-Twin-for-Diabetes.git
cd Digital-Twin-for-Diabetes

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📝 License & References

This project is developed as part of the **Major Project for B.Tech Semester 7**. Data derived from the **CGMacros** study.

