# 📡 Telecom Customer Churn Prediction & Retention Analytics

> **An end-to-end ML-powered analytics platform** that predicts customer churn, segments customers by risk, and simulates retention campaigns — all through an interactive Dash dashboard.

---

## 🧠 What It Does

Telecom companies lose thousands of customers silently. This platform converts raw customer data into **actionable retention intelligence** by:

- Training a calibrated **XGBoost classifier** to predict the probability each customer will churn
- Segmenting all customers into **High / Medium / Low risk tiers** based on churn probability
- Generating **personalized retention actions** per customer based on their contract, tenure, payment method, and monthly charges
- Simulating **pilot retention campaigns** with adjustable offer rates and success rates to project churn reduction before spending a rupee

---

## 📸 Application Screens

### 🏠 Home — Live Churn Overview
Displays total churned vs. retained customers with real percentages computed directly from model output.

![Home Page](https://raw.githubusercontent.com/placeholder/home.png)

> **7,043 customers | 1,869 churned (26.5%) | 5,174 retained (73.5%)**

---

### 📊 Dashboard — Churn Analytics
Interactive filter by churn status with four analytics views:

| Visual | Insight |
|--------|---------|
| Churn Rate by Contract Type | Month-to-month contracts show ~4× higher churn than two-year contracts |
| Churn Rate by Tenure Bucket | Early-tenure customers (0–6 months) are highest risk |
| Calibrated Probability Distribution | Right-skewed: majority of customers have low churn probability |
| Contract × Tenure Heatmap | Month-to-month + 0–6 months = highest risk cell (deep red) |

---

### 🎯 Customer Segmentation
Segments customers dynamically using configurable filters:

- **Risk Tier Filter** — view All / High / Medium / Low segments
- **Top N by predicted probability** — surface your most at-risk customers
- **Segment Summary bar chart** — 416 High risk | ~1,600 Medium | ~5,000 Low
- **Downloadable CSV** — export segmented customer list with retention actions attached

Each row shows: `pred_prob`, `RiskTier`, `RetentionAction`, `Contract`, `tenure`, `MonthlyCharges`, `PaymentMethod`

---

### 🔁 Retention Panel — Pilot Simulation
Simulate a retention campaign before deploying it:

**Inputs:**
- Pilot offer % for High-risk customers (default: 20%)
- Assumed offer success rate (default: 55%)
- Random seed for reproducibility

**Outputs:**
- Pilot Offer Distribution by Risk Tier
- Retention Outcome for offered customers
- Before vs. After churn rate comparison
- Top Retention Actions by success rate
- Full monitoring summary

**Sample simulation result (seed=42, pilot=20%, success=55%):**
```
Total Customers:               7,043
High Risk Customers:             416
Pilot Offers Sent:                96
Retention Success Rate:        48.96%
Churn Rate Before:             19.92%
Churn Rate After:               5.24%
Estimated Churn Reduction:     14.68%
```

---

## ⚙️ Technical Architecture

```
churn_app/
├── app.py                        # Dash app — layout, routing, callbacks
├── model/
│   ├── churn_dataset.csv         # Telco dataset (7,043 records, 21 features)
│   └── xgboost_churn_model.joblib  # Pre-trained XGBoost model
├── utils/
│   └── preprocess.py             # Data loading, encoding, model calibration
├── output_playbooks/
│   └── telco_with_next_steps.csv # Simulation output CSV
└── assets/
    └── style.css                 # Dashboard styling
```

---

## 🔬 ML Pipeline

### 1. Data Preprocessing (`utils/preprocess.py`)
- Loads raw telco CSV with `pandas`
- Coerces `TotalCharges` to numeric, imputes missing values with **column median**
- Drops `customerID` (non-predictive identifier)
- Encodes binary target: `Churn` → `{Yes: 1, No: 0}`
- Applies `LabelEncoder` across all categorical features for model compatibility

### 2. Model Training & Calibration
- Loads a pre-trained `XGBoost` classifier via `joblib`
- Wraps it in `CalibratedClassifierCV` with **isotonic regression** and **5-fold CV**
- Produces calibrated `pred_prob` — reliable probability estimates, not just raw scores

> Calibration is critical for churn: uncalibrated XGBoost probabilities are often overconfident. Isotonic regression corrects the probability curve against held-out folds.

### 3. Customer Segmentation Engine
Risk tiers assigned by thresholding `pred_prob`:

| Tier | Probability Range | Count |
|------|-------------------|-------|
| 🔴 High | > 0.70 | ~416 |
| 🟡 Medium | 0.40 – 0.70 | ~1,600 |
| 🟢 Low | < 0.40 | ~5,000 |

### 4. Retention Action Logic
Rule-based engine assigns personalized actions per customer using:

```python
if prob > 0.7:
    if month-to-month contract  → "Offer 15% discount for 1-year plan"
    if electronic/mailed check  → "Incentivize AutoPay ($5 credit)"
    if monthly_charges > 80 and tenure < 12 → "10% bill reduction for 3 months"
    if tenure < 6               → "Welcome loyalty package"
elif prob > 0.4:
    if tenure < 24              → "Recommend value bundle / free trial"
    else                        → "Satisfaction survey with targeted offer"
else:
    → "Loyalty communications and newsletters"
```

### 5. Pilot Simulation Engine
Monte Carlo-style stochastic simulation:
- Randomly assigns pilot offers to `pilot_pct`% of High-risk customers using `numpy.random.RandomState` (seeded for reproducibility)
- Simulates retention outcomes with `success_pct`% probability per offered customer
- Computes before/after churn rates and ranks retention actions by success rate
- Exports full simulation results as downloadable CSV

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Model** | XGBoost | Gradient boosted churn classifier |
| **Calibration** | Scikit-learn `CalibratedClassifierCV` | Reliable probability estimation |
| **Data Processing** | Pandas, NumPy | Feature engineering & preprocessing |
| **Encoding** | Scikit-learn `LabelEncoder` | Categorical feature transformation |
| **Model Persistence** | Joblib | Serialized model loading |
| **Dashboard** | Plotly Dash | Interactive web app |
| **Visualizations** | Plotly Express | Charts, heatmaps, histograms |
| **Styling** | CSS | Custom dashboard theme |
| **Runtime** | Python 3.11, Flask (via Dash) | App server |

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd churn_app

# 2. Install dependencies
pip install dash plotly pandas numpy scikit-learn xgboost joblib

# 3. Run the app
python app.py

# 4. Open in browser
# http://127.0.0.1:8050
```

---

## 📊 Dataset

**Telco Customer Churn Dataset**
- **Records:** 7,043 customers
- **Features:** 21 (demographics, services, contract, billing)
- **Target:** `Churn` (binary: Yes/No → 1/0)
- **Class split:** ~26.5% churned, ~73.5% retained

Key predictive features: `Contract`, `tenure`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, `InternetService`, `TechSupport`

---

## 📈 Key Results

```
Churn Rate (dataset):           26.5%
High Risk Customers:             416  (5.9% of base)
Simulated Churn Reduction:     ~14.7% with 20% pilot offer coverage
Top Retention Action:          15% discount + 10% bill reduction → 57% success rate
```

---

## 👤 Author

**Atla Mayank Rao**
[LinkedIn](https://linkedin.com/in/mayank-rao) · [GitHub](https://github.com/mayank-rao) · mayankatla13@gmail.com

---

*Built as part of Telecom Churn Prediction & Retention Analytics System — Datathon 2K25 (3rd Place)*
