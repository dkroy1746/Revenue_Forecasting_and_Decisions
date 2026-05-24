# 📊 Sales Revenue Forecasting & Decision Intelligence

A comprehensive end-to-end data science solution for sales revenue forecasting, anomaly detection, root-cause analysis, and 30-day forward revenue prediction — built on a real-world invoice ledger dataset spanning **April 2022 to May 2026**.

---

## 📁 Project Structure

```
├── Solution_Notebook.ipynb      # Main solution notebook
├── revenue_ledger.csv           # Source dataset (invoice-level transactions)
├── TotalrevenueABmonthly.png    # Monthly revenue chart (Account A vs B)
├── revtrendsovertimeA.png       # Daily/weekly/monthly revenue trend (Acc A)
├── revbypartsA.png              # Top 10 parts by revenue (Acc A)
├── AnomalyA.png                 # Anomaly detection chart
├── ProphetPredictedvsActual.png # Prophet model: actual vs predicted
├── forecast30day.png            # 30-day forward forecast with confidence intervals
└── forecast30dayweekly.png      # Forecast components (macro trend + weekly cycle)
└── Account_B                    # All the same things related to Account B
```

---

## 🎯 Objective

Analyze a multi-account sales revenue ledger to:
1. **Understand** the data through exploratory analysis
2. **Forecast** future revenue using time-series models
3. **Engineer** meaningful temporal and lag features
4. **Interpret** business insights and anomalies
5. **Perform** root-cause and what-if scenario analysis

---

## 📦 Dataset

**File:** `revenue_ledger.csv`  
**Shape:** 234,516 rows × 8 columns

| Column | Description |
|---|---|
| `Account` | Account identifier (Account A or Account B) |
| `Cust Code` | Customer code (848 unique customers) |
| `Part Code` | Part/product code (2,519 unique parts) |
| `Inv Date` | Invoice date (Apr 2022 – May 2026) |
| `Quantity` | Quantity sold (can be negative for returns) |
| `Price` | Unit price (INR) |
| `Tax` | Tax charged (10.7% missing values) |
| `Others` | Other charges (5.4% missing values) |

**Key data characteristics:**
- Two accounts: **Account A** (active from Apr 2022) and **Account B** (active from Apr 2025)
- Revenue computed as: `Total Revenue = Quantity × Price + Tax + Others`
- Negative revenue rows (returns) handled separately
- 19,335 outliers flagged via IQR method

---

## 🔬 Tasks Covered

### Task 1 — Data Understanding & EDA
- Data type conversion, missing value imputation (Tax, Others → 0)
- Separate handling of negative revenue (returns) vs. positive revenue (sales)
- Outlier flagging using the IQR method (preserved, not deleted)
- Account-level revenue segmentation (Account A vs. B)
- Monthly revenue trends, tax rate, and "Others" rate analysis
- Seasonal analysis: day-of-week and monthly patterns
- Top-10 parts by revenue (Pareto distribution observed)
- Anomaly detection using Z-score (>3σ flagged as anomalies on specific dates)

### Task 2 — Forecasting Model
**Primary Model: Meta's Prophet**  
Reasons for selection:
- Native **changepoint detection** for structural breaks (e.g., tax regime shift in Jan 2026)
- **Robust to outliers** present in the data
- **Native weekly seasonality** support (Monday/Sunday lows, Friday highs)

**Model training pipeline:**
- Chronological train/test split (last 30 days held out as test set)
- Hyperparameter tuning via grid search over 64 candidates with Prophet's cross-validation
- Best params: `changepoint_prior_scale=0.01`, `weekly_seasonality=True`, `monthly_seasonality=True`
- Final model retrained on all available history for forward forecasting

**📈 Prophet Model Results:**
| Metric | Value |
|---|---|
| RMSE | ₹2,396,115.58 |
| MAE | ₹1,845,200.40 |
| Standard MAPE | 79.82% |
| Weighted MAPE (wMAPE) | **46.51%** |

**Comparison Model: XGBoost (with TimeSeriesSplit CV)**

| Metric | Value |
|---|---|
| RMSE | ₹3,877,672.21 |
| MAE | ₹3,605,097.75 |
| Standard MAPE | 147.03% |
| Weighted MAPE (wMAPE) | 90.87% |

> Prophet significantly outperforms XGBoost on this dataset due to the presence of structural breaks and strong weekly seasonality patterns.

### Task 3 — Feature Engineering
Temporal and lag features created for the XGBoost model:
- `day_of_week`, `month`, `quarter`, `is_weekend`
- `lag_30`, `lag_31`, `lag_36` (30-day horizon-aware lag features)
- `rolling_mean_30`, `rolling_std_30` (7-day rolling stats over 30-day shifted window)

### Task 4 — Business Interpretation
Key findings from the forecast:
- **Highest revenue risk:** Final weeks of the 30-day forecast window due to a macro declining trend
- **Weekly vulnerability:** Sundays and Mondays consistently drag revenue down by ~₹2M
- **Planning recommendation:** Scale down production/inventory on Sun–Mon, maximize availability on Fridays

### Task 5 — Root Cause & What-If Analysis
**Anomaly identified:** Sharp structural break in Account A starting January 2026
- Tax rate crashed from a stable ~20% to volatile 5–10%, then flatlined to 0%
- Monthly revenue dropped from ~₹240M baseline to ~₹170M–₹50M

**Root cause:** Business migration / account cannibalization
- Account B (launched Apr 2025, with 0% tax rate) absorbed high-volume operations
- India's GST policy shifts also contributed to the tax rate collapse in Account A

**What-if scenario:** Without the migration anomaly, Account A would have maintained ~₹240M/month — representing an estimated loss of ₹70M–₹190M/month in Account A alone (offset by Account B growth).

**Platform suggestion (PlantNxt):**
- Deploy drift-detection on `Tax Rate` and `Others Rate` as leading indicators
- Implement relational analytics to flag account-level cannibalization events

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scipy.stats` | Z-score anomaly detection |
| `scikit-learn` | `StandardScaler`, `LabelEncoder`, `GridSearchCV`, `TimeSeriesSplit`, metrics |
| `xgboost` | XGBoost regression model |
| `prophet` | Time-series forecasting |
| `statsmodels` | Statistical analysis |
| `joblib` | Model persistence |
| `polars` | Supplementary fast data processing |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sales-revenue-forecasting.git
cd sales-revenue-forecasting
```

### 2. Set up the environment
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost prophet statsmodels joblib polars
```
> **Note:** Prophet requires `cmdstanpy` as a backend. Install it via `pip install prophet` which handles this automatically.

### 3. Run the notebook
```bash
jupyter notebook Solution_Notebook.ipynb
```
Ensure `revenue_ledger.csv` is placed in the same directory as the notebook.

---

## 📊 Key Visualizations

| Chart | Description |
|---|---|
| Monthly Total Revenue — Account A vs B | Stacked bar chart showing revenue contribution by account over time |
| Tax Rate & Others Rate | Time-series of effective tax/other charge rates per account |
| Daily/Weekly/Monthly Revenue Trend (Acc A) | Multi-resolution revenue trend for Account A |
| Top 10 Parts by Revenue (Acc A) | Horizontal bar chart showing Pareto distribution |
| Daily Revenue with Anomalies | Z-score based anomaly scatter plot |
| Prophet Actual vs. Predicted | 30-day test set evaluation |
| 30-Day Forward Forecast | Future forecast with 80% confidence intervals |
| Forecast Components | Macro trend decomposition + weekly cycle revenue impact |

---

## 💡 Business Insights Summary

- **Account A** experienced a structural revenue decline from Jan 2026 driven by account migration to Account B and GST policy changes.
- **Account B** fully absorbed and exceeded Account A's volume by mid-2026.
- A **Pareto effect** exists: a tiny fraction of parts drive the majority of Account A's revenue.
- Revenue follows a **strong weekly cycle** — Fridays peak, Sundays and Mondays trough.
- The **macro trend is declining** through May 2026, signaling the need for conservative production planning.

---

## 📌 Notes

- All monetary values are in **Indian Rupees (INR)**.
- Negative revenue rows (returns) were excluded from the forecasting pipeline.
- Outliers were **flagged but not removed** to preserve true revenue totals for EDA; anomalies were removed for model training.
- The final Prophet model is retrained on the **entire available history** before generating the 30-day forward forecast.
