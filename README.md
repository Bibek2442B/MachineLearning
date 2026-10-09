# Netflix (NFLX) Stock Closing Price Prediction

A supervised machine learning pipeline designed to forecast daily closing prices (`Close`) for Netflix, Inc. (`NFLX`). The project benchmarks Multivariate Linear Regression against K-Nearest Neighbors (KNN), strictly enforcing chronological time-series constraints to eliminate lookahead bias while evaluating technical market indicators.

---

## Overview

Predicting market price dynamics requires rigorous data preparation to avoid data leakage. This project implements an end-to-end regression workflow on historical market data (covering 2023–2024), utilizing both intraday price metrics and engineered lagged momentum signals without relying on future information.

---

## Key Features

- **Leak-Free Temporal Split**:
  - Implements a chronological train-test split (80% historical training, 20% future testing) without random shuffling, preserving the directional nature of market time-series.
  - Ensures the target variable (`Close`) is strictly withheld from model feature sets.

- **Feature Engineering & Technical Indicators**:
  - **Intraday Volatility**: Daily trading range/amplitude ($\text{High} - \text{Low}$).
  - **Lagged Price Action**: Prior-day closing prices ($\text{Close}_{t-1}$) and percentage daily returns.
  - **Trend Filters**: Short-term trailing moving averages (SMA-5, SMA-10) computed exclusively on preceding intervals to capture recent momentum without lookahead bias.

- **Model Implementations & Hyperparameter Tuning**:
  - **Multivariate Linear Regression**: Evaluates linear relationships between price volume, intraday spreads, and lagged momentum features.
  - **K-Nearest Neighbors (KNN) Regressor**: Tests local neighborhood estimation across varying values of $k$ to identify optimal bias-variance balance in price space.
  - **Feature Ablation Study**: Comparative evaluation contrasting baseline models (raw features) against feature-engineered models.

- **Standardized Evaluation Metrics**:
  - Assesses model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).
  - Exports sequential test-set predictions to `precos_estimados.csv` for out-of-sample inspection.

---

## Project Structure

```text
├── data/
│   ├── NFLX.csv                 # Raw historical price & volume dataset (2023–2024)
│   └── precos_estimados.csv     # Exported sequential test-set predictions
├── notebooks/
│   └── solution.ipynb           # Complete EDA, feature engineering, and modeling pipeline
└── README.md
