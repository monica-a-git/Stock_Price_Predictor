# Stock Price Predictor

A hybrid stock price forecasting pipeline combining sequential deep learning with gradient boosting regression.

## Project Overview

This project predicts the next trading day close price for selected US stocks using a two-stage hybrid model:

- **LSTM** for sequence modeling of recent price and volume history
- **LightGBM** for tabular regression using technical indicators and the LSTM output

The streamlit dashboard `app.py` displays the latest predictions and historical price charts for each ticker.

Deployed on : https://stockpricedetector-monica-a-git.streamlit.app/

## Models and Methodology

### 1. Data Acquisition

- Historical data is stored in `data/` as CSV files for tickers: `AAPL`, `GOOGL`, `MSFT`, `AMZN`, `TSLA`.
- `setup_history.py` can bootstrap history using Yahoo Finance (`yfinance`) and save cleaned OHLCV data.
- `src/data_loader.py` refreshes daily data using Alpha Vantage (`TIME_SERIES_DAILY`) and preserves cached data when API limits or errors occur.

### 2. Feature Engineering

The project computes technical indicators from raw OHLCV data in `src/features.py`:

- **RSI (Relative Strength Index)** with a 14-day window
- **EMA (Exponential Moving Average)** with a 20-day span
- **Bollinger Bands** using a 20-day moving average and 2 standard deviations

After generating indicators, the pipeline scales the base OHLCV feature set with `MinMaxScaler` and builds 60-day sequences for the LSTM.

### 3. Hybrid Model Architecture

The hybrid model is implemented in `src/models.py` as `HybridModel`.

#### LSTM component

- Sequential model with:
  - 2 LSTM layers (50 units each)
  - Dropout layers at 20%
  - Dense layers to regress a single scalar
- Input shape is `(60, 5)` representing the past 60 days of `[open, high, low, close, volume]`
- Trained with mean squared error loss and the Adam optimizer

#### LightGBM component

- Uses LightGBM regression to learn from:
  - technical indicators derived from the full dataset
  - LSTM predictions generated on the same training sequences
- This effectively stacks the LSTM output as a feature, letting LightGBM adjust the final close price estimate using indicator context.

### 4. Prediction Flow

The main pipeline in `main.py` follows these steps for each ticker:

1. Load or refresh historical data with `DataLoader`
2. Add technical indicators via `FeatureEngineer`
3. Prepare LSTM training sequences and scaled targets
4. Train the LSTM and save it to `models/{ticker}_lstm.h5`
5. Build a stacked LightGBM dataset using indicators + LSTM predictions
6. Train LightGBM and save it to `models/{ticker}_lgbm.pkl`
7. Predict tomorrow's close using the last 60 scaled days and latest indicators
8. Write final predictions into `latest_predictions.csv`

### 5. Online Learning and Update Strategy

- The pipeline is designed for repeated execution, refreshing history and retraining models as new daily market data arrives.
- `src/data_loader.py` appends new Alpha Vantage data to the local CSV cache and avoids service interruptions by falling back to cached history when API limits are hit.
- Each run retrains both the LSTM and LightGBM models on the most recent dataset, which is a form of batch-online updating.
- This keeps the forecast model aligned with recent price behavior and technical indicator changes.

### 6. Forecast Interpretation

- The LSTM captures recent temporal patterns in OHLCV history.
- The LightGBM stage brings in handcrafted technical signals to correct short-term trend or momentum bias.
- The final output is a predicted close price for the next trading day.

## Files and Structure

- `app.py` — Streamlit dashboard for viewing predictions and price charts
- `main.py` — end-to-end training and prediction pipeline
- `src/config.py` — configuration constants, tickers, directory paths, and hyperparameters
- `src/data_loader.py` — Alpha Vantage fetcher and CSV persistence logic
- `src/features.py` — technical indicator engineering and LSTM sequence preparation
- `src/models.py` — hybrid LSTM + LightGBM model implementation
- `data/` — historical stock CSVs
- `models/` — saved model artifacts
- `latest_predictions.csv` — generated prediction output for dashboard consumption

## Dependencies

Key libraries used in the project:

- `pandas`, `numpy` — data handling
- `scikit-learn` — scaling and preprocessing
- `tensorflow-cpu` — LSTM model training and inference
- `lightgbm` — gradient boosting regression
- `joblib` — model persistence
- `streamlit`, `plotly` — dashboard UI and charts
- `requests` — API data fetching

## Running the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Alpha Vantage API key in `ALPHA_VANTAGE_KEY`.
3. Optionally run `setup_history.py` once to download full historical CSVs.
4. Run the pipeline:
   ```bash
   python main.py
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Notes

- The model is designed for one-day-ahead close price forecasting.
- Hybrid stacking blends the strengths of sequential learning and indicator-based regression.
- The pipeline uses compact Alpha Vantage data fetches to stay within free-tier limits.
