import time
import numpy as np
import pandas as pd
from datetime import datetime
from src.config import Config
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models import HybridModel

def run_pipeline():
    print(f"--- Pipeline Started: {datetime.now()} ---")
    
    loader = DataLoader()
    results = {}

    for ticker in Config.TICKERS:
        try:
            # 1. Ingest Data
            df = loader.fetch_data(ticker)
            if df is None or len(df) < 200:
                print(f"[{ticker}] Skipped (Not enough data)")
                continue
            
            # 2. Feature Engineering
            fe = FeatureEngineer()
            df_enriched = fe.add_technical_indicators(df)
            
            # Get LSTM Data (X, y) and the Scaler fitted on history
            X_lstm, y_lstm, full_scaled_data = fe.prepare_lstm_data(df_enriched)
            
            # 3. Initialize & Train Hybrid Model
            model = HybridModel(ticker)
            model.build_lstm(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
            
            # Pass df_enriched so the model can extract indicators for LightGBM
            model.train(X_lstm, y_lstm, df_enriched)
            
            # 4. Predict Tomorrow
            # Grab last 60 days from full_scaled_data for LSTM
            last_60_days = full_scaled_data[-Config.SEQ_LEN:]
            last_60_days = last_60_days.reshape(1, Config.SEQ_LEN, 5)
            
            # Grab last row of indicators for LightGBM
            last_indicators = df_enriched.iloc[[-1]] # Keep as DataFrame
            
            prediction = model.predict_tomorrow(last_60_days, last_indicators, fe.scaler)
            
            results[ticker] = round(prediction, 2)
            print(f"[{ticker}] PREDICTION: ${results[ticker]}")
            
            # Rate limit respect
            time.sleep(12) 
            
        except Exception as e:
            print(f"[{ticker}] Pipeline Failed: {e}")

    # 5. Save Results to CSV for Streamlit
    # Create a DataFrame for predictions
    pred_df = pd.DataFrame(list(results.items()), columns=['Ticker', 'Predicted_Close'])
    pred_df['Date'] = datetime.now().strftime("%Y-%m-%d")
    pred_df.to_csv("latest_predictions.csv", index=False)
            
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    run_pipeline()