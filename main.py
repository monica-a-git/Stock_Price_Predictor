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
            X_lgbm, y_lgbm, aligned_dates = model.train(X_lstm, y_lstm, df_enriched)
            
            # 3b. Evaluate on Historical Data
            scaled_preds = model.lgbm_model.predict(X_lgbm)
            
            dummy_preds = np.zeros((len(scaled_preds), 5))
            dummy_preds[:, 3] = scaled_preds
            unscaled_preds = fe.scaler.inverse_transform(dummy_preds)[:, 3]
            
            dummy_actuals = np.zeros((len(y_lgbm), 5))
            dummy_actuals[:, 3] = y_lgbm
            unscaled_actuals = fe.scaler.inverse_transform(dummy_actuals)[:, 3]
            
            errors_pct = np.abs(unscaled_actuals - unscaled_preds) / unscaled_actuals * 100
            accuracies_pct = 100 - errors_pct
            
            hist_df = pd.DataFrame({
                'timestamp': aligned_dates,
                'Latest Actual Close': np.round(unscaled_actuals, 2),
                'Predicted Close': np.round(unscaled_preds, 2),
                'Error (%)': np.round(errors_pct, 2),
                'Accuracy (%)': np.round(accuracies_pct, 2)
            })
            hist_df.to_csv(f"data/{ticker}_predictions.csv", index=False)

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
            print("Waiting 15 seconds for API rate limit...")
            time.sleep(15) 
            
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