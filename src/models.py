import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import lightgbm as lgb
import joblib
import os
from .config import Config

class HybridModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.lstm_model = None
        self.lgbm_model = None
        self.lstm_path = os.path.join(Config.MODEL_DIR, f"{ticker}_lstm.h5")
        self.lgbm_path = os.path.join(Config.MODEL_DIR, f"{ticker}_lgbm.pkl")

    def build_lstm(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model

    def train(self, X_lstm, y_lstm, df_features):
        print(f"[{self.ticker}] Training LSTM...")
        # 1. Train LSTM
        self.lstm_model.fit(X_lstm, y_lstm, batch_size=32, epochs=5, verbose=0)
        self.lstm_model.save(self.lstm_path)
        
        # 2. Get LSTM Predictions on Training Data (for stacking)
        lstm_preds_train = self.lstm_model.predict(X_lstm, verbose=0)
        
        # 3. Prepare LightGBM Data
        # We need to create the LightGBM feature set from the DataFrame + LSTM Preds
        # FeatureEngineer helper logic handles alignment in main.py, 
        # but here we assume X_lgbm is passed correctly or constructed.
        
        # Note: We need y_lgbm matching the length. 
        # y_lstm matches X_lstm.
        
        # Construct LGBM Input
        # Drop OHLCV cols to get just indicators
        indicators = df_features.drop(columns=Config.FEATURE_COLS).iloc[Config.SEQ_LEN:]
        
        # Align lengths
        min_len = min(len(indicators), len(lstm_preds_train))
        X_lgbm = np.hstack((indicators.iloc[:min_len].values, lstm_preds_train[:min_len]))
        y_lgbm = y_lstm[:min_len]
        
        print(f"[{self.ticker}] Training LightGBM...")
        params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
        d_train = lgb.Dataset(X_lgbm, label=y_lgbm)
        self.lgbm_model = lgb.train(params, d_train, num_boost_round=100)
        joblib.dump(self.lgbm_model, self.lgbm_path)
        
        # Return features, true labels, and the aligned dates for historical analysis
        aligned_dates = df_features.index[Config.SEQ_LEN : Config.SEQ_LEN + min_len]
        return X_lgbm, y_lgbm, aligned_dates

    def predict_tomorrow(self, last_sequence_scaled, last_indicators_df, scaler):
        """
        last_sequence_scaled: The last 60 days of scaled OHLCV data (Shape: 1, 60, 5)
        last_indicators_df: The DataFrame row containing indicators for today
        scaler: The scaler object to inverse transform the result
        """
        # 1. LSTM Prediction
        lstm_pred = self.lstm_model.predict(last_sequence_scaled, verbose=0)
        
        # 2. LightGBM Prediction
        # Drop OHLCV cols from the single row DataFrame
        indicators_val = last_indicators_df.drop(columns=Config.FEATURE_COLS).values
        
        # Combine
        combined_feat = np.hstack((indicators_val, lstm_pred))
        
        # Final Scaled Prediction
        scaled_final_price = self.lgbm_model.predict(combined_feat)[0]
        
        # 3. Inverse Scale
        # We need to construct a dummy array to inverse transform just the 'Close' column
        dummy = np.zeros((1, 5))
        dummy[0, 3] = scaled_final_price # 3 is index of Close
        
        final_price = scaler.inverse_transform(dummy)[0, 3]
        return final_price