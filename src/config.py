import os

class Config:
    # API & Tickers
    API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
    TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    # Paths
    DATA_DIR = "data"
    MODEL_DIR = "models"
    
    # Hyperparameters
    SEQ_LEN = 60          # Lookback window for LSTM (Past 60 days)
    PREDICT_STEPS = 1     # Predict 1 day ahead
    
    # Feature Columns (Must match order in data_loader)
    FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']