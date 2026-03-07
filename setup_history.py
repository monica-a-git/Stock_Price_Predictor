import os
import yfinance as yf
import pandas as pd
from src.config import Config

# Ensure data directory exists
if not os.path.exists(Config.DATA_DIR):
    os.makedirs(Config.DATA_DIR)

print("--- Downloading Historical Data (Yahoo Finance) ---")

for ticker in Config.TICKERS:
    print(f"Fetching history for {ticker}...")
    
    # Download Max History
    # multi_level_index=False ensures we get simple column names like 'Open', 'Close'
    df = yf.download(ticker, period="max", multi_level_index=False)
    
    # Check if empty
    if df.empty:
        print(f"Warning: No data found for {ticker}")
        continue

    # 1. Reset Index to make Date a column
    df.reset_index(inplace=True)
    
    # 2. Fix Column Names (Handle potential MultiIndex or capitalization)
    # This specifically fixes the "AttributeError: tuple" by forcing strings
    df.columns = [str(c).lower() for c in df.columns]
    
    # 3. Rename 'date' to 'timestamp' to match Alpha Vantage
    if 'date' in df.columns:
        df.rename(columns={'date': 'timestamp'}, inplace=True)
    
    # 4. Set Index
    df.set_index('timestamp', inplace=True)
    
    # 5. Filter only required columns
    # We verify columns exist to prevent errors
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in required_cols if c in df.columns]
    
    if len(available_cols) < 5:
        print(f"Warning: Missing columns for {ticker}. Found: {df.columns}")
        continue
        
    df = df[available_cols]
    
    # Save to CSV
    save_path = f"{Config.DATA_DIR}/{ticker}.csv"
    df.to_csv(save_path)
    print(f"Saved {ticker}.csv ({len(df)} rows)")

print("--- History Download Complete ---")