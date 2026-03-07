import os
import pandas as pd
import requests
from .config import Config

class DataLoader:
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_data(self, symbol):
        """Fetches data from API and updates local CSV"""
        csv_path = os.path.join(Config.DATA_DIR, f"{symbol}.csv")
        
        # FIX: Always use 'compact' for Free Tier
        # 'compact' returns the latest 100 data points, which is enough for daily updates.
        output_size = 'compact'
        
        print(f"[{symbol}] Fetching daily update (Alpha Vantage)...")
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": Config.API_KEY,
            "outputsize": output_size,
            "datatype": "csv"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            
            # 1. Check for API Errors (JSON response instead of CSV)
            content = response.text.strip()
            if content.startswith("{"):
                print(f"[{symbol}] API Warning: {content[:100]}...")
                # If API fails, we try to return existing data so the pipeline doesn't crash
                if os.path.exists(csv_path):
                    print(f"[{symbol}] Using cached data instead.")
                    return pd.read_csv(csv_path, index_col=0, parse_dates=True)
                return None
            
            # 2. Parse New Data
            temp_path = f"temp_{symbol}.csv"
            with open(temp_path, "wb") as f:
                f.write(response.content)
                
            try:
                new_df = pd.read_csv(temp_path)
            except:
                print(f"[{symbol}] Error parsing CSV.")
                return None

            if 'timestamp' not in new_df.columns:
                print(f"[{symbol}] Error: 'timestamp' column missing.")
                os.remove(temp_path)
                return None

            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            new_df = new_df.sort_values('timestamp')
            new_df.set_index('timestamp', inplace=True)
            
            # 3. Merge with History
            if os.path.exists(csv_path):
                old_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                # Append new data
                combined = pd.concat([old_df, new_df])
                # Remove duplicates
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.to_csv(csv_path)
                final_df = combined
            else:
                # Fallback if setup_history.py wasn't run
                new_df.to_csv(csv_path)
                final_df = new_df
            
            os.remove(temp_path)
            print(f"[{symbol}] Database updated. Total rows: {len(final_df)}")
            return final_df

        except Exception as e:
            print(f"[{symbol}] Update Failed: {e}")
            return None