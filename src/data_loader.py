import os
import time
import pandas as pd
import requests
from .config import Config

class DataLoader:
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_data(self, symbol):
        """Fetches data from API and updates local CSV"""
        csv_path = os.path.join(Config.DATA_DIR, f"{symbol}.csv")
        
        # Check if we already have data to decide API mode
        file_exists = os.path.exists(csv_path)
        output_size = 'compact' if file_exists else 'full'
        
        print(f"[{symbol}] Fetching data (Mode: {output_size})...")
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": Config.API_KEY,
            "outputsize": output_size,
            "datatype": "csv"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            
            # API Error Handling
            if "Error" in response.text or "limit" in response.text:
                print(f"[{symbol}] API Error: {response.text}")
                return None

            # Save to temporary file to parse
            temp_path = f"temp_{symbol}.csv"
            with open(temp_path, "wb") as f:
                f.write(response.content)
                
            new_df = pd.read_csv(temp_path)
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            new_df = new_df.sort_values('timestamp')
            new_df.set_index('timestamp', inplace=True)
            
            # Merge with existing data
            if file_exists:
                old_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                combined = pd.concat([old_df, new_df])
                # Remove duplicates based on Index (Date)
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.to_csv(csv_path)
                final_df = combined
            else:
                new_df.to_csv(csv_path)
                final_df = new_df
            
            os.remove(temp_path)
            print(f"[{symbol}] Data saved. Total rows: {len(final_df)}")
            return final_df

        except Exception as e:
            print(f"[{symbol}] Critical Error: {e}")
            return None

    def get_data(self, symbol):
        """Just loads the CSV without fetching (if needed)"""
        csv_path = os.path.join(Config.DATA_DIR, f"{symbol}.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return None