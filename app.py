import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Page Config
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 Hybrid AI Stock Predictor (LSTM + LightGBM)")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])

# Load Predictions
if os.path.exists("latest_predictions.csv"):
    preds = pd.read_csv("latest_predictions.csv")
    last_update = preds['Date'].iloc[0]
    st.sidebar.info(f"Last AI Update: {last_update}")
else:
    st.error("No predictions found. The system might be running for the first time.")
    preds = pd.DataFrame()

# Main Display
col1, col2 = st.columns(2)

# 1. Show Prediction for Selected Ticker
if not preds.empty:
    row = preds[preds['Ticker'] == ticker]
    if not row.empty:
        prediction = row['Predicted_Close'].values[0]
        with col1:
            st.metric(label=f"Predicted Close for Tomorrow ({ticker})", value=f"${prediction}")
    else:
        st.warning(f"No prediction available for {ticker}")

# 2. Show Historical Data & Chart
csv_path = f"data/{ticker}.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Get last 90 days for clearer view
    recent_df = df.tail(90)

    # Plot Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(x=recent_df['timestamp'],
                    open=recent_df['open'],
                    high=recent_df['high'],
                    low=recent_df['low'],
                    close=recent_df['close'])])
    
    fig.update_layout(title=f"{ticker} - Last 90 Days Trend", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recent Data (OHLCV)")
        st.dataframe(recent_df.tail(10)[['timestamp', 'open', 'high', 'low', 'close', 'volume']], hide_index=True)

else:
    st.warning(f"No historical data found for {ticker}. The pipeline hasn't run yet.")

# Footer
st.markdown("---")
st.caption("System Architecture: Alpha Vantage API -> GitHub Actions (Daily Retraining) -> LSTM/LightGBM -> Streamlit Cloud")