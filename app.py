import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock AI Predictor", layout="wide")
st.title("📈 Hybrid AI Stock Predictor (LSTM + LightGBM)")
st.markdown("Returns the predicted **Close Price** for the next trading day.")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])

# --- LOAD PREDICTIONS ---
if os.path.exists("latest_predictions.csv"):
    preds = pd.read_csv("latest_predictions.csv")
    
    # Filter for selected ticker
    row = preds[preds['Ticker'] == ticker]
    
    if not row.empty:
        prediction = row['Predicted_Close'].values[0]
        # Display Big Metric
        st.metric(label=f"Predicted Price ({ticker})", value=f"${prediction}")
    else:
        st.error(f"No prediction found for {ticker}")
else:
    st.warning("No predictions file found. Run the pipeline first.")

# --- LOAD HISTORICAL DATA ---
csv_path = f"data/{ticker}.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Tabs for Chart vs Data
    tab1, tab2 = st.tabs(["📉 Price Chart", "📄 Raw Data"])

    with tab1:
        # Interactive Candlestick Chart (Last 180 Days)
        st.subheader(f"{ticker} - Last 6 Months")
        recent_df = df.tail(120) 
        
        fig = go.Figure(data=[go.Candlestick(
            x=recent_df['timestamp'],
            open=recent_df['open'],
            high=recent_df['high'],
            low=recent_df['low'],
            close=recent_df['close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(df.tail(10).sort_values('timestamp', ascending=False), hide_index=True)

else:
    st.error(f"Data file for {ticker} not found.")

st.markdown("---")
st.caption("Automated by GitHub Actions | Models: LSTM + LightGBM")