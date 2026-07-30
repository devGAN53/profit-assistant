import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

st.title("💰 Private Profit Assistant")

# --- USER INPUTS ---
purchase_price = st.number_input("Enter your purchase price:", value=5.08, step=0.01)
symbol = st.text_input("Enter stock symbol:", value="5067.KL")

# --- CACHED DATA FETCHING (Prevents Rate Limit Errors) ---
@st.cache_data(ttl=3600)  # Caches data for 1 hour
def load_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="6mo")
    return df

if st.button("Run Prediction"):
    try:
        data = load_data(symbol)
        
        if data.empty:
            st.error("No data found for this ticker symbol. Please check the stock code.")
        else:
            # --- DATA PREPARATION ---
            data['Target'] = data['Close'].shift(-1)
            data_clean = data.dropna().copy()
            
            # Simple feature setup using historical OHLCV
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            X = data_clean[features]
            y = data_clean['Target']
            
            # --- MODEL TRAINING ---
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Predict next day's price based on the latest available market data
            latest_features = X.iloc[[-1]]
            predicted_price = model.predict(latest_features)[0]
            
            st.subheader(f"Predicted Target Price: RM {predicted_price:.2f}")
            
            # --- SIGNAL LOGIC (FIXED) ---
            # Triggers BUY as long as the prediction beats your purchase price
            if predicted_price > purchase_price:
                st.success(f"🟢 Signal: BUY / HOLD FOR TARGET PROFIT")
                st.write(f"The model predicts a rise above your entry price of RM {purchase_price:.2f}.")
            elif predicted_price < purchase_price:
                st.error(f"🔴 Signal: SELL / AVOID")
                st.write(f"The model predicts the price will stay below your entry price of RM {purchase_price:.2f}.")
            else:
                st.warning("🟡 Signal: NEUTRAL / HOLD")

    except Exception as e:
        st.error(f"Error fetching data or running prediction: {e}")
