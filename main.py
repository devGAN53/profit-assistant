import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

st.title("💰 Private Profit Assistant (Pro)")

# --- USER INPUTS ---
purchase_price = st.number_input("Enter your purchase price (RM):", value=0.20, step=0.01)
symbol = st.text_input("Enter stock symbol (e.g. 0285.KL):", value="0285.KL")

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="1y")
    return df

@st.cache_data(ttl=3600)
def load_market_data():
    # Fetches FTSE Bursa Malaysia KLCI index data for market sentiment filter
    klci = yf.Ticker("^KLSE")
    return klci.history(period="1y")

if st.button("Run Prediction"):
    try:
        data = load_data(symbol)
        market_data = load_market_data()
        
        if data.empty:
            st.error("No data found for this ticker symbol. Please check the stock code.")
        else:
            # --- 1. TECHNICAL INDICATORS (MACD & ATR) ---
            # Moving Averages
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            
            # Daily Returns
            data['Returns'] = data['Close'].pct_change()
            
            # RSI (14 Days)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            data['RSI'] = 100 - (100 / (1 + rs))

            # MACD Line & Signal Line
            ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # ATR (Average True Range - Volatility Measure)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['ATR'] = tr.rolling(window=14).mean()

            # --- 2. BURSA MARKET TREND FILTER ---
            market_sma_20 = market_data['Close'].rolling(window=20).mean().iloc[-1]
            market_current = market_data['Close'].iloc[-1]
            is_market_healthy = market_current >= market_sma_20

            # --- 3. TARGET & DATA PREPARATION ---
            data['Target'] = data['Close'].shift(-1)
            data_clean = data.dropna().copy()
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_20', 
                        'Returns', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
            X = data_clean[features]
            y = data_clean['Target']
            
            # --- 4. MODEL TRAINING ---
            model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
            model.fit(X, y)
            
            latest_features = X.iloc[[-1]]
            predicted_price = model.predict(latest_features)[0]
            current_atr = data_clean['ATR'].iloc[-1]
            
            # Target buffer: predicted price must cover expected price movement noise
            required_min_target = purchase_price + (0.5 * current_atr)

            st.subheader(f"Predicted Target Price: RM {predicted_price:.2f}")

            # --- 5. ENHANCED SIGNAL LOGIC ---
            if not is_market_healthy:
                st.warning("⚠️ Market Warning: Overall Bursa Malaysia Index (^KLSE) is in a short-term downtrend. Trade with extra caution.")

            if predicted_price >= required_min_target:
                st.success("🟢 Signal: BUY / STRONG HOLD")
                st.write(f"Predicted price exceeds your entry price (RM {purchase_price:.2f}) with sufficient profit buffer.")
            elif predicted_price > purchase_price and predicted_price < required_min_target:
                st.warning("🟡 Signal: WEAK BUY / HOLD")
                st.write(f"Predicted price is slightly above entry price (RM {purchase_price:.2f}), but profit buffer is narrow relative to daily volatility.")
            else:
                st.error("🔴 Signal: SELL / AVOID")
                st.write(f"Predicted price is below or too close to your entry price of RM {purchase_price:.2f}.")

            # --- 6. RISK MANAGEMENT (STOP-LOSS & TAKE-PROFIT) ---
            stop_loss_price = purchase_price - (1.5 * current_atr)
            take_profit_price = predicted_price

            st.markdown("---")
            st.subheader("📊 Risk Management Parameters")
            col1, col2 = st.columns(2)
            col1.metric("🛡️ Suggested Stop-Loss", f"RM {max(0.01, stop_loss_price):.2f}")
            col2.metric("🎯 Suggested Take-Profit Target", f"RM {take_profit_price:.2f}")

    except Exception as e:
        st.error(f"Error fetching data or running prediction: {e}")
