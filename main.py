import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

st.set_page_config(page_title="Private Profit Assistant Pro", layout="wide")
st.title("💰 Private Profit Assistant (Pro)")

tab1, tab2 = st.tabs(["📉 Quantitative ML Signal", "🏢 Fundamental Health Checker"])

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    return stock.history(period="1y")

@st.cache_data(ttl=3600)
def load_market_data():
    klci = yf.Ticker("^KLSE")
    return klci.history(period="1y")

# ==========================================
# TAB 1: QUANTITATIVE ML SIGNAL & RISK
# ==========================================
with tab1:
    st.header("Technical Momentum & Risk Analysis")
    
    # Execution Mode Selection
    trade_mode = st.radio(
        "Select Execution Strategy:",
        ["Exchange / Limit Order (Normal Way)", "Instant Buy / Market Order"],
        horizontal=True
    )
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        purchase_price = st.number_input("Enter target/purchase price (RM):", value=0.20, step=0.01, key="quant_price")
    with col_b:
        symbol = st.text_input("Enter stock symbol (e.g. BTC-USD, 0285.KL):", value="BTC-USD", key="quant_symbol")

    if st.button("Run Quant Prediction", key="run_quant"):
        try:
            data = load_data(symbol)
            
            if data.empty:
                st.error("No data found for this ticker symbol. Please check the stock code.")
            else:
                data['SMA_10'] = data['Close'].rolling(window=10).mean()
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['Returns'] = data['Close'].pct_change()
                
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-9)
                data['RSI'] = 100 - (100 / (1 + rs))

                ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = ema_12 - ema_26
                data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['ATR'] = tr.rolling(window=14).mean()

                data['Target'] = data['Close'].shift(-1)
                data_clean = data.dropna().copy()
                
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_20', 
                            'Returns', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
                X = data_clean[features]
                y = data_clean['Target']
                
                model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
                model.fit(X, y)
                
                latest_features = X.iloc[[-1]]
                predicted_price = model.predict(latest_features)[0]
                current_atr = data_clean['ATR'].iloc[-1]
                
                # Apply Fee Adjustments Based on Mode Selection
                if "Exchange" in trade_mode:
                    fee_pct = 0.0035  # ~0.35% exchange limit order fee
                    execution_label = "Limit Order Target"
                else:
                    fee_pct = 0.0200  # ~2.0% instant buy fee
                    execution_label = "Instant Buy Estimate"

                break_even_price = purchase_price * (1 + fee_pct)
                required_min_target = break_even_price + (0.5 * current_atr)

                st.subheader(f"Predicted Target Price: RM {predicted_price:.2f}")
                st.caption(f"⚙️ Mode: **{trade_mode}** | Est. Fee Shift: **{fee_pct*100:.2f}%** | Break-even Entry: **RM {break_even_price:.2f}**")

                if predicted_price >= required_min_target:
                    st.success("🟢 Signal: BUY / STRONG HOLD")
                    st.write("Predicted gain comfortably covers fee overhead and volatility bounds.")
                elif predicted_price > break_even_price:
                    st.warning("🟡 Signal: WEAK BUY / HOLD")
                    st.write("Predicted target covers transaction fees, but profit margin is thin relative to market noise.")
                else:
                    st.error("🔴 Signal: AVOID / SELL")
                    st.write("Predicted target price is below the required break-even point after fees.")

                # Risk Management Calculations
                stop_loss_price = purchase_price - (1.5 * current_atr)

                st.markdown("---")
                st.subheader("📊 Execution & Risk Parameters")
                col1, col2 = st.columns(2)
                col1.metric("🛡️ Suggested Stop-Loss", f"RM {max(0.01, stop_loss_price):.2f}")
                col2.metric(f"🎯 {execution_label}", f"RM {predicted_price:.2f}")

        except Exception as e:
            st.error(f"Error executing quantitative prediction: {e}")
