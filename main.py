streamlit
yfinance>=0.2.54
scikit-learn
numpy
pandas
requests

Same for this 

import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import requests

st.set_page_config(page_title="Private Profit Assistant Pro", layout="wide")
st.title("💰 Private Profit Assistant (Pro)")

# Create Tab Navigation
tab1, tab2 = st.tabs(["📉 Quantitative ML Signal", "🏢 Fundamental Health Checker"])

# --- SESSION SETUP TO PREVENT RATE LIMITS ---
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    session = get_yf_session()
    stock = yf.Ticker(ticker_symbol, session=session)
    df = stock.history(period="1y")
    return df

@st.cache_data(ttl=3600)
def load_market_data():
    session = get_yf_session()
    klci = yf.Ticker("^KLSE", session=session)
    return klci.history(period="1y")

@st.cache_data(ttl=3600)
def load_fundamental_info(ticker_symbol):
    session = get_yf_session()
    stock = yf.Ticker(ticker_symbol, session=session)
    return stock.info


# ==========================================
# TAB 1: QUANTITATIVE ML SIGNAL & RISK
# ==========================================
with tab1:
    st.header("Technical Momentum & Risk Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        purchase_price = st.number_input("Enter your purchase price (RM):", value=0.20, step=0.01, key="quant_price")
    with col_b:
        symbol = st.text_input("Enter stock symbol (e.g. 0285.KL):", value="0285.KL", key="quant_symbol")

    if st.button("Run Quant Prediction", key="run_quant"):
        try:
            data = load_data(symbol)
            market_data = load_market_data()
            
            if data.empty:
                st.error("No data found for this ticker symbol. Please check the stock code.")
            else:
                # 1. Technical Indicators (MACD & ATR)
                data['SMA_10'] = data['Close'].rolling(window=10).mean()
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['Returns'] = data['Close'].pct_change()
                
                # RSI
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-9)
                data['RSI'] = 100 - (100 / (1 + rs))

                # MACD
                ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = ema_12 - ema_26
                data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

                # ATR
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['ATR'] = tr.rolling(window=14).mean()

                # 2. Bursa Market Filter
                market_sma_20 = market_data['Close'].rolling(window=20).mean().iloc[-1]
                market_current = market_data['Close'].iloc[-1]
                is_market_healthy = market_current >= market_sma_20

                # 3. Model Training
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
                
                required_min_target = purchase_price + (0.5 * current_atr)

                st.subheader(f"Predicted Target Price: RM {predicted_price:.2f}")

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

                # 4. Risk Parameters
                stop_loss_price = purchase_price - (1.5 * current_atr)
                take_profit_price = predicted_price

                st.markdown("---")
                st.subheader("📊 Risk Management Parameters")
                col1, col2 = st.columns(2)
                col1.metric("🛡️ Suggested Stop-Loss", f"RM {max(0.01, stop_loss_price):.2f}")
                col2.metric("🎯 Suggested Take-Profit Target", f"RM {take_profit_price:.2f}")

        except Exception as e:
            st.error(f"Error executing quantitative prediction: {e}")


# ==========================================
# TAB 2: FUNDAMENTAL HEALTH CHECKER
# ==========================================
with tab2:
    st.header("Financial Health & Valuation Screening")
    fund_symbol = st.text_input("Enter stock symbol for screening (e.g. 0285.KL):", value="0285.KL", key="fund_symbol")
    
    if st.button("Check Financial Health", key="run_fund"):
        try:
            info = load_fundamental_info(fund_symbol)
            
            if not info or len(info) <= 1:
                st.error("Rate limit active. Please wait 1–2 minutes before clicking check again.")
            else:
                company_name = info.get("longName", fund_symbol)
                pe_ratio = info.get("trailingPE", None)
                roe = info.get("returnOnEquity", None)
                debt_to_equity = info.get("debtToEquity", None)
                current_ratio = info.get("currentRatio", None)
                profit_margins = info.get("profitMargins", None)
                
                st.subheader(f"Results for: {company_name}")
                
                # --- SCORING SYSTEM (Out of 10) ---
                score = 0
                max_score = 10
                checklist = []

                # 1. ROE (Return on Equity) -> Max 3 pts
                if roe is not None:
                    roe_pct = roe * 100
                    if roe_pct >= 15:
                        score += 3
                        checklist.append(("🟢 ROE", f"{roe_pct:.1f}% (Excellent >= 15%)"))
                    elif roe_pct >= 8:
                        score += 2
                        checklist.append(("🟡 ROE", f"{roe_pct:.1f}% (Moderate 8%-15%)"))
                    else:
                        checklist.append(("🔴 ROE", f"{roe_pct:.1f}% (Weak < 8%)"))
                else:
                    checklist.append(("⚪ ROE", "Data unavailable"))

                # 2. Debt-to-Equity Ratio -> Max 3 pts
                if debt_to_equity is not None:
                    de_val = debt_to_equity if debt_to_equity < 10 else debt_to_equity / 100
                    if de_val <= 0.5:
                        score += 3
                        checklist.append(("🟢 Debt-to-Equity", f"{de_val:.2f}x (Low Risk <= 0.5x)"))
                    elif de_val <= 1.0:
                        score += 2
                        checklist.append(("🟡 Debt-to-Equity", f"{de_val:.2f}x (Moderate Risk 0.5x-1.0x)"))
                    else:
                        checklist.append(("🔴 Debt-to-Equity", f"{de_val:.2f}x (High Debt > 1.0x)"))
                else:
                    checklist.append(("⚪ Debt-to-Equity", "Data unavailable"))

                # 3. Current Ratio (Liquidity) -> Max 2 pts
                if current_ratio is not None:
                    if current_ratio >= 1.5:
                        score += 2
                        checklist.append(("🟢 Current Ratio", f"{current_ratio:.2f}x (Strong Liquidity >= 1.5x)"))
                    elif current_ratio >= 1.0:
                        score += 1
                        checklist.append(("🟡 Current Ratio", f"{current_ratio:.2f}x (Acceptable 1.0x-1.5x)"))
                    else:
                        checklist.append(("🔴 Current Ratio", f"{current_ratio:.2f}x (Liquidity Risk < 1.0x)"))
                else:
                    checklist.append(("⚪ Current Ratio", "Data unavailable"))

                # 4. Profit Margins -> Max 2 pts
                if profit_margins is not None:
                    pm_pct = profit_margins * 100
                    if pm_pct >= 10:
                        score += 2
                        checklist.append(("🟢 Profit Margin", f"{pm_pct:.1f}% (Healthy >= 10%)"))
                    elif pm_pct > 0:
                        score += 1
                        checklist.append(("🟡 Profit Margin", f"{pm_pct:.1f}% (Thin Profit 0%-10%)"))
                    else:
                        checklist.append(("🔴 Profit Margin", f"{pm_pct:.1f}% (Unprofitable < 0%)"))
                else:
                    checklist.append(("⚪ Profit Margin", "Data unavailable"))

                # Display Overall Score
                st.markdown("---")
                if score >= 7:
                    st.success(f"### Overall Fundamental Score: {score}/{max_score} — STRONG HEALTH")
                    st.write("This company demonstrates solid profitability, low balance sheet risk, and good liquidity.")
                elif score >= 4:
                    st.warning(f"### Overall Fundamental Score: {score}/{max_score} — MODERATE HEALTH")
                    st.write("This company has acceptable fundamentals, but pay attention to specific red flags below.")
                else:
                    st.error(f"### Overall Fundamental Score: {score}/{max_score} — WEAK / HIGH RISK")
                    st.write("Caution: This company suffers from weak profitability, high debt, or poor liquidity.")

                # Metrics Table / Grid
                st.markdown("---")
                st.subheader("📋 Financial Metric Breakdown")
                c1, c2 = st.columns(2)
                
                for idx, (label, status) in enumerate(checklist):
                    if idx % 2 == 0:
                        c1.metric(label, status)
                    else:
                        c2.metric(label, status)
                        
                if pe_ratio is not None:
                    st.caption(f"ℹ️ **Trailing P/E Ratio:** {pe_ratio:.2f}x")

        except Exception as e:
            st.error(f"Could not load fundamental data for {fund_symbol}. Error: {e}")


Same for this

import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import requests

st.set_page_config(page_title="Private Profit Assistant Pro", layout="wide")
st.title("💰 Private Profit Assistant (Pro)")

# Create Tab Navigation
tab1, tab2 = st.tabs(["📉 Quantitative ML Signal", "🏢 Fundamental Health Checker"])

# --- SESSION SETUP TO PREVENT RATE LIMITS ---
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    session = get_yf_session()
    stock = yf.Ticker(ticker_symbol, session=session)
    df = stock.history(period="1y")
    return df

@st.cache_data(ttl=3600)
def load_market_data():
    session = get_yf_session()
    klci = yf.Ticker("^KLSE", session=session)
    return klci.history(period="1y")

@st.cache_data(ttl=3600)
def load_fundamental_info(ticker_symbol):
    session = get_yf_session()
    stock = yf.Ticker(ticker_symbol, session=session)
    return stock.info


# ==========================================
# TAB 1: QUANTITATIVE ML SIGNAL & RISK
# ==========================================
with tab1:
    st.header("Technical Momentum & Risk Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        purchase_price = st.number_input("Enter your purchase price (RM):", value=0.20, step=0.01, key="quant_price")
    with col_b:
        symbol = st.text_input("Enter stock symbol (e.g. 0285.KL):", value="0285.KL", key="quant_symbol")

    if st.button("Run Quant Prediction", key="run_quant"):
        try:
            data = load_data(symbol)
            market_data = load_market_data()
            
            if data.empty:
                st.error("No data found for this ticker symbol. Please check the stock code.")
            else:
                # Technical Indicators
                data['SMA_10'] = data['Close'].rolling(window=10).mean()
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['Returns'] = data['Close'].pct_change()
                
                # RSI
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-9)
                data['RSI'] = 100 - (100 / (1 + rs))

                # MACD
                ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = ema_12 - ema_26
                data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

                # ATR
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['ATR'] = tr.rolling(window=14).mean()

                # Bursa Market Filter
                market_sma_20 = market_data['Close'].rolling(window=20).mean().iloc[-1]
                market_current = market_data['Close'].iloc[-1]
                is_market_healthy = market_current >= market_sma_20

                # Model Training
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
                
                required_min_target = purchase_price + (0.5 * current_atr)

                st.subheader(f"Predicted Target Price: RM {predicted_price:.2f}")

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

                # Risk Parameters
                stop_loss_price = purchase_price - (1.5 * current_atr)
                take_profit_price = predicted_price

                st.markdown("---")
                st.subheader("📊 Risk Management Parameters")
                col1, col2 = st.columns(2)
                col1.metric("🛡️ Suggested Stop-Loss", f"RM {max(0.01, stop_loss_price):.2f}")
                col2.metric("🎯 Suggested Take-Profit Target", f"RM {take_profit_price:.2f}")

        except Exception as e:
            st.error(f"Error executing quantitative prediction: {e}")


# ==========================================
# TAB 2: FUNDAMENTAL HEALTH CHECKER
# ==========================================
with tab2:
    st.header("Financial Health & Valuation Screening")
    fund_symbol = st.text_input("Enter stock symbol for screening (e.g. 0285.KL):", value="0285.KL", key="fund_symbol")
    
    if st.button("Check Financial Health", key="run_fund"):
        try:
            info = load_fundamental_info(fund_symbol)
            
            if not info or len(info) <= 1:
                st.error("Rate limit active. Please wait 1–2 minutes before clicking check again.")
            else:
                company_name = info.get("longName", fund_symbol)
                pe_ratio = info.get("trailingPE", None)
                roe = info.get("returnOnEquity", None)
                debt_to_equity = info.get("debtToEquity", None)
                current_ratio = info.get("currentRatio", None)
                profit_margins = info.get("profitMargins", None)
                
                st.subheader(f"Results for: {company_name}")
                
                score = 0
                max_score = 10
                checklist = []

                if roe is not None:
                    roe_pct = roe * 100
                    if roe_pct >= 15:
                        score += 3
                        checklist.append(("🟢 ROE", f"{roe_pct:.1f}% (Excellent >= 15%)"))
                    elif roe_pct >= 8:
                        score += 2
                        checklist.append(("🟡 ROE", f"{roe_pct:.1f}% (Moderate 8%-15%)"))
                    else:
                        checklist.append(("🔴 ROE", f"{roe_pct:.1f}% (Weak < 8%)"))
                else:
                    checklist.append(("⚪ ROE", "Data unavailable"))

                if debt_to_equity is not None:
                    de_val = debt_to_equity if debt_to_equity < 10 else debt_to_equity / 100
                    if de_val <= 0.5:
                        score += 3
                        checklist.append(("🟢 Debt-to-Equity", f"{de_val:.2f}x (Low Risk <= 0.5x)"))
                    elif de_val <= 1.0:
                        score += 2
                        checklist.append(("🟡 Debt-to-Equity", f"{de_val:.2f}x (Moderate Risk 0.5x-1.0x)"))
                    else:
                        checklist.append(("🔴 Debt-to-Equity", f"{de_val:.2f}x (High Debt > 1.0x)"))
                else:
                    checklist.append(("⚪ Debt-to-Equity", "Data unavailable"))

                if current_ratio is not None:
                    if current_ratio >= 1.5:
                        score += 2
                        checklist.append(("🟢 Current Ratio", f"{current_ratio:.2f}x (Strong Liquidity >= 1.5x)"))
                    elif current_ratio >= 1.0:
                        score += 1
                        checklist.append(("🟡 Current Ratio", f"{current_ratio:.2f}x (Acceptable 1.0x-1.5x)"))
                    else:
                        checklist.append(("🔴 Current Ratio", f"{current_ratio:.2f}x (Liquidity Risk < 1.0x)"))
                else:
                    checklist.append(("⚪ Current Ratio", "Data unavailable"))

                if profit_margins is not None:
                    pm_pct = profit_margins * 100
                    if pm_pct >= 10:
                        score += 2
                        checklist.append(("🟢 Profit Margin", f"{pm_pct:.1f}% (Healthy >= 10%)"))
                    elif pm_pct > 0:
                        score += 1
                        checklist.append(("🟡 Profit Margin", f"{pm_pct:.1f}% (Thin Profit 0%-10%)"))
                    else:
                        checklist.append(("🔴 Profit Margin", f"{pm_pct:.1f}% (Unprofitable < 0%)"))
                else:
                    checklist.append(("⚪ Profit Margin", "Data unavailable"))

                st.markdown("---")
                if score >= 7:
                    st.success(f"### Overall Fundamental Score: {score}/{max_score} — STRONG HEALTH")
                    st.write("This company demonstrates solid profitability, low balance sheet risk, and good liquidity.")
                elif score >= 4:
                    st.warning(f"### Overall Fundamental Score: {score}/{max_score} — MODERATE HEALTH")
                    st.write("This company has acceptable fundamentals, but pay attention to specific red flags below.")
                else:
                    st.error(f"### Overall Fundamental Score: {score}/{max_score} — WEAK / HIGH RISK")
                    st.write("Caution: This company suffers from weak profitability, high debt, or poor liquidity.")

                st.markdown("---")
                st.subheader("📋 Financial Metric Breakdown")
                c1, c2 = st.columns(2)
                
                for idx, (label, status) in enumerate(checklist):
                    if idx % 2 == 0:
                        c1.metric(label, status)
                    else:
                        c2.metric(label, status)
                        
                if pe_ratio is not None:
                    st.caption(f"ℹ️ **Trailing P/E Ratio:** {pe_ratio:.2f}x")

        except Exception as e:
            st.error(f"Could not load fundamental data for {fund_symbol}. Error: {e}")
