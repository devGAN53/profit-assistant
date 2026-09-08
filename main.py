import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

st.set_page_config(page_title="Private Profit Assistant Pro", layout="wide")
st.title("💰 Private Profit Assistant (Pro)")

# Create Tab Navigation
tab1, tab2 = st.tabs(["📉 Quantitative ML Signal", "🏢 Fundamental Health Checker"])

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="1y")
    return df

@st.cache_data(ttl=3600)
def load_market_data():
    klci = yf.Ticker("^KLSE")
    return klci.history(period="1y")

@st.cache_data(ttl=3600)
def load_fundamental_metrics(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    metrics = {
        "company_name": ticker_symbol,
        "pe_ratio": None,
        "roe": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "profit_margins": None
    }
    
    # 1. Try standard info dictionary
    try:
        info = stock.info
        if info and isinstance(info, dict):
            metrics["company_name"] = info.get("longName", ticker_symbol)
            metrics["pe_ratio"] = info.get("trailingPE")
            metrics["roe"] = info.get("returnOnEquity")
            metrics["debt_to_equity"] = info.get("debtToEquity")
            metrics["current_ratio"] = info.get("currentRatio")
            metrics["profit_margins"] = info.get("profitMargins")
    except Exception:
        pass

    # 2. Fallback: Parse Financial Statements manually if metrics are missing
    try:
        bs = stock.balance_sheet
        inc = stock.financials
        
        if not bs.empty and not inc.empty:
            # Current Ratio
            if metrics["current_ratio"] is None:
                if "Current Assets" in bs.index and "Current Liabilities" in bs.index:
                    ca = bs.loc["Current Assets"].iloc[0]
                    cl = bs.loc["Current Liabilities"].iloc[0]
                    if cl and cl != 0:
                        metrics["current_ratio"] = ca / cl

            # Debt to Equity
            if metrics["debt_to_equity"] is None:
                if "Total Debt" in bs.index and "Stockholders Equity" in bs.index:
                    tot_debt = bs.loc["Total Debt"].iloc[0]
                    equity = bs.loc["Stockholders Equity"].iloc[0]
                    if equity and equity != 0:
                        metrics["debt_to_equity"] = tot_debt / equity

            # Profit Margin
            if metrics["profit_margins"] is None:
                if "Net Income" in inc.index and "Total Revenue" in inc.index:
                    net_inc = inc.loc["Net Income"].iloc[0]
                    rev = inc.loc["Total Revenue"].iloc[0]
                    if rev and rev != 0:
                        metrics["profit_margins"] = net_inc / rev

            # ROE
            if metrics["roe"] is None:
                if "Net Income" in inc.index and "Stockholders Equity" in bs.index:
                    net_inc = inc.loc["Net Income"].iloc[0]
                    equity = bs.loc["Stockholders Equity"].iloc[0]
                    if equity and equity != 0:
                        metrics["roe"] = net_inc / equity
    except Exception:
        pass

    return metrics


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

                market_sma_20 = market_data['Close'].rolling(window=20).mean().iloc[-1]
                market_current = market_data['Close'].iloc[-1]
                is_market_healthy = market_current >= market_sma_20

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
            data = load_fundamental_metrics(fund_symbol)
            
            company_name = data["company_name"]
            pe_ratio = data["pe_ratio"]
            roe = data["roe"]
            debt_to_equity = data["debt_to_equity"]
            current_ratio = data["current_ratio"]
            profit_margins = data["profit_margins"]
            
            st.subheader(f"Results for: {company_name}")
            
            score = 0
            max_score = 0
            checklist = []

            # 1. ROE
            if roe is not None and not np.isnan(roe):
                max_score += 3
                roe_pct = roe * 100 if abs(roe) < 5 else roe
                if roe_pct >= 15:
                    score += 3
                    checklist.append(("🟢 ROE", f"{roe_pct:.1f}% (Excellent >= 15%)"))
                elif roe_pct >= 8:
                    score += 2
                    checklist.append(("🟡 ROE", f"{roe_pct:.1f}% (Moderate 8%-15%)"))
                else:
                    checklist.append(("🔴 ROE", f"{roe_pct:.1f}% (Weak < 8%)"))
            else:
                checklist.append(("⚪ ROE", "Not reported on Yahoo Finance"))

            # 2. Debt-to-Equity
            if debt_to_equity is not None and not np.isnan(debt_to_equity):
                max_score += 3
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
                checklist.append(("⚪ Debt-to-Equity", "Not reported on Yahoo Finance"))

            # 3. Current Ratio
            if current_ratio is not None and not np.isnan(current_ratio):
                max_score += 2
                if current_ratio >= 1.5:
                    score += 2
                    checklist.append(("🟢 Current Ratio", f"{current_ratio:.2f}x (Strong Liquidity >= 1.5x)"))
                elif current_ratio >= 1.0:
                    score += 1
                    checklist.append(("🟡 Current Ratio", f"{current_ratio:.2f}x (Acceptable 1.0x-1.5x)"))
                else:
                    checklist.append(("🔴 Current Ratio", f"{current_ratio:.2f}x (Liquidity Risk < 1.0x)"))
            else:
                checklist.append(("⚪ Current Ratio", "Not reported on Yahoo Finance"))

            # 4. Profit Margin
            if profit_margins is not None and not np.isnan(profit_margins):
                max_score += 2
                pm_pct = profit_margins * 100 if abs(profit_margins) < 5 else profit_margins
                if pm_pct >= 10:
                    score += 2
                    checklist.append(("🟢 Profit Margin", f"{pm_pct:.1f}% (Healthy >= 10%)"))
                elif pm_pct > 0:
                    score += 1
                    checklist.append(("🟡 Profit Margin", f"{pm_pct:.1f}% (Thin Profit 0%-10%)"))
                else:
                    checklist.append(("🔴 Profit Margin", f"{pm_pct:.1f}% (Unprofitable < 0%)"))
            else:
                checklist.append(("⚪ Profit Margin", "Not reported on Yahoo Finance"))

            st.markdown("---")
            if max_score > 0:
                final_percentage = (score / max_score) * 100
                st.info(f"### Adjusted Fundamental Score: {score}/{max_score} ({final_percentage:.0f}%)")
            else:
                st.warning("⚠️ Yahoo Finance does not supply financial statements for this stock ticker. Refer to the Quantitative ML Signal tab for price predictions.")

            st.markdown("---")
            st.subheader("📋 Financial Metric Breakdown")
            c1, c2 = st.columns(2)
            
            for idx, (label, status) in enumerate(checklist):
                if idx % 2 == 0:
                    c1.metric(label, status)
                else:
                    c2.metric(label, status)
                    
            if pe_ratio is not None and not np.isnan(pe_ratio):
                st.caption(f"ℹ️ **Trailing P/E Ratio:** {pe_ratio:.2f}x")

        except Exception as e:
            st.error(f"Fundamental data processing error for {fund_symbol}: {e}")
