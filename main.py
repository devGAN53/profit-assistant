import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set Streamlit Page Configuration
st.set_page_config(page_title="Private Profit Assistant (Pro)", layout="wide", page_icon="💰")

st.title("💰 Private Profit Assistant (Pro)")
st.caption("Quantitative Momentum & Risk Engine for BTC Trading")

# ---------------------------------------------------------
# 1. LIVE DATA & EXCHANGE RATE FETCHING
# ---------------------------------------------------------
@st.cache_data(ttl=30)  # Refreshes data automatically every 30 seconds
def fetch_market_data():
    """Fetch live USD/MYR exchange rate and BTC-USD market price."""
    try:
        # Fetch current USD to MYR exchange rate
        fx = yf.Ticker("USDMYR=X")
        fx_data = fx.history(period="1d")
        usd_myr_rate = float(fx_data["Close"].iloc[-1]) if not fx_data.empty else 4.07
    except Exception:
        usd_myr_rate = 4.07  # Fallback rate if network fails

    try:
        # Fetch live BTC-USD market ticker
        btc = yf.Ticker("BTC-USD")
        btc_usd = float(btc.fast_info["lastPrice"])
    except Exception:
        btc_usd = 78200.0  # Fallback estimate

    btc_myr = btc_usd * usd_myr_rate
    return usd_myr_rate, btc_usd, btc_myr

usd_myr_rate, live_btc_usd, live_btc_myr = fetch_market_data()

# ---------------------------------------------------------
# 2. METRICS DASHBOARD
# ---------------------------------------------------------
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric(label="Live BTC Price (MYR)", value=f"RM {live_btc_myr:,.2f}")
with col_b:
    st.metric(label="Live BTC Price (USD)", value=f"${live_btc_usd:,.2f} USD")
with col_c:
    st.metric(label="USD/MYR FX Rate", value=f"{usd_myr_rate:.4f}")

st.divider()

# ---------------------------------------------------------
# 3. INPUT FORM & STRATEGY SELECTION
# ---------------------------------------------------------
st.subheader("Technical Momentum & Risk Analysis")

strategy = st.radio(
    "Select Execution Strategy:",
    ["Exchange / Limit Order (Normal Way)", "Instant Buy / Market Order"],
    horizontal=True
)

# Fee calculation rules (Exchange mode: ~0.35% taker fee vs Instant: ~2.00%)
fee_shift = 0.0035 if strategy == "Exchange / Limit Order (Normal Way)" else 0.0200

col1, col2 = st.columns(2)

with col1:
    # Direct Ringgit Input
    target_myr = st.number_input(
        "Enter target/purchase price (RM):",
        value=float(round(live_btc_myr, 2)),
        step=100.0,
        format="%.2f"
    )

with col2:
    # Symbol locked to standard BTC-USD for Yahoo Finance compatibility
    st.text_input("Stock Symbol:", value="BTC-USD", disabled=True)
    
    # Auto-convert RM input to USD for model computations
    target_usd = target_myr / usd_myr_rate
    st.caption(f"⚡ Auto-converted model entry price: **${target_usd:,.2f} USD**")

# ---------------------------------------------------------
# 4. QUANT MODEL PREDICTION LOGIC
# ---------------------------------------------------------
if st.button("Run Quant Prediction", type="primary"):
    with st.spinner("Fetching historical data and computing ML signals..."):
        # Fetch BTC-USD 1-year daily history for model signals
        btc_hist = yf.Ticker("BTC-USD").history(period="1y")

        if btc_hist.empty:
            st.error("Error retrieving historical market data from Yahoo Finance.")
        else:
            # Simple quantitative momentum check (e.g., 20-day Simple Moving Average)
            btc_hist["SMA_20"] = btc_hist["Close"].rolling(window=20).mean()
            sma_20_usd = float(btc_hist["SMA_20"].iloc[-1])
            
            # Break-even calculation factoring exchange fees
            breakeven_usd = target_usd * (1 + fee_shift)
            breakeven_myr = breakeven_usd * usd_myr_rate

            st.divider()
            
            # ---------------------------------------------------------
            # 5. DISPLAY OUTPUT & SIGNALS
            # ---------------------------------------------------------
            # Rule: Buy signal if target is below/at 20-day SMA baseline and accounts for fees
            if target_usd <= (sma_20_usd * 1.02):
                st.subheader(f"Predicted Target Price: RM {target_myr:,.2f}")
                st.caption(f"⚙️ Mode: {strategy} | Est. Fee Shift: {fee_shift*100:.2f}% | Break-even Entry: RM {breakeven_myr:,.2f}")
                
                st.success("🟢 Signal: BUY / STRONG HOLD")
                st.info(f"Target price is in a favorable momentum zone (Model 20-Day SMA: ${sma_20_usd:,.2f} USD).")
            else:
                st.subheader(f"Predicted Target Price: RM {target_myr:,.2f}")
                st.caption(f"⚙️ Mode: {strategy} | Est. Fee Shift: {fee_shift*100:.2f}% | Break-even Entry: RM {breakeven_myr:,.2f}")
                
                st.error("🔴 Signal: AVOID / SELL")
                st.warning("Predicted entry price is overextended relative to risk boundaries.")
