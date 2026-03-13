import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1. Page Title
st.title("💰 Private Profit Assistant")

# 2. Web-based User Inputs
# These create input boxes in your browser, not the terminal
buy_price = st.number_input("Enter your purchase price:", min_value=0.0, value=250.0)
symbol = st.text_input("Enter stock symbol:", "AAPL")

# 3. Execution Button
# The app will only calculate when you click this button
if st.button("Run Prediction"):
    # Fetch Data
    stock = yf.Ticker(symbol)
    data = stock.history(period="3mo")
    
    # Feature Engineering
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    # Model Training
    X = data[['Close', 'SMA_5']].values
    y = data['Target'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Prediction
    prediction = model.predict([[data['Close'].iloc[-1], data['SMA_5'].iloc[-1]]])[0]
    profit_potential = ((prediction - buy_price) / buy_price) * 100

    # 4. Display Results
    st.write(f"### Strategy Report for {symbol}")
    st.write(f"Predicted Price: **{prediction:.2f}**")
    st.write(f"Potential Return: **{profit_potential:.2f}%**")

    # Signal Logic
    if prediction > buy_price * 1.05:
        st.success("SIGNAL: Sell now for target profit!")
    elif prediction < buy_price:
        st.warning("SIGNAL: Hold, price is below your buy price.")
    else:
        st.info("SIGNAL: Hold, waiting for more growth.")

    # 5. Visualization
    fig, ax = plt.subplots()
    ax.plot(data['Close'].values, label='Actual Price')
    ax.axhline(y=prediction, color='r', linestyle='--', label='Predicted Price')
    ax.legend()
    st.pyplot(fig)
