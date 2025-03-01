import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# ---- Streamlit App Title ----
st.title("📈 Stock Price Prediction with RSI")

# ---- User Input ----
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

# ---- Fetch Stock Data ----
st.write(f"Fetching stock data for: {ticker}")
try:
    stock_data = yf.download(ticker, period=period, auto_adjust=False)

    if stock_data.empty:
        st.error("⚠ No data found! Please enter a valid ticker.")
    else:
        st.success("✅ Data Loaded Successfully!")
        
        # ---- Compute RSI ----
        def calculate_rsi(data, window=14):
            delta = data["Close"].diff(1)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
            avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return pd.Series(rsi).fillna(50)  # Fill NaNs with neutral RSI value

        stock_data["RSI"] = calculate_rsi(stock_data)

        # ---- Normalize Data ----
        scaler = MinMaxScaler()
        stock_data[["Close", "RSI"]] = scaler.fit_transform(stock_data[["Close", "RSI"]])

        # ---- Train Simple Linear Regression Model ----
        if len(stock_data) > 5:
            X = stock_data[["Close", "RSI"]].iloc[:-5].values  # Convert to NumPy array
            y = stock_data["Close"].shift(-1).dropna().iloc[:-5].values  # Ensure correct index
            
            model = LinearRegression()
            model.fit(X, y)

            # ---- Predict Next 5 Days ----
            future_X = stock_data[["Close", "RSI"]].iloc[-5:].values
            future_predictions = model.predict(future_X)

            # ---- Add Predictions to DataFrame ----
            if "Predicted Price" not in stock_data.columns:
                stock_data["Predicted Price"] = np.nan  # Create column if missing
            
            stock_data.iloc[-5:, stock_data.columns.get_loc("Predicted Price")] = future_predictions

            # ---- Display Data ----
            st.subheader("Stock Data with RSI & Predictions")
            st.write(stock_data.tail(10))

            # ---- Plot Stock Prices ----
            fig, ax = plt.subplots(figsize=(12, 6))
            stock_data["Close"].plot(ax=ax, label="Actual Price", color="blue")
            stock_data["Predicted Price"].plot(ax=ax, label="Predicted Price", color="red", linestyle="dashed")
            plt.title(f"{ticker} Stock Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Normalized Price")
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("⚠ Not enough data to train the model. Try selecting a longer time period.")

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
