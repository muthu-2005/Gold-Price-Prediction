import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('rnn_gold_price_model.h5')

# Load your historical data
df = pd.read_csv('gold_prices_daily_past_2_years.csv')

# Assuming the date column is in 'YYYY-MM-DD' format
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

def predict_future_gold_prices(end_date, gram_amount):
    # Get today's date
    start_date = datetime.today()

    # Calculate the number of days between today and the end date
    end_date = pd.to_datetime(end_date)
    num_days = (end_date - start_date).days

    # Check if today's date is within the historical data range
    if start_date not in df.index:
        # Find the closest available date
        closest_date = df.index[df.index < start_date].max()
        if closest_date is pd.NaT:
            st.warning("No available data prior to today.")
            return None
        st.info(f"Using closest historical date: {closest_date.date()}")
        start_date = closest_date

    # Get data up to the start date
    historical_data = df.loc[:start_date]

    # Check if there's enough historical data for predictions
    if len(historical_data) < 60:
        st.warning("Not enough historical data to make predictions.")
        return None

    # Get the last 60 days of data for prediction
    last_60_days = historical_data['10 Gram Price (₹)'].values[-60:].reshape(-1, 1)

    # Normalize the last 60 days data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_last_60_days = scaler.fit_transform(last_60_days)

    # Prepare input for the model
    x_input = scaled_last_60_days.reshape(1, 60, 1)

    future_prices = []

    for _ in range(num_days):
        # Predict the next day's price
        predicted_price_scaled = model.predict(x_input)

        # Inverse transform to get the original price
        predicted_price_10g = scaler.inverse_transform(predicted_price_scaled)[0][0]
        predicted_price_per_gram = predicted_price_10g / 10  # 1 gram price

        # Calculate the price based on user input
        predicted_price = predicted_price_per_gram * gram_amount

        # Store the predicted price
        future_prices.append(predicted_price)

        # Update the input for the next prediction
        new_data = np.append(scaled_last_60_days, predicted_price_scaled)[1:]  # remove the first element
        x_input = new_data.reshape(1, 60, 1)

    return future_prices, num_days

# Streamlit user interface
st.title('Gold Price Prediction')

# Input for end date and gram amount
end_date = st.date_input("Select the ending date for prediction:", datetime.today())
gram_amount = st.number_input("Enter the amount in grams (1, 8, 10, etc.):", min_value=1.0, value=10.0)

if st.button("Predict Gold Prices"):
    future_prices, num_days = predict_future_gold_prices(end_date, gram_amount)

    # Print the predicted future prices
    if future_prices is not None:
        st.write(f"**Predicted Gold Prices from Today to {end_date} for {gram_amount} grams:**")
        for i in range(num_days):
            st.write(f"Day {i + 1}: ₹{future_prices[i]:.2f}")

        # Prepare data for plotting
        # Get last 10 days of historical data
        last_10_days = df.loc[pd.date_range(end=pd.Timestamp(datetime.now().date()) - timedelta(days=10), periods=10)]

        # Create a date range for the historical data
        historical_dates = last_10_days.index.tolist()

        # Create combined list of historical and future prices
        combined_prices = last_10_days['10 Gram Price (₹)'].tolist()[-10:] + future_prices
        combined_dates = historical_dates + pd.date_range(start=datetime.today(), periods=num_days).tolist()

        # Plotting the results
        plt.figure(figsize=(12, 6))
        plt.plot(combined_dates, combined_prices, label=f'Price for {gram_amount} Grams', marker='o', color='blue')
        plt.axvline(x=datetime.now().date(), color='red', linestyle='--', label='Today')
        plt.title('Predicted Gold Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(plt)
