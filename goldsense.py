import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained LSTM model
model = load_model("gold_price_prediction_model.keras")  # Adjust the path as necessary

# Load the dataset
data = pd.read_csv("gold_prices_daily_past_5_years.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['1 Gram Price (₹)']])

# Function to predict prices until a specific date
def predict_until_date(model, data, start_date, end_date, time_step, scaler, grams):
    # Convert start_date to a Pandas Timestamp
    start_date = pd.Timestamp(start_date)
    closest_date = data.index[data.index <= start_date][-1]
    start_idx = data.index.get_loc(closest_date)
    
    if start_idx < time_step:
        raise ValueError("Not enough historical data to make predictions.")
    
    last_sequence = data.iloc[start_idx - time_step:start_idx]['1 Gram Price (₹)'].values
    last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    
    num_days = (end_date - start_date).days
    predicted_prices = []
    
    for _ in range(num_days):
        X_input = last_sequence.reshape((1, time_step, 1))
        predicted_price = model.predict(X_input)
        predicted_price_rescaled = scaler.inverse_transform(predicted_price)
        
        predicted_price_for_grams = predicted_price_rescaled[0, 0] * grams
        predicted_prices.append(predicted_price_for_grams)
        
        last_sequence = np.append(last_sequence[1:], predicted_price, axis=0)
    
    future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=num_days)
    return pd.DataFrame({'Date': future_dates, 'Predicted Price (₹)': predicted_prices})

# Streamlit app UI
st.title("Gold Price Prediction App")

# Get the current date and user input date
now_date = dt.datetime.now().date()  # Keep this as a date
now_date_ts = pd.Timestamp(now_date)  # Convert to Pandas Timestamp
user_end_date = st.date_input("Enter the end date (YYYY/MM/DD)", min_value=now_date)
user_end_date_ts = pd.Timestamp(user_end_date)  # Convert to Pandas Timestamp
grams = st.number_input("Enter the number of grams for prediction", min_value=1.0, step=0.1)

if st.button("Predict Gold Prices"):
    # Ensure the user end date is after today
    if user_end_date_ts <= now_date_ts:
        st.error("End date must be after today.")
    else:
        time_step = 60  # Time step for LSTM
        
        # Predict the gold prices from today until the user-entered date
        predictions = predict_until_date(model, data, now_date_ts, user_end_date_ts, time_step, scaler, grams)
        
        # Format the dates to YYYY/MM/DD
        predictions['Date'] = predictions['Date'].dt.strftime('%Y/%m/%d')
        
        # Display the results
        st.write("Predicted Prices:")
        st.dataframe(predictions)

        # Plot the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(predictions['Date'], predictions['Predicted Price (₹)'], marker='o', label='Predicted Price for {} Grams'.format(grams))
        plt.xlabel('Date')
        plt.ylabel('Gold Price (₹)')
        plt.title('Gold Price Prediction from Now to User-Entered Date for {} Grams'.format(grams))
        plt.legend()
        
        # Render the plot in Streamlit
        st.pyplot(plt)

# Run the app with: streamlit run gold_price_prediction_app.py
