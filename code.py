import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
from datetime import timedelta

# Load the dataset
gold_price_data = pd.read_csv("C:/Users/muthu/Downloads/gold_price_per_pavan.csv")

# Sort the dataset by date to ensure it's in chronological order
gold_price_data['Date'] = pd.to_datetime(gold_price_data['Date'])
gold_price_data.sort_values('Date', inplace=True)

# Feature Engineering: Create lag features (e.g., previous day's price)
gold_price_data['Prev_Day_Price'] = gold_price_data['Price_Per_Pavan'].shift(1)
gold_price_data['3_Day_MA'] = gold_price_data['Price_Per_Pavan'].rolling(window=3).mean()
gold_price_data['7_Day_MA'] = gold_price_data['Price_Per_Pavan'].rolling(window=7).mean()

# Drop the first few rows with NaN due to lagging and rolling
gold_price_data.dropna(inplace=True)

# Define the features and target
features = ['Prev_Day_Price', '3_Day_MA', '7_Day_MA']
X = gold_price_data[features]
y = gold_price_data['Price_Per_Pavan']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions for evaluation
y_pred = model.predict(X_test)

# Evaluate the model (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit app UI
st.title("Gold Price Prediction")


# User input for future date prediction
input_future_date_str = st.text_input("Enter a future date (YYYY-MM-DD):")

# Function to predict future price
def predict_future_price(input_future_date_str):
    try:
        # Convert the user input future date to a datetime object
        input_future_date = pd.to_datetime(input_future_date_str)

        # Get the last available date and the corresponding data
        last_row = gold_price_data.iloc[-1]
        last_date = last_row['Date']
        last_price = last_row['Price_Per_Pavan']
        prev_day_price = last_row['Prev_Day_Price']
        moving_avg_3 = last_row['3_Day_MA']
        moving_avg_7 = last_row['7_Day_MA']

        # Calculate how many days into the future we need to predict
        days_to_predict = (input_future_date - last_date).days

        if days_to_predict <= 0:
            st.write("The input date should be a future date.")
            return

        st.write(f"Predicting for {days_to_predict} days into the future...")

        predictions = []

        # Iteratively predict the next day's price until the input future date
        for day in range(days_to_predict):
            # Prepare the input features for the next day
            next_day_input = np.array([[prev_day_price, moving_avg_3, moving_avg_7]])

            # Predict the next day's price
            next_day_price = model.predict(next_day_input)[0]

            # Print the predicted price
            predicted_date = last_date + timedelta(days=day + 1)
            predictions.append([predicted_date.date(), next_day_price])

            # Update the values for the next iteration (simulating the future)
            prev_day_price = next_day_price
            moving_avg_3 = (moving_avg_3 * 2 + next_day_price) / 3  # Simple update for 3-day MA
            moving_avg_7 = (moving_avg_7 * 6 + next_day_price) / 7  # Simple update for 7-day MA

        return predictions
    except Exception as e:
        st.error(f"Error: {e}")

# When a user provides a future date, show the predictions in a table
if input_future_date_str:
    predictions = predict_future_price(input_future_date_str)
    if predictions:
        # Create a DataFrame for better visualization in a table
        df_predictions = pd.DataFrame(predictions, columns=["Date", "Predicted Price of Gold(INR)"])
        st.write("Predicted Prices for Future Dates:")
        st.dataframe(df_predictions)  # Display the table in a row x column format
