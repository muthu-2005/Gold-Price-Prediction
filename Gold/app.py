import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model("C:/Users/muthu/Downloads/gold_price_prediction_model.keras")  # Adjust the path as necessary

# Load the dataset
data = pd.read_csv("C:/Users/muthu/Downloads/gold_prices_daily_past_5_years.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['1 Gram Price (₹)']])

# Function to predict prices until a specific date
def predict_until_date(model, data, start_date, end_date, time_step, scaler, grams):
    closest_date = data.index[data.index <= start_date][-1]
    start_idx = data.index.get_loc(closest_date)
    
    if start_idx < time_step:
        raise ValueError("Not enough historical data to make predictions.")
    
    # Prepare the last sequence for prediction
    last_sequence = data.iloc[start_idx - time_step:start_idx]['1 Gram Price (₹)'].values
    last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    
    num_days = (end_date - start_date).days
    predicted_prices = []
    
    for _ in range(num_days):
        X_input = last_sequence.reshape((1, time_step, 1))
        predicted_price = model.predict(X_input)
        predicted_price_rescaled = scaler.inverse_transform(predicted_price)
        
        # Calculate predicted price for the specified grams
        predicted_price_for_grams = predicted_price_rescaled[0, 0] * grams
        predicted_prices.append(predicted_price_for_grams)
        
        # Update the last_sequence for the next prediction
        last_sequence = np.append(last_sequence[1:], predicted_price, axis=0)
    
    future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=num_days)
    return pd.DataFrame({'Date': future_dates, 'Predicted Price (₹)': predicted_prices})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_end_date = request.form['end_date']
        grams = float(request.form['grams'])
        
        now_date = pd.Timestamp.now().normalize()  # Normalize current date

        # Convert user_end_date to Timestamp for comparison
        user_end_date_timestamp = pd.to_datetime(user_end_date)

        # Check if user_end_date is in the past
        if user_end_date_timestamp <= now_date:
            return redirect(url_for('index'))

        time_step = 60  # Time step for LSTM
        
        # Predict the gold prices from now until the user-entered date
        predictions = predict_until_date(model, data, now_date, user_end_date_timestamp, time_step, scaler, grams)
        
        # Plot the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(predictions['Date'], predictions['Predicted Price (₹)'], marker='o', label='Predicted Price for {} Grams'.format(grams))
        plt.xlabel('Date')
        plt.ylabel('Gold Price (₹)')
        plt.title('Gold Price Prediction from Now to User-Entered Date for {} Grams'.format(grams))
        plt.legend()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close()  # Close the figure to free memory

        return render_template('index.html', predictions=predictions.to_html(index=False), plot_url=plot_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
