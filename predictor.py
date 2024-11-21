import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Define the date range for one year of data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Fetch historical data for Reliance Industries (ticker: RELIANCE.NS)
data = yf.download(tickers='RELIANCE.NS', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Calculate target
data['Target'] = data['Adj Close'] - data['Open']
data['Target'] = data['Target'].shift(-1)
data['TargetNextClose'] = data['Adj Close'].shift(-1)

# Drop unnecessary columns except 'Date'
data.drop(['Volume', 'Close'], axis=1, inplace=True)

# Data scaling without 'Date' column
data_set = data[['Open', 'High', 'Low', 'Adj Close']]  # Only essential columns for scaling
sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data_set)

# Prepare features and target
X, y = [], data_set_scaled[30:, -1]  # Using last column as target
for i in range(30, data_set_scaled.shape[0]):
    X.append(data_set_scaled[i-30:i, :])  # Selecting a 30-day sequence of features for each sample

# Convert to numpy arrays
X = np.array(X)
y = np.reshape(y, (len(y), 1))

# Split data into training and testing sets
split_limit = int(len(X) * 0.8)
X_train, X_test = X[:split_limit], X[split_limit:]
y_train, y_test = y[:split_limit], y[split_limit:]

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Make predictions
y_pred = model.predict(X_test)

# Transform predictions and test set back to original scale
y_pred_original = sc.inverse_transform(np.concatenate([np.zeros((len(y_pred), X.shape[2] - 1)), y_pred], axis=1))[:, -1]
y_test_original = sc.inverse_transform(np.concatenate([np.zeros((len(y_test), X.shape[2] - 1)), y_test], axis=1))[:, -1]

# Calculate RMSE for the test set
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"RMSE for the test set: {rmse:.2f}")

# Extract test dates for plotting
test_dates = data.index[split_limit + 30:].to_list()

# Visualization with original prices and date-based x-axis
plt.figure(figsize=(16, 8))
plt.plot(test_dates, y_test_original, color='black', label='Actual Prices')
plt.plot(test_dates, y_pred_original, color='green', label='Predicted Prices')
plt.title(f'Reliance Stock Price Prediction (Last 1 Year) - RMSE: {rmse:.2f}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()
