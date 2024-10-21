import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf


file_path = "data/NepseData.xlsx"
data = pd.read_excel(file_path)


data['Month'] = data['Date'].dt.month
data['Day of Week'] = data['Date'].dt.dayofweek
data['Quarter'] = data['Date'].dt.quarter
data['Range'] = data['High'] - data['Low']

sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Range'], label='Daily Range (High-Low)', color='orange')
plt.title("NEPSE Stock Daily Range (Volatility)")
plt.xlabel("Date")
plt.ylabel("Range (NPR)")
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Change'], label='Change (NPR)', color='purple')
plt.title("NEPSE Daily Change (NPR)")
plt.xlabel("Date")
plt.ylabel("Change (NPR)")
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Per Change (%)'], label='Percent Change (%)', color='red')
plt.title("NEPSE Percentage Change Over Time")
plt.xlabel("Date")
plt.ylabel("Percentage Change (%)")
plt.legend()
plt.show()

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
plt.plot(data['Date'], data['SMA_50'], label='50-day SMA', color='green')
plt.plot(data['Date'], data['EMA_50'], label='50-day EMA', color='red')
plt.title("NEPSE Stock Price with 50-day SMA and EMA")
plt.xlabel("Date")
plt.ylabel("Price (NPR)")
plt.legend()
plt.show()

data.set_index('Date', inplace=True)
mpf.plot(data, type='candle', style='charles', title='NEPSE Candlestick Chart', volume=False)

monthly_trends = data.groupby('Month').mean()
plt.figure(figsize=(14, 7))
plt.plot(monthly_trends.index, monthly_trends['Close'], marker='o', label='Average Close Price')
plt.title('Average Monthly Close Price')
plt.xlabel('Month')
plt.ylabel('Average Close Price (NPR)')
plt.xticks(range(1, 13))
plt.legend()
plt.show()

quarterly_trends = data.groupby('Quarter').mean()
plt.figure(figsize=(14, 7))
plt.plot(quarterly_trends.index, quarterly_trends['Close'], marker='o', label='Average Close Price')
plt.title('Average Quarterly Close Price')
plt.xlabel('Quarter')
plt.ylabel('Average Close Price (NPR)')
plt.xticks([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4'])
plt.legend()
plt.show()

day_of_week_trends = data.groupby('Day of Week').mean()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(14, 7))
plt.plot(day_of_week_trends.index, day_of_week_trends['Close'], marker='o', label='Average Close Price')
plt.title('Average Close Price by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Close Price (NPR)')
plt.xticks(ticks=range(7), labels=weekdays)
plt.legend()
plt.show()

data['Prediction'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = data[['High', 'Low', 'Close', 'Change', 'Per Change (%)']]
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
print(f"Training R^2 Score: {train_score}")

predictions = model.predict(X_test)

comparison = pd.DataFrame({'Date': data.index[-len(y_test):], 'Actual': y_test, 'Predicted': predictions})

print("Predicted Stock Prices:")
print(comparison)

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test, label="Actual Price", color='blue')
plt.plot(data.index[-len(y_test):], predictions, label="Predicted Price", color='red')

future_days = 10
future_data = pd.DataFrame(columns=['Date', 'Predicted'])

last_row = data.iloc[-1]
last_high = last_row['High']
last_low = last_row['Low']
last_close = last_row['Close']
last_change = last_row['Change']
last_per_change = last_row['Per Change (%)']

for i in range(1, future_days + 1):
    X_future = pd.DataFrame([[last_high, last_low, last_close, last_change, last_per_change]],
                             columns=['High', 'Low', 'Close', 'Change', 'Per Change (%)'])

    predicted_price = model.predict(X_future)[0]

    next_date = last_row.name + pd.Timedelta(days=i)

    new_prediction = pd.DataFrame({'Date': [next_date], 'Predicted': [predicted_price]})
    future_data = pd.concat([future_data, new_prediction], ignore_index=True)

    last_high = last_low = last_close = predicted_price
    last_change = predicted_price - data['Close'].iloc[-1]
    last_per_change = (last_change / data['Close'].iloc[-1]) * 100 if data['Close'].iloc[-1] != 0 else 0

print("Future Stock Price Predictions:")
print(future_data)

plt.plot(future_data['Date'], future_data['Predicted'], label="Future Predictions", color='orange', linestyle='--')
plt.title("NEPSE Stock Price Prediction vs Actual & Future Predictions")
plt.xlabel("Date")
plt.ylabel("Price (NPR)")
plt.legend()
plt.show()
