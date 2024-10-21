import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/NepseData.xlsx"
data = pd.read_excel(file_path)

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

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

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
plt.plot(data['Date'], data['SMA_50'], label='50-day SMA', color='green')
plt.plot(data['Date'], data['EMA_50'], label='50-day EMA', color='red')
plt.title("NEPSE Stock Price with 50-day SMA and EMA")
plt.xlabel("Date")
plt.ylabel("Price (NPR)")
plt.legend()
plt.show()
data['Prediction'] =data['Close'].shift(-1)
data.dropna(inplace=True)

X = data[['Close']]
y = data['Prediction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)



model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
print(f"Training R^2 Score: {train_score}")
predictions = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(comparison.head())

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
plt.figure(figsize=(14, 7))
plt.plot(data['Date'].iloc[-len(y_test):], y_test, label="Actual Price", color='blue')
plt.plot(data['Date'].iloc[-len(y_test):], predictions, label="Predicted Price", color='red')
plt.title("NEPSE Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (NPR)")
plt.legend()
plt.show()

data['Prediction'] = data['Close'].shift(-1)
data.dropna(inplace=True)


X = data[['High', 'Low', 'Close', 'Change', 'Per Change (%)']]
y = data['Prediction']


X_train, X
