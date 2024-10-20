import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

file_path =r"C:\Users\DELL\Downloads\NepseData.xlsx"
data = pd.read_excel(file_path)

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

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
