import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

# Define the stock symbol, start date, and end date
symbol = 'SPY'
start_date = '2013-01-01'
end_date = '2023-03-30'

# Retrieve the historical stock data using yfinance
stock_data = yf.download(symbol, start=start_date, end=end_date)
sp500_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()

# Create a matrix of predictor variables
X = np.column_stack((sp500_returns[:-1],))

# Create a vector of target variables
y = sp500_returns[1:]

# Define a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Print the coefficients of the model
print('Intercept: {:.4f}'.format(model.intercept_))
print('Coefficient: {:.4f}'.format(model.coef_[0]))
