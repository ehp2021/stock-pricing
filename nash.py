import pandas as pd
import numpy as np
import yfinance as yf

def nash_equilibrium(df, n):
    # Calculate the mean and standard deviation of daily returns
    mu = np.mean(df['returns'])
    sigma = np.std(df['returns'])
    
    # Generate n random values from a normal distribution
    rand_vals = np.random.normal(mu, sigma, n)
    
    # Calculate the payoff for each value in the random distribution
    payoffs = []
    for val in rand_vals:
        num_higher = len(df[df['returns'] > val])
        num_lower = len(df[df['returns'] < val])
        payoff = num_higher / (num_higher + num_lower)
        payoffs.append(payoff)
    
    # Calculate the Nash equilibrium
    nash = np.mean(payoffs)
    
    return nash

# Define the ticker symbol and time period
ticker = "TSLA"
start_date = "2020-01-01"
end_date = "2021-01-01"

# Download the data for the specified ticker and time period
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily returns
data['returns'] = np.log(data['Close'] / data['Open'])

# Calculate the Nash equilibrium
nash = nash_equilibrium(data, 365)
print('Nash equilibrium for TSLA:', nash)

