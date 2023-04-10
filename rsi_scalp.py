import alpaca_trade_api as tradeapi
import yfinance as yf
import talib

# Define Alpaca API credentials
api_key = 'PK4MW9Z1GG9FZLU3XXKG'
api_secret = 'wMsgrnWnPhUfKpnPzCgKd6FlVdZftOHbxmn2XPA9'
base_url = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API client
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Define stock symbol and time frame
symbol = 'AAPL'
timeframe = '1Min'

# Define RSI period and threshold levels
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30

# Get historical data using yfinance
df = yf.download(symbol, period='1d', interval=timeframe)

# Calculate RSI using Talib
rsi = talib.RSI(df['Close'], timeperiod=rsi_period)

# Get the last RSI value
last_rsi = rsi[-1]

# Get current price from Alpaca API
current_price = api.get_last_trade(symbol).price

# If RSI is above the overbought threshold, sell
if last_rsi > rsi_overbought:
    api.submit_order(
        symbol=symbol,
        qty=1,
        side='sell',
        type='limit',
        time_in_force='gtc',
        limit_price=current_price
    )

# If RSI is below the oversold threshold, buy
elif last_rsi < rsi_oversold:
    api.submit_order(
        symbol=symbol,
        qty=1,
        side='buy',
        type='limit',
        time_in_force='gtc',
        limit_price=current_price
    )
