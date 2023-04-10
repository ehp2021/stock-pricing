# $ pip3 install alpaca-trade-api talib
# https://github.com/alpacahq/alpaca-trade-api-python
#https://alpaca.markets/deprecated/docs/api-documentation/how-to/orders/


import talib as ta
from alpaca_trade_api import REST, TimeFrame

API_KEY = 'Your_API_Key'
SECRET_KEY = 'Your_Secret_Key'

rest_client = REST(API_KEY, SECRET_KEY)

# Historical market data
bars = rest_client.get_bars("SPY", TimeFrame.Day, "2021-06-01", "2021-10-01").df

bars

# Simple Moving Average with TA-Lib
bars['30_Day_SMA'] = ta.SMA(bars['close'], timeperiod=30)

# plotly imports
import plotly.graph_objects as go
import plotly.express as px

# SPY bar data candlestick plot
candlestick_fig = go.Figure(data=[go.Candlestick(x=bars.index,
                open=bars['open'],
                high=bars['high'],
                low=bars['low'],
                close=bars['close'])])

# creating a line plot for our sma
sma_fig = px.line(x=bars.index, y=30_Day_SMA)

# adding both plots onto one chart
fig = go.Figure(data=candlestick_fig.data + sma_fig.data)

# displaying our chart
fig.show()

#bollinger bands
bars['upper_band'], bars['middle_band'], bars['lower_band'] =   
                         ta.BBANDS(bars['Close'], timeperiod =30)


# creating a line plot for our sma
upper_line_fig = px.line(x=bars.index, y=upper_bands)
# creating a line plot for our sma
Lower_line_fig = px.line(x=bars.index, y=lower_bands)

# adding both plots onto one chart
fig = go.Figure(data=candlestick_fig.data + sma_fig.data + upper_line_fig.data + lower_line_fig.data)

# displaying our chart
fig.show()





# MAKE TRADES
import alpaca_trade_api as tradeapi

api = tradeapi.REST()

# Submit a market order to buy 1 share of Apple at market price
api.submit_order(
    symbol='AAPL',
    qty=1,
    side='buy',
    type='market',
    time_in_force='gtc'
)

# Submit a limit order to attempt to sell 1 share of AMD at a
# particular price ($20.50) when the market opens
api.submit_order(
    symbol='AMD',
    qty=1,
    side='sell',
    type='limit',
    time_in_force='opg',
    limit_price=20.50
)