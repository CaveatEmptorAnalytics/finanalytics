# dash imports
from os import X_OK
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

# pandas imports
import pandas as pd
import numpy as np
import scipy.stats as stat
import statsmodels.api as sm

from pandas_datareader import data
from scipy.stats import norm

'''
Static charts
'''

# initialising datetime
from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')

# helper functions
def load_data(ticker):
    start_date = '2019-01-01'
    end_date = today

    crypto_data = data.DataReader(ticker, 'yahoo', start_date, end_date)

    return crypto_data

def chart_analysis(crypto, years):
    crypto_close = crypto['Close']
    
    # percentage change
    close_pct_change = crypto_close.pct_change()
    
    # calculating return series
    close_return_series = (1 + close_pct_change).cumprod() - 1
    # annualized_returns = (1 + close_return_series.tail(1))**(1/years)-1
    
    # calculating annual volatility
    # volatility = np.sqrt(np.log(crypto_close / crypto_close.shift(1)).var()) * np.sqrt(252)
    ahv = np.sqrt(252) * pd.DataFrame.rolling(np.log(crypto_close / crypto_close.shift(1)),window=20).std()
    
    # calculating sharpe ratio
    risk_free_rate = 0
    returns_ts = close_pct_change.dropna()
    # avg_daily_returns = returns_ts.mean()
    
    returns_ts['Risk Free Rate'] = risk_free_rate/252
    # avg_rfr_ret = returns_ts['Risk Free Rate'].mean()
    returns_ts['Excess Returns'] = returns_ts - returns_ts['Risk Free Rate']

    return close_pct_change, close_return_series, ahv

def all_funcs(ticker, years):
    crypto = load_data(ticker)
    return chart_analysis(crypto, years)

# receiving chart data
ticker = 'BTC-USD'
years = 1
close_pct_change, close_return_series, ahv = all_funcs(ticker, years)

close_pct_change = pd.DataFrame(close_pct_change)
close_pct_change = close_pct_change.reset_index()

close_return_series =  pd.DataFrame(close_return_series)
close_return_series = close_return_series.reset_index()

ahv = pd.DataFrame(ahv)
ahv = ahv.reset_index()

'''
Dash Implementation
'''
# static charts
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children="Dashboard"),
    dcc.Graph(
        id="close_pct_change",
        figure = {
            'data': [
                {'x': close_pct_change['Date'], 'y': close_pct_change['Close'],'type':'line'}
            ],
            'layout' : {
                'title': 'Bitcoin Close Percentage Change'
            }
        },
        style = {
            'display':'inline-block',
            'width': '500px',
            'height': '400px'
        }
    ),
    dcc.Graph(
        id="close_return_series",
        figure = {
            'data': [
                {'x': close_return_series['Date'], 'y': close_return_series['Close'],'type':'line'}
            ],
            'layout' : {
                'title': 'Bitcoin Close Return Series'
            }
        },
        style = {
            'display':'inline-block',
            'width': '500px',
            'height': '400px'
        }
    )
    ,
    dcc.Graph(
        id="ahv",
        figure = {
            'data': [
                {'x': ahv['Date'], 'y': ahv['Close'],'type':'line'}
            ],
            'layout' : {
                'title': 'Annual Historical Volatility'
            }
        },
        style = {
            'display':'inline-block',
            'width': '500px',
            'height': '400px'
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)