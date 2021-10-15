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

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px

# set up parameters
N_PORTFOLIOS = 10 ** 5
N_DAYS = 365
RISKY_ASSETS = ['BTC-USD', 'ETH-USD', 'WAVES-USD', 'XEM-USD', 'ADA-USD', 'LBC-USD','XMR-USD', 'BNB-USD', 'OMG-USD', 'MTL-USD']
START_DATE = '2020-01-01'
END_DATE = '2020-12-31'

n_assets = len(RISKY_ASSETS)

prices_df = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE, adjusted=True)

# calculate annualized average returns and corresponding standard deviation
returns_df = prices_df['Close'].pct_change().dropna()

avg_returns = returns_df.mean() * N_DAYS
cov_mat = returns_df.cov() * N_DAYS

# Simulate random portfolio weights
np.random.seed(42)
weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
weights /=  np.sum(weights, axis=1)[:, np.newaxis]

# Calculate portfolio metrics
portf_rtns = np.dot(weights, avg_returns)

portf_vol = []
for i in range(0, len(weights)):
    portf_vol.append(np.sqrt(np.dot(weights[i].T, 
                                    np.dot(cov_mat, weights[i]))))
portf_vol = np.array(portf_vol)  
portf_sharpe_ratio = portf_rtns / portf_vol

# Create a joint DataFrame with all data
portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                 'volatility': portf_vol,
                                 'sharpe_ratio': portf_sharpe_ratio})

# Locate the points creating the Efficient Frontier
N_POINTS = 100
portf_vol_ef = []
indices_to_skip = []

portf_rtns_ef = np.linspace(portf_results_df.returns.min(), 
                            portf_results_df.returns.max(), 
                            N_POINTS)
portf_rtns_ef = np.round(portf_rtns_ef, 2)    
portf_rtns = np.round(portf_rtns, 2)

for point_index in range(N_POINTS):
    if portf_rtns_ef[point_index] not in portf_rtns:
        indices_to_skip.append(point_index)
        continue
    matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
    portf_vol_ef.append(np.min(portf_vol[matched_ind]))
    
portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)

# Plot using plotly (the chart is here)
fig = px.scatter(portf_results_df, x="volatility", y="returns", color='sharpe_ratio')
fig.update_traces(marker=dict(size=10,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()

# dash implementation
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children="Efficient Frontier"),
    dcc.Graph(
        id="efficient_frontier",
        figure = {
            'data': [
                {'x': portf_results_df['volatility'], 'y': portf_results_df['returns'],'type':'scatter'}
            ]
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