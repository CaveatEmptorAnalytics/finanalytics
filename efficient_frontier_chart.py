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
import plotly.graph_objects as go

# set up parameters
N_PORTFOLIOS = 10 ** 5
N_DAYS = 365
RISKY_ASSETS = ['ETH-USD', 'DGD-USD', 'XMR-USD', 'ADX-USD', 'BTC-USD', 'KNC-USD', 'BNT-USD', 'ICX-USD', 'LBC-USD', 'EMC2-USD']
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
    portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))

portf_vol = np.array(portf_vol)  

portf_sharpe_ratio = portf_rtns / portf_vol

# Create a joint DataFrame with all data
portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                 'volatility': portf_vol,
                                 'sharpe_ratio': portf_sharpe_ratio})

# display max sharpe ratio portfolio
max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
max_sharpe_portf_weights = weights[max_sharpe_ind]

# display max returns portfolio
max_rtns_ind = np.argmax(portf_results_df.returns)
max_rtns_portf = portf_results_df.loc[max_rtns_ind]
max_rtns_portf_weights = weights[max_rtns_ind]

# display lowest volatility portfolio
min_vol_ind = np.argmin(portf_results_df.volatility)
min_vol_portf = portf_results_df.loc[min_vol_ind]
min_vol_portf_weights = weights[min_vol_ind]





# Plot using plotly (the chart is here)
fig = px.scatter(portf_results_df, x="volatility", y="returns", color='sharpe_ratio')
fig.update_traces(marker=dict(size=10,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

# dash implementation
app = dash.Dash()

volatilities = [0.8, 0.9, 1.0]

app.layout = html.Div([
    html.H1(children="Efficient Frontier"),
    dcc.Dropdown(
        id='volatility', 
        options=[{"value": x, "label": x} 
                 for x in volatilities],
        value='1'
    ),
    dcc.Graph(id="graph", figure=fig),
    dcc.Graph(id="portfolio_tables")
])

@app.callback(
    Output("portfolio_tables", "figure"), 
    [Input("volatility", "value")])
def change_volatility(vol):
    num_portfolios = N_PORTFOLIOS
    portfolios_consider = {}
    for i in range(num_portfolios):
        if portf_results_df['volatility'][i].round(2) == float(vol):
            portfolios_consider[i] = portf_results_df['returns'][i]
    
    portf_index_from_vol = max(portfolios_consider, key=portfolios_consider.get)

    portf_ind_from_vol_weights = weights[portf_index_from_vol]

    fig = go.Figure(data=[go.Table(header=dict(values=['Ticker','Weight (%)']),
                 cells=dict(values=[RISKY_ASSETS, (portf_ind_from_vol_weights*100).round(2)]))])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)