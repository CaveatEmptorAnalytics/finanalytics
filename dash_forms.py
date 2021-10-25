import dash
from dash.exceptions import PreventUpdate
from dash import html
from dash import dcc
from dash import dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import plotly.graph_objects as go

from pandas_datareader import data
from scipy.stats import norm

from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')

from dash.dependencies import Input, Output, State

tickers = ['BTC-USD','ETH-USD','HEX-USD','ADA-USD','BNB-USD','XRP-USD','SOL1-USD','DOT1-USD','USDC-USD',
           'DOGE-USD','LUNA1-USD','UNI3-USD','LTC-USD','AVAX-USD','SHIB-USD','LINK-USD','BCH-USD','ALGO-USD',
           'MATIC-USD','XLM-USD','FIL-USD','ICP1-USD','ATOM1-USD','VET-USD','AXS-USD','ETC-USD','TRX-USD',
           'FTT1-USD','DAI1-USD','THETA-USD','XTZ-USD','FTM-USD','HBAR-USD','XMR-USD','CRO-USD','EGLD-USD',
           'EOS-USD','CAKE-USD','MIOTA-USD','AAVE-USD','QNT-USD','BSV-USD','GRT2-USD','NEO-USD','KSM-USD',
           'WAVES-USD','BTT1-USD','ONE2-USD','MKR-USD','STX1-USD','AMP1-USD','AR-USD','OMG-USD','DASH-USD',
           'HNT1-USD','CHZ-USD','CELO-USD','DCR-USD','RUNE-USD','COMP-USD','XEM-USD','HOT1-USD','TFUEL-USD',
           'ZEC-USD','XDC-USD','ICX-USD','CTC1-USD','CEL-USD','MANA-USD','SUSHI-USD','ENJ-USD','TUSD-USD',
           'QTUM-USD', 'OMI-USD','BTG-USD','YFI-USD','CRV-USD','ZIL-USD','SNX-USD','RVN-USD','BAT-USD',
           'SRM-USD','IOST-USD','CCXX-USD','SC-USD','CELR-USD','BNT-USD','ZRX-USD','ZEN-USD','ONT-USD',
           'DFI-USD','DGB-USD','NANO-USD','XWC-USD','RAY-USD','ANKR-USD', 'IOTX-USD','SAND-USD','VGX-USD',
           'UMA-USD','SKL-USD','C98-USD','FET-USD','KAVA-USD','GLM-USD','1INCH-USD','STORJ-USD','LRC-USD',
           'WAXP-USD','RSR-USD','SXP-USD','BCD-USD','GNO-USD','COTI-USD','NMR-USD','LSK-USD','ARRR-USD',
           'XVG-USD','WIN1-USD','CKB-USD','VTHO-USD','MED-USD','ARDR-USD','TWT-USD','ETN-USD','CVC-USD',
           'STMX-USD', 'SNT-USD','ERG-USD','EWT-USD','HIVE-USD','VLX-USD','STRAX-USD','ROSE-USD','KDA-USD',
           'RLC-USD','REP-USD','VRA-USD','BAND-USD','ARK-USD','DERO-USD','MAID-USD','CTSI-USD','XCH-USD',
           'STEEM-USD','NKN-USD','OXT-USD','MLN-USD','MTL-USD','FUN-USD','MIR1-USD','PHA-USD','DAG-USD',
           'SAPP-USD','ACH-USD','NU-USD','TOMO-USD', 'ANT-USD','WAN-USD','SYS-USD','AVA-USD','MCO-USD',
           'CLV-USD','BAL-USD','META-USD','RBTC-USD','KIN-USD','BTS-USD','ZNN-USD','KMD-USD','IRIS-USD',
           'TT-USD','XHV-USD','NYE-USD','GAS-USD','ABBC-USD','MONA-USD','DIVI-USD','FIRO-USD','PAC-USD',
           'XNC-USD','NRG-USD', 'DNT-USD','ELA-USD','AION-USD','BTM-USD','GRS-USD','FRONT-USD','ZEL-USD',
           'BEPRO-USD','HNS-USD','REV-USD','WTC-USD','RDD-USD','SBD-USD','ADX-USD','DMCH-USD','BEAM-USD',
           'FIO-USD','BCN-USD','APL-USD','WOZX-USD','CUT-USD','XCP-USD','DGD-USD','CRU-USD','AXEL-USD',
           'SERO-USD','NIM-USD', 'NULS-USD','MARO-USD','ADK-USD','PIVX-USD','CUDOS-USD','PCX-USD','VERI-USD',
           'NXS-USD','GXC-USD','FSN-USD','PPT-USD','LOKI-USD','ATRI-USD','VITE-USD','CET-USD','VSYS-USD',
           'CTXC-USD','MWC-USD','GO-USD','ZANO-USD','PART-USD','GRIN-USD','KRT-USD','FO-USD','AE-USD',
           'MASS-USD','SRK-USD','VTC-USD','NAV-USD','VAL1-USD','WICC-USD','HC-USD','SOLVE-USD','MHC-USD',
           'QASH-USD','SKY-USD','BTC2-USD','WABI-USD','PPC-USD','NEBL-USD','NAS-USD','NMC-USD','GAME-USD',
           'RSTR-USD','GBYTE-USD','PAI-USD','AMB-USD','LBC-USD','SALT-USD', 'LCC-USD','PZM-USD','OBSR-USD',
           'DCN-USD','BIP-USD','UBQ-USD','MAN-USD','BHP-USD','INSTAR-USD','TRTL-USD','EMC2-USD','PLC-USD',
           'YOYOW-USD','TRUE-USD','DMD-USD','PAY-USD','CHI-USD','MRX-USD','HPB-USD','BLOCK-USD','SCC3-USD',
           'NLG-USD','POA-USD','RINGX-USD','SFT-USD','QRK-USD','FCT-USD', 'DNA1-USD','PI-USD','NVT-USD',
           'BHD-USD','ZYN-USD','SMART-USD','ACT-USD','INT-USD','WGR-USD','XDN-USD','AEON-USD','CMT1-USD',
           'HTML-USD','GHOST1-USD','VEX-USD','HTDF-USD','XMY-USD','DYN-USD','VIA-USD','IDNA-USD','FTC-USD',
           'XMC-USD','BCA-USD','FLO-USD','PMEER-USD','NYZO-USD', 'TERA-USD','SCP-USD','GRC-USD','BLK-USD',
           'WINGS-USD','BTX-USD','GLEEC-USD','MIR-USD','OTO-USD','XST-USD','VIN-USD','PHR-USD','OWC-USD',
           'ILC-USD','IOC-USD','POLIS-USD','CURE-USD','USNBT-USD', 'LEDU-USD','GHOST-USD','AYA-USD',
           'FAIR-USD','CRW-USD','COLX-USD','TUBE-USD', 'SUB-USD','GCC1-USD','DIME-USD','XLT-USD','NIX-USD',
           'BPS-USD','MGO-USD','FTX-USD','DDK-USD','XRC-USD','SONO1-USD','HYC-USD','EDG-USD','MBC-USD',
           'ERK-USD','XBY-USD','XAS-USD','BPC-USD','SNGLS-USD','ATB-USD','FRST-USD','COMP1-USD','OURO-USD',
           'UNO-USD','ECC-USD','CLAM-USD','MOAC-USD', 'ECA-USD','NLC2-USD','BDX-USD','ALIAS-USD','FLASH-USD',
           'CSC-USD','LKK-USD','BONO-USD','XUC-USD','HONEY3-USD','DUN-USD','RBY-USD','HNC-USD','DACC-USD',
           'SPHR-USD','AIB-USD','MINT-USD','SHIFT-USD','CCA-USD','MTC2-USD','MIDAS-USD','JDC-USD','SLS-USD',
           'DCY-USD','GRN-USD','KNC-USD','LRG-USD','BRC-USD','SFMS-USD','BONFIRE-USD','VBK-USD', 'DTEP-USD',
           'QRL-USD','ETP-USD','NXT-USD','XSN-USD']

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
# fig = px.scatter(portf_results_df, x="volatility", y="returns", color='sharpe_ratio')
# fig.update_traces(marker=dict(size=10,
#                               line=dict(width=1,
#                                         color='DarkSlateGrey')),
#                   selector=dict(mode='markers'))

volatilities = [0.8, 0.9, 1.0]

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Investment Amount", html_for="investment"),
                        dbc.Input(
                            type="text",
                            id="investment",\
                        ),
                    ],
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(["Select Your Portfolio"], className="text-center"),
                        dbc.Row(
                            [
                                html.Div(
                                    [
                                        dbc.Label("Select crypto", html_for="dropdown"),
                                        dcc.Dropdown(
                                            id="dropdown",
                                            options=[{"value": x, "label": x} 
                                                    for x in tickers]
                                        ),
                                    ],
                                    className="col-4"
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Portfolio Weight", html_for="portfolio_weight"),
                                        dbc.Input(
                                            id="portfolio_weight",
                                            type="number"
                                        ),
                                    ],
                                    className="col-4"
                                ),
                                html.Div(
                                    [
                                        dbc.Button("Add Crypto", color="dark", className="mt-1 align-self-end", id="submit_crypto", n_clicks=0),
                                    ],
                                    className="col-4 d-flex"
                                ),
                            ],
                            style={
                                'padding-top': 10,
                                'padding-bottom': 10,
                            }
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            dash_table.DataTable(
                                                id='portfolio_table',
                                                columns=[{'name': 'Crypto', 'id': 'crypto', 'presentation': 'dropdown', 'deletable': False, 'renamable': False},
                                                        {'name': 'weightage', 'id': 'weightage', 'deletable': False, 'renamable': False}
                                                ],
                                                data=[{'crypto': 'Select your crypto ticker from the dropdown', 'weightage': 'Input your desired weights for the crypto'},
                                                ],
                                                # editable=True,                  # allow user to edit data inside tabel
                                                row_deletable=True,             # allow user to delete rows
                                                # sort_action="native",           # give user capability to sort columns
                                                sort_mode="single",             # sort across 'multi' or 'single' columns
                                                # filter_action="native",         # allow filtering of columns
                                                page_action='none',             # render all of the data at once. No paging.
                                                # style_table={'height': '200px', 'overflowY': 'auto'},
                                                style_cell={'textAlign': 'center'},
                                                style_data={
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',
                                                },
                                            ),   
                                        ],
                                    ),
                                )
                            ],
                            style={
                                'padding-top': 20,
                                'padding-bottom': 30
                            },
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H3(["Our Recommended Portfolio"], className="text-center"),
                        dbc.Row(
                            [
                                html.Div(
                                    [
                                        dbc.Label("Select Volatility", html_for="volatility"),
                                        dcc.Dropdown(
                                            id='volatility', 
                                            options=[{"value": x, "label": x} 
                                                    for x in volatilities],
                                            value='1'
                                        ),
                                    ],
                                ),
                            ],
                            style={
                                'padding-top': 10,
                                'padding-bottom': 10
                            },
                        ),
                        dbc.Row(
                            [
                                html.Div(id="portfolio_tables"),
                            ],
                            style={
                                'text-align': 'center',
                                'padding-top': 20
                            },
                        )
                        
                    ]
                )
            ],
            style={
                'padding-top': 30,
                'padding-bottom': 30
            },
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Select the crypto you would like it analyze", html_for="crypto_dropdown"),
                        dcc.Dropdown(
                                id="crypto_dropdown"
                            ),
                    ],
                    width=4,
                    style={
                        'padding': 10
                    },
                ),
                dbc.Col(
                    [
                        dbc.Label("Select the start date you would like to analyze", html_for="start_analyze_date"),
                        dbc.Input(
                            type="date",
                            id="start_analyze_date",
                        ),
                    ],
                    width=4,
                    style={
                        'padding': 10
                    },
                ),
                dbc.Col(
                    [
                        dbc.Label("Select the end date you would like to analyze", html_for="end_analyze_date"),
                        dbc.Input(
                            type="date",
                            id="end_analyze_date",
                        ),
                    ],
                    width=4,
                    style={
                        'padding': 10
                    },
                ),
            ],
        ),
        dbc.Button("Generate Analysis Charts", color="dark", className="mr-1", id="submit_analysis_charts", n_clicks=0 ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="close_pct_change")
                ),
                dbc.Col(
                    html.Div(id="close_return_series")
                ),
                dbc.Col(
                    html.Div(id="ahv")
                ),
            ],
            className="g-0",
            style={
                'padding': 10
            },
        ),

        dbc.Button("Generate Portfolio Comparison", color="dark", className="mr-1", id="submit_comparison", n_clicks=0),
        
        # weightage comparison
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [],
                        id="gut_feel_weights"
                    ),
                    width=6,
                ),
                dbc.Col(
                    html.Div(
                        [],
                        id="recommended_weights"
                    ),
                    width=6,
                ),
            ]
        ),

        # returns comparison
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="recommended_returns"),
                    width=6,
                ),
                dbc.Col(
                    html.Div(id="gut_feel_returns"),
                    width=6,
                ),
            ]
        ),

        # sharpe ratio comparison
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="recommended_sharpe"),
                    width=6,
                ),
                dbc.Col(
                    html.Div(id="gut_feel_sharpe"),
                    width=6,
                ),
            ]
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(html.Div("One of three columns")),
        #         dbc.Col(html.Div("One of three columns")),
        #         dbc.Col(html.Div("One of three columns")),
        #     ]
        # ),
    ]
)

# add a callback to update dataframe holding all the cryptos
# checks for dataframe: 1. if the dataframe receives the first input, replace the dummy input, 2. if the dataframe already has 10 inputs, it will not update

# dash video guide https://www.youtube.com/watch?v=mTsZL-VmRVE

@app.callback(
    Output("portfolio_table", 'data'),
    Input('submit_crypto', 'n_clicks'),
    [State('dropdown', 'value'),
    State('portfolio_weight', 'value'),
    State('portfolio_table', 'data'),
    State('portfolio_table', 'columns')],
    prevent_initial_call = True
)
def add_row(n_clicks, crypto, weight, rows, columns):
    if n_clicks == 1:
        rows = [{'crypto': crypto, 'weightage': weight}]
    else:
        if len(rows)<=10:
            rows.append({'crypto': crypto, 'weightage': weight})

    return rows

@app.callback(
    Output("crypto_dropdown", 'options'),
    Input('portfolio_table', 'data'),
    prevent_initial_call = True
)
def add_dropdown_options(crypto_data):
    output = []
    for data in crypto_data:
        output.append(data["crypto"])
    
    options = [{"value": x, "label": x} for x in output]

    return options


@app.callback(
    [Output('close_pct_change', 'children'),
    Output('close_return_series', 'children'),
    Output('ahv', 'children')],
    Input('submit_analysis_charts', 'n_clicks'),
    [State('crypto_dropdown', 'value'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def update_charts(clicks, crypto, start_date, end_date):
    # receiving chart data
    ticker = crypto
    years = 1
    close_pct_change, close_return_series, ahv = all_funcs(ticker, years, start_date, end_date)

    close_pct_change = pd.DataFrame(close_pct_change)
    close_pct_change = close_pct_change.reset_index()

    close_return_series =  pd.DataFrame(close_return_series)
    close_return_series = close_return_series.reset_index()

    ahv = pd.DataFrame(ahv)
    ahv = ahv.reset_index()
    
    df1 = close_pct_change
    close_pct_change = dcc.Graph(
                        figure = {
                            'data': [
                                {'x': df1['Date'], 'y': df1['Close'],'type':'line'}
                            ],
                            'layout' : {
                                'title': str(crypto) + ' Close Percentage Change'
                            }
                        },
                        style = {
                            'display':'inline-block',
                            'width': '400px',
                            'height': '400px'
                        }
                    ),

    close_return_series = dcc.Graph(
                            figure = {
                                'data': [
                                    {'x': close_return_series['Date'], 'y': close_return_series['Close'],'type':'line'}
                                ],
                                'layout' : {
                                    'title': str(crypto) + ' Close Return Series'
                                }
                            },
                            style = {
                                'display':'inline-block',
                                'width': '400px',
                                'height': '400px'
                            }
                        )

    ahv = dcc.Graph(
            figure = {
                'data': [
                    {'x': ahv['Date'], 'y': ahv['Close'],'type':'line'}
                ],
                'layout' : {
                    'title': str(crypto) + ' Annual Historical Volatility'
                }
            },
            style = {
                'display':'inline-block',
                'width': '400px',
                'height': '400px'
            }
        )

    return close_pct_change, close_return_series, ahv

def load_data(ticker, startdate, enddate):
    start_date = startdate
    end_date = enddate

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

def all_funcs(ticker, years, start_date, end_date):
    crypto = load_data(ticker, start_date, end_date)
    return chart_analysis(crypto, years)

@app.callback(
    [Output('recommended_weights', 'children'),
    Output('gut_feel_weights', 'children'),
    Output('recommended_returns', 'children'),
    Output('gut_feel_returns', 'children'),
    Output('recommended_sharpe', 'children'),
    Output('gut_feel_sharpe', 'children')],
    Input('submit_comparison', 'n_clicks'),
    [State('portfolio_table', 'data'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def update_portfolio_comparison(clicks, gut_feel_data, start_date, end_date):
    recommended_weights = ""
    gut_feel_weights = dcc.Graph(figure=px.pie(gut_feel_data, names='crypto', values='weightage'))

    recommended_returns = ""
    gut_feel_returns = ""

    recommended_sharpe = ""
    gut_feel_sharpe = ""
    
    return recommended_weights, gut_feel_weights, recommended_returns, gut_feel_returns, recommended_sharpe, gut_feel_sharpe

@app.callback(
    Output("portfolio_tables", "children"), 
    [Input("volatility", "value")])
def change_volatility(vol):
    num_portfolios = N_PORTFOLIOS
    portfolios_consider = {}
    for i in range(num_portfolios):
        if portf_results_df['volatility'][i].round(2) == float(vol):
            portfolios_consider[i] = portf_results_df['returns'][i]
    
    portf_index_from_vol = max(portfolios_consider, key=portfolios_consider.get)

    portf_ind_from_vol_weights = weights[portf_index_from_vol]

    table_data = []
    for i in range(len(RISKY_ASSETS)):
        pair = [RISKY_ASSETS[i], portf_ind_from_vol_weights[i].round(2)]
        table_data.append(pair)
    
    recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])

    fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)

    # fig = go.Figure(data=[go.Table(header=dict(values=['Ticker','Weight (%)']),
    #              cells=dict(values=[RISKY_ASSETS, (portf_ind_from_vol_weights*100).round(2)]))])
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)


# dashboard that takes in start and end date for each input and returns weightage for top 10 crypto
# 2 pages 