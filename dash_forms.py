import dash
from dash.exceptions import PreventUpdate
from dash import html
from dash import dcc
from dash import dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats.stats import skew
import yfinance as yf
import plotly.graph_objects as go
import statsmodels.api as sm
import tensorflow as tf

from pandas_datareader import data
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from datetime import timedelta, date
from plotly.subplots import make_subplots


from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')

from dash.dependencies import Input, Output, State

TICKERS = [
    '1INCH-USD', 'AAVE-USD', 'ABBC-USD', 'ACH-USD', 'ACT-USD', 'ADA-USD', 'ADK-USD', 'ADX-USD', 'AE-USD', 'AEON-USD', 'AIB-USD', 'AION-USD', 'ALGO-USD', 'ALIAS-USD', 'AMB-USD', 'AMP1-USD', 'ANKR-USD', 'ANT-USD', 'APL-USD', 'AR-USD', 'ARDR-USD', 'ARK-USD', 'ARRR-USD', 'ATB-USD', 'ATOM1-USD', 'ATRI-USD', 'AVA-USD', 'AVAX-USD', 'AXEL-USD', 'AXS-USD', 'AYA-USD', 'BAL-USD', 'BAND-USD', 'BAT-USD', 'BCA-USD', 'BCD-USD', 'BCH-USD', 'BCN-USD', 'BDX-USD', 'BEAM-USD', 'BEPRO-USD', 'BHD-USD', 'BHP-USD', 'BIP-USD', 'BLK-USD', 'BLOCK-USD', 'BNB-USD', 'BNT-USD', 'BONFIRE-USD', 'BONO-USD', 'BPC-USD', 'BPS-USD', 'BRC-USD', 'BSV-USD', 'BTC-USD', 'BTC2-USD', 'BTG-USD', 'BTM-USD', 'BTS-USD', 'BTT1-USD', 'BTX-USD', 'C98-USD', 'CAKE-USD', 'CCA-USD', 'CCXX-USD', 'CEL-USD', 'CELO-USD', 'CELR-USD', 'CET-USD', 'CHI-USD', 'CHZ-USD', 'CKB-USD', 'CLAM-USD', 'CLV-USD', 'CMT1-USD', 'COLX-USD', 'COMP-USD', 'COMP1-USD', 'COTI-USD', 'CRO-USD', 'CRU-USD', 'CRV-USD', 'CRW-USD', 'CSC-USD', 'CTC1-USD', 'CTSI-USD', 'CTXC-USD', 'CUDOS-USD', 'CURE-USD', 'CUT-USD', 'CVC-USD', 'DACC-USD', 'DAG-USD', 'DAI1-USD', 'DASH-USD', 'DCN-USD', 'DCR-USD', 'DCY-USD', 'DDK-USD', 'DERO-USD', 'DFI-USD', 'DGB-USD', 'DGD-USD', 'DIME-USD', 'DIVI-USD', 'DMCH-USD', 'DMD-USD', 'DNA1-USD', 'DNT-USD', 'DOGE-USD', 'DOT1-USD', 'DTEP-USD', 'DUN-USD', 'DYN-USD', 'ECA-USD', 'ECC-USD', 'EDG-USD', 'EGLD-USD', 'ELA-USD', 'EMC2-USD', 'ENJ-USD', 'EOS-USD', 'ERG-USD', 'ERK-USD', 'ETC-USD', 'ETH-USD', 'ETN-USD', 'ETP-USD', 'EWT-USD', 'FAIR-USD', 'FCT-USD', 'FET-USD', 'FIL-USD', 'FIO-USD', 'FIRO-USD', 'FLASH-USD', 'FLO-USD', 'FO-USD', 'FRONT-USD','FRST-USD','FSN-USD', 'FTC-USD', 'FTM-USD', 'FTT1-USD', 'FTX-USD', 'FUN-USD', 'GAME-USD', 'GAS-USD', 'GBYTE-USD', 'GCC1-USD', 'GHOST-USD', 'GHOST1-USD', 'GLEEC-USD', 'GLM-USD', 'GNO-USD', 'GO-USD', 'GRC-USD', 'GRIN-USD', 'GRN-USD', 'GRS-USD', 'GRT2-USD', 'GXC-USD', 'HBAR-USD', 'HC-USD', 'HEX-USD', 'HIVE-USD', 'HNC-USD', 'HNS-USD', 'HNT1-USD', 'HONEY3-USD', 'HOT1-USD', 'HPB-USD', 'HTDF-USD', 'HTML-USD', 'HYC-USD', 'ICP1-USD', 'ICX-USD', 'IDNA-USD', 'ILC-USD', 'INSTAR-USD', 'INT-USD', 'IOC-USD', 'IOST-USD', 'IOTX-USD', 'IRIS-USD', 'JDC-USD', 'KAVA-USD', 'KDA-USD', 'KIN-USD', 'KMD-USD', 'KNC-USD', 'KRT-USD', 'KSM-USD', 'LBC-USD', 'LCC-USD', 'LEDU-USD', 'LINK-USD', 'LKK-USD', 'LOKI-USD', 'LRC-USD', 'LRG-USD', 'LSK-USD', 'LTC-USD', 'LUNA1-USD', 'MAID-USD', 'MAN-USD', 'MANA-USD', 'MARO-USD', 'MASS-USD', 'MATIC-USD', 'MBC-USD', 'MCO-USD', 'MED-USD', 'META-USD', 'MGO-USD', 'MHC-USD', 'MIDAS-USD', 'MINT-USD', 'MIOTA-USD', 'MIR-USD', 'MIR1-USD', 'MKR-USD', 'MLN-USD', 'MOAC-USD', 'MONA-USD', 'MRX-USD', 'MTC2-USD', 'MTL-USD', 'MWC-USD', 'NANO-USD', 'NAS-USD', 'NAV-USD', 'NEBL-USD', 'NEO-USD', 'NIM-USD', 'NIX-USD', 'NKN-USD', 'NLC2-USD', 'NLG-USD', 'NMC-USD', 'NMR-USD', 'NRG-USD', 'NU-USD', 'NULS-USD', 'NVT-USD', 'NXS-USD', 'NXT-USD', 'NYE-USD', 'NYZO-USD', 'OBSR-USD', 'OMG-USD', 'OMI-USD', 'ONE2-USD', 'ONT-USD', 'OTO-USD', 'OURO-USD', 'OWC-USD', 'OXT-USD', 'PAC-USD', 'PAI-USD', 'PART-USD', 'PAY-USD', 'PCX-USD', 'PHA-USD', 'PHR-USD', 'PI-USD', 'PIVX-USD', 'PLC-USD', 'PMEER-USD', 'POA-USD', 'POLIS-USD', 'PPC-USD', 'PPT-USD', 'PZM-USD', 'QASH-USD', 'QNT-USD', 'QRK-USD', 'QRL-USD', 'QTUM-USD', 'RAY-USD', 'RBTC-USD', 'RBY-USD', 'RDD-USD', 'REP-USD', 'REV-USD', 'RINGX-USD', 'RLC-USD', 'ROSE-USD', 'RSR-USD', 'RSTR-USD', 'RUNE-USD', 'RVN-USD', 'SALT-USD', 'SAND-USD', 'SAPP-USD', 'SBD-USD', 'SC-USD', 'SCC3-USD', 'SCP-USD', 'SERO-USD', 'SFMS-USD', 'SFT-USD', 'SHIB-USD', 'SHIFT-USD', 'SKL-USD', 'SKY-USD','SLS-USD', 'SMART-USD', 'SNGLS-USD', 'SNT-USD', 'SNX-USD', 'SOL1-USD', 'SOLVE-USD', 'SONO1-USD', 'SPHR-USD', 'SRK-USD', 'SRM-USD', 'STEEM-USD', 'STMX-USD', 'STORJ-USD', 'STRAX-USD', 'STX1-USD', 'SUB-USD', 'SUSHI-USD', 'SXP-USD', 'SYS-USD', 'TERA-USD', 'TFUEL-USD', 'THETA-USD', 'TOMO-USD', 'TRTL-USD', 'TRUE-USD', 'TRX-USD', 'TT-USD', 'TUBE-USD', 'TUSD-USD', 'TWT-USD', 'UBQ-USD', 'UMA-USD', 'UNI3-USD', 'UNO-USD', 'USDC-USD', 'USNBT-USD', 'VAL1-USD', 'VBK-USD', 'VERI-USD', 'VEX-USD', 'VGX-USD', 'VIA-USD', 'VIN-USD', 'VITE-USD', 'VLX-USD', 'VRA-USD', 'VSYS-USD', 'VTC-USD', 'VTHO-USD', 'WABI-USD', 'WAN-USD', 'WAVES-USD', 'WAXP-USD', 'WGR-USD', 'WICC-USD', 'WIN1-USD', 'WINGS-USD', 'WOZX-USD', 'WTC-USD', 'XAS-USD', 'XBY-USD', 'XCH-USD', 'XCP-USD', 'XDC-USD', 'XDN-USD', 'XEM-USD', 'XHV-USD', 'XLM-USD', 'XLT-USD', 'XMC-USD', 'XMR-USD', 'XMY-USD', 'XNC-USD','XRC-USD', 'XRP-USD', 'XSN-USD', 'XST-USD', 'XTZ-USD', 'XUC-USD', 'XVG-USD', 'XWC-USD', 'YFI-USD', 'YOYOW-USD', 'ZANO-USD', 'ZEC-USD', 'ZEL-USD', 'ZEN-USD', 'ZIL-USD', 'ZNN-USD', 'ZRX-USD', 'ZYN-USD'
 ]

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.Div(
            dbc.Container(
                [
                    html.H1("Input the following information", className="display-3"),
                    html.Hr(className="my-2"),
                    # html.P(
                    #     "Use Containers to create a jumbotron to call attention to "
                    #     "featured content or information.",
                    #     className="lead",
                    # ),
                    # this section is for input investment amount
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Investment Amount", html_for="investment"),
                                    dbc.Input(
                                        type="text",
                                        id="investment",
                                    ),
                                ],
                            ),
                        ]
                    ),
                    # this section is for the input start and end date
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Select the start date you would like to analyze", html_for="start_analyze_date"),
                                    dbc.Input(
                                        type="date",
                                        id="start_analyze_date",
                                        value="2020-01-01"
                                    ),
                                ],
                                width=6,
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
                                        value="2020-12-31"
                                    ),
                                ],
                                width=6,
                                style={
                                    'padding': 10
                                },
                            ),
                            dbc.Col(
                                [
                                    dbc.Button("Confirm", color="dark", className="mr-1", id="pull_data", n_clicks=0 ),
                                ],
                                width = 1,
                                style={
                                    "padding": 10
                                }
                            ),
                            dbc.Col(
                                [
                                    dbc.Button("Get Recommended Portfolio", color="dark", className="mr-1", id="get_reco_portf", n_clicks=0 ),
                                ],
                                width = 1,
                                style={
                                    "padding": 10
                                }
                            )
                        ]
                    ),
                ],
                fluid=True,
                className="py-3",
            ),
            className="p-3 text-white bg-dark rounded-3",
        ),
        dcc.Store(id="memory"),
        # this section is for them to select the crypto ticker that they want to analyse
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Select the crypto you would like it analyze", html_for="crypto_dropdown"),
                        dcc.Dropdown(
                                id="crypto_dropdown",
                                options=[{"value": x, "label": x} 
                                            for x in TICKERS]
                            ),
                    ],
                    width=4,
                    style={
                        'padding-top': 50,
                        'padding-bottom': 10
                    },
                ),
            ],
        ),
        dbc.Button("Generate Analysis Charts", color="dark", className="mr-1", id="submit_analysis_charts", n_clicks=0 ),

        # this section displays the analysis charts
        dbc.Row(id="display_statistics", className="g-0"),
        html.Div(id="display_charts",
            className="g-0",
            style={
                'padding-top': 50,
                'padding-bottom': 100
            },
        ),

        dbc.Button("Toggle View MPT Chart", color="dark", className="mr-1", id="toggle_mpt", n_clicks=0 ),
        dbc.Spinner(
            html.Div(
                id="mpt_chart",
                style={
                    'padding-bottom': 100
                },
            ),
        ),

        dbc.Row(
            [
                # this section adds their gut feel portfolio
                dbc.Col(
                    [
                        html.H3(["Select Your Portfolio"], className="text-center"),
                        dbc.Row(
                            [
                                html.Div(
                                    [
                                        dbc.Label("Select crypto", html_for="dropdown"),
                                        dcc.Dropdown(
                                            id="gut_feel_dropdown",
                                            options=[{"value": x, "label": x} 
                                            for x in TICKERS]
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
                                html.Div(id="gut_feel_portfolio_metrics"),
                                html.Div(
                                    [
                                        dbc.Alert(
                                            id="alert_auto",
                                            is_open=False,
                                            duration=2000,
                                            style={
                                                'padding-top': 20,
                                                'padding-bottom': 20
                                            }
                                        ),
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
                            ],
                            style={
                                'padding-top': 20,
                                'padding-bottom': 30,
                                'text-align': 'center'
                            },
                        ),
                    ],
                    width=6,
                ),
                # this section displays our recommended portfolio
                dbc.Col(
                    [
                        html.H3(["Our Recommended Portfolio"], className="text-center"),
                        # this section switches the views of recommended portfolio
                        dbc.Tabs(
                            [
                                # tab 1
                                dbc.Tab(
                                    label="Best Portfolio",
                                    tab_id = "best"
                                ),
                                # tab 2
                                dbc.Tab(
                                    label="Adjust Volatility",
                                    tab_id = "volatility"
                                ),
                                # tab 3
                                dbc.Tab(
                                    label="Adjust Returns",
                                    tab_id = "returns"
                                ),
                            ],
                            id="tabs",
                            active_tab = "best"
                        ),
                        dbc.Spinner(
                            dbc.Row(
                                [   
                                    html.Div(
                                        [
                                            dbc.Label(id="vol_rtn_label", html_for="vol_rtn_input", align="start", style={
                                                "padding": 10
                                            }),
                                            dbc.Input(
                                                type="number",
                                                id="vol_rtn_input",
                                                min = 0,
                                                max = 0,
                                                value = 0
                                            ),
                                        ],
                                        className = "col-11"
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button("Send", color="dark", className="mt-1 align-self-end", id="submit_vol_rtn_input", n_clicks=0)
                                        ],
                                        className = "col-1 d-flex"
                                    ),

                                    html.Div(id="portfolio_metrics"),
                                    html.Div(
                                        [
                                            dbc.Alert(
                                                id="recommended_alert_auto",
                                                is_open=False,
                                                duration=2000,
                                                style={
                                                    'padding-top': 20,
                                                    'padding-bottom': 20
                                                }
                                            ),
                                            dash_table.DataTable(
                                                id='portfolio_tables',
                                                columns=[{'name': 'Crypto', 'id': 'crypto', 'presentation': 'dropdown', 'deletable': False, 'renamable': False},
                                                        {'name': 'weightage', 'id': 'weightage', 'deletable': False, 'renamable': False}
                                                ],
                                                data=[],
                                                # editable=True,                  # allow user to edit data inside tabel
                                                # row_deletable=True,             # allow user to delete rows
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
                                ],
                                style={
                                    'text-align': 'center',
                                    'padding-top': 20
                                },
                            )
                        )
                    ]
                )
            ],
            style={
                'padding-bottom': 100
            },
        ),
        
        

        dbc.Button("Generate Portfolio Comparison", color="dark", className="mr-1", id="submit_comparison", n_clicks=0),
        
        # weightage comparison
        dbc.Spinner(
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
        ),
        
        # returns comparison
        dbc.Spinner(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="gut_feel_returns"),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(id="recommended_returns"),
                        width=6,
                    ),
                ]
            ),
        ),

        # sharpe ratio comparison
        dbc.Spinner(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="gut_feel_vol"),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(id="recommended_vol"),
                        width=6,
                    ),
                ]
            ),
        ),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
            ]
        ),
        dbc.Row(
            [
                dbc.Button("Generate LSTM Predictions", color="dark", className="mr-1", id="lstm_button", n_clicks=0),
                html.Div(id="lstm_weights")
            ]
        )
    ]
)


# add a callback to update dataframe holding all the cryptos
# checks for dataframe: 1. if the dataframe receives the first input, replace the dummy input, 2. if the dataframe already has 10 inputs, it will not update

# dash video guide https://www.youtube.com/watch?v=mTsZL-VmRVE

# this section pulls data depending on the date
@app.callback(
    Output("memory", "data"),
    Input("pull_data", "n_clicks"),
    [State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value'),
    State("memory", 'data')]
)
def get_data(n_clicks, start_date, end_date, data):
    if not data:
        prices_df = yf.download(TICKERS, start=start_date, end=end_date, adjusted=True)
        tickers_vol = prices_df['Volume']
        tickers_dropdown = prices_df['Close'].columns.tolist()

        data = {"tickers volume": tickers_vol, "tickers dropdown": tickers_dropdown}
        return data

    else:
        return data

# this section controls callbacks for gut feel portfolio
@app.callback(
    [Output("portfolio_table", 'data'),
    Output("alert_auto", 'children'),
    Output("alert_auto", 'color'),
    Output("alert_auto", 'is_open'),
    Output("gut_feel_portfolio_metrics", 'children')],
    Input('submit_crypto', 'n_clicks'),
    [State('gut_feel_dropdown', 'value'),
    State('portfolio_weight', 'value'),
    State('portfolio_table', 'data'),
    State('alert_auto', 'is_open'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def add_row(n_clicks, crypto, weight, rows, is_open, start_date, end_date):
    cryptos_list = []
    weights_list = []
    for i in range(len(rows)):
        cryptos_list.append(rows[i]["crypto"])
        weights_list.append(rows[i]["weightage"])

    if n_clicks == 1:
        if crypto == None:
            text = "Please select your crypto"
            color = "danger"
        
        elif weight == None:
            text = "Please enter your weights for the crypto"
            color = "danger"

        elif weight > 1:
            text = "Your total weightage is more than 1"
            color = "danger"

        else:
            rows = [{'crypto': crypto, 'weightage': weight}]
            cryptos_list = [crypto]
            weights_list = [weight]
            text = "Your crypto has been added"
            color = "success"

    else:
        if crypto == None:
            text = "Please select your crypto"
            color = "danger"   

        elif weight == None:
            text = "Please enter your weights for the crypto"
            color = "danger"

        elif len(rows) < 10:
            if crypto in cryptos_list:
                text = "You have added this crypto before"
                color = "danger"

            elif sum(weights_list) + weight > 1:
                text = "Your total weightage is more than 1"
                color = "danger"

            else:
                rows.append({'crypto': crypto, 'weightage': weight})
                cryptos_list.append(crypto)
                weights_list.append(weight)
                text = "Your crypto has been added"
                color = "success"
            
        is_open = True

    # this section generates the sharpe, returns and vol table
    if sum(weights_list) == 1:
        returns, volatility, sharpe_ratio = get_metrics(cryptos_list, weights_list, start_date, end_date)
        
        recommended_metrics = pd.DataFrame({"Returns": [returns], "Volatility": [volatility], "Sharpe Ratio": [sharpe_ratio]})

        fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
    
    else:
        recommended_metrics = pd.DataFrame({"Returns": [], "Volatility": [], "Sharpe Ratio": []})

        fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)

    print(rows)
    return rows, text, color, is_open, fig

def get_metrics(tickers, weights, start_date, end_date):
    prices_df = yf.download(tickers, start=start_date, end=end_date, adjusted=True)
    
    returns_df = prices_df['Close'].pct_change().dropna()
    
    avg_returns = returns_df.mean() * 365
    cov_mat = returns_df.cov() * 365
    
    # RETURNS / EXPECTED RETURNS
    weights = np.array(weights)
    portf_rtns = np.dot(weights, avg_returns)

    # VOLATILITY / STANDARD DEVIATION
    portf_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))

    portf_vol = np.array(portf_vol)

    # SHARPE RATIO
    portf_sharpe_ratio = portf_rtns / portf_vol
    
    return portf_rtns.round(2), portf_vol.round(2), portf_sharpe_ratio.round(2)
    
# this section controls callbacks for chart analysis
@app.callback(
    [Output('display_statistics', 'children'),
    Output('display_charts', 'children')],
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
    close_pct_change, close_return_series, ahv, volatility, sharpe_ratio, annualized_returns, statistics, skewness, kurtosis = all_funcs(ticker, years, start_date, end_date)

    stats_dict = {
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Annualized Returns": annualized_returns,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
    }

    statistics_df = pd.DataFrame(stats_dict)
    # statistics_df = statistics_df.reset_index()

    close_pct_change = pd.DataFrame(close_pct_change)
    close_pct_change = close_pct_change.reset_index()

    close_return_series =  pd.DataFrame(close_return_series)
    close_return_series = close_return_series.reset_index()

    ahv = pd.DataFrame(ahv)
    ahv = ahv.reset_index()
    
    df1 = close_pct_change

    display_stats = html.Div(
                        [
                            html.H3(["Crypto Statistics"], className="text-center"),
                            dbc.Table.from_dataframe(statistics_df, striped=True, bordered=True, hover=True),
                        ]
                    ),
    display_charts = dbc.Row(
                        [
                            html.H3(["Crypto Charts"], className="text-center"),
                            dbc.Col(
                                dcc.Graph(
                                    figure = {
                                        'data': [
                                            {'x': df1['Date'], 'y': df1['Close'],'type':'line'}
                                        ],
                                        'layout' : {
                                            'title': str(crypto) + ' Close Percentage Change'
                                        }
                                    }
                                ),
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    figure = {
                                        'data': [
                                            {'x': close_return_series['Date'], 'y': close_return_series['Close'],'type':'line'}
                                        ],
                                        'layout' : {
                                            'title': str(crypto) + ' Close Return Series'
                                        }
                                    }
                                ),
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    figure = {
                                        'data': [
                                            {'x': close_return_series['Date'], 'y': close_return_series['Close'],'type':'line'}
                                        ],
                                        'layout' : {
                                            'title': str(crypto) + ' Close Return Series'
                                        }
                                    }
                                )
                            )
                        ]
                    )

    return display_stats, display_charts

def load_data(ticker, startdate, enddate):
    start_date = startdate
    end_date = enddate

    crypto_data = data.DataReader(ticker, 'yahoo', start_date, end_date)

    return crypto_data

def chart_analysis(crypto, years):
    crypto_close = crypto['Close']
    statistics = crypto_close.describe()
    
    # percentage change
    close_pct_change = crypto_close.pct_change()
    
    skewness = close_pct_change.skew()
    kurtosis = close_pct_change.kurtosis()
    
    # calculating return series
    close_return_series = (1 + close_pct_change).cumprod() - 1
    annualized_returns = (1 + close_return_series.tail(1))**(1/years)-1
    
    # calculating annual volatility
    volatility = np.sqrt(np.log(crypto_close / crypto_close.shift(1)).var()) * np.sqrt(365)
    ahv = np.sqrt(252) * pd.DataFrame.rolling(np.log(crypto_close / crypto_close.shift(1)),window=20).std()
    
    # calculating sharpe ratio
    risk_free_rate = 0
    returns_ts = close_pct_change.dropna()
    avg_daily_returns = returns_ts.mean()
    
    returns_ts['Risk Free Rate'] = risk_free_rate/252
    avg_rfr_ret = returns_ts['Risk Free Rate'].mean()
    returns_ts['Excess Returns'] = returns_ts - returns_ts['Risk Free Rate']

    sharpe_ratio = ((avg_daily_returns - avg_rfr_ret) /returns_ts['Excess Returns'].std())*np.sqrt(252)

    return close_pct_change, close_return_series, ahv, volatility, sharpe_ratio, annualized_returns, statistics, skewness, kurtosis

def all_funcs(ticker, years, start_date, end_date):
    crypto = load_data(ticker, start_date, end_date)
    return chart_analysis(crypto, years)

# this section controls callbacks for mpt chart
@app.callback(
    Output("mpt_chart", "children"),
    Input("toggle_mpt", "n_clicks"),
    [State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def display_mpt(n_clicks, start_date, end_date):
    # print(fig)
    if n_clicks % 2 != 0:
        top_10 = get_top_10(start_date, end_date)
        # top_10 = ['XWC-USD', 'DGD-USD', 'LRC-USD', 'ANT-USD', 'ADA-USD', 'KNC-USD','DNT-USD', 'ICX-USD','DGB-USD', 'LTC-USD']
        n_portfolios = 10 ** 5
        n_assets = len(top_10)

        prices_df = yf.download(top_10, start=start_date, end=end_date, adjusted=True)

        # calculate annualized average returns and corresponding standard deviation
        returns_df = prices_df['Close'].pct_change().dropna()

        avg_returns = returns_df.mean() * 365
        cov_mat = returns_df.cov() * 365

        # Simulate random portfolio weights
        np.random.seed(42)
        weights = np.random.random(size=(n_portfolios, n_assets))
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

        print(portf_results_df.head(5))

        fig = dcc.Graph(
            figure = px.scatter(portf_results_df, x="volatility", y="returns", color='sharpe_ratio')
        ) 
        
    else:
        # fig = dcc.Graph(
        #     figure = {}
        # )
        fig = html.Div()

    return fig

def get_top_10(start_date, end_date):
    prices_df = yf.download(TICKERS, start=start_date, end=end_date, adjusted=True)
    vol_df = prices_df['Volume']

    nan_value = float("NaN")
    vol_df.replace(0, nan_value, inplace=True)
    vol_df = vol_df.dropna(1)
    
    mean_vol = vol_df.mean(axis=0)
    mean_vol_df = pd.DataFrame(mean_vol, columns = ['Average Volume'])

    top100_avg_vol = mean_vol_df.nlargest(100, 'Average Volume')
    tickers = top100_avg_vol.index.values

    sharpe_ratio_set = {"ticker" : (tickers) , "Volatility" : ([]),'Annualised Returns' : ([]),'Sharpe Ratio' : ([])}

    risk_free_rate = 0
    years = 1

    for ticker in tickers:
        panel_data = yf.download(ticker, start=start_date, end=end_date, adjusted=True)
        
        #close price series
        close_df = panel_data['Close']
        
        #calculate close return series
        close_pct_change = close_df.pct_change()
        close_return_series = ((1 + close_pct_change).cumprod() - 1)
        close_returns_df = close_return_series.to_frame()

        #calculate annualised returns
        annualized_returns_df = (1 + close_returns_df.tail(1))**(1/years)-1
        annualised_returns = annualized_returns_df.iloc[0][0]
        sharpe_ratio_set['Annualised Returns'].append(annualised_returns)
        
        #calculate volatility
        volatility = np.sqrt(np.log(close_df/close_df.shift(1)).var()) * np.sqrt(365)
        sharpe_ratio_set['Volatility'].append(volatility)

        #calculate annualised historical volatility
        annualised_historical_volatility = np.sqrt(365) * pd.DataFrame.rolling(np.log(close_df / close_df.shift(1)),window=20).std()
    
        #calculate sharpe ratio
        returns_ts = close_pct_change.dropna()
        returns_ts = returns_ts.to_frame()
        avg_daily_returns = returns_ts['Close'].mean()
        returns_ts['Risk Free Rate'] = risk_free_rate/365
        avg_rfr_ret = returns_ts['Risk Free Rate'].mean()
        returns_ts['Excess Returns'] = returns_ts['Close'] - returns_ts['Risk Free Rate']
        sharpe_ratio = ((avg_daily_returns - avg_rfr_ret) /(returns_ts['Excess Returns'].std()))*np.sqrt(365)
        sharpe_ratio_set['Sharpe Ratio'].append(sharpe_ratio) 

    sharpe_ratio_df = pd.DataFrame(sharpe_ratio_set).sort_values(by='Sharpe Ratio',ascending=False)
    positive_sharpe_ratio_df = sharpe_ratio_df[sharpe_ratio_df['Sharpe Ratio'] > 0]
    positive_sharpe_ratio_df = positive_sharpe_ratio_df.reset_index()

    first_ticker = positive_sharpe_ratio_df['ticker'].values[0]
    top_list = []
    for i in positive_sharpe_ratio_df['ticker'].values:
        top_list.append(i)

    confirmed_top_list = top_list
    top_df = yf.download(top_list, start=start_date, end=end_date, adjusted=True)
    #get return series of top 10 sharpe ratio
    top_close = top_df['Close']
    close_pct_change = top_close.pct_change()
    close_return_df = ((1 + close_pct_change).cumprod() - 1)
    close_return_df.fillna(0,inplace = True)
    corr_matrix = close_return_df.corr()

    top_pairs = get_top_abs_correlations(close_return_df, 10000)
    top_pairs = pd.DataFrame(top_pairs, columns = ['Correlation'])
    removed_top_pairs = top_pairs[top_pairs['Correlation']>0.9]
    removed_top_pairs = removed_top_pairs.index

    removed_list = []
    confirmed_list = []
    for ticker in top_list:
        for i in removed_top_pairs:
            ticker1 = i[0]
            ticker2 = i[1]
            if ticker == ticker1 and ticker2 not in removed_list:
                removed_list.append(ticker2)
            if ticker == ticker2 and ticker1 not in removed_list:
                removed_list.append(ticker1)
        if ticker not in removed_list:
            confirmed_list.append(ticker)
        top_list = [crypto for crypto in top_list if crypto not in removed_list] 
        top_list = [crypto for crypto in top_list if crypto not in confirmed_list] 

    if confirmed_top_list[0] not in confirmed_list:
        confirmed_list = list(confirmed_top_list[0]) + confirmed_list

    final_list = confirmed_list[:10]
    return final_list

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=35):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# this section controls callbacks for portfolio comparisons
@app.callback(
    [Output('recommended_weights', 'children'),
    Output('gut_feel_weights', 'children'),
    Output('recommended_returns', 'children'),
    Output('gut_feel_returns', 'children'),
    Output('recommended_vol', 'children'),
    Output('gut_feel_vol', 'children')],
    Input('submit_comparison', 'n_clicks'),
    [State('portfolio_table', 'data'),
    State('portfolio_tables', 'data'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def update_portfolio_comparison(clicks, gut_feel_data, reco_data, start_date, end_date):
    # print(reco_data)
    reco_tickers = []
    reco_weights = []
    for i in range(len(reco_data)):
        reco_tickers.append(reco_data[i]['crypto'])
        reco_weights.append(reco_data[i]['weightage'])
    
    gut_feel_tickers = []
    gut_feel_weights = []
    for k in range(len(gut_feel_data)):
        gut_feel_tickers.append(gut_feel_data[k]['crypto'])
        gut_feel_weights.append(gut_feel_data[k]['weightage'])
    
    # print(gut_feel_tickers)
    # print(gut_feel_weights)

    recommended_pie = dcc.Graph(figure=px.pie(reco_data, names='crypto', values='weightage'))
    gut_feel_pie = dcc.Graph(figure=px.pie(gut_feel_data, names='crypto', values='weightage'))

    recommended_returns_series = get_rtn_series_df(reco_tickers, reco_weights, start_date, end_date)
    recommended_returns_df = recommended_returns_series.to_frame().reset_index()
    recommended_returns = dcc.Graph(
                                figure = {
                                    'data': [
                                        {'x': recommended_returns_df['Date'], 'y': recommended_returns_df[0],'type':'line'}
                                    ],
                                    'layout' : {
                                        'title': 'Recommended Portfolio Returns'
                                    }
                                }
                            )

    gut_feel_returns_series = get_rtn_series_df(gut_feel_tickers, gut_feel_weights, start_date, end_date)
    gut_feel_returns_df = gut_feel_returns_series.to_frame().reset_index()
    gut_feel_returns = dcc.Graph(
                                figure = {
                                    'data': [
                                        {'x': gut_feel_returns_df['Date'], 'y': gut_feel_returns_df[0],'type':'line'}
                                    ],
                                    'layout' : {
                                        'title': 'Selected Portfolio Returns'
                                    }
                                }
                            )

    recommended_vol_df = get_rolling_volatility_df(recommended_returns_series, 20).to_frame().reset_index()
    recommended_vol = dcc.Graph(
                            figure = {
                                'data': [
                                    {'x': recommended_vol_df['Date'], 'y': recommended_vol_df[0],'type':'line'}
                                ],
                                'layout' : {
                                    'title': 'Recommended Portfolio Volatility'
                                }
                            }
                        )
    # recommended_vol = ""

    
    gut_feel_vol_df = get_rolling_volatility_df(gut_feel_returns_series, 20).to_frame().reset_index()
    gut_feel_vol = dcc.Graph(
                        figure = {
                            'data': [
                                {'x': gut_feel_vol_df['Date'], 'y': gut_feel_vol_df[0],'type':'line'}
                            ],
                            'layout' : {
                                'title': 'Selected Portfolio Volatility'
                            }
                        }
                    )
    
    # gut_feel_vol = ""
    
    return recommended_pie, gut_feel_pie, recommended_returns, gut_feel_returns, recommended_vol, gut_feel_vol

def get_rtn_series_df(tickers, weights, start_date, end_date):
    prices_df = yf.download(tickers, start=start_date, end=end_date, adjusted=True)
    close_df = prices_df['Close']
    rtn_series = (close_df.pct_change()+1).cumprod() - 1 
    # print(len(weights))
    # print(len(rtn_series))
    weighted_rtn_series = weights * rtn_series
    final_rtn_series = weighted_rtn_series.sum(axis=1)
    
    return final_rtn_series

def get_rolling_volatility_df(rtn_series, rolling_window):
    return pd.DataFrame.rolling(np.log((rtn_series+1)/(rtn_series+1).shift(1)),window=rolling_window).std() * np.sqrt(365)

# this section controls callbacks for recommended portfolio
@app.callback(
    [Output("vol_rtn_input", "min"),
    Output("vol_rtn_input", "max"),
    Output("vol_rtn_input", "value"),
    Output("portfolio_tables", "data"),
    Output("portfolio_metrics", "children"),
    Output("vol_rtn_input", "disabled"),
    Output("vol_rtn_label", "children"),
    Output("recommended_alert_auto", 'children'),
    Output("recommended_alert_auto", 'color'),
    Output("recommended_alert_auto", 'is_open'),],
    [Input("tabs", "active_tab"),
    Input('submit_vol_rtn_input', 'n_clicks'),
    Input("get_reco_portf", 'n_clicks')],
    [State('vol_rtn_input', 'value'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value'),
    State('recommended_alert_auto', 'is_open'),
    State('memory', 'data')],
    prevent_initial_call = True
)
def switch_tab(at, n_clicks, n_clicks2, slider_val, start_date, end_date, is_open, mem_data):
    # print(slider_val)
    # print(at)
    n_portfolios = 10 ** 5
    n_days = 365
    # implement again later
    risky_assets = get_risky_assets(mem_data, start_date, end_date)
    # risky_assets = ['XWC-USD', 'DGD-USD', 'LRC-USD', 'ANT-USD', 'ADA-USD', 'KNC-USD','DNT-USD', 'ICX-USD','DGB-USD', 'LTC-USD']
    n_assets = len(risky_assets)

    # Simulate random portfolio weights
    np.random.seed(42)
    weights = np.random.random(size=(n_portfolios, n_assets))
    weights /=  np.sum(weights, axis=1)[:, np.newaxis]

    is_open = True

    # for best sharpe ratio
    if at == "best":
        portf_results_df = return_mpt_df(start_date,end_date, n_portfolios, n_days, n_assets, risky_assets, weights)
        max_sharpe_ind, max_sharpe_portf_weights = max_sharpe(portf_results_df, weights)

        table_data = []
        for i in range(n_assets):
            pair = {'crypto': risky_assets[i], 'weightage': max_sharpe_portf_weights[i].round(2)}
            # print(pair)
            table_data.append(pair)

        metrics = portf_results_df.loc[max_sharpe_ind]
        # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
        recommended_metrics = pd.DataFrame({"Returns": [metrics[0].round(2)], "Volatility": [metrics[1].round(2)], "Sharpe Ratio": [metrics[2].round(2)]})
        
        fig  = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
        # fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
        
        min_vol_value = 0
        max_vol_value = 0
        min_vol_value = 0
        label_text = "Input disabled"
        text = "Success!"
        color = "success"
        return min_vol_value, max_vol_value, min_vol_value, table_data, fig, True, label_text, text, color, is_open

    # adjusting volatility
    elif at == "volatility":
        portf_results_df = return_mpt_df(start_date,end_date, n_portfolios, n_days, n_assets, risky_assets, weights)
        min_vol_value, max_vol_value = get_min_max_vol(portf_results_df)

        if port_exists_vol_input(slider_val, n_portfolios, portf_results_df):
            index_from_vol = portf_ind_from_vol(slider_val, n_portfolios, portf_results_df)
            weights_from_vol = weights[index_from_vol]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_vol[i].round(2)}
                # print(pair)
                table_data.append(pair)
            
            metrics = portf_results_df.loc[index_from_vol]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Returns": [metrics[0].round(2)], "Volatility": [metrics[1].round(2)], "Sharpe Ratio": [metrics[2].round(2)]})

            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            label_text = "Input volatility"
            text = "Success!"
            color = "success"

            return min_vol_value, max_vol_value, slider_val, table_data, fig, False, label_text, text, color, is_open

        else:
            index_from_vol = portf_ind_from_vol(min_vol_value, n_portfolios, portf_results_df)
            weights_from_vol = weights[index_from_vol]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_vol[i].round(2)}
                # print(pair)
                table_data.append(pair)

            metrics = portf_results_df.loc[index_from_vol]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Returns": [metrics[0].round(2)], "Volatility": [metrics[1].round(2)], "Sharpe Ratio": [metrics[2].round(2)]})

            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            label_text = "Input volatility"
            text = "No portfolio for specified volatility"
            color = "danger"

            return min_vol_value, max_vol_value, min_vol_value, table_data, fig, False, label_text, text, color, is_open

    # adjusting returns
    elif at == "returns":
        portf_results_df = return_mpt_df(start_date,end_date, n_portfolios, n_days, n_assets, risky_assets, weights)
        min_vol_value, max_vol_value = get_min_max_vol(portf_results_df)
        min_vol_ind = portf_ind_from_vol(min_vol_value, n_portfolios, portf_results_df)
        min_rtn_value, max_rtn_value = get_min_max_rtns(portf_results_df, min_vol_ind)

        if port_exists_rtn_input(slider_val, n_portfolios, portf_results_df):
            index_from_rtn = portf_ind_from_rtn(slider_val, n_portfolios, portf_results_df)
            weights_from_rtn = weights[index_from_rtn]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_rtn[i].round(2)}
                # print(pair)
                table_data.append(pair)

            metrics = portf_results_df.loc[index_from_rtn]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Returns": [metrics[0].round(2)], "Volatility": [metrics[1].round(2)], "Sharpe Ratio": [metrics[2].round(2)]})

            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            label_text = "Input returns"
            text = "Success!"
            color = "success"

            return min_rtn_value, max_rtn_value, slider_val, table_data, fig, False, label_text, text, color, is_open

        else: 
            index_from_rtn = portf_ind_from_rtn(max_rtn_value, n_portfolios, portf_results_df)
            weights_from_rtn = weights[index_from_rtn]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_rtn[i].round(2)}
                # print(pair)
                table_data.append(pair)

            metrics = portf_results_df.loc[index_from_rtn]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Returns": [metrics[0].round(2)], "Volatility": [metrics[1].round(2)], "Sharpe Ratio": [metrics[2].round(2)]})

            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            text = "No portfolio for specified returns"
            color = "danger"
            label_text = "Input returns"

            return min_rtn_value, max_rtn_value, max_rtn_value, table_data, fig, False, label_text, text, color, is_open

def get_risky_assets(mem_data, start_date, end_date):
    vol_df = mem_data["tickers volume"]

    nan_value = float("NaN")
    vol_df.replace(0, nan_value, inplace=True)
    vol_df = vol_df.dropna(1)
    
    mean_vol = vol_df.mean(axis=0)
    mean_vol_df = pd.DataFrame(mean_vol, columns = ['Average Volume'])

    top100_avg_vol = mean_vol_df.nlargest(100, 'Average Volume')
    tickers = top100_avg_vol.index.values

    sharpe_ratio_set = {"ticker" : (tickers) , "Volatility" : ([]),'Annualised Returns' : ([]),'Sharpe Ratio' : ([])}

    risk_free_rate = 0
    years = 1

    for ticker in tickers:
        panel_data = yf.download(ticker, start=start_date, end=end_date, adjusted=True)
        
        #close price series
        close_df = panel_data['Close']
        
        #calculate close return series
        close_pct_change = close_df.pct_change()
        close_return_series = ((1 + close_pct_change).cumprod() - 1)
        close_returns_df = close_return_series.to_frame()

        #calculate annualised returns
        annualized_returns_df = (1 + close_returns_df.tail(1))**(1/years)-1
        annualised_returns = annualized_returns_df.iloc[0][0]
        sharpe_ratio_set['Annualised Returns'].append(annualised_returns)
        
        #calculate volatility
        volatility = np.sqrt(np.log(close_df/close_df.shift(1)).var()) * np.sqrt(365)
        sharpe_ratio_set['Volatility'].append(volatility)

        #calculate annualised historical volatility
        annualised_historical_volatility = np.sqrt(365) * pd.DataFrame.rolling(np.log(close_df / close_df.shift(1)),window=20).std()
    
        #calculate sharpe ratio
        returns_ts = close_pct_change.dropna()
        returns_ts = returns_ts.to_frame()
        avg_daily_returns = returns_ts['Close'].mean()
        returns_ts['Risk Free Rate'] = risk_free_rate/365
        avg_rfr_ret = returns_ts['Risk Free Rate'].mean()
        returns_ts['Excess Returns'] = returns_ts['Close'] - returns_ts['Risk Free Rate']
        sharpe_ratio = ((avg_daily_returns - avg_rfr_ret) /(returns_ts['Excess Returns'].std()))*np.sqrt(365)
        sharpe_ratio_set['Sharpe Ratio'].append(sharpe_ratio) 

    sharpe_ratio_df = pd.DataFrame(sharpe_ratio_set).sort_values(by='Sharpe Ratio',ascending=False)
    positive_sharpe_ratio_df = sharpe_ratio_df[sharpe_ratio_df['Sharpe Ratio'] > 0]
    positive_sharpe_ratio_df = positive_sharpe_ratio_df.reset_index()

    first_ticker = positive_sharpe_ratio_df['ticker'].values[0]
    top_list = []
    for i in positive_sharpe_ratio_df['ticker'].values:
        top_list.append(i)

    confirmed_top_list = top_list
    top_df = yf.download(top_list, start=start_date, end=end_date, adjusted=True)
    #get return series of top 10 sharpe ratio
    top_close = top_df['Close']
    close_pct_change = top_close.pct_change()
    close_return_df = ((1 + close_pct_change).cumprod() - 1)
    close_return_df.fillna(0,inplace = True)
    corr_matrix = close_return_df.corr()

    top_pairs = get_top_abs_correlations(close_return_df, 10000)
    top_pairs = pd.DataFrame(top_pairs, columns = ['Correlation'])
    removed_top_pairs = top_pairs[top_pairs['Correlation']>0.9]
    removed_top_pairs = removed_top_pairs.index

    removed_list = []
    confirmed_list = []
    for ticker in top_list:
        for i in removed_top_pairs:
            ticker1 = i[0]
            ticker2 = i[1]
            if ticker == ticker1 and ticker2 not in removed_list:
                removed_list.append(ticker2)
            if ticker == ticker2 and ticker1 not in removed_list:
                removed_list.append(ticker1)
        if ticker not in removed_list:
            confirmed_list.append(ticker)
        top_list = [crypto for crypto in top_list if crypto not in removed_list] 
        top_list = [crypto for crypto in top_list if crypto not in confirmed_list] 

    if confirmed_top_list[0] not in confirmed_list:
        confirmed_list = list(confirmed_top_list[0]) + confirmed_list

    final_list = confirmed_list[:10]
    return final_list

def return_mpt_df(start_date, end_date, n_portfolios, n_days, n_assets, risky_assets, weights):
    prices_df = yf.download(risky_assets, start=start_date, end=end_date, adjusted=True)

    # calculate annualized average returns and corresponding standard deviation
    returns_df = prices_df['Close'].pct_change().dropna()
    avg_returns = returns_df.mean() * n_days
    cov_mat = returns_df.cov() * n_days

    # RETURNS / EXPECTED RETURNS
    portf_rtns = np.dot(weights, avg_returns)
    
    # VOLATILITY / STANDARD DEVIATION
    portf_vol = []
    for i in range(0, len(weights)):
        portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))

    portf_vol = np.array(portf_vol)  

    # SHARPE RATIO
    portf_sharpe_ratio = portf_rtns / portf_vol

    # Create a joint DataFrame with all data
    portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                    'volatility': portf_vol,
                                    'sharpe_ratio': portf_sharpe_ratio})

    return portf_results_df 

def max_sharpe(portf_results_df, weights):
    max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
    max_sharpe_portf_weights = weights[max_sharpe_ind]
    
    return max_sharpe_ind, max_sharpe_portf_weights

def get_min_max_vol(portf_results_df):
    min_vol_ind = np.argmin(portf_results_df.volatility)
    min_vol_value = portf_results_df['volatility'][min_vol_ind].round(2)

    max_vol_ind = np.argmax(portf_results_df.volatility)
    max_vol_value = portf_results_df['volatility'][max_vol_ind].round(2)
    return min_vol_value, max_vol_value

def port_exists_vol_input(vol_input, n_portfolios, portf_results_df):
    for i in range(n_portfolios):
        if portf_results_df['volatility'][i].round(2) == vol_input:
            return True
    return False

def portf_ind_from_vol(vol_input, n_portfolios, portf_results_df):
    portfolios_consider = {}
    for i in range(n_portfolios):
        if portf_results_df['volatility'][i].round(2) == vol_input:
            portfolios_consider[i] = portf_results_df['returns'][i]
            found = True


    max_key = max(portfolios_consider, key=portfolios_consider.get)
    return max_key

def get_min_max_rtns(portf_results_df, min_vol_ind):
    min_rtn_value = portf_results_df['returns'][min_vol_ind].round(2)

    max_rtn_ind = np.argmax(portf_results_df.returns)
    max_rtn_value = portf_results_df['returns'][max_rtn_ind].round(2)
    return min_rtn_value,max_rtn_value

def port_exists_rtn_input(rtn_input, n_portfolios, portf_results_df):
    for i in range(n_portfolios):
        if portf_results_df['returns'][i].round(2) == rtn_input:
            return True
    return False

def portf_ind_from_rtn(rtn_input, n_portfolios, portf_results_df):
    portfolios_consider = {}
    for i in range(n_portfolios):
        if portf_results_df['returns'][i].round(2) == rtn_input:
            portfolios_consider[i] = portf_results_df['volatility'][i]
    
    min_key = min(portfolios_consider, key=portfolios_consider.get)

    return min_key

@app.callback(
    Output("lstm_weights", "children"),
    Input("lstm_button", "n_clicks"),
    [State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value'),
    State('portfolio_tables', 'data')]
)
def get_lstm_weights(n_clicks, start_date, end_date, reco_data):
    crypto_list = []
    weights_list = []
    for i in range(len(reco_data)):
        crypto_list.append(reco_data[i]['crypto'])
        weights_list.append(reco_data[i]['weightage'])

    prices_df = yf.download(crypto_list, start=start_date, end=end_date, adjusted=True)
    prices_df = prices_df['Close']
    results = []

    original_price = []
    for ticker in crypto_list:
        current_price = prices_df[ticker].iloc[-1]
        original_price.append(current_price)
    original_price
    dataFrames_arr = []

    for ticker in crypto_list:

        close_df = prices_df[ticker]


        scaler=MinMaxScaler(feature_range=(0,1))
        close_df=scaler.fit_transform(np.array(close_df).reshape(-1,1))

        training_size=int(len(close_df)*0.65)
        test_size=len(close_df)-training_size
        train_data,test_data=close_df[0:training_size,:],close_df[training_size:len(close_df),:1]


        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        model= Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam') 
        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=5,batch_size = 32,verbose =1)

        shape_input = test_data.shape[0]-100


        x_input=test_data[shape_input:].reshape(1,-1)

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()



        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
            
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                # print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                # print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


        df3=close_df.tolist()
        df3.extend(lst_output)

        df3=scaler.inverse_transform(df3).tolist()
        dataFrames_arr.append(df3)

        results.append(df3[-1])

    future_prices = []

    for pr in results:
        future_prices.append(pr[0])

    array = []

    for i in range(len(original_price)):
        eachTicker = []
        eachTicker.append(original_price[i])
        eachTicker.append(future_prices[i])
        array.append(eachTicker)


    time_extension = len(df3) - 1

    EndDate = prices_df.index[0] + timedelta(days=time_extension)

    dateIndexes = pd.date_range(start='2020-01-01', end=EndDate)

    df = pd.DataFrame(data = array, 
                    index = crypto_list, 
                    columns = ['original_prices','future_prices'])
    df['pct_change'] = ((df['future_prices'] - df['original_prices'])/df['original_prices']) * 100
    df.sort_values(by='pct_change',ascending = False,inplace = True)

    new_df = df.copy()
    new_df['weights'] = [24, 21, 15, 13, 9, 7, 5, 3, 2 , 1 ]

    name = crypto_list[0]
    column_name = name + " Price"
    each_df = pd.DataFrame(data = dataFrames_arr[0], 
                    index = dateIndexes, 
                    columns = [column_name])
    


    for j in range(1, len(dataFrames_arr)):
        name = crypto_list[j]
        column_name = name + " Price"
        each_df2 = pd.DataFrame(data = dataFrames_arr[j], 
                        index = dateIndexes, 
                        columns = [column_name])
        each_df = pd.merge(each_df, each_df2, left_index=True, right_index=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for k in range(len(crypto_list)):
        name = crypto_list[k]
        column_name = name + " Price"

        # Add traces
        fig.add_trace(
            go.Scatter(x=dateIndexes, y=each_df[column_name], name=column_name),
            secondary_y=False,
        )

    figure = dcc.Graph(
        figure = fig
    )
    lstm_table = dcc.Table.from_dataframe(new_df, striped=True, bordered=True, hover=True)

    return [figure, lstm_table]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
        
    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
    app.run_server(debug=True)