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
                        ]
                    ),
                ],
                fluid=True,
                className="py-3",
            ),
            className="p-3 text-white bg-dark rounded-3",
        ),
        # this section is for them to select the crypto ticker that they want to analyse
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Select the crypto you would like it analyze", html_for="crypto_dropdown"),
                        dcc.Dropdown(
                                id="crypto_dropdown",
                                options=[{"value": x, "label": x} 
                                            for x in tickers]
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
        # this section displays the analysis charts
        dbc.Row(
            [
                html.Div(id="display_statistics", className="col-4"),
                html.Div(id="display_charts", className="col-8"),
            ],
            className="g-0",
            style={
                'padding-top': 50,
                'padding-bottom': 100
            },
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
                        dbc.Row(
                            [
                                dcc.Slider(id="slider_bar", min=0, max=10, step=0.01, value=0),
                                # html.Div(id="slider_bar"),
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
                'padding-bottom': 100
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

# this section controls callbacks for gut feel portfolio
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

    stats_dict = {"labels": ["Volatility", "Sharpe Ratio", "Annualized Returns", "Skewness", "Kurtosis"], "values": [volatility, sharpe_ratio, annualized_returns, skewness, kurtosis]}

    statistics_df = pd.DataFrame(stats_dict, index=stats_dict["labels"])
    # statistics_df = statistics_df.reset_index()

    close_pct_change = pd.DataFrame(close_pct_change)
    close_pct_change = close_pct_change.reset_index()

    close_return_series =  pd.DataFrame(close_return_series)
    close_return_series = close_return_series.reset_index()

    ahv = pd.DataFrame(ahv)
    ahv = ahv.reset_index()
    
    df1 = close_pct_change

    display_stats = dbc.Col(
                        [
                            html.H3(["Crypto Statistics"], className="text-center"),
                            dbc.Table.from_dataframe(statistics_df, striped=True, bordered=True, hover=True),
                        ]
                    ),
    display_charts = dbc.Col(
                        [
                            html.H3(["Crypto Charts"], className="text-center"),
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
    volatility = np.sqrt(np.log(crypto_close / crypto_close.shift(1)).var()) * np.sqrt(252)
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

# this section controls callbacks for portfolio comparisons
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
    recommended_weights = "get_reco_df(gut_feel_data)"
    gut_feel_weights = dcc.Graph(figure=px.pie(gut_feel_data, names='crypto', values='weightage'))

    recommended_returns = ""
    gut_feel_returns = ""

    recommended_sharpe = ""
    gut_feel_sharpe = ""
    
    return recommended_weights, gut_feel_weights, recommended_returns, gut_feel_returns, recommended_sharpe, gut_feel_sharpe

# this section controls callbacks for recommended portfolio
@app.callback(
    [Output("slider_bar", "min"),
    Output("slider_bar", "max"),
    Output("slider_bar", "value"),
    Output("portfolio_tables", "children")],
    [Input("tabs", "active_tab"),
    Input('slider_bar', 'value')],
    [State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')]
)
def switch_tab(at, slider_val, start_date, end_date):
    print(slider_val)
    n_portfolios = 10 ** 5
    n_days = 365
    # implement again later
    # risky_assets = get_risky_assets()
    risky_assets = ['XWC-USD', 'DGD-USD', 'LRC-USD', 'ANT-USD', 'ADA-USD', 'KNC-USD','DNT-USD', 'ICX-USD','DGB-USD', 'LTC-USD']
    n_assets = len(risky_assets)

    # Simulate random portfolio weights
    np.random.seed(42)
    weights = np.random.random(size=(n_portfolios, n_assets))
    weights /=  np.sum(weights, axis=1)[:, np.newaxis]

    if at == "best":
        portf_results_df = return_mpt_df(start_date,end_date, n_portfolios, n_days, n_assets, risky_assets, weights)
        max_sharpe_ind, max_sharpe_portf_weights = max_sharpe(portf_results_df, weights)

        table_data = []
        for i in range(n_assets):
            pair = [risky_assets[i], (max_sharpe_portf_weights[i]*100).round(2)]
            # print(pair)
            table_data.append(pair)

        recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
        fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
        min_vol_value = 0
        max_vol_value = 0
        min_vol_value = 0
        return min_vol_value, max_vol_value, min_vol_value, fig

    elif at == "volatility":
        portf_results_df = return_mpt_df(start_date,end_date, n_portfolios, n_days, n_assets, risky_assets, weights)
        min_vol_value, max_vol_value = get_min_max_vol(portf_results_df)

        if slider_val == 0:
            index_from_vol = portf_ind_from_vol(min_vol_value, n_portfolios, portf_results_df)
            weights_from_vol = weights[index_from_vol]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = [risky_assets[i], (weights_from_vol[i]*100).round(2)]
                # print(pair)
                table_data.append(pair)

            recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            return min_vol_value, max_vol_value, min_vol_value, fig

        elif port_exists_vol_input(slider_val, n_portfolios, portf_results_df):
            index_from_vol = portf_ind_from_vol(slider_val, n_portfolios, portf_results_df)
            weights_from_vol = weights[index_from_vol]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = [risky_assets[i], (weights_from_vol[i]*100).round(2)]
                # print(pair)
                table_data.append(pair)

            recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            return min_vol_value, max_vol_value, slider_val, fig

        else:
            print("does not exist")

    elif at == "returns":
        portf_results_df = return_mpt_df(start_date,end_date, n_portfolios, n_days, n_assets, risky_assets, weights)
        min_vol_value, max_vol_value = get_min_max_vol(portf_results_df)
        min_vol_ind = portf_ind_from_vol(min_vol_value, n_portfolios, portf_results_df)
        min_rtn_value, max_rtn_value = get_min_max_rtns(portf_results_df, min_vol_ind)

        if slider_val == 0:    
            index_from_rtn = portf_ind_from_rtn(max_rtn_value, n_portfolios, portf_results_df)
            weights_from_rtn = weights[index_from_rtn]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = [risky_assets[i], (weights_from_rtn[i]*100).round(2)]
                # print(pair)
                table_data.append(pair)

            recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            return min_rtn_value, max_rtn_value, max_rtn_value, fig
        
        elif port_exists_rtn_input(slider_val, n_portfolios, portf_results_df):
            index_from_rtn = portf_ind_from_rtn(slider_val, n_portfolios, portf_results_df)
            weights_from_rtn = weights[index_from_rtn]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = [risky_assets[i], (weights_from_rtn[i]*100).round(2)]
                # print(pair)
                table_data.append(pair)

            recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            return min_rtn_value, max_rtn_value, slider_val, fig
        
        else:
            print("does not exist")
    
    pass

def get_risky_assets():
    pass

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

# this section controls callbacks for the sliders
# @app.callback(
#     Output("portfolio_tables", "children"),
#     [Input("slider_bar", "value"),
#     Input("tabs", "active_tab")],
#     prevent_initial_call = True
# )
# def update_tables(value, tab):
#     n_portfolios = 10 ** 5
#     n_days = 365
#     # implement again later
#     # risky_assets = get_risky_assets()
#     risky_assets = ['XWC-USD', 'DGD-USD', 'LRC-USD', 'ANT-USD', 'ADA-USD', 'KNC-USD','DNT-USD', 'ICX-USD','DGB-USD', 'LTC-USD']
#     n_assets = len(risky_assets)

#     # Simulate random portfolio weights
#     np.random.seed(42)
#     weights = np.random.random(size=(n_portfolios, n_assets))
#     weights /=  np.sum(weights, axis=1)[:, np.newaxis]
    
#     if tab == "volatility":
#         portf_results_df = return_mpt_df(start_date,end_date, n_portfolios, n_days, n_assets, risky_assets, weights)
#         min_vol_value, max_vol_value = get_min_max_vol(portf_results_df)
#         # slider = dcc.Slider(id="volatility_slider", min=min_vol_value, max=max_vol_value, step=0.01, value=min_vol_value)

#         index_from_vol = portf_ind_from_vol(min_vol_value, n_portfolios, portf_results_df)
#         weights_from_vol = weights[index_from_vol]
        
#         # display tables
#         table_data = []
#         for i in range(n_assets):
#             pair = [risky_assets[i], (weights_from_vol[i]*100).round(2)]
#             # print(pair)
#             table_data.append(pair)

#         recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
#         fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
        
#         return min_vol_value, max_vol_value, min_vol_value, fig

#     if tab == "returns":
#         pass

#     print(value)
#     pass

# def return_mpt_df(start_date, end_date, n_portfolios, n_days, n_assets, risky_assets, weights):
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

if __name__ == "__main__":
    app.run_server(debug=True)


# dashboard that takes in start and end date for each input and returns weightage for top 10 crypto
# 2 pages 