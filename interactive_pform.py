import dash
from dash.dependencies import Input, Output, State
from dash import dash_table

# import dash_core_components as dcc
# import dash_html_components as html


from dash import html
from dash import dcc
import pandas as pd
import plotly.express as px


# initialising datetime
from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')

def get_crypto_prices_df():
    START_DATE = '2020-01-01'
    # END_DATE = '2020-12-31'
    END_DATE = '2021-01-01'

    plt.style.use('seaborn')
    sns.set_palette('cubehelix')
    # plt.style.use('seaborn-colorblind') #alternative
    plt.rcParams['figure.figsize'] = [8, 4.5]
    plt.rcParams['figure.dpi'] = 300
    warnings.simplefilter(action='ignore', category=FutureWarning)

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

    prices_df = yf.download(tickers, start=START_DATE,end=END_DATE, adjusted=True)
    return prices_df

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        # dcc.Input(
        #     id='adding-rows-name',
        #     placeholder='Enter a column name...',
        #     value='',
        #     style={'padding': 10}
        # ),
         html.Button('Add Row', id='editing-rows-button', n_clicks=0)
        # html.Button('Add Column', id='adding-columns-button', n_clicks=0)
    ], style={'height': 50}),


dash_table.DataTable(
        id='our-table',
        columns=[{'name': 'Product', 'id': 'Product', 'deletable': False, 'renamable': False},
                 {'name': 'weightage', 'id': 'weightage', 'deletable': True, 'renamable': True}
        ],
        data={'Product': 'BTC', 'weightage': '0.1'},
        ],[
        editable=True,                  # allow user to edit data inside tabel
        row_deletable=True,             # allow user to delete rows
        sort_action="native",           # give user capability to sort columns
        sort_mode="single",             # sort across 'multi' or 'single' columns
        filter_action="native",         # allow filtering of columns
        page_action='none',             # render all of the data at once. No paging.
        style_table={'height': '200px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'left', 'minWidth': '50px', 'width': '50px', 'maxWidth': '50px'},
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'right'
            } for c in ['Product', 'weightage']
        ]
    ),   
   


    # Create notification when saving to excel
    html.Div(id='placeholder', children=[]),
    dcc.Store(id="store", data=0),
    dcc.Interval(id='interval', interval=1000),

    dcc.Graph(id='my_graph')
])


# @app.callback(

#     Output('our-table', 'columns'),
#     [Input('adding-columns-button', 'n_clicks')],
#     [State('adding-rows-name', 'value'),
#      State('our-table', 'columns')],
# )
# def add_columns(n_clicks, value, existing_columns):
#     print(existing_columns)
#     if n_clicks > 0:
#         existing_columns.append({
#             'name': value, 'id': value,
#             'renamable': True, 'deletable': True
#         })
#     print(existing_columns)
#     return existing_columns


@app.callback(
    Output('our-table', 'data'),
    [Input('editing-rows-button', 'n_clicks')],
    [State('our-table', 'data'),
     State('our-table', 'columns')],
)
def add_row(n_clicks, rows, columns):
    # print(rows)
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    # print(rows)
    return rows


@app.callback(
    Output('my_graph', 'figure'),
    [Input('our-table', 'data')])
def display_graph(data):
    df_fig = pd.DataFrame(data)
    # print(df_fig)
    # fig = px.bar(df_fig, x='Product', y='weightage')
    fig = px.pie(df_fig, names='Product', values='weightage')

    # fig.show()
    # fig2.show()
    return fig

# @app.callback(
#     Output('my_graph', 'figure'),
#     [Input('our-table', 'data')])
# def display(data):
#     df_fig = pd.DataFrame(data)
#     fig2 = px.pie(df_fig, names='Product', values='weightage')
#     return fig2

if __name__ == '__main__':
    app.run_server(debug=True)