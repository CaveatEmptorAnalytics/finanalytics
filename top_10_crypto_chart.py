import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
import operator
import plotly.express as px
import plotly.graph_objects as go

from os import X_OK
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

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

START_DATE = '2020-01-01'
END_DATE = '2021-01-01'

prices_df = yf.download(tickers, start=START_DATE, 
                        end=END_DATE, adjusted=True)

vol_df = prices_df['Volume']
nan_value = float("NaN")
vol_df.replace(0, nan_value, inplace=True)
vol_df = vol_df.dropna(1)

mean_vol = vol_df.mean(axis=0)
mean_vol_df = pd.DataFrame(mean_vol, columns = ['Average Volume'])

top100_avg_vol = mean_vol_df.nlargest(100, 'Average Volume')

tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'BCH-USD', 'EOS-USD','TRX-USD', 'ETC-USD', 'LINK-USD', 
          'DASH-USD', 'XMR-USD', 'NEO-USD','ZEC-USD', 'ADA-USD', 'XLM-USD', 'QTUM-USD', 'BNB-USD', 'OMG-USD',
          'XTZ-USD', 'BAT-USD', 'DOGE-USD', 'WAVES-USD', 'XEM-USD','ZRX-USD', 'KNC-USD', 'DCR-USD', 'BNT-USD', 
          'ICX-USD', 'WTC-USD','STORJ-USD', 'BTG-USD', 'MCO-USD', 'REP-USD', 'LRC-USD', 'HC-USD','SNT-USD', 
          'BTM-USD', 'MIOTA-USD', 'CVC-USD', 'NULS-USD','ANT-USD', 'DGB-USD', 'BTS-USD', 'GXC-USD', 'AE-USD',
          'GAS-USD','PAY-USD', 'NANO-USD', 'MTL-USD', 'MONA-USD', 'STEEM-USD','LSK-USD', 'ZEN-USD', 'AION-USD', 
          'SC-USD', 'NAS-USD', 'KMD-USD','ARDR-USD', 'ACT-USD', 'XVG-USD', 'RLC-USD', 'GRS-USD', 'ARK-USD',
          'PPT-USD', 'DNT-USD', 'ADK-USD', 'ADX-USD', 'MGO-USD', 'SBD-USD','XUC-USD', 'DGD-USD', 'SYS-USD', 
          'LBC-USD', 'MLN-USD', 'EMC2-USD','XAS-USD', 'NXT-USD', 'ETP-USD', 'XWC-USD', 'AMB-USD', 'FUN-USD',
          'VTC-USD', 'PIVX-USD', 'ETN-USD', 'PZM-USD', 'SKY-USD','YOYOW-USD', 'NEBL-USD', 'SNGLS-USD', 
          'FCT-USD', 'VIA-USD','QRL-USD', 'MAID-USD', 'NAV-USD', 'GNO-USD', 'SMART-USD','RDD-USD', 'DCN-USD',
          'HNC-USD', 'NXS-USD']

START_DATE = '2020-01-01'
END_DATE = '2021-01-01'

sharpe_ratio_set = {"ticker" : (tickers) , "Volatility" : ([]),'Annualised Returns' : ([]),'Sharpe Ratio' : ([])}

risk_free_rate = 0
years = 1

for ticker in tickers:
    panel_data = yf.download(ticker, start=START_DATE, end=END_DATE, adjusted=True)
    
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

sharpe_ratio_num = 35
top_35df = sharpe_ratio_df.nlargest(sharpe_ratio_num, 'Sharpe Ratio')

top35_list = ['XWC-USD', 'DGD-USD', 'ETH-USD', 'BTC-USD', 'GNO-USD', 'LRC-USD','XEM-USD', 'ANT-USD', 
              'ADA-USD', 'LINK-USD', 'MLN-USD', 'HNC-USD','WAVES-USD', 'XMR-USD', 'KNC-USD', 'DNT-USD', 
              'BNT-USD', 'ICX-USD','ADX-USD', 'DGB-USD', 'MAID-USD', 'LTC-USD', 'OMG-USD', 'CVC-USD',
              'SBD-USD', 'BNB-USD', 'SNT-USD', 'STORJ-USD', 'EMC2-USD','XLM-USD', 'QRL-USD', 'ARK-USD', 
              'DCR-USD', 'SYS-USD', 'LBC-USD']

top_35_df = yf.download(top35_list, start=START_DATE, 
                        end=END_DATE, adjusted=True)

top_35_close = top_35_df['Close']
close_pct_change = top_35_close.pct_change()
close_return_df = ((1 + close_pct_change).cumprod() - 1)

close_return_df.fillna(0,inplace = True)

corr_matrix = close_return_df.corr()

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

top_pairs = get_top_abs_correlations(close_return_df, 20)
top_pairs = pd.DataFrame(top_pairs, columns = ['Correlation'])

top_pairs_index = top_pairs.index

crypto_list = []
for i in top_pairs_index:
    for j in i:
        crypto_list.append(j)

top_crypto_dict = {}

for i in crypto_list: 
    if i not in top_crypto_dict.keys():
        top_crypto_dict[i] = 1
    else: 
        top_crypto_dict[i] += 1

crypto_sort = sorted(top_crypto_dict.items(),key = operator.itemgetter(1),reverse = True)

n = 5
top5_crypto_list = []
for crypto in range (0, n):
    top5_crypto_list.append(crypto_sort[crypto][0])

def get_bottom_correlations(df, n):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=True)
    return au_corr[0:n]

all_pairs = get_bottom_correlations(close_return_df, sharpe_ratio_num*sharpe_ratio_num-sharpe_ratio_num)
all_pairs = pd.DataFrame(all_pairs, columns = ['Correlation'])
all_pairs['row num'] = np.arange(len(all_pairs))

next5_list = []

for i in top5_crypto_list:
    index_count = -1
    count = 0
    for ticker in all_pairs.index:
        index_count += 1 
        if i in ticker: 
            count += 1
            if count < 8:
                next5_list.append(index_count)

bottom_pairs = all_pairs.loc[all_pairs['row num'].isin(next5_list)]
bottom_pairs_index = bottom_pairs.index

bottom_crypto_list = []
for i in bottom_pairs_index:
    for j in i:
        bottom_crypto_list.append(j)

bottom_crypto_dict = {}

for i in bottom_crypto_list: 
    if i not in bottom_crypto_dict.keys():
        bottom_crypto_dict[i] = 1
    else: 
        bottom_crypto_dict[i] += 1

bottom_crypto_sort = sorted(bottom_crypto_dict.items(),key = operator.itemgetter(1),reverse = True)

revised_bottom_crypto_sort = []
for crypto in bottom_crypto_sort: 
    if crypto[0] not in top5_crypto_list:
        revised_bottom_crypto_sort.append(crypto)

n = 5
bottom5_crypto_list = []
for crypto in range (0, n):
    bottom5_crypto_list.append(revised_bottom_crypto_sort[crypto][0])

top10 = top5_crypto_list + bottom5_crypto_list

# this is the chart
values = ['0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1']
fig = go.Figure(data=[go.Pie(labels=top10, values=values)])
fig.show()

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children="Top 10 Cryptos"),
    dcc.Graph(
        id="top_10_cryptos",
        figure = {
            'data': [
                {'x': values, 'y': top10,'type':'pie'}
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