import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import statsmodels.api as sm
from pandas_datareader import data
from matplotlib.ticker import FuncFormatter 
from scipy.stats import norm
from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d')
start_date = '2020-01-01'
end_date = '2020-12-31'
end_date = today
tickers = ['ETH-USD','HEX-USD', 'BNB-USD','XRP-USD','SOL1-USD','DOT1-USD','DOGE-USD','UNI3-USD','AVAX-USD','LUNA1-USD','LINK-USD','ALGO-USD','LTC-USD','BCH-USD','ATOM1-USD','MATIC-USD','ICP1-USD','XLM-USD','FIL-USD','TRX-USD','ETC-USD']
def get_yahoo():
    df = pd.DataFrame()
    for ticker in tickers:
        panel_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
        panel_data[ticker] = panel_data['Close']

        if df.empty:
            df = panel_data[[ticker]]
        else:
            df = df.join(panel_data[[ticker]], how='outer')
    return df

