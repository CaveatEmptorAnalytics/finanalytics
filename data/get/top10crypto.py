import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import yfinance as yf


def get_top_10_crypto():
    START_DATE = '2020-01-01'
    # END_DATE = '2020-12-31'
    END_DATE = '2021-01-01'

    plt.style.use('seaborn')
    sns.set_palette('cubehelix')
    # plt.style.use('seaborn-colorblind') #alternative
    plt.rcParams['figure.figsize'] = [8, 4.5]
    plt.rcParams['figure.dpi'] = 300
    warnings.simplefilter(action='ignore', category=FutureWarning)
    tickers = ['ETH-USD', 'DGD-USD', 'XMR-USD', 'ADX-USD', 'BTC-USD', 'KNC-USD', 'BNT-USD', 'ICX-USD', 'LBC-USD', 'EMC2-USD']


    prices_df = yf.download(tickers, start=START_DATE,end=END_DATE, adjusted=True)
    return prices_df