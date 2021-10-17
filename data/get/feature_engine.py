from top10crypto import get_top_10_crypto
import pandas as pd
prices_df = get_top_10_crypto()
# def create_single_crypto_df(crypto_ticker):

#     single_crypto_df['Adj Close'] = prices_df['Adj Close'][crypto_ticker]

#     return single_crypto_df

def create_single_crypto_df(crypto_ticker):
    single_crypto_df = pd.DataFrame()
    single_crypto_df['Adj Close'] = prices_df['Adj Close'][crypto_ticker]
    #1month returns
    single_crypto_df['mkt_1m'] = single_crypto_df['Adj Close'].pct_change(21)
    #forward market returns
    single_crypto_df['fwd_mkt_ret_1m'] = single_crypto_df['mkt_1m'] .shift(periods=-21)

    return single_crypto_df
