# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from pandas_datareader import data
# from matplotlib.ticker import FuncFormatter
# # render the figures in this notebook
# %matplotlib inline
tickers = ['BTC-USD']

start_date = '2014-01-01'
end_date = '2020-12-31'# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from pandas_datareader import data
# from matplotlib.ticker import FuncFormatter
# # render the figures in this notebook
# %matplotlib inline
tickers = ['BTC-USD']

start_date = '2014-01-01'
end_date = '2020-12-31'

# Use pandas_reader.data.DataReader to load data
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

close_price = panel_data['Close']

close_price.plot(figsize=(16,9), title = 'BTC-USD 2020')

# Use pandas_reader.data.DataReader to load data
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

close_price = panel_data['Close']

close_price.plot(figsize=(16,9), title = 'BTC-USD 2020')