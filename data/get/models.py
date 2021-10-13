import sys
sys.path.append("data/get")
from yahoo import get_yahoo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#LR model for BTC tweets and prices

dropped_df = df.drop(['Date','Unnamed: 0'],axis=1)
X_btc = dropped_df['Decred - BTC Tweets']
y_btc = dropped_df['BTC-USD']

X_train, X_test, y_train, y_test = train_test_split(X_btc, y_btc, test_size=0.20, random_state=4)
#scaling
scaler = MinMaxScaler()
normalised_X_train = scaler.fit_transform(np.array(X_train).reshape(-1,1) )
normalised_X_test = scaler.fit_transform(np.array(X_test).reshape(-1,1) )
normalised_y_train=scaler.fit_transform(np.array(y_train).reshape(-1,1) )
normalised_y_test=scaler.fit_transform(np.array(y_test).reshape(-1,1) )
model = LinearRegression()
LR = model.fit(normalised_X_train,normalised_y_train)
y_pred = LR.predict(normalised_X_test)
r2 = r2_score(y_pred,normalised_y_test)




#LR model for ETH tweets and prices
dropped_df = df.drop(['Date','Unnamed: 0'],axis=1)
X_eth = dropped_df['Decred - ETH Tweets']
y_eth = dropped_df['ETH-USD']

X_train, X_test, y_train, y_test = train_test_split(X_eth, y_eth, test_size=0.20, random_state=4)
#scaling
scaler = MinMaxScaler()
normalised_X_train = scaler.fit_transform(np.array(X_train).reshape(-1,1) )
normalised_X_test = scaler.fit_transform(np.array(X_test).reshape(-1,1) )
normalised_y_train=scaler.fit_transform(np.array(y_train).reshape(-1,1) )
normalised_y_test=scaler.fit_transform(np.array(y_test).reshape(-1,1) )
#LR model
model = LinearRegression()
LR = model.fit(normalised_X_train,normalised_y_train)
y_pred = LR.predict(normalised_X_test)
r2 = r2_score(y_pred,normalised_y_test)
