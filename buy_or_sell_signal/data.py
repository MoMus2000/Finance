# https://www.kaggle.com/shtrausslearning/building-an-asset-trading-strategy/output#2.-Exploratory-Data-Analysis-
# Thanks for the great code !

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import sklearn
import pickle

def generate_data(ticker):
    data  = yf.Ticker(ticker)
    df = data.history(period="max")
    df = df[['Open', 'High', 'Low', 'Close']]
    return df


def create_target_variable(df):
    df['SMA1'] = df['Close'].rolling(window=10, min_periods=1, center=False).mean()
    df['SMA2'] = df['Close'].rolling(window=30, min_periods=1, center=False).mean()
    conditions  = [ df['SMA1']  > df['SMA2'] + df['SMA1']*(3/100) , df['SMA1']  < df['SMA2'] - df['SMA1']*(3/100), True]
    #buy, sell, hold
    choices     = [ 1, -1, 0 ]
    df['Signal'] = np.select(conditions, choices, default=np.nan)

    print(df['Signal'].value_counts())
    # df.to_csv("/Users/a./Desktop/prophet_stock_pipeline/buy_or_sell_signal/est.csv")
    # df['Signal'] = np.where(df['SMA1'] > df['SMA2'],  1.0, 0.0)
    return df


def moving_average(df, n):
    return pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name=f'MA_{n}')

def exp_moving_average(df, n):
    return pd.Series(df['Close'].ewm(n, min_periods=n).mean(), name=f"EMA_{n}")

def price_momentum(df, n):
    return pd.Series(df.diff(n), name=f"Momentum_{n}")

def rate_of_change(df, n):
    l = df.shift(n - 1)
    m = df.diff(n - 1)
    return pd.Series((m/l)*100, name=f"ROC{n}")

def relative_strength_index(df, period):
    delta = df.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] )
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] )
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def stochastic_oscillator(close, low, high, n, ids):
    stok = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    if ids == 0:
        return stok
    return stok.rolling(3).mean()


def bollinger_bands(df, period):
    sma = df['Close'].rolling(window=period).mean()

    # calculate the standar deviation
    rstd = df['Close'].rolling(window=period).std()

    upper_band = sma + 2 * rstd
    lower_band = sma - 2 * rstd

    return upper_band, lower_band


def average_true_rating(df, period = 14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    atr = true_range.rolling(period).sum()/14

    return atr



def average_directional_index(df, period=14):
    """
    Computes the ADX indicator.
    """
    df = df.copy()
    alpha = 1/period
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM']/df['ATR'])*100
    df['-DMI'] = (df['S-DM']/df['ATR'])*100
    del df['S+DM'], df['S-DM']
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']

    return df['ADX']

def drop_orignal_feature(df):
    return df.drop(['Open', 'High', 'Close', 'Low'])


def time_series_split(df, split_id = None, test_id=False, cut_id=None):
    if cut_id != None:
        df = df.iloc[-cut_id:]
        t1 = df.index.max()
        t0 = df.index.min()

    if split_id != None:
        train_df, pred_df = train_test_split(df, test_size=split_id, shuffle=False)

    return train_df, pred_df


def add_indicators(df):
    df['SMA1'] = df['Close'].rolling(window=10, min_periods=1, center=False).mean()
    df['SMA2'] = df['Close'].rolling(window=30, min_periods=1, center=False).mean()

    df['MA21'] = moving_average(df, 10)
    df['MA63'] = moving_average(df, 30)
    df['MA252'] = moving_average(df, 200)

    bands = bollinger_bands(df, 20)

    df['BUP'] = bands[0]
    df['BDO'] = bands[1]


    df['ADX14'] =  average_directional_index(df)
    df['ADX05'] =  average_directional_index(df, period=5)

    df['ATR10'] = average_true_rating(df, period=10)
    df['ATR20'] = average_true_rating(df, period=20)

    df['EMA10'] = exp_moving_average(df, 10)
    df['EMA30'] = exp_moving_average(df, 30)
    df['EMA20'] = exp_moving_average(df, 200)

    df['MOM10'] = price_momentum(df['Close'], 10)
    df['MOM30'] = price_momentum(df['Close'], 30)

    df['RSI10'] = relative_strength_index(df['Close'], 10)
    df['RSI30'] = relative_strength_index(df['Close'], 30)
    df['RSI200'] = relative_strength_index(df['Close'], 200)

    #Slow stochastic oscillators
    df['%K10'] = stochastic_oscillator(df['Close'], df['Low'], df['High'], 5, 0)
    df['%K30'] = stochastic_oscillator(df['Close'], df['Low'], df['High'], 10, 0)
    df['%K200'] = stochastic_oscillator(df['Close'], df['Low'], df['High'], 20, 0)

    #Fast stochastic oscillators
    df['%D10'] = stochastic_oscillator(df['Close'], df['Low'], df['High'], 10, 1)
    df['%D30'] = stochastic_oscillator(df['Close'], df['Low'], df['High'], 30, 1)
    df['%D200'] = stochastic_oscillator(df['Close'], df['Low'], df['High'], 200, 1)

    return df

def drop_useless_feats(df):
    df = df.dropna() 
    return df.drop(['High', 'Low', 'Open'], axis=1)

def evaluate(train, test):
    y_train = train['Signal']
    y_test = test['Signal']
    X_train = train.loc[:, train.columns != 'Signal']
    X_test = test.loc[:, test.columns != 'Signal']

    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = pd.concat([y_train, y_test], axis=0)

    print("Training ...")

    start = time.time()

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    res = model.fit(X_train, y_train)

    end = time.time()

    print(f"Completed in {end - start} s ...")

    print(sklearn.metrics.accuracy_score(res.predict(X_test), y_test))
    data_to_predict = list(X_test.iloc[-1])
    print(model.predict([data_to_predict]))






if __name__ == "__main__":
    df = generate_data("IBM")
    df_tr, df_te = time_series_split(df, split_id = 0.2)
    df_tr1 = df_tr.copy()
    df_te1 = df_te.copy()

    df_tr1 = create_target_variable(df_tr1)
    df_te1 = create_target_variable(df_te1)

    df_tr2 = df_tr1.copy()
    df_te2 = df_te1.copy()

    df_tr2 = add_indicators(df_tr2)
    df_te2 = add_indicators(df_te2)

    df_tr2 = drop_useless_feats(df_tr2)
    df_te2 = drop_useless_feats(df_te2)

    evaluate(df_tr2, df_te2)



