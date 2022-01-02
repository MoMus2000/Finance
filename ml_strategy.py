from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def download_data(stocks, period, intercept):
  data_frame = yf.download(stocks, period=period, intercept=intercept, \
                           group_by='ticker')
  data = pd.DataFrame()
  
  for title in stocks:
    if len(stocks) == 1:
      data[title] = data_frame['Adj Close']
    else:
      data[title] = data_frame[title]['Adj Close']
  
  return data

num_of_stocks = 300

def calculate_return(stock):
  returns = np.log(stock/stock.shift(1))
  return returns

def calculate_portfolio_return(weights, returns):
  return np.dot(weights.T, returns.mean()*252)

def calculate_portfolio_variance(weights, returns):
  return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

def initialize_weights():
  weights = np.random.random(num_of_stocks)
  weights /= np.sum(weights)
  return weights

def generate_portfolios(weights, returns):
  port_returns = []
  port_variances = []

  for i in range(1000):
    weights = np.random.random(num_of_stocks)
    weights /= np.sum(weights)
    port_returns.append(np.sum(returns.mean()*weights)*252)
    port_variances.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() \
                                                           * 252, weights))))
  
  port_returns = np.array(port_returns)
  port_variances = np.array(port_variances)

  return port_returns, port_variances

def calculate_shape_ratio(weights, returns):
  r = calculate_portfolio_return(weights, returns)
  v = calculate_portfolio_variance(weights, returns)
  return r/v


def random_forest_strategy(data, index="SPY"):
    data.dropna(axis=0, how='any', inplace=True)
    data.dropna(axis=1, how='any', inplace=True)

    returns = pd.DataFrame()
    for name in data.columns:
        returns[name] = np.log(data[name]).diff()

    returns[index] = returns[index].shift(-1)

    split = 200

    train = returns.iloc[1:len(returns)-split]
    test = returns.iloc[len(returns)-split:-1]

    stock_names = list(data.columns)[:-1]

    x_train = train[stock_names]
    y_train = train[index]

    x_test = test[stock_names]
    y_test = test[index]

    c_train = (y_train > 0)
    c_test = (y_test > 0)

    model = RandomForestClassifier(n_estimators=100)

    model.fit(x_train, c_train)

    p_x_train = model.predict(x_train)
    p_x_test = model.predict(x_test)

    returns['Position'] = 0

    returns.loc[1:len(returns)-split, "Position"] = p_x_train
    returns.loc[len(returns)-split:-1, "Position"] = p_x_test

    returns['Algo Return'] = returns['Position'] * returns[index]

    print(f'Algo Return on train set: {returns.iloc[1:len(returns)-split]["Algo Return"].sum()}')

    print(f'Algo Return on test set: {returns.iloc[len(returns)-split:-1]["Algo Return"].sum()}')

    print(f'Buy and hold on train: {y_train.sum()}')

    print(f'Buy and hold on test: {y_test.sum()}')

    return returns.iloc[len(returns)-split:-1]["Algo Return"].sum()

def optimize_portfolio(weights, returns):
  constraints=({'type':'eq', 'fun': lambda x: np.sum(x)-1})
               # Choose 0.05 of the first stock
              #  {'type':'eq', 'fun': lambda x: x[0]-0.05})
  bounds = tuple((0, 1) for x in range(num_of_stocks))
  optimum = optimize.minimize(fun=calculate_shape_ratio, x0=weights, \
                              args=returns \
                              , method='SLSQP',bounds=bounds, \
                              constraints=constraints)
  return optimum


def print_results(stock_data, optimum, returns):
  optimum = optimum['x']
  print(optimum)
  for idx, opt in enumerate(optimum):
    if opt.round(3) > 0:
      print(stock_data[idx], opt.round(3))
  
  return_val = calculate_portfolio_return(weights, returns)
  total_risk = calculate_portfolio_variance(weights, returns)
  print(f"Total return: {return_val}")
  print(f"Total risk: {total_risk}")
  print(f"Sharpe ratio {return_val/total_risk}")

  return return_val, total_risk

def run_optimization():
    with open("/content/stocks.txt", "r") as f:
        stock_data = f.read().splitlines()

    data = download_data(stock_data, "1y", "1d")

    weights = initialize_weights()
    returns = calculate_return(data)

    v = calculate_portfolio_variance(weights, returns)
    r = calculate_portfolio_return(weights, returns)

    sharpe = calculate_shape_ratio(weights, returns)

    optimum = optimize_portfolio(weights, returns)

    ret, risk = print_results(stock_data, optimum, returns)

if __name__ == "__main__":
    stocks = download_data(["AAPL", "GOOG", "FB", "TSLA", "MSFT", "SPY"], "1y", "1d")
    ret = []
    for i in range(0, 500):
        predicted_return = random_forest_strategy(stocks)
        ret.append(predicted_return)
    ret = np.array(ret)

    low = ret.mean() - 1.96 * ret.std() / np.sqrt(len(ret))
    high = ret.mean() + 1.96 * ret.std() / np.sqrt(len(ret))

    print(f"Expected Return {ret.mean()} between {low} and {high}")

    print(f"Expected Risk {ret.std()}")

    plt.plot(ret)
    plt.show()