import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import copy
import heapq
import random

class MarkovitzModel:
    def __init__(self, period, interval, stonks = None):
        if stonks == None:
            raise ValueError("Please provide a list of stocks")

        self.period = period
        self.interval = interval

        
        data_frame = yf.download(stonks, period = self.period, interval = self.interval, group_by='ticker', threads=True)
        self.data_frame = pd.DataFrame()
        
        for titles in stonks:
            self.data_frame[titles] = data_frame[titles]['Adj Close']

        self.stocks = [stock for stock in self.data_frame.columns]
        self.calculate_returns()


    def plot_data(self):
        self.data_frame.plot(figsize=(10,5))    
        plt.show()

    def calculate_returns(self):
        data = copy.deepcopy(self.data_frame)
        self.returns = np.log(data/data.shift(1))

    def plot_daily_returns(self):
        if self.returns == None:
            self.calculate_returns()
        self.returns.plot(figsize=(10,5))
        plt.show()

    def show_stats(self):
        if self.returns == None:
            self.calculate_returns()
        days = self.returns.shape[0]
        days = 252
        print(days)
        print(self.returns.mean()*days)
        print(self.returns.cov()*days)

    def _initialize_weights(self):
        weights = np.random.random(len(self.stocks))
        weights /= np.sum(weights)
        self.weights = weights

    def _calculate_portfolio_returns(self):
        portfolio_return = np.sum(self.returns.mean()*self.weights)*252
        print("Expected portfolio returns:", portfolio_return)

    def _calculate_portfolio_variance(self):
        portfolio_varience= np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
        print("Expected variance:", portfolio_varience)


    def stats(self, weights, returns):
        portfolio_return = np.dot(returns.mean()*252,weights.T)
        portfolio_volatility = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
        sharpe_ratio = portfolio_return/portfolio_volatility
        return np.array([portfolio_return,portfolio_volatility,sharpe_ratio])

    def minimizer(self, weights, returns):
        return -self.stats(weights,returns)[2]

    def optimizer(self):
        constraints=({'type':'eq','fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for x in range(len(self.stocks)))
        optimum = optimize.minimize(fun=self.minimizer,x0=self.weights,args=self.returns,
            method='SLSQP',bounds=bounds,constraints=constraints, options={'disp':True})
        return optimum


    def print_optimum_values(self, optimum):
        weights = optimum['x'].round(3)
        heap = []
        for i in range(0, len(self.stocks)):
            heap.append((weights[i], self.stocks[i]))

        largests = heapq.nlargest(20, heap)

        i = 0
        while(largests[i][0] != 0):
            print(f"{largests[i][1]}  {largests[i][0]}")
            i+=1

        print("expected return, volatility and Sharpe Ratio:",self.stats(weights,self.returns))

    def run(self):
        self._initialize_weights()
        optimizer = self.optimizer()
        self.print_optimum_values(optimizer)




if __name__ == "__main__":
    with open("/Users/a./Desktop/markovitz_model/stocks.txt", 'r') as f:
        stocks = f.read().splitlines()

    stocks = stocks[0:30]
    markovitz = MarkovitzModel("5y", "1d", stocks)
    markovitz.run()
