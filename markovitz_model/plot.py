import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


CAPITAL = 10000

def backtest(df):
    first_values = []
    last_values = []
    for titles in df.columns:
        first_values.append(df[titles].iloc[0])
        # print(first_values)
        last_values.append(df[titles].iloc[len(df)-1])
        # print(last_values)

    profit = []
    for i in range(0, len(first_values)):
        profit.append((last_values[i] - first_values[i])/first_values[i])

    return first_values, last_values, profit



if __name__ == "__main__":
    with open("/Users/a./Desktop/markovitz_model/backtest.txt", 'r') as f:
        stocks = f.read().splitlines()

    df = pd.DataFrame()

    columns1 = []
    columns2 = []
    INITIAL_PORTFOLIO = []
    for val in stocks:
        tup = val.split(" ")
        name = tup[0]
        price = tup[2]
        columns1.append(name)
        columns2.append(price)
        INITIAL_PORTFOLIO.append(CAPITAL*float(price))



    data_frame = yf.download(columns1, period = "5y", interval = "1d", group_by='ticker', threads=True)
    data = pd.DataFrame()
        
    for titles in columns1:
        data[titles] = data_frame[titles]['Adj Close']

    data.fillna(method='backfill', inplace=True)

    data.plot()
    plt.show()

    price, sold, profit = backtest(data)

    total_profit = 0

    for i in range(0, len(price)):
        total_profit += INITIAL_PORTFOLIO[i]*profit[i] - price[i]
        print(f"{columns1[i]} bought at {price[i]} & sold at {sold[i]} & with final profit of {INITIAL_PORTFOLIO[i]*profit[i] - price[i]}")


    print(f"For a total profit of {total_profit}")



