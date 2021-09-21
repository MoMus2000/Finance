import yfinance as yf
from timeseries import TimeSeries
import pandas as pd
from multiprocessing import Pool, cpu_count



def generate_stock_dataframe(ticker):
		data  = yf.Ticker(ticker)
		df = data.history(period="6mo")

		print(df)

		train_frame = pd.DataFrame()

		train_frame['y'] = df['Close']
		train_frame.reset_index(level=0, inplace=True)
		try:
			train_frame.rename(columns={"Date":"ds"}, inplace=True)
		except:
			train_frame.rename(columns={"Date":"ds"}, inplace=True)

		train_frame['ds'] = pd.to_datetime(train_frame['ds'])

		train_frame['y'] = train_frame['y'].astype('float')

		return train_frame

def generate_predictions(data, stock_name):
		ts = TimeSeries(data, stock_name)
		days_ahead = 2
		ts.predict(days_ahead, to_csv = False, plot=True, to_db = False)


def train_model(portfolio):

	dfs = []

	for stock in portfolio:
		data = generate_stock_dataframe(stock)

		dfs.append((data, stock))

	print("starting training ... ")

	print(f"total cpus {cpu_count()}")

	pool = Pool(cpu_count())

	res = pool.starmap_async(generate_predictions, dfs)

	res.get()

	pool.close()

	pool.join()

if __name__ == "__main__":
	train_model(portfolio = ["AAPL"])