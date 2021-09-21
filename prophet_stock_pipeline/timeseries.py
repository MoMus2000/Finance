from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from multiprocessing import Pool, cpu_count
import pytz
from utils import create_connection_to_db, create_table

class TimeSeries:
	def __init__(self, data, name):
		self.data = data
		self.model = Prophet()
		self.model.fit(self.data)
		self.name = name


	def predict(self, time_period, plot = False, to_db = False, to_csv = False):
		future = self.model.make_future_dataframe(periods=time_period, freq='D')
		print(future.tail())
		forecast = self.model.predict(future)

		if plot:
			self.model.plot(forecast)
			plt.show()

		if to_db:
			self.write_to_db(forecast)

		if to_csv:
			forecast.to_csv(f"{os.getcwd()}/forecast_{self.name}_{time.time()}.csv")

	def write_to_db(self, forecast_df):
		conn = create_connection_to_db()
		last_row = forecast_df.iloc[-1]
		name = self.name
		date = last_row['ds']
		y_hat = last_row['yhat']
		y_hat_upper = last_row['yhat_upper']
		y_hat_lower = last_row['yhat_lower']
		create_table(name)



		query = f"""
		INSERT INTO {name} VALUES (?, ?, ?, ?, ?, ?)
		"""

		c = conn.cursor()

		try:
			c.execute(query, (None, str(name), str(date), y_hat, y_hat_upper, y_hat_lower))
		except Exception as e:
			print(e)

		conn.commit()

		return


if __name__ == "__main__":
	pass


