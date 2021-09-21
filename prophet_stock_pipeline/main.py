import yfinance as yf
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot
import sqlite3
from sqlite3 import Error


def create_connection():
    """ create a database connection to a database that resides
        in the memory
    """
    conn = None;
    try:
        conn = sqlite3.connect('/Users/a./Desktop/learning-prophet/db/algo.db')
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

    return conn

query = """
CREATE TABLE IF NOT EXISTS fbprophet_preds(
    id integer PRIMARY KEY,
    name text NOT NULL,
    begin_date text,
    y_hat integer NOT NULL,
    y_hat_up integer NOT NULL,
    y_hat_low integer NOT NULL

);
"""

def create_table(query):
    conn = sqlite3.connect('/Users/a./Desktop/learning-prophet/db/algo.db')
    try:
        c = conn.cursor()
        c.execute(query)
    except Error as e:
        print(e)


if __name__ == "__main__":
    

    # conn = create_connection()
    create_table(query)
    # data  = yf.Ticker("AAPL")
    # df = data.history(period = "1mo", interval = "30m")

    # print(df)
    # AAPL = df[['Close']]

    # print(AAPL)

    # train_frame = pd.DataFrame()

    # train_frame['y'] = AAPL['Close']
    # train_frame.reset_index(level=0, inplace=True)
    # try:
    #   train_frame.rename(columns={"Datetime":"ds"}, inplace=True)
    # except:
    #   train_frame.rename(columns={"Date":"ds"}, inplace=True)

    # train_frame['ds'] = pd.to_datetime(train_frame['ds'])
    # train_frame['ds'] = train_frame['ds'].dt.tz_convert(None)

    # train_frame['y'] = train_frame['y'].astype('float')

    # m = Prophet()

    # m.fit(train_frame)

    # future = m.make_future_dataframe(periods=2)

    # print(future.tail())

    # a()

    # forecast = m.predict(future)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # m.plot(forecast)
    # pyplot.show()
