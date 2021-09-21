import sqlite3
from sqlite3 import Error
import os

def create_connection_to_db():
    try:
        conn = sqlite3.connect(f'{os.getcwd()}/db/algo.db')
    except:
        print("Error connecting to db")

    return conn


def create_table(name):
    conn = create_connection_to_db()

    query = f"""
        CREATE TABLE IF NOT EXISTS {name}(
            id integer PRIMARY KEY AUTOINCREMENT,
            name text NOT NULL,
            begin_date text,
            y_hat integer NOT NULL,
            y_hat_up integer NOT NULL,
            y_hat_low integer NOT NULL

        );
        """

    try:
        c = conn.cursor()
        c.execute(query)
    except Error as e:
        print(e)

    return

if __name__ == "__main__":
    create_table("FB")


