"""
Backtesting model to see if it works
"""

import pandas as pd
import numpy as np
import time
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from data import generate_data, time_series_split, create_target_variable, add_indicators, drop_useless_feats

def evaluate(train, test):

    y_train = train['Signal']
    y_test = test['Signal']
    X_train = train.loc[:, train.columns != 'Signal']
    X_test = test.loc[:, test.columns != 'Signal']

    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = pd.concat([y_train, y_test], axis=0)

    print("Training ...")

    start = time.time()

    model = RandomForestClassifier(n_estimators=800, n_jobs=-1)

    # model = best_params(model, X_train, y_train)

    res = model.fit(X_train, y_train)

    end = time.time()

    print(f"Completed in {end - start} s ...")

    back_test(model, X_train, X_test, y_test, X_combined, y_combined, len(X_train))

    print(sklearn.metrics.accuracy_score(res.predict(X_test), y_test))



def best_params(model, X_train, y_train):
        # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)

    return rf_random.best_estimator_


def back_test(model, X_train, X_test, y_test, X_combined, y_combined, start):
    start_off = X_combined.iloc[len(X_train)-1]

    profit = 0

    buy_price = start_off['Close']

    error = 0

    cool_down = False
    cool_down_index = 0

    for i in range(1, len(X_test)):

        if i - cool_down_index < 10 and cool_down == True:
            print("cooling down")
            continue

        cool_down = False

        today = X_test.iloc[i]
        decision = model.predict([list(today)])
        
        if decision[0] == 1:
            print("Price going up")
            buy_price = min(buy_price, today['Close'])

        if decision[0] == 0:
            print("Holding")

        if decision[0] == -1:
            print("Price falling")
            if buy_price != float('inf'):
                profit += today['Close'] - buy_price
                cool_down = True
                cool_down_index = i
                buy_price = float('inf')




    print(f"total profit {profit}")



if __name__ == "__main__":
    #Backtesting on a 15day window

    df = generate_data("GOOGL")
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
