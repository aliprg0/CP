import tensorflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tvDatafeed import TvDatafeed,Interval
import yfinance as yf

# import load_model from tensorflow
from tensorflow.keras.models import load_model
import os

def load_models():
    global models
    models = {}
    for file in os.listdir("models"):
        if file.endswith(".h5"):
            models[file.split(".")[0]] = load_model(f"models/{file}")
    

def scaler(row):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    row = scaler.fit_transform(row)
    return row


def process_for_prediction(data, index):
        i = index
        if "symbol" in data.columns:
              data.drop("symbol", axis=1, inplace=True)
        if "datetime" in data.columns:
              data.drop("datetime", axis=1, inplace=True)
        if "Adj Close" in data.columns:
              data.drop("Adj Close", axis=1, inplace=True)

        data = np.array(data)
        grow = []
        ggggrow = []
        for x in range(1, 31):
            grow.append([data[i-x][0] - data[i-(1+x)][0]])
            ggggrow.append([data[i-x][3] - data[i-(1+x)][3]])
        arr = np.array(grow).flatten()
        arr4 = np.array(ggggrow).flatten()
        arr = np.concatenate((arr, arr4), axis=0).reshape(-1, 1)
        arr = scaler(arr.reshape(-1, 1))
        return arr

def make_prediction_for_yf(symbol, period, timeframe,index):
    raw_data = process_for_prediction(yf.download(symbol, period=period, interval=timeframe),index)
    for model in models_reverse:
      results.append(
          f"YF : {models_reverse.get(model)} : {model.predict(np.array(raw_data).reshape(1,-1))}")

def make_prediction_for_tv(symbol, exchange, timeframe,tindex):
   tv = TvDatafeed()
   raw_data = process_for_prediction(tv.get_hist(symbol=symbol, exchange=exchange, interval=timeframe,n_bars=100),tindex)
   for model in models_reverse:
      results.append(
          f"TVB : {models_reverse.get(model)} : {model.predict(np.array(raw_data).reshape(1,-1))}")

def make_prediction(ysymbol, period, timeframe,tsymbol,texchange,ttimeframe,tindex,index):
    results.append(make_prediction_for_yf(ysymbol, period, timeframe,index))
    results.append(make_prediction_for_tv(tsymbol, texchange, ttimeframe,tindex))


if __name__ == "__main__":
    global results
    results = []
    load_models()
    global models_reverse
    models_reverse = {v: k for k, v in models.items()}
    make_prediction("btc-usd", "max", "1d",
     "btcusdt", "binance", Interval.in_daily, -1, -1)
    print(results)
