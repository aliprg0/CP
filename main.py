import tensorflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tvDatafeed import TvDatafeed,Interval
import yfinance as yf
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
    data = yf.download(symbol, period=period, interval=timeframe)
    raw_data = process_for_prediction(data,index)
    for model in models_reverse:
      yresults.append(
          f"{models_reverse.get(model)} : {model.predict(np.array(raw_data).reshape(1,-1))}")

def make_prediction_for_tv(symbol, exchange, timeframe,tindex):
   tv = TvDatafeed()
   raw_data = process_for_prediction(tv.get_hist(symbol=symbol, exchange=exchange, interval=timeframe,n_bars=33),tindex)
   for model in models_reverse:
      tresults.append(
          f"{models_reverse.get(model)} : {model.predict(np.array(raw_data).reshape(1,-1))}")


def make_prediction(ysymbol, period, timeframe, tsymbol, texchange, ttimeframe, tindex, index, tsymbol2):
    make_prediction_for_yf(ysymbol, period, timeframe,index)
    try:
       make_prediction_for_tv(tsymbol, texchange, ttimeframe,tindex)
    except:
       make_prediction_for_tv(tsymbol2, texchange, ttimeframe, tindex)

def interval_handler(timeframe):
    if timeframe == "15m":
        return Interval.in_15_minute
    elif timeframe == "1h":
        return Interval.in_1_hour
    elif timeframe == "1d":
        return Interval.in_daily
    elif timeframe == "1wk":
        return Interval.in_weekly
    elif timeframe == "1mo":
        return Interval.in_monthly

def period_handler(period):
    if period == "15m":
        return "10d"
    elif period == "1h":
        return "10d"
    elif period=="1d":
        return "50d"
    elif period=="1wk":
        return "max"
    elif period=="1mo":
        return "max"
    
def symbol_handler(symbol):
    return f"{symbol}-usd", f"{symbol}usd", f"{symbol}usdt"

def yf_data_handler(symbol, period, timeframe):
    data = yf.download(symbol, period=period, interval=timeframe)
    print(data.tail( ))
    index_c = data.index.values
    index = index_c[-3:]
    
    if timeframe == "15m":
        mins = []
        for i in index:
         mins.append( str(i).split(":")[1])
        for i in mins:
            if int(i) % 15 != 0:
                return -2
        return -1
    if timeframe == "1h":
        if str(index[-1]).strip().split(":")[0][-2:] == str(index[-2]).strip().split(":")[0][-2:]:
            return -2
        else :
            return -1
      
    if timeframe == "1d":
       return -1
    
    if timeframe == "1wk":
        print(str(index[-2]).strip().split("-")[-1][:2])
        if int(str(index[-2]).strip().split("-")[-1][:2]) - int(str(index[-1]).strip().split("-")[-1][:2]) != 7:
            return -2
        else:
            return -1

    if timeframe == "1mo":
        if int(str(index[-2]).split("-")[-2]) == int(str(index[-1]).split("-")[-2]):
            return -2
        else:
            return -1

      
def predict(symbol,exchange,timeframe,future_prediction=False):
    global yresults
    global tresults
    global models_reverse

    load_models()
    yresults = []
    tresults = []


    models_reverse = {v: k for k, v in models.items()}
    ttimeframe = interval_handler(timeframe)
    period = period_handler(timeframe)
    ysymbol,tsymbol,tsymbol2 = symbol_handler(symbol)

    
    tindex = -1
    index = yf_data_handler(ysymbol, period, timeframe)
    
    if future_prediction:
        tindex = 0
        index = index + 1

    make_prediction(ysymbol, period, timeframe,tsymbol,exchange,ttimeframe,tindex,index,tsymbol2)
    return yresults,tresults


yresults, tresults = predict("btc", "binance", "1mo", False)
print(yresults)
print(tresults)

    
