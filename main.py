import tensorflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tvDatafeed import TvDatafeed,Interval
import yfinance as yf
from tensorflow.keras.models import load_model
import os
from datetime import datetime

def load_models(timeframe):
    global models
    models = {}
    if timeframe == "all":
        for root, dirs, files in os.walk("./models"):
            for file in files:
                if file.endswith(".h5"):
                    models[file] = load_model(os.path.join(root, file))
    else:
        for root, dirs, files in os.walk(f"./models/{timeframe}"):
            for file in files:
                if file.endswith(".h5"):
                    models[file] = load_model(os.path.join(root, file))
    
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
    data = yf.download(symbol, period=period, interval=timeframe,progress=False)
    raw_data = process_for_prediction(data,index)
    for model in models_reverse:
      prediction = model.predict(np.array(raw_data).reshape(1, -1))
      yresults.append(
          f"{models_reverse.get(model)} : {prediction}")
      y_only_num.append(prediction)

def make_prediction_for_tv(symbol, exchange, timeframe,tindex):
   tv = TvDatafeed()
   raw_data = process_for_prediction(tv.get_hist(symbol=symbol, exchange=exchange, interval=timeframe,n_bars=33),tindex)
   for model in models_reverse:
      prediction = model.predict(np.array(raw_data).reshape(1, -1))
      tresults.append(
          f"{models_reverse.get(model)} : {prediction}")
      t_only_num.append(prediction)

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
    data = yf.download(symbol, period=period, interval=timeframe,progress=False)
    index_c = data.index.values
    index = index_c[-3:]
    if timeframe == "15m":
        mins = []
        for i in index:
         mins.append( str(i).split(":")[1])
        for i in mins:
            if int(i) % 15 != 0:
                return -2,index
        return -1,index
    if timeframe == "1h":
        if str(index[-1]).strip().split(":")[0][-2:] == str(index[-2]).strip().split(":")[0][-2:]:
            return -2,index
        else :
            return -1,index
    if timeframe == "1d":
       return -1,index
    if timeframe == "1wk":
        print(str(index[-2]).strip().split("-")[-1][:2])
        if int(str(index[-2]).strip().split("-")[-1][:2]) - int(str(index[-1]).strip().split("-")[-1][:2]) != 7:
            return -2,index
        else:
            return -1,index
    if timeframe == "1mo":
        if int(str(index[-2]).split("-")[-2]) == int(str(index[-1]).split("-")[-2]):
            return -2,index
        else:
            return -1,index

def predict(symbol,timeframe,future_prediction=False,using_all_models=False):
    global yresults
    global tresults
    global models_reverse
    global y_only_num
    global t_only_num
    if using_all_models:
        load_models("all")
    else:
        load_models(timeframe)
    yresults = []
    tresults = []
    y_only_num = []
    t_only_num = []
    exchange = "binance"
    models_reverse = {v: k for k, v in models.items()}
    ttimeframe = interval_handler(timeframe)
    period = period_handler(timeframe)
    ysymbol,tsymbol,tsymbol2 = symbol_handler(symbol)
    tindex = -1
    index,last_three = yf_data_handler(ysymbol, period, timeframe)
    if future_prediction:
        tindex = 0
        index = index + 1
    make_prediction(ysymbol, period, timeframe,tsymbol,exchange,ttimeframe,tindex,index,tsymbol2)
    y_buy = 0
    y_sell = 0
    for i in y_only_num:
       if list(i)[0][0] > list(i)[0][1]:
          y_buy += 1
       else:
            y_sell += 1
    t_buy = 0
    t_sell = 0
    for i in t_only_num:
       if list(i)[0][0] > list(i)[0][1]:
          t_buy += 1
       else:
           t_sell += 1
    all_buy = y_buy + t_buy
    all_sell = y_sell + t_sell
    plus = all_buy - all_sell
    if y_buy > y_sell:
        suggestion = "BUY"
    else:
        suggestion = "SELL"

    info = f'''
        YF index: {index}
        TV index: {tindex}
        Current_utc_time: {str(datetime.utcnow())}
        Last_three: {last_three}

        Timeframe: {timeframe}
        Yf_buy: {y_buy}
        Yf_sell: {y_sell}

        Exchange: {exchange}
        Number of predictions: {len(yresults) + len(tresults)}
        Yf_prediction: {yresults}
        Tv_prediction: {tresults}

        Buys: {all_buy}
        Sells: {all_sell}
        All: {plus}
        Suggestion: {suggestion}'''
    return info

info = predict("btc", "15m", future_prediction=False, using_all_models=True)
print(info)
    
